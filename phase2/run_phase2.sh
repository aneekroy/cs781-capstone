#!/usr/bin/env bash
# CS781 Phase 2 launcher for your 2x RTX 3090 server.
#
# Copy phase2/ (and this script) into $HOME/cs781_phase1 next to phase1/, then:
#   bash run_phase2.sh access   # check which HF account your token is and whether it can read the dataset
#   bash run_phase2.sh fetch    # download Phase_2_LNER (asks once for a token; --reset-token to change it)
#   bash run_phase2.sh check    # list checkpoints, verify one shared tokenizer, data stats + alignment ceiling
#   bash run_phase2.sh round1   # E2 enc-dec full FT (KD teacher)  ||  E1 dense_lm_v2 + LoRMA+ (causal)
#   bash run_phase2.sh now      # E3 gqa LoRA + E1 dense LoRMA right now, next to a running E2
#   bash run_phase2.sh round2   # E3 gqa_local + LoRA (causal)     ||  E4 hybrid KD student (teacher = E2)
#   bash run_phase2.sh extras   # X1-X7 ablations, one variable each (list: bash run_phase2.sh list)
#   bash run_phase2.sh one NAME [GPU]   # any single experiment
#   bash run_phase2.sh adapt-data       # CPU: tokenize ~1B official tokens for continued pretraining
#   bash run_phase2.sh round3           # E4f (KD, leak fixed, teacher X1) || X9 no-KD baseline, then X10 || X8
#   bash run_phase2.sh adapt NAME [GPU] # continued pretraining: encdec_mlm | dense_clm | gqa_clm | hybrid_clm
#   bash run_phase2.sh adapted          # NER (A1-A4) on the adapted backbones
#   bash run_phase2.sh round4   # X20 O penalty || X17 header, then X18 augmentation || X19 header+augmentation
#   bash run_phase2.sh round5   # X21 dense: all + self-KD || X22 adapted gqa: header+augmentation
#   bash run_phase2.sh round6   # X23: X1 recipe + CV-tuned entity-entry bonus, 4 seeds
#   bash run_phase2.sh repost R..  # re-choose post-processing for finished runs with the current rules (CPU)
#   bash run_phase2.sh errors RUN [LABEL..] # real out-of-fold misses / false hits with context
#   bash run_phase2.sh round7   # X24 dense / X25 gqa trained on merged roles (PARTY, COUNSEL), sides resolved by rules
#   bash run_phase2.sh selfkd   # X14 X1->X1 self-distillation || X15 X1->adapted gqa (test only: KD CV is leaked)
#   bash run_phase2.sh seeds    # X1 x3 more seeds + A3 x1 more seed, for a seed ensemble
#   bash run_phase2.sh ensemble # average token log-probs (default: X1 E1 X2 E3 -> X11_ensemble_top4)
#   bash run_phase2.sh vote     # span-voting ensemble, members + threshold chosen on CV (CPU)
#   bash run_phase2.sh overlap  # how many NER documents appear in the Phase-1 corpus
#   bash run_phase2.sh summary  # CV table of every run so far
#
# Every knob below can be overridden from the environment, e.g. GPU_A=2 GPU_B=3 bash run_phase2.sh round1
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID          # GPU ids below = nvidia-smi ids (as in your Phase-1 notebooks)

PROJECT=${PROJECT:-$HOME/cs781_phase1}
CACHE=${CACHE:-$HOME/cs781_cache}
DATA=${DATA:-$CACHE/raw/Phase_2_LNER}
OUT=${OUT:-$PROJECT/runsP2}
GPU_A=${GPU_A:-0}
GPU_B=${GPU_B:-1}
PY=${PY:-python}

# ---- Phase-1 checkpoints (paths from your run1 / run2 directory listings) ----
RUNS=$PROJECT/runs
IMP=$PROJECT/improvements/dense_v2/runs
DENSE_ORIG=${DENSE_ORIG:-$RUNS/dense_gpt_full_395280384_s42_jupyter24gb_v1/best.pt}
GQA=${GQA:-$RUNS/gqa_local_full_57507840_s42_jupyter24gb_v1/best.pt}
ENCDEC=${ENCDEC:-$RUNS/encoder_decoder_full_428507136_s42_jupyter24gb_v1/best.pt}
HYBRID=${HYBRID:-$RUNS/recurrent_hybrid_full_78217216_s42_jupyter24gb_v1/best.pt}
DENSE_LM=${DENSE_LM:-$IMP/dense_lm_dense_v2_s42/best.pt}            # continued LM on cleaned SC/HC: dense backbone
DENSE_SUFFIX=${DENSE_SUFFIX:-$IMP/dense_suffix_dense_v2_s42/latest.pt}  # Phase-1 winner (suffix objective)
UNLABELED=${UNLABELED:-$CACHE/raw/Phase_1_Pretraining/Train/train_supreme_court.json}

# ---- training defaults (CV picks the epoch count; patience stops early) ----
WINDOW=${WINDOW:-512}
STRIDE=${STRIDE:-256}
BATCH=${BATCH:-8}
EPOCHS=${EPOCHS:-20}
PATIENCE=${PATIENCE:-6}
KD_LAYERS=${KD_LAYERS:-2}                    # 2 of 10 hybrid layers: 37.1M student vs 64.6M E2 teacher (57%)
TEACHER=${TEACHER:-$OUT/X1_denseorig_lorma/model.pt}   # best NER model so far (test 0.596, 109.8M params)

# ---- continued pretraining (DAPT/TAPT) before NER: more official tokens for under-trained backbones
RAW_TRAIN=${RAW_TRAIN:-$CACHE/raw/Phase_1_Pretraining/Train}
ADAPT_DATA=${ADAPT_DATA:-$OUT/adapt_data}
ADAPT_TOKENS=${ADAPT_TOKENS:-1000000000}     # unique tokens to tokenize (2 bytes each on disk)
ADAPT_HOURS=${ADAPT_HOURS:-10}               # wall-clock budget per backbone
ADAPTED=$OUT/adapt
TEST=$DATA/Test/fold3-text.json

# The repository root (phase1/ ... phase7/) comes first on PYTHONPATH; $PROJECT supplies runs and checkpoints.
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # repository root (this script lives in phase2/)
[[ -f $CODE/phase2/run.py ]] || { echo "phase2/ not found in $CODE"; exit 1; }
[[ -f $PROJECT/phase1/models.py ]] || { echo "phase1/ not found in PROJECT=$PROJECT (set PROJECT=...)"; exit 1; }
export PYTHONPATH="$CODE:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
cd "$CODE"                                   # python -m puts the cwd first: use this checkout
mkdir -p "$OUT"
COMMON=(--data "$DATA" --project-root "$PROJECT" --window "$WINDOW" --stride "$STRIDE"
        --batch-size "$BATCH" --epochs "$EPOCHS" --patience "$PATIENCE" --final)

need() { for f in "$@"; do [[ -f "$f" ]] || { echo "MISSING: $f"; exit 1; }; done; }

run() {  # run <gpu> <name> <phase2.run cv args...>   (skips a run that already has a submission)
  local gpu=$1 name=$2; shift 2
  if [[ -f "$OUT/$name/submission/submission.zip" ]]; then echo "[skip] $name already finished"; return 0; fi
  exec {lock}>"$OUT/$name.lock"                      # a second launch of the same run skips instead of colliding
  flock -n "$lock" || { echo "[busy] $name is already running elsewhere"; return 0; }
  echo "[GPU $gpu] $name  (log: $OUT/$name.log)"
  CUDA_VISIBLE_DEVICES=$gpu $PY -m phase2.run cv "${COMMON[@]}" --output "$OUT/$name" "$@" \
    > "$OUT/$name.log" 2>&1 || { echo "FAILED: $name — tail of log:"; tail -20 "$OUT/$name.log"; return 1; }
  grep -h "CV raw" "$OUT/$name.log" | tail -1 | sed "s/^/$name: /" | cut -c1-200 || true
}

# ---- experiment table: name -> arguments. Decoder-only models run --mode causal: they were
# trained left-to-right, and removing the mask without an adaptation stage gave F1 0.0 (E1, bidirectional,
# lr 1e-3). Right-hand context comes from the BiLSTM head instead.
LR_PEFT=${LR_PEFT:-3e-4}
HEADER=${HEADER:-160}        # document tokens prepended to every later window (parties + roles)
AUGMENT=${AUGMENT:-2}        # entity-replaced copies per training document
PEFT=(--rank 16 --alpha 32 --lr-peft "$LR_PEFT")
spec() {
  case $1 in
    E2_encdec_full)          ARGS=(--checkpoint "$ENCDEC" --method full) ;;
    E1_denselm_lorma_causal) ARGS=(--checkpoint "$DENSE_LM" --method lorma --mode causal "${PEFT[@]}") ;;
    E3_gqa_lora_causal)      ARGS=(--checkpoint "$GQA" --method lora --mode causal "${PEFT[@]}") ;;
    E4_hybrid_kd)            need "$TEACHER"
                             ARGS=(--checkpoint "$HYBRID" --method full --mode causal --keep-layers "$KD_LAYERS"
                                   --teacher "$TEACHER" --kd-unlabeled "$UNLABELED" --kd-alpha 0.5 --kd-temperature 2) ;;
    X1_denseorig_lorma)      ARGS=(--checkpoint "$DENSE_ORIG" --method lorma --mode causal "${PEFT[@]}") ;;   # run-2 continued LM: worth it?
    X2_densesuffix_lorma)    ARGS=(--checkpoint "$DENSE_SUFFIX" --method lorma --mode causal "${PEFT[@]}") ;; # suffix objective
    X3_denselm_lorma_bidir)  ARGS=(--checkpoint "$DENSE_LM" --method lorma --mode bidirectional "${PEFT[@]}") ;; # mask, same LR as E1
    X4_encdec_span)          ARGS=(--checkpoint "$ENCDEC" --method full --span --max-width 96) ;;          # span head vs BIOES-CRF
    X6_encdec_full_bpeseg)   ARGS=(--checkpoint "$ENCDEC" --method full --segment bpe) ;;                 # punct vs BPE segmentation
    X7_encdec_lorma)         ARGS=(--checkpoint "$ENCDEC" --method lorma "${PEFT[@]}") ;;                 # LoRMA vs full FT, same backbone
    X8_hybrid_kd4)           need "$TEACHER"                                                               # bigger student: 53.6M (83%)
                             ARGS=(--checkpoint "$HYBRID" --method full --mode causal --keep-layers 4
                                   --teacher "$TEACHER" --kd-unlabeled "$UNLABELED" --kd-alpha 0.5 --kd-temperature 2) ;;
    A2_encdec_mlm_full)      ARGS=(--checkpoint "$ADAPTED/encdec_mlm/best.pt" --method full) ;;
    A1_dense_clm_lorma)      ARGS=(--checkpoint "$ADAPTED/dense_clm/best.pt" --method lorma --mode causal "${PEFT[@]}") ;;
    A3_gqa_clm_lora)         ARGS=(--checkpoint "$ADAPTED/gqa_clm/best.pt" --method lora --mode causal "${PEFT[@]}") ;;
    A4_hybrid_clm_kd)        need "$TEACHER"
                             ARGS=(--checkpoint "$ADAPTED/hybrid_clm/best.pt" --method full --mode causal --keep-layers "$KD_LAYERS"
                                   --teacher "$TEACHER" --kd-unlabeled "$UNLABELED" --kd-alpha 0.5 --kd-temperature 2) ;;
    E4f_hybrid_kd)           need "$TEACHER"                       # E4 rerun: teacher X1, NER-overlapping KD text removed
                             ARGS=(--checkpoint "$HYBRID" --method full --mode causal --keep-layers "$KD_LAYERS"
                                   --teacher "$TEACHER" --kd-unlabeled "$UNLABELED" --kd-alpha 0.5 --kd-temperature 2) ;;
    X10_denseorig_lorma_bpeseg) ARGS=(--checkpoint "$DENSE_ORIG" --method lorma --mode causal "${PEFT[@]}" --segment bpe) ;;
    X14_dense_selfkd)        need "$TEACHER"   # self-distillation: X1 recipe + X1 as teacher on 3x more teacher-labelled SC text
                             ARGS=(--checkpoint "$DENSE_ORIG" --method lorma --mode causal "${PEFT[@]}"
                                   --teacher "$TEACHER" --kd-unlabeled "$UNLABELED" --kd-alpha 0.5 --kd-temperature 2
                                   --kd-unlabeled-docs 1000 --kd-unlabeled-windows 6000) ;;
    X15_gqa_clm_kd)          need "$TEACHER"   # adapted gqa (A3, 0.574) as a KD student of X1: 57.5M vs 109.8M
                             ARGS=(--checkpoint "$ADAPTED/gqa_clm/best.pt" --method lora --mode causal "${PEFT[@]}"
                                   --teacher "$TEACHER" --kd-unlabeled "$UNLABELED" --kd-alpha 0.5 --kd-temperature 2
                                   --kd-unlabeled-docs 1000 --kd-unlabeled-windows 6000) ;;
    # ---- round 4: one change each on the X1 recipe (header context / augmentation / O penalty)
    X20_denseorig_obias)     ARGS=(--checkpoint "$DENSE_ORIG" --method lorma --mode causal "${PEFT[@]}" --o-bias auto) ;;
    X17_denseorig_header)    ARGS=(--checkpoint "$DENSE_ORIG" --method lorma --mode causal "${PEFT[@]}" --o-bias auto
                                   --header-tokens "$HEADER") ;;
    X18_denseorig_aug)       ARGS=(--checkpoint "$DENSE_ORIG" --method lorma --mode causal "${PEFT[@]}" --o-bias auto
                                   --augment "$AUGMENT") ;;
    X19_denseorig_header_aug) ARGS=(--checkpoint "$DENSE_ORIG" --method lorma --mode causal "${PEFT[@]}" --o-bias auto
                                   --header-tokens "$HEADER" --augment "$AUGMENT") ;;
    # ---- round 5: everything on the two best backbones (+ self-distillation on dense)
    X21_dense_all_kd)        need "$TEACHER"
                             ARGS=(--checkpoint "$DENSE_ORIG" --method lorma --mode causal "${PEFT[@]}" --o-bias auto
                                   --header-tokens "$HEADER" --augment "$AUGMENT"
                                   --teacher "$TEACHER" --kd-unlabeled "$UNLABELED" --kd-alpha 0.5 --kd-temperature 2
                                   --kd-unlabeled-docs 1000 --kd-unlabeled-windows 6000) ;;
    X22_gqa_clm_header_aug)  ARGS=(--checkpoint "$ADAPTED/gqa_clm/best.pt" --method lora --mode causal "${PEFT[@]}"
                                   --o-bias auto --header-tokens "$HEADER" --augment "$AUGMENT") ;;
    # ---- round 6: the X1 recipe with the CV-tuned entity-entry bonus (X20 used the withdrawn 'O' penalty)
    X23_denseorig_entry)     ARGS=(--checkpoint "$DENSE_ORIG" --method lorma --mode causal "${PEFT[@]}" --o-bias auto) ;;
    # ---- round 7: merged roles (PARTY, COUNSEL) + rule-based side resolution
    X24_dense_coarse)        ARGS=(--checkpoint "$DENSE_ORIG" --method lorma --mode causal "${PEFT[@]}" --coarse-roles) ;;
    X25_gqa_clm_coarse)      ARGS=(--checkpoint "$ADAPTED/gqa_clm/best.pt" --method lora --mode causal "${PEFT[@]}" --coarse-roles) ;;
    X9_hybrid2_nokd)         ARGS=(--checkpoint "$HYBRID" --method full --mode causal --keep-layers "$KD_LAYERS") ;;  # E4 without KD
    *) echo "unknown experiment: $1"; experiments; exit 1 ;;
  esac
}
experiments() { echo "experiments: E2_encdec_full E1_denselm_lorma_causal E3_gqa_lora_causal E4_hybrid_kd" \
                     "X1_denseorig_lorma X2_densesuffix_lorma X3_denselm_lorma_bidir X4_encdec_span X6_encdec_full_bpeseg X7_encdec_lorma" \
                     "X8_hybrid_kd4 X9_hybrid2_nokd" \
                     "| any name + _s<N> = another seed (X1_denseorig_lorma_s7)" \
                     "| X14_dense_selfkd X15_gqa_clm_kd X17_denseorig_header X18_denseorig_aug X19_denseorig_header_aug" \
                     "X20_denseorig_obias X21_dense_all_kd X22_gqa_clm_header_aug X23_denseorig_entry X24_dense_coarse X25_gqa_clm_coarse" \
                     "| fixes: E4f_hybrid_kd X10_denseorig_lorma_bpeseg | after adapt: A2_encdec_mlm_full A1_dense_clm_lorma A3_gqa_clm_lora A4_hybrid_clm_kd"; }
one() {  # any experiment name + optional _s<N> suffix = same recipe, another seed (X1_denseorig_lorma_s7)
  local base=$1 seed=()
  if [[ $1 =~ ^(.+)_s([0-9]+)$ ]]; then base=${BASH_REMATCH[1]}; seed=(--seed "${BASH_REMATCH[2]}"); fi
  spec "$base"; need "${ARGS[1]}"; run "$2" "$1" "${ARGS[@]}" ${seed[@]+"${seed[@]}"}
}

adapt_spec() {   # backbone name -> parent checkpoint and objective
  case $1 in
    encdec_mlm)  PARENT=$ENCDEC;   OBJECTIVE=mlm ;;
    dense_clm)   PARENT=$DENSE_ORIG; OBJECTIVE=clm ;;   # run-1 dense: the best NER starting point (X1)
    gqa_clm)     PARENT=$GQA;      OBJECTIVE=clm ;;
    hybrid_clm)  PARENT=$HYBRID;   OBJECTIVE=clm ;;
    *) echo "unknown backbone: $1 (encdec_mlm dense_clm gqa_clm hybrid_clm)"; exit 1 ;;
  esac
}

case "${1:-}" in
  access|fetch)  # token: $HF_TOKEN, else $HF_HOME/token (your own cs781_cache), else a hidden prompt
    export HF_HOME=${HF_HOME:-$CACHE/huggingface}   # never the machine-wide ~/.cache/huggingface token
    extra=(); [[ $1 == access ]] && extra+=(--check-only)
    $PY -m phase2.fetch --output "$CACHE/raw" --token-file "$HF_HOME/token" ${extra[@]+"${extra[@]}"} "${@:2}"
    ;;

  check)
    need "$DENSE_ORIG" "$GQA" "$ENCDEC" "$HYBRID" "$DENSE_LM" "$DENSE_SUFFIX"
    ref=$(sha256sum "$(dirname "$DENSE_ORIG")/tokenizer.json" | cut -c1-64)
    for ck in "$DENSE_ORIG" "$GQA" "$ENCDEC" "$HYBRID" "$DENSE_LM" "$DENSE_SUFFIX"; do
      h=$(sha256sum "$(dirname "$ck")/tokenizer.json" | cut -c1-64)
      [[ $h == "$ref" ]] && tag="same tokenizer" || tag="DIFFERENT TOKENIZER (no KD/ensemble with others)"
      printf '%-80s %s  %s\n' "${ck#$PROJECT/}" "$(du -h "$ck" | cut -f1)" "$tag"
    done
    need "$DATA/Train/labels.txt" "$DATA/Train/fold1-gold.json" "$TEST"
    $PY -m phase2.run stats --data "$DATA" --checkpoint "$DENSE_LM" --project-root "$PROJECT" --output "$OUT/stats.json"
    ;;

  round1)   # enc-dec (full FT, also the KD teacher) || dense (LoRMA)
    one E2_encdec_full "$GPU_A" & one E1_denselm_lorma_causal "$GPU_B" & wait
    ;;

  now)      # start the decoder-only runs immediately, alongside whatever is already running
    one E3_gqa_lora_causal "$GPU_A" & one E1_denselm_lorma_causal "$GPU_B" & wait
    ;;

  round2)   # gqa (LoRA) || hybrid KD student (needs E2's model.pt as teacher)
    one E3_gqa_lora_causal "$GPU_A" & one E4_hybrid_kd "$GPU_B" & wait
    ;;

  extras)   # one variable changed per run, for the report
    one X1_denseorig_lorma "$GPU_A" & one X2_densesuffix_lorma "$GPU_B" & wait
    one X3_denselm_lorma_bidir "$GPU_A" & one X4_encdec_span "$GPU_B" & wait
    one X6_encdec_full_bpeseg "$GPU_A" & one X7_encdec_lorma "$GPU_B" & wait
    one X9_hybrid2_nokd "$GPU_A" & one X8_hybrid_kd4 "$GPU_B" & wait         # KD effect and student size
    ;;

  one)      # bash run_phase2.sh one <experiment> [gpu]
    [[ -n ${2:-} ]] || { echo "usage: $0 one <experiment> [gpu]"; experiments; exit 1; }
    one "$2" "${3:-$GPU_A}"
    ;;

  list) experiments ;;

  adapt-data)   # CPU only: tokenize official Phase-1 text (+ NER training-fold text) with the NER segmentation
    raws=(); weights=()
    for pair in supreme_court:0.5 high_courts:0.3 acts:0.1 tribunals:0.1; do
      f=$RAW_TRAIN/train_${pair%%:*}.json
      [[ -f $f ]] && { raws+=("$f"); weights+=("${pair##*:}"); }
    done
    (( ${#raws[@]} )) || { echo "no train_*.json under $RAW_TRAIN"; exit 1; }
    printf 'using %s\n' "${raws[@]}"
    $PY -m phase2.adapt prepare --raw "${raws[@]}" --weights "${weights[@]}" --ner-data "$DATA" \
      --tokenizer "$(dirname "$DENSE_LM")/tokenizer.json" --tokens "$ADAPT_TOKENS" --output "$ADAPT_DATA"
    ;;

  adapt)        # bash run_phase2.sh adapt <encdec_mlm|denselm_clm|gqa_clm|hybrid_clm> [gpu]
    adapt_spec "${2:-}"; need "$PARENT" "$ADAPT_DATA/metadata.json"
    gpu=${3:-$GPU_A}; out=$ADAPTED/$2; mkdir -p "$out"
    exec {lock}>"$out.lock"; flock -n "$lock" || { echo "[busy] adapt $2 is already running"; exit 0; }
    [[ -f $out/adapt_summary.json ]] && { echo "[skip] adapt $2 already finished"; exit 0; }
    resume=(); [[ -f $out/last.pt ]] && resume=(--resume)
    echo "[GPU $gpu] adapt $2 ($OBJECTIVE, ${ADAPT_HOURS} h)  (log: $out.log)"
    CUDA_VISIBLE_DEVICES=$gpu $PY -m phase2.adapt train --checkpoint "$PARENT" --data "$ADAPT_DATA" --output "$out" \
      --objective "$OBJECTIVE" --hours "$ADAPT_HOURS" --project-root "$PROJECT" ${resume[@]+"${resume[@]}"} >> "$out.log" 2>&1 \
      || { echo "FAILED: adapt $2"; tail -20 "$out.log"; exit 1; }
    tail -1 "$out.log"
    ;;

  round4)       # X1 recipe + one change each: O penalty, header context, augmentation, header+augmentation
    one X20_denseorig_obias "$GPU_A" & one X17_denseorig_header "$GPU_B" & wait
    one X18_denseorig_aug "$GPU_A" & one X19_denseorig_header_aug "$GPU_B" & wait
    ;;

  round5)       # the combination on dense (with self-distillation) and on the adapted gqa
    one X21_dense_all_kd "$GPU_A" & one X22_gqa_clm_header_aug "$GPU_B" & wait
    ;;

  round6)       # X1 recipe + entry bonus, 4 seeds (for the final vote)
    one X23_denseorig_entry "$GPU_A" & one X23_denseorig_entry_s13 "$GPU_B" & wait
    one X23_denseorig_entry_s21 "$GPU_A" & one X23_denseorig_entry_s7 "$GPU_B" & wait
    ;;

  round7)       # merged-role training: the tagger learns PARTY / COUNSEL, sides come from the header and cues
    one X24_dense_coarse "$GPU_A" & one X25_gqa_clm_coarse "$GPU_B" & wait
    ;;

  selfkd)       # KD aimed at the score: X1 teaching itself and the adapted gqa, on more teacher-labelled text
    one X14_dense_selfkd "$GPU_A" & one X15_gqa_clm_kd "$GPU_B" & wait
    ;;

  seeds)        # best recipes with new seeds, for a seed ensemble (X1 = best test score, A3 = best gqa)
    one X1_denseorig_lorma_s7 "$GPU_A" & one X1_denseorig_lorma_s13 "$GPU_B" & wait
    one A3_gqa_clm_lora_s7 "$GPU_A" & one X1_denseorig_lorma_s21 "$GPU_B" & wait
    ;;

  round3)       # KD with the leak fixed + its no-KD control, then segmentation on the best model + 4-layer student
    one E4f_hybrid_kd "$GPU_A" & one X9_hybrid2_nokd "$GPU_B" & wait
    one X10_denseorig_lorma_bpeseg "$GPU_A" & one X8_hybrid_kd4 "$GPU_B" & wait
    ;;

  overlap)      # how many NER documents are inside the Phase-1 corpus (CPU)
    raws=(); for f in "$RAW_TRAIN"/train_*.json; do [[ -f $f ]] && raws+=("$f"); done
    $PY -m phase2.overlap --ner "$DATA" --raw "${raws[@]}"
    ;;

  adapted)      # NER on the adapted backbones (same recipes as E1-E4, new starting weights)
    one A2_encdec_mlm_full "$GPU_A" & one A1_dense_clm_lorma "$GPU_B" & wait
    one A3_gqa_clm_lora "$GPU_A" & one A4_hybrid_clm_kd "$GPU_B" & wait
    ;;

  ensemble)  # all members share one tokenizer; members must be token-head runs with --final
    members=()
    for name in ${MEMBERS:-X1_denseorig_lorma E1_denselm_lorma_causal X2_densesuffix_lorma E3_gqa_lora_causal}; do
      f=$OUT/$name/submission/token_logprobs.pt; need "$f"; members+=("$f")
    done
    name=${ENSEMBLE_NAME:-X11_ensemble_top4}
    [[ -f $OUT/$name/submission.zip ]] && { echo "[skip] $name exists; set ENSEMBLE_NAME=... for a new one"; exit 0; }
    $PY -m phase2.run ensemble --reference-model "$OUT/${REFERENCE:-X1_denseorig_lorma}/model.pt" --test "$TEST" \
      --output "$OUT/$name" --logprobs "${members[@]}"
    ;;

  vote)       # span-voting ensemble whose members and threshold are chosen on out-of-fold CV (CPU)
    name=${VOTE_NAME:-V1_vote}
    [[ -f $OUT/$name/submission.zip ]] && { echo "[skip] $name exists; set VOTE_NAME=... for a new one"; exit 0; }
    runs=()
    if [[ -n ${VOTE_RUNS:-} ]]; then for r in $VOTE_RUNS; do runs+=("$OUT/$r"); done
    else for d in "$OUT"/*/; do [[ -f $d/oof_predictions.pt ]] && runs+=("${d%/}"); done; fi
    $PY -m phase2.vote --data "$DATA" --runs "${runs[@]}" --output "$OUT/$name" ${VOTE_ARGS:-}
    ;;

  repost)     # re-choose post-processing with the current rules for finished runs (CPU): repost RUN [RUN ...]
    [[ -n ${2:-} ]] || { echo "usage: $0 repost <run> [run ...]"; exit 1; }
    for r in "${@:2}"; do $PY -m phase2.repost --data "$DATA" --run "$OUT/$r" ${REPOST_ARGS:-}; done
    ;;

  errors)     # real out-of-fold errors with context: errors RUN [LABEL ...]   (training text only)
    [[ -n ${2:-} ]] || { echo "usage: $0 errors <run> [label ...]"; exit 1; }
    labs=("${@:3}"); [[ ${#labs[@]} -gt 0 ]] || labs=(APP RESP WIT)
    $PY -m phase2.errors --data "$DATA" --run "$OUT/$2" --labels "${labs[@]}" --n "${N:-12}"
    ;;

  summary)
    $PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
rows = []
for p in sorted(Path(sys.argv[1]).glob("*/cv_summary.json")):
    s = json.loads(p.read_text())
    info = s["folds"][0]["info"]
    per = {}
    for f in s["folds"]:
        for k, v in f["per_label"].items():
            per.setdefault(k, []).append(v["f1"])
    worst = sorted(per, key=lambda k: sum(per[k]))[:3]
    rows.append((p.parent.name, s["cv_macro_f1_raw"], s.get("cv_macro_f1_post"), info["architecture"],
                 info["peft"]["method"], info["parameters"]["trainable"], s["best_epochs"],
                 (p.parent / "submission" / "submission.zip").exists(), worst))
print(f"{'run':28s} {'CV raw':>7s} {'CV post':>7s}  {'arch':17s} {'method':7s} {'trainable':>11s} epochs  zip  weakest labels")
for r in sorted(rows, key=lambda r: -(r[2] or r[1])):
    print(f"{r[0]:28s} {r[1]:7.4f} {(r[2] or 0):7.4f}  {r[3]:17s} {r[4]:7s} {r[5]:11,d} {str(r[6]):7s} {'yes' if r[7] else 'no':4s} {', '.join(r[8])}")
EOF
    ;;

  *) sed -n '2,/^# Every knob/p' "$0"; exit 1;;
esac
