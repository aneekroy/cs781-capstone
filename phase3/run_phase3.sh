#!/usr/bin/env bash
# CS781 Phase 3 (rhetorical roles, CL + IT) launcher for the 2x RTX 3090 server.
#
# Lives in phase3/ of the repository; run from the repository root as bash phase3/run_phase3.sh <command>:
#   bash run_phase3.sh fetch      # download Phase_3_RR with your own token (same as Phase 2), or copy it to $DATA
#   bash run_phase3.sh check      # data stats, test-id checks, dev truth files + all-Fact smoke test (official scorer)
#   bash run_phase3.sh list       # the experiment table
#   bash run_phase3.sh round1     # P1 dense LoRMA + document context || P2 same backbone, sentence-only (ablation)
#   bash run_phase3.sh round2     # P3 adapted dense (DAPT) || P4 adapted gqa LoRA + domain heads
#   bash run_phase3.sh round3     # P5 Phase-2 NER model as the start (X24) || P6 enc-dec LoRMA, transformer context
#   bash run_phase3.sh round4     # P7 P1 + weighted CE (loss axis) || P8 P1 + rare-segment oversampling + aux CE
#   bash run_phase3.sh kd         # K1 hybrid 4-layer student <- P1 || K2 gqa student <- P1 (+ teacher-labelled SC text)
#   bash run_phase3.sh one NAME [GPU]   # any single experiment; NAME_s<N> = another seed
#   bash run_phase3.sh ensemble   # posterior-averaging ensemble, members chosen on OOF (CPU)
#   bash run_phase3.sh summary    # OOF table of every run
#   bash run_phase3.sh install-fast   # FlashAttention-2 + numba (once)
#   bash run_phase3.sh r1         # R1A || R1B: ensemble teacher + whole-document context + aux heads (+ two-pass)
#   R1_BEST=R1A_gqa_ens_doc bash run_phase3.sh r2   # noisy-student round 2 (gqa || dense student)
#
# Finding the best model to fine-tune (section 9.2.1):
#   bash run_phase3.sh screen      # every Phase-1/2 backbone with the same technique (LoRMA), 3 folds, no test fit
#   BEST=dense_dapt bash run_phase3.sh techniques   # 9 techniques on the best backbone: LoRA rsLoRA QLoRA AdaLoRA
#                                  # LoRMA Prefix Prompt IA3 + full fine-tuning (reference); KD runs via `kd`
#   bash run_phase3.sh report      # markdown tables (backbones, techniques) -> runsP3/report.md
#
# Model choice uses the pooled out-of-fold macro-F1 over ALL train+dev documents (5 folds) - the same
# pooled number the leaderboard computes, on 9x more documents than the 10 dev documents alone.
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}   # less fragmentation on 24 GB

PROJECT=${PROJECT:-$HOME/cs781_phase1}
CACHE=${CACHE:-$HOME/cs781_cache}
DATA=${DATA:-$CACHE/raw/Phase_3_RR}
OUT2=${OUT2:-$PROJECT/runsP2}               # Phase-2 runs (NER models, adapted backbones)
OUT=${OUT:-$PROJECT/runsP3}
GPU_A=${GPU_A:-0}
GPU_B=${GPU_B:-1}
PY=${PY:-python}

RUNS=$PROJECT/runs
DENSE_ORIG=${DENSE_ORIG:-$RUNS/dense_gpt_full_395280384_s42_jupyter24gb_v1/best.pt}
GQA=${GQA:-$RUNS/gqa_local_full_57507840_s42_jupyter24gb_v1/best.pt}
ENCDEC=${ENCDEC:-$RUNS/encoder_decoder_full_428507136_s42_jupyter24gb_v1/best.pt}
HYBRID=${HYBRID:-$RUNS/recurrent_hybrid_full_78217216_s42_jupyter24gb_v1/best.pt}
IMP=$PROJECT/improvements/dense_v2/runs
DENSE_LM=${DENSE_LM:-$IMP/dense_lm_dense_v2_s42/best.pt}
DENSE_SUFFIX=${DENSE_SUFFIX:-$IMP/dense_suffix_dense_v2_s42/latest.pt}
ADAPTED=$OUT2/adapt                          # Phase-2 continued pretraining on official legal text
adapted() {  # first existing checkpoint among the candidate adapt/ directories (best.pt, else latest.pt)
  local d f
  for d in "$@"; do for f in "$ADAPTED"/$d/best.pt "$ADAPTED"/$d/latest.pt; do
    [[ -f $f ]] && { echo "$f"; return; }; done; done
  echo "$ADAPTED/$1/best.pt"                 # not found: reported as [missing] and skipped
}
DENSE_DAPT=${DENSE_DAPT:-$(adapted dense_clm denselm_clm 'dense*')}
GQA_DAPT=${GQA_DAPT:-$(adapted gqa_clm 'gqa*')}
ENCDEC_DAPT=${ENCDEC_DAPT:-$(adapted encdec_mlm 'encdec*')}
HYBRID_DAPT=${HYBRID_DAPT:-$(adapted hybrid_clm 'hybrid*')}
P2_NER=${P2_NER:-$OUT2/X24_dense_coarse/model.pt}   # best Phase-2 single NER model
P2_NER2=${P2_NER2:-$OUT2/X1_denseorig_lorma/model.pt}
UNLABELED=${UNLABELED:-$CACHE/raw/Phase_1_Pretraining/Train/train_supreme_court.json}
# judgment text for in-domain silver data: Phase-1 SC / HC / tribunal judgments (not the bare Acts) and, once
# fetched (bash run_phase3.sh fetch-more), the training judgments of Phases 4 (ILDC), 6 (In-Abs) and 7 (IL-PCR)
silver_sources() {
  local f
  for f in "$CACHE"/raw/Phase_1_Pretraining/Train/train_tribunals.json \
           "$CACHE"/raw/Phase_1_Pretraining/Train/train_high_courts.json \
           "$CACHE"/raw/Phase_1_Pretraining/Train/train_supreme_court.json \
           "$CACHE"/raw/Phase_4_CJPE/Train/train.json "$CACHE"/raw/Phase_6_Summ/Train/train.json \
           "$CACHE"/raw/Phase_7_PCR/Train/train_candidates.json "$CACHE"/raw/Phase_7_PCR/Train/train.json; do
    [[ -f $f ]] && printf '%s ' "$f"
  done
}
read -r -a UNLABELED_ALL <<< "${UNLABELED_ALL:-$(silver_sources)}"
[[ ${#UNLABELED_ALL[@]} -gt 0 ]] || UNLABELED_ALL=("$UNLABELED")

PROTOCOL=${PROTOCOL:-cv}                     # cv (K folds over train+dev) | dev (train -> official dev, 5x faster)
FOLDS=${FOLDS:-5}
EPOCHS=${EPOCHS:-12}
PATIENCE=${PATIENCE:-4}
SEG=${SEG:-48}                               # sentences per training segment (document context window)
WINDOW=${WINDOW:-512}                        # tokens per packed encoder window
LR_PEFT=${LR_PEFT:-3e-4}
TEACHER=${TEACHER:-$OUT/P1_dense_lorma_ctx}
TEACHER_M=${TEACHER_M:-$OUT/P5_p2ner_lorma}       # teacher for the data-mixing KD runs (best OOF)
MIX=(--repeat-labels "${REPEAT_LABELS:-Dissent=3,Issue=3,PrecedentNotReliedUpon=2,None=2,RulingByPresentCourt=2}"
     --splice "${SPLICE:-1.0}")
[[ -n ${DOMAIN_WEIGHTS:-} ]] && MIX+=(--domain-weights "$DOMAIN_WEIGHTS")   # e.g. DOMAIN_WEIGHTS=CL=1,IT=1.3
SILVER=(--kd-unlabeled "${UNLABELED_ALL[@]}" --kd-select-domain --kd-take-prob 0.5 --kd-unlabeled-docs "${SILVER_DOCS:-400}"
        --kd-alpha 0.5 --kd-temperature 2)
# ---- R1 (rethought pipeline): ensemble teacher, whole-document context, auxiliary heads, noisy student
TEACHERS_DEFAULT="T_lorma P5_p2ner_lorma M4_gqa_dapt_lora_kd_mix"   # fold-matched 5-fold runs (not the 3-fold S_*)
teachers() {   # teachers NAME... -> TEACHER_DIRS=(existing 5-fold runs with fold models)
  TEACHER_DIRS=()
  local n
  for n in "$@"; do
    if [[ -f $OUT/$n/folds.json && -f $OUT/$n/folds/fold$((FOLDS - 1))/model.pt && -f $OUT/$n/final/model.pt ]]; then
      TEACHER_DIRS+=("$OUT/$n"); else echo "[teacher skipped] $n (needs a finished 5-fold run with --final)" >&2; fi
  done
  [[ ${#TEACHER_DIRS[@]} -gt 0 ]] || { echo "no usable teacher among: $*" >&2; exit 1; }
}
SILVER_R1=(--kd-unlabeled "${UNLABELED_ALL[@]}" --kd-select-domain --kd-select-similar --kd-similar-pool 3
           --kd-take-prob 0.5 --kd-unlabeled-docs "${SILVER_DOCS_R1:-600}" --kd-unlabeled-sentences "${SILVER_SENTS_R1:-30000}"
           --kd-alpha 0.5 --kd-temperature 2 --kd-confidence --kd-silver-schedule decay)
R1_MODEL=(--context lstm_tf --doc-context --segment-sentences "${SEG_R1:-64}" --aux-shift 0.3 --aux-coarse 0.3
          --token-dropout 0.05 --domain-heads)
read -r -a R1X <<< "${R1_EXTRA:-}"            # e.g. R1_EXTRA="--gradient-checkpointing" if a run still runs out of memory
R1_MODEL+=(${R1X[@]+"${R1X[@]}"})
FINAL=${FINAL:-1}                            # 1: after CV fit on all labelled docs and write submission/
BEST=${BEST:-dense_orig}                     # backbone key for `techniques` (pick it from `screen`)
SCREEN_FOLDS=${SCREEN_FOLDS:-3}
TECH_FOLDS=${TECH_FOLDS:-$FOLDS}

CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # repository root (this script lives in phase3/)
[[ -f $CODE/phase3/run.py && -f $CODE/phase2/model.py ]] || { echo "phase3/ and phase2/ must sit in the repository root $CODE"; exit 1; }
[[ -f $CODE/phase1/models.py || -f $PROJECT/phase1/models.py ]] || { echo "phase1/ not found in $CODE or PROJECT=$PROJECT"; exit 1; }
export PYTHONPATH="$CODE:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
cd "$CODE"
mkdir -p "$OUT"
COMMON=(--data "$DATA" --project-root "$PROJECT" --protocol "$PROTOCOL" --folds "$FOLDS" --epochs "$EPOCHS"
        --patience "$PATIENCE" --segment-sentences "$SEG" --window "$WINDOW")

need() { for f in "$@"; do [[ -e "$f" ]] || { echo "MISSING: $f"; exit 1; }; done; }

run() {  # run <gpu> <name> <phase3.run cv args...>   (skips a finished run)
  local gpu=$1 name=$2; shift 2
  local final=(); [[ $FINAL == 1 ]] && final=(--final)
  if [[ -f "$OUT/$name/submission/submission.zip" || ( $FINAL != 1 && -f "$OUT/$name/cv_summary.json" ) ]]; then
    echo "[skip] $name already finished"; return 0; fi
  exec {lock}>"$OUT/$name.lock"
  flock -n "$lock" || { echo "[busy] $name is already running elsewhere"; return 0; }
  echo "[GPU $gpu] $name  (log: $OUT/$name.log)"
  CUDA_VISIBLE_DEVICES=$gpu $PY -m phase3.run cv "${COMMON[@]}" ${final[@]+"${final[@]}"} --output "$OUT/$name" "$@" \
    > "$OUT/$name.log" 2>&1 || { echo "FAILED: $name - tail of log:"; tail -20 "$OUT/$name.log"; return 1; }
  grep -h "OOF pooled macro-F1 with it" "$OUT/$name.log" | tail -1 | sed "s/^/$name: /" || true
}

PEFT=(--rank 16 --alpha 32 --lr-peft "$LR_PEFT")
backbone() {   # key -> BB=(checkpoint/init args + mode)
  case $1 in
    dense_orig)   BB=(--checkpoint "$DENSE_ORIG" --mode causal) ;;           # Phase-1 run-1 dense
    dense_lm)     BB=(--checkpoint "$DENSE_LM" --mode causal) ;;             # Phase-1 continued LM (dense_v2)
    dense_suffix) BB=(--checkpoint "$DENSE_SUFFIX" --mode causal) ;;         # Phase-1 winner (suffix objective)
    dense_dapt)   BB=(--checkpoint "$DENSE_DAPT" --mode causal) ;;   # Phase-2 domain-adapted
    gqa)          BB=(--checkpoint "$GQA" --mode causal) ;;
    gqa_dapt)     BB=(--checkpoint "$GQA_DAPT" --mode causal) ;;
    encdec)       BB=(--checkpoint "$ENCDEC" --mode encoder) ;;
    encdec_dapt)  BB=(--checkpoint "$ENCDEC_DAPT" --mode encoder) ;;
    hybrid)       BB=(--checkpoint "$HYBRID" --mode causal) ;;
    hybrid_dapt)  BB=(--checkpoint "$HYBRID_DAPT" --mode causal) ;;
    p2_x24)       BB=(--init-phase2 "$P2_NER" --mode causal) ;;              # Phase-2 NER model, LoRMA merged
    p2_x1)        BB=(--init-phase2 "$P2_NER2" --mode causal) ;;
    *) echo "unknown backbone: $1 ($BACKBONES)"; exit 1 ;;
  esac
}
BACKBONES="dense_orig dense_lm dense_suffix dense_dapt gqa gqa_dapt encdec encdec_dapt hybrid hybrid_dapt p2_x24 p2_x1"
TECHNIQUES="lora rslora qlora adalora lorma prefix prompt ia3 full"
spec() {
  case $1 in
    S_*)   backbone "${1#S_}"; ARGS=("${BB[@]}" --method lorma --rank 16 --alpha 32) ;;      # backbone screen
    T_*)   backbone "$BEST"; m=${1#T_}                                                      # technique comparison
           ARGS=("${BB[@]}" --method "$m" --rank 16 --alpha 32)
           [[ $m == full ]] && ARGS+=(--gradient-checkpointing) ;;
    # ---- the main recipe: Phase-1 dense LM + LoRMA, sentences packed with left context, BiLSTM-CRF over the document
    P1_dense_lorma_ctx)    ARGS=(--checkpoint "$DENSE_ORIG" --method lorma --mode causal "${PEFT[@]}") ;;
    # ---- ablation: same backbone, each sentence alone, no document context, no position (sentence classifier)
    P2_dense_lorma_sent)   ARGS=(--checkpoint "$DENSE_ORIG" --method lorma --mode causal "${PEFT[@]}"
                                 --packing isolated --context none --no-position --loss ce) ;;
    # ---- backbones after Phase-2 domain-adaptive pretraining (official legal text)
    P3_adapted_dense_lorma) ARGS=(--checkpoint "$DENSE_DAPT" --method lorma --mode causal "${PEFT[@]}") ;;
    P4_adapted_gqa_lora_dom) ARGS=(--checkpoint "$GQA_DAPT" --method lora --mode causal "${PEFT[@]}"
                                 --domain-heads) ;;
    # ---- transfer: start from the Phase-2 NER model (its LoRMA merged into the weights), new LoRMA on top
    P5_p2ner_lorma)        ARGS=(--init-phase2 "$P2_NER" --method lorma --mode causal "${PEFT[@]}") ;;
    # ---- bidirectional encoder (enc-dec) + transformer document context
    P6_encdec_lorma_tf)    ARGS=(--checkpoint "$ENCDEC" --method lorma --mode encoder "${PEFT[@]}" --context transformer) ;;
    # ---- loss axis on the P1 recipe: class-weighted CE (offsets tuned after, as for every run)
    P7_dense_lorma_wce)    ARGS=(--checkpoint "$DENSE_ORIG" --method lorma --mode causal "${PEFT[@]}" --loss ce
                                 --class-weights sqrt) ;;
    # ---- rare-class pressure on the P1 recipe: CRF + weighted CE auxiliary + rare-segment oversampling
    P8_dense_lorma_rare)   ARGS=(--checkpoint "$DENSE_ORIG" --method lorma --mode causal "${PEFT[@]}" --aux-ce 0.5
                                 --class-weights sqrt --rare-repeat 3 --domain-heads) ;;
    # ---- knowledge distillation (fold-matched teacher = P1; students are smaller Phase-1 models)
    K1_hybrid4_kd)         need "$TEACHER/folds.json"
                           ARGS=(--checkpoint "$HYBRID" --method full --mode causal --keep-layers 4
                                 --teacher "$TEACHER" --kd-unlabeled "$UNLABELED" --kd-alpha 0.5 --kd-temperature 2) ;;
    K2_gqa_lora_kd)        need "$TEACHER/folds.json"
                           ARGS=(--checkpoint "$GQA" --method lora --mode causal "${PEFT[@]}"
                                 --teacher "$TEACHER" --kd-unlabeled "$UNLABELED" --kd-alpha 0.5 --kd-temperature 2) ;;
    # ---- data mixing: label-targeted oversampling + block splicing (+ in-domain silver judgments for KD)
    M1_gqa_lora_kd_mix)    need "$TEACHER_M/folds.json"                    # K2 recipe (best on test) + mixing
                           ARGS=(--checkpoint "$GQA" --method lora --mode causal "${PEFT[@]}" "${MIX[@]}"
                                 --teacher "$TEACHER_M" "${SILVER[@]}") ;;
    M2_p2ner_lorma_mix)    ARGS=(--init-phase2 "$P2_NER" --method lorma --mode causal "${PEFT[@]}" "${MIX[@]}") ;;
    M3_dense_lorma_rare_mix) ARGS=(--checkpoint "$DENSE_ORIG" --method lorma --mode causal "${PEFT[@]}" "${MIX[@]}"
                                 --aux-ce 0.5 --class-weights sqrt --domain-heads) ;;
    M4_gqa_dapt_lora_kd_mix) need "$TEACHER_M/folds.json"
                           ARGS=(--checkpoint "$GQA_DAPT" --method lora --mode causal "${PEFT[@]}" "${MIX[@]}"
                                 --domain-heads --teacher "$TEACHER_M" "${SILVER[@]}") ;;
    K0_hybrid4_nokd)       ARGS=(--checkpoint "$HYBRID" --method full --mode causal --keep-layers 4) ;;  # K1 without KD
    # ---- R1: ensemble teacher (TEACHERS) + in-domain silver ranked by similarity + whole-document BiLSTM+attention
    #      context + label-shift / coarse-role heads + noisy student (token dropout, splicing, silver decay)
    R1A_gqa_ens_doc)       teachers ${TEACHERS:-$TEACHERS_DEFAULT}
                           ARGS=(--checkpoint "$GQA_DAPT" --method lora --mode causal "${PEFT[@]}" "${MIX[@]}"
                                 "${R1_MODEL[@]}" --teacher "${TEACHER_DIRS[@]}" "${SILVER_R1[@]}") ;;
    R1B_gqa_ens_doc_2pass) teachers ${TEACHERS:-$TEACHERS_DEFAULT}                     # + right-context pass
                           ARGS=(--checkpoint "$GQA_DAPT" --method lora --mode causal "${PEFT[@]}" "${MIX[@]}"
                                 "${R1_MODEL[@]}" --two-pass --teacher "${TEACHER_DIRS[@]}" "${SILVER_R1[@]}") ;;
    # ---- round 2 of the noisy student: the better R1 run joins the teachers (R1_BEST=R1A_... or R1B_...)
    R2_gqa_noisy)          teachers ${TEACHERS:-$TEACHERS_DEFAULT} "${R1_BEST:-R1A_gqa_ens_doc}"
                           extra=(); [[ ${R1_BEST:-} == R1B* ]] && extra=(--two-pass)
                           ARGS=(--checkpoint "$GQA_DAPT" --method lora --mode causal "${PEFT[@]}" "${MIX[@]}"
                                 "${R1_MODEL[@]}" ${extra[@]+"${extra[@]}"} --teacher "${TEACHER_DIRS[@]}" "${SILVER_R1[@]}") ;;
    R2_dense_noisy)        teachers ${TEACHERS:-$TEACHERS_DEFAULT} "${R1_BEST:-R1A_gqa_ens_doc}"   # bigger student
                           ARGS=(--checkpoint "$DENSE_DAPT" --method lorma --mode causal "${PEFT[@]}" "${MIX[@]}"
                                 "${R1_MODEL[@]}" --teacher "${TEACHER_DIRS[@]}" "${SILVER_R1[@]}") ;;
    # ---- ablations of R1A for the write-up (each removes one component)
    R1A_no_doc)            teachers ${TEACHERS:-$TEACHERS_DEFAULT}
                           ARGS=(--checkpoint "$GQA_DAPT" --method lora --mode causal "${PEFT[@]}" "${MIX[@]}"
                                 --aux-shift 0.3 --aux-coarse 0.3 --token-dropout 0.05 --domain-heads
                                 --teacher "${TEACHER_DIRS[@]}" "${SILVER_R1[@]}") ;;
    R1A_no_aux)            teachers ${TEACHERS:-$TEACHERS_DEFAULT}
                           ARGS=(--checkpoint "$GQA_DAPT" --method lora --mode causal "${PEFT[@]}" "${MIX[@]}"
                                 --context lstm_tf --doc-context --segment-sentences "${SEG_R1:-64}" --token-dropout 0.05
                                 --domain-heads --teacher "${TEACHER_DIRS[@]}" "${SILVER_R1[@]}") ;;
    R1A_one_teacher)       teachers "${TEACHER_M##*/}"
                           ARGS=(--checkpoint "$GQA_DAPT" --method lora --mode causal "${PEFT[@]}" "${MIX[@]}"
                                 "${R1_MODEL[@]}" --teacher "${TEACHER_DIRS[@]}" "${SILVER_R1[@]}") ;;
    *) echo "unknown experiment: $1"; experiments; exit 1 ;;
  esac
}
experiments() { echo "backbones (S_<key>): $BACKBONES | techniques (T_<method>, on \$BEST): $TECHNIQUES"
               echo "data mixing: M1_gqa_lora_kd_mix M2_p2ner_lorma_mix M3_dense_lorma_rare_mix M4_gqa_dapt_lora_kd_mix"
               echo "R1 pipeline: R1A_gqa_ens_doc R1B_gqa_ens_doc_2pass | R2_gqa_noisy R2_dense_noisy |" \
                    "ablations R1A_no_doc R1A_no_aux R1A_one_teacher"
               echo "experiments: P1_dense_lorma_ctx P2_dense_lorma_sent P3_adapted_dense_lorma P4_adapted_gqa_lora_dom" \
                     "P5_p2ner_lorma P6_encdec_lorma_tf P7_dense_lorma_wce P8_dense_lorma_rare" \
                     "K0_hybrid4_nokd K1_hybrid4_kd K2_gqa_lora_kd | NAME_s<N> = another seed"; }
one() {
  local base=$1 seed=()
  if [[ $1 =~ ^(.+)_s([0-9]+)$ ]]; then base=${BASH_REMATCH[1]}; seed=(--seed "${BASH_REMATCH[2]}"); fi
  spec "$base"
  if [[ ! -e ${ARGS[1]} ]]; then echo "[missing] $1: ${ARGS[1]}"; return 0; fi
  run "$2" "$1" "${ARGS[@]}" ${seed[@]+"${seed[@]}"}
}
queue() {   # queue NAME... : two GPUs, each working through its half of the list
  local a=() b=() i=0
  for n in "$@"; do if (( i++ % 2 == 0 )); then a+=("$n"); else b+=("$n"); fi; done
  ( for n in ${a[@]+"${a[@]}"}; do one "$n" "$GPU_A" || true; done ) & local pa=$!
  ( for n in ${b[@]+"${b[@]}"}; do one "$n" "$GPU_B" || true; done ) & local pb=$!
  wait $pa; wait $pb
}
pair() { one "$1" "$GPU_A" & local a=$!; one "$2" "$GPU_B" & local b=$!; wait $a; wait $b; }

case "${1:-}" in
  fetch)
    export HF_HOME=$CACHE/huggingface
    $PY -m phase2.fetch --output "$CACHE/raw" --token-file "${HF_TOKEN_FILE:-$CACHE/huggingface/token}" --folder Phase_3_RR "${@:2}" ;;
  fetch-all)    # every folder of the course dataset (all phases) into $CACHE/raw
    export HF_HOME=$CACHE/huggingface
    $PY -m phase2.fetch --output "$CACHE/raw" --token-file "${HF_TOKEN_FILE:-$CACHE/huggingface/token}" --all "${@:2}" ;;
  fetch-more)   # other official phases' judgments (unlabeled text only) for silver KD data
    export HF_HOME=$CACHE/huggingface
    for f in Phase_4_CJPE Phase_6_Summ Phase_7_PCR; do
      $PY -m phase2.fetch --output "$CACHE/raw" --token-file "${HF_TOKEN_FILE:-$CACHE/huggingface/token}" --folder "$f" || echo "could not fetch $f"
    done
    echo "silver sources now: $(silver_sources)" ;;
  check)
    need "$DATA/Train/label_vocab.json" "$DATA/Test/CL_test.json" "$DATA/Test/IT_test.json"
    $PY -m phase3.run stats --data "$DATA"
    $PY -m phase3.run truth --data "$DATA" --output "$OUT/truth"
    if [[ -f $CODE/phase3/evaluate_phase3.py ]]; then
      $PY "$CODE/phase3/evaluate_phase3.py" --truth-cl "$OUT/truth/cl_dev_truth.json" --truth-it "$OUT/truth/it_dev_truth.json" \
        --prediction "$OUT/truth/dev_all_fact_prediction.json" --labels "$DATA/Train/label_vocab.json"
    fi
    for ck in "$DENSE_ORIG" "$DENSE_LM" "$DENSE_SUFFIX" "$GQA" "$ENCDEC" "$HYBRID" "$DENSE_DAPT" "$GQA_DAPT" \
              "$ENCDEC_DAPT" "$HYBRID_DAPT" "$P2_NER" "$P2_NER2"; do
      [[ -f $ck ]] && echo "found   $ck" || echo "missing $ck"
    done ;;
  list) experiments ;;
  one) [[ -n ${2:-} ]] || { experiments; exit 1; }; one "$2" "${3:-$GPU_A}" ;;
  round1) pair P1_dense_lorma_ctx P2_dense_lorma_sent ;;
  round2) pair P3_adapted_dense_lorma P4_adapted_gqa_lora_dom ;;
  round3) pair P5_p2ner_lorma P6_encdec_lorma_tf ;;
  round4) pair P7_dense_lorma_wce P8_dense_lorma_rare ;;
  kd)     pair K1_hybrid4_kd K2_gqa_lora_kd ;;
  screen)      # same technique (LoRMA r16) on every backbone; 3 folds, no test fit: only to choose the backbone
    names=(); for k in $BACKBONES; do names+=("S_$k"); done
    FINAL=0; COMMON=(--data "$DATA" --project-root "$PROJECT" --protocol cv --folds "$SCREEN_FOLDS"
      --epochs "$EPOCHS" --patience "$PATIENCE" --segment-sentences "$SEG" --window "$WINDOW")
    queue "${names[@]}"; $PY -m phase3.report "$OUT" ;;
  techniques)  # every technique on $BEST, same protocol (TECH_FOLDS), with a test fit each
    names=(); for m in $TECHNIQUES; do names+=("T_$m"); done
    COMMON=(--data "$DATA" --project-root "$PROJECT" --protocol cv --folds "$TECH_FOLDS" --epochs "$EPOCHS"
      --patience "$PATIENCE" --segment-sentences "$SEG" --window "$WINDOW")
    echo "backbone for the technique comparison: $BEST"
    queue "${names[@]}"; $PY -m phase3.report "$OUT" ;;
  report) $PY -m phase3.report "$OUT" ;;
  mix)    pair M1_gqa_lora_kd_mix M2_p2ner_lorma_mix ;;
  mix2)   pair M3_dense_lorma_rare_mix M4_gqa_dapt_lora_kd_mix ;;
  token)       # which token and account the fetch commands use (never prints the token)
    $PY -m phase2.fetch --output "$CACHE/raw" --token-file "${HF_TOKEN_FILE:-$CACHE/huggingface/token}" --check-only ;;
  install-fast)  # FlashAttention-2 (sliding window + GQA kernels) and numba (fast Viterbi); safe to rerun
    $PY -m pip install numba
    # prebuilt wheel for this torch / CUDA / Python / C++ ABI (no nvcc needed); compile from source only as a fallback
    whl=$($PY - <<'PYEOF'
import sys, torch
tv = ".".join(torch.__version__.split("+")[0].split(".")[:2])
cu = (torch.version.cuda or "").split(".")[0]
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
v = "2.7.4.post1"
print(f"https://github.com/Dao-AILab/flash-attention/releases/download/v{v}/flash_attn-{v}+cu{cu}torch{tv}cxx11abi{abi}-{py}-{py}-linux_x86_64.whl")
PYEOF
)
    echo "flash-attn wheel: $whl"
    $PY -m pip install "$whl" || $PY -m pip install flash-attn --no-build-isolation \
      || echo "flash-attn did not install: runs fall back to SDPA with expanded KV heads"
    $PY -c "from phase3.flash import enable; print(enable('auto'))" ;;
  r1)     pair R1A_gqa_ens_doc R1B_gqa_ens_doc_2pass ;;            # the rethought pipeline (both GPUs)
  r2)     pair R2_gqa_noisy R2_dense_noisy ;;                       # noisy-student round 2 (set R1_BEST first)
  r1abl)  queue R1A_no_doc R1A_no_aux R1A_one_teacher ;;            # ablations for the write-up
  finalize)    # final fit + submission for finished CV runs without one (the S_* screen runs); two GPUs
    todo=(); for d in "$OUT"/*/; do n=$(basename "$d")
      [[ -f $d/cv_summary.json && -f $d/args.json && ! -f $d/submission/submission.zip && $n != E* ]] && todo+=("$n"); done
    [[ -n ${RUNS_ONLY:-} ]] && todo=($RUNS_ONLY)
    echo "finalizing: ${todo[*]:-nothing to do}"
    fin() { CUDA_VISIBLE_DEVICES=$1 $PY -m phase3.finalize --run "$OUT/$2" --data "$DATA" > "$OUT/$2.final.log" 2>&1 \
              && grep -h "^submission:" "$OUT/$2.final.log" | sed "s|^|$2: |" | cut -c1-160 \
              || { echo "FAILED: $2"; tail -5 "$OUT/$2.final.log"; }; }
    a=(); b=(); i=0; for n in ${todo[@]+"${todo[@]}"}; do if (( i++ % 2 == 0 )); then a+=("$n"); else b+=("$n"); fi; done
    ( for n in ${a[@]+"${a[@]}"}; do fin "$GPU_A" "$n"; done ) & pa=$!
    ( for n in ${b[@]+"${b[@]}"}; do fin "$GPU_B" "$n"; done ) & pb=$!
    wait $pa; wait $pb
    bash "$0" collect ;;
  collect)     # every run's submission.zip, renamed by run, in one folder for upload
    mkdir -p "$OUT/submissions"
    for z in "$OUT"/*/submission/submission.zip; do
      [[ -f $z ]] || continue
      n=$(basename "$(dirname "$(dirname "$z")")")
      cp "$z" "$OUT/submissions/submission_${n}.zip"
      honest=$($PY -c "import json,sys;print(json.load(open(sys.argv[1])).get('oof_macro_f1_honest', json.load(open(sys.argv[1])).get('oof_macro_f1','?')))" \
               "$OUT/$n/cv_summary.json" 2>/dev/null || $PY -c "import json,sys;print(json.load(open(sys.argv[1]))['oof_macro_f1_honest'])" "$OUT/$n/ensemble_summary.json" 2>/dev/null || echo "?")
      printf '%-36s OOF %s\n' "submission_${n}.zip" "$honest"
    done | sort -k3 -r
    echo "-> $OUT/submissions/" ;;
  stack)       # stacked ensemble (second-level classifier): STACK_NAME, STACK_RUNS="A B ..."
    name=${STACK_NAME:-ST1_stack}
    runs=(); for d in "$OUT"/*/; do n=$(basename "$d"); [[ -f $d/oof.pt && -f $d/test.pt && $n != E* && $n != V* && $n != ST* ]] && runs+=("$d"); done
    [[ -n ${STACK_RUNS:-} ]] && { runs=(); for r in $STACK_RUNS; do runs+=("$OUT/$r"); done; }
    $PY -m phase3.stack --data "$DATA" --runs "${runs[@]}" --output "$OUT/$name" ${STACK_ARGS:-} 2>&1 | tee "$OUT/$name.log" ;;
  vote)        # sentence-level voting ensemble (phase-2 style): VOTE_NAME, VOTE_RUNS="A B ...", VOTE_ARGS=--fixed
    name=${VOTE_NAME:-V1_vote}
    runs=(); for d in "$OUT"/*/; do n=$(basename "$d"); [[ -f $d/oof.pt && $n != E* && $n != V* ]] && runs+=("$d"); done
    [[ -n ${VOTE_RUNS:-} ]] && { runs=(); for r in $VOTE_RUNS; do runs+=("$OUT/$r"); done; }
    $PY -m phase3.vote --data "$DATA" --runs "${runs[@]}" --output "$OUT/$name" ${VOTE_ARGS:-} 2>&1 | tee "$OUT/$name.log" ;;
  redecode)    # re-choose decoding + rules (Dissent) for every finished run, rewrite submission/ (CPU, minutes)
    runs=(); for d in "$OUT"/*/; do [[ -f $d/oof.pt && $(basename "$d") != E* ]] && runs+=("$d"); done
    [[ -n ${RUNS_ONLY:-} ]] && { runs=(); for r in $RUNS_ONLY; do runs+=("$OUT/$r"); done; }
    $PY -m phase3.redecode --data "$DATA" --runs "${runs[@]}" ${REDECODE_ARGS:-} ;;
  seeds)  pair "${S1:-P1_dense_lorma_ctx}_s7" "${S2:-P1_dense_lorma_ctx}_s13" ;;
  lexical)     # TF-IDF + logreg member (CPU, ~10-20 min): lexical errors for the ensembles -> L1_tfidf
    $PY -m phase3.lexical --data "$DATA" --output "$OUT/L1_tfidf" 2>&1 | tee "$OUT/L1_tfidf.log" | grep -E "^fold|^OOF|honest" ;;
  push)        # leaderboard push on finished runs (CPU): Dissent fix everywhere, then sequence-decoded ensembles
    bash "$0" redecode
    [[ -f $OUT/L1_tfidf/oof.pt ]] || bash "$0" lexical || true
    DIV=${DIV:-"M4_gqa_dapt_lora_kd_mix M3_dense_lorma_rare_mix K2_gqa_lora_kd S_p2_x1 T_ia3 S_hybrid_dapt P5_p2ner_lorma T_lorma P7_dense_lorma_wce"}
    ENS_NAME=E6_div_seq ENS_ARGS=--fixed ENS_RUNS="$DIV" bash "$0" ensemble 2>&1 | tee "$OUT/E6_div_seq.log" | grep -E "sequence|ensemble of|submission"
    ENS_NAME=E7_greedy_seq ENS_ARGS="--max-members 9" bash "$0" ensemble 2>&1 | tee "$OUT/E7_greedy_seq.log" | grep -E "^start|^\+|sequence|ensemble of|submission"
    ENS_NAME=E8_div_lex ENS_ARGS=--fixed ENS_RUNS="$DIV L1_tfidf" bash "$0" ensemble 2>&1 | tee "$OUT/E8_div_lex.log" | grep -E "sequence|ensemble of|submission"
    VOTE_NAME=V6_div VOTE_ARGS=--fixed VOTE_RUNS="$DIV" bash "$0" vote | tail -2
    STACK_NAME=ST3_div STACK_RUNS="$DIV L1_tfidf" bash "$0" stack | tail -4
    bash "$0" collect | grep -E "E6|E7|E8|V6|ST3|M4|E2" ;;
  ensemble)
    name=${ENS_NAME:-E1_ensemble}
    runs=(); for d in "$OUT"/*/; do n=$(basename "$d"); [[ -f $d/oof.pt && $n != E* && $n != V* ]] && runs+=("$d"); done
    [[ -n ${ENS_RUNS:-} ]] && { runs=(); for r in $ENS_RUNS; do runs+=("$OUT/$r"); done; }
    $PY -m phase3.ensemble --data "$DATA" --runs "${runs[@]}" --output "$OUT/$name" ${ENS_ARGS:-} ;;
  predict)  # bash run_phase3.sh predict RUN [dev|test]
    $PY -m phase3.run predict --data "$DATA" --model "$OUT/$2/final/model.pt" --split "${3:-test}" \
      --output "$OUT/$2/predict_${3:-test}" ;;
  summary)
    $PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
rows = []
for f in sorted(Path(sys.argv[1]).glob("*/cv_summary.json")):
    s = json.loads(f.read_text())
    d = s["decoding"]
    rows.append((s["oof_macro_f1_honest"], f.parent.name, s["oof_macro_f1_plain"], s["oof_macro_f1"], s.get("dev_docs_macro_f1"),
                 s["oof_by_domain"], f"{d['mode']}+{d['offsets']}", s["best_epochs"],
                 (f.parent / "submission" / "submission.zip").exists()))
print(f"{'run':28s} {'plain':>6s} {'tuned':>6s} {'honest':>6s} {'devdocs':>7s} {'CL':>6s} {'IT':>6s}  decoding          epochs  test")
for honest, name, plain, tuned, dev, dom, dec, ep, sub in sorted(rows, reverse=True):
    print(f"{name:28s} {plain:6.4f} {tuned:6.4f} {honest:6.4f} {dev if dev is not None else float('nan'):7.4f} "
          f"{dom['CL']:6.4f} {dom['IT']:6.4f}  {dec:17s} {ep}  {'yes' if sub else '-'}")
EOF
    ;;
  *) sed -n '2,20p' "$0" ;;
esac
