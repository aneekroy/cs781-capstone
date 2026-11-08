#!/usr/bin/env bash
# CS781 Phase 6 (abstractive legal summarization, IN-Abs; BERTScore-F1 primary) launcher, 2x RTX 3090.
#
# Unzip next to phase2/ phase3/ phase5/ (same folder as run_phase5.sh), then:
#   bash run_phase6.sh fetch          # Phase_6_Summ with your own token (cs781_cache/huggingface/token)
#   bash run_phase6.sh stats          # lengths, compression, length model, chunks per document, alignment profile
#   bash run_phase6.sh baseline       # lead + supervised extractive (CPU) + extractor.pkl              -> B0_extractive
#   bash run_phase6.sh arch           # LoRMA on all four Phase-1 models (chunked) + enc-dec FiD        -> L1..L5
#   bash run_phase6.sh techniques     # on $BEST: full, LoRA, QLoRA, AdaLoRA, Houlsby, Pfeiffer, Prefix, Prompt -> T_*
#   bash run_phase6.sh kd             # word-level KD + sequence-level KD into gqa <- $TEACHER (L1)     -> K1, K2
#   bash run_phase6.sh kd-small       # 6-layer dense student with / without KD from L1                -> K3, K0
#   bash run_phase6.sh kd-spread      # 6 evenly spaced dense layers, lr 1e-4: word KD / none / seq KD  -> K4, K5, K6
#   bash run_phase6.sh encdec         # encoder-decoder full fine-tuning || LoRMA r64                   -> L4b, L4c
#   bash run_phase6.sh combine        # best factors per architecture: 512-token chunks, full / LoRMA     -> C1..C5
#   bash run_phase6.sh chunks         # chunk-size sweep on gqa LoRMA: 384 || 256                        -> C6, C7
#   bash run_phase6.sh longer         # dense LoRMA || dense full, 512-token chunks, 5 epochs           -> C8, C9
#   bash run_phase6.sh dapt           # hybrid / enc-dec on their Phase-2 adapted backbones, full, c512 -> D1, D2
#   bash run_phase6.sh ensembles      # dense (C3+C1+L1) || gqa (C2+A_chunk512+T_full) checkpoint ensembles -> E_dense, E_gqa
#   bash run_phase6.sh ens NAME R1 R2 # any ensemble of finished runs
#   bash run_phase6.sh ablations      # head-only, extract-then-abstract, overlap, alignment, chunk size, source-LM, scratch
#   bash run_phase6.sh tune NAME      # decoding / length variants on held-out docs (+ test submission of the best)
#   bash run_phase6.sh one NAME GPU   # any single run (NAME_s<N> = another seed)
#   bash run_phase6.sh summary | collect | check FILE
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PROJECT=${PROJECT:-$HOME/cs781_phase1}
CACHE=${CACHE:-$HOME/cs781_cache}
DATA=${DATA:-$CACHE/raw/Phase_6_Summ}
OUT2=${OUT2:-$PROJECT/runsP2}
OUT=${OUT:-$PROJECT/runsP6}
GPU_A=${GPU_A:-0}
GPU_B=${GPU_B:-1}
PY=${PY:-python}
export PROJECT HF_HOME=${HF_HOME_P6:-$CACHE/huggingface}   # BERTScore model cache (scoring only)

RUNS=$PROJECT/runs
DENSE_ORIG=${DENSE_ORIG:-$RUNS/dense_gpt_full_395280384_s42_jupyter24gb_v1/best.pt}
GQA_ORIG=${GQA_ORIG:-$RUNS/gqa_local_full_57507840_s42_jupyter24gb_v1/best.pt}
HYBRID=${HYBRID:-$RUNS/recurrent_hybrid_full_78217216_s42_jupyter24gb_v1/best.pt}
ENCDEC=${ENCDEC:-$RUNS/encoder_decoder_full_428507136_s42_jupyter24gb_v1/best.pt}
ADAPTED=$OUT2/adapt
adapted() { local d f; for d in "$@"; do for f in "$ADAPTED"/$d/best.pt "$ADAPTED"/$d/latest.pt; do
  [[ -f $f ]] && { echo "$f"; return; }; done; done; echo "$ADAPTED/$1/best.pt"; }
DENSE=${DENSE:-$(adapted dense_clm denselm_clm 'dense*')}
GQA=${GQA:-$(adapted gqa_clm 'gqa*')}
[[ -f $DENSE ]] || DENSE=$DENSE_ORIG          # legal-domain-adapted (Phase 2) backbones if present, else Phase-1
[[ -f $GQA ]] || GQA=$GQA_ORIG
# Phase-2 continued pretraining of the two backbones that had none here (bash run_phase2.sh adapt hybrid_clm | encdec_mlm)
HYBRID_DAPT=${HYBRID_DAPT:-$(adapted hybrid_clm 'hybrid*')}
ENCDEC_DAPT=${ENCDEC_DAPT:-$(adapted encdec_mlm 'encdec*')}

EPOCHS=${EPOCHS:-3}
CHUNK=${CHUNK:-1024}            # source tokens per chunk (cut to fit the Phase-1 context automatically)
BEAMS=${BEAMS:-4}
EVAL_DOCS=${EVAL_DOCS:-100}     # held-out documents summarized for the dev scores (of DEV_SIZE)
DEV_SIZE=${DEV_SIZE:-200}
GEN_BATCH=${GEN_BATCH:-8}
MAX_TRAIN=${MAX_TRAIN:-0}       # e.g. 1500 for a quick pass
BEST=${BEST:-gqa}               # backbone of the technique comparison and ablations: dense | gqa | hybrid | encdec
TEACHER=${TEACHER:-$OUT/C3_dense_full_c512}   # the strongest chunk-strategy run (dev BERTScore 0.861)
EXTRACTOR=$OUT/B0_extractive/extractor.pkl
BERT=(); [[ ${NO_BERTSCORE:-0} == 1 ]] && BERT=(--no-bertscore)

CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # repository root (this script lives in phase6/)
[[ -f $CODE/phase6/run.py && -f $CODE/phase3/model.py ]] || { echo "phase6/, phase3/, phase2/ must sit in the repository root $CODE"; exit 1; }
export PYTHONPATH="$CODE:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
cd "$CODE"
mkdir -p "$OUT"
FT=(--data "$DATA" --project-root "$PROJECT" --dev-size "$DEV_SIZE" --epochs "$EPOCHS" --chunk "$CHUNK"
    --beams "$BEAMS" --eval-docs "$EVAL_DOCS" --gen-batch "$GEN_BATCH" --max-train-docs "$MAX_TRAIN" ${BERT[@]+"${BERT[@]}"})
LORMA=(--method lorma --rank 16 --alpha 32)

backbone() {   # key -> BB (checkpoint + memory settings)
  case $1 in
    dense)  BB=(--checkpoint "$DENSE"  --micro-tokens 4096) ;;
    gqa)    BB=(--checkpoint "$GQA"    --micro-tokens 8192) ;;
    hybrid) BB=(--checkpoint "$HYBRID" --micro-tokens 6144) ;;
    encdec) BB=(--checkpoint "$ENCDEC" --micro-tokens 4096) ;;
    *) echo "unknown backbone $1 (dense|gqa|hybrid|encdec)"; exit 1 ;;
  esac
}
spec() {
  case $1 in
    # ---- all four Phase-1 architectures, LoRMA (compulsory); sentence-aligned chunks, learned length budgets
    L1_dense_lorma)      backbone dense;  ARGS=("${BB[@]}" "${LORMA[@]}") ;;
    L2_gqa_lorma)        backbone gqa;    ARGS=("${BB[@]}" "${LORMA[@]}") ;;
    L3_hybrid_lorma)     backbone hybrid; ARGS=("${BB[@]}" "${LORMA[@]}") ;;
    L4_encdec_lorma)     backbone encdec; ARGS=("${BB[@]}" "${LORMA[@]}") ;;
    L5_encdec_fid_lorma) backbone encdec; ARGS=("${BB[@]}" "${LORMA[@]}" --strategy fid --fid-chunks 8 --gradient-checkpointing) ;;
    # ---- the technique comparison (section 12.2.1) on $BEST
    T_full)      backbone "$BEST"; ARGS=("${BB[@]}" --method full --gradient-checkpointing) ;;
    T_lora)      backbone "$BEST"; ARGS=("${BB[@]}" --method lora --rank 16 --alpha 32) ;;
    T_qlora)     backbone "$BEST"; ARGS=("${BB[@]}" --method qlora --rank 16 --alpha 32) ;;
    T_adalora)   backbone "$BEST"; ARGS=("${BB[@]}" --method adalora --alpha 32) ;;
    T_houlsby)   backbone "$BEST"; ARGS=("${BB[@]}" --method houlsby --bottleneck 64) ;;
    T_pfeiffer)  backbone "$BEST"; ARGS=("${BB[@]}" --method pfeiffer --bottleneck 128) ;;
    T_prefix)    backbone "$BEST"; ARGS=("${BB[@]}" --method prefix --prefix-length 24) ;;
    T_prompt)    backbone "$BEST"; ARGS=("${BB[@]}" --method prompt --prompt-tokens 32) ;;
    # ---- knowledge distillation (the teacher must be a chunk-strategy run: L1 or L4)
    K1_gqa_kd_logit)  backbone gqa;   ARGS=("${BB[@]}" "${LORMA[@]}" --teacher "$TEACHER" --kd logit) ;;    # control: L2
    K2_gqa_kd_seq)    backbone gqa;   ARGS=("${BB[@]}" "${LORMA[@]}" --teacher "$TEACHER" --kd seq) ;;      # control: L2
    K3_dense6_kd)     backbone dense; ARGS=("${BB[@]}" --method full --keep-layers 6 --teacher "$OUT/L1_dense_lorma" --kd logit) ;;
    K0_dense6_nokd)   backbone dense; ARGS=("${BB[@]}" --method full --keep-layers 6) ;;
    # 6 of 12 dense layers, evenly spaced incl. the last, higher learning rate (the halved network must re-learn)
    K4_dense6s_kd)    backbone dense; ARGS=("${BB[@]}" --method full --keep-layers 6 --layer-pick spread --lr-full 1e-4 --chunk 512 --teacher "$TEACHER" --kd logit) ;;
    K5_dense6s_nokd)  backbone dense; ARGS=("${BB[@]}" --method full --keep-layers 6 --layer-pick spread --lr-full 1e-4 --chunk 512) ;;
    K6_dense6s_kdseq) backbone dense; ARGS=("${BB[@]}" --method full --keep-layers 6 --layer-pick spread --lr-full 1e-4 --chunk 512 --teacher "$TEACHER" --kd seq --kd-seq-docs 3000) ;;
    # encoder-decoder: full fine-tuning and a stronger LoRMA (its LoRMA run lags far behind the decoders)
    L4b_encdec_full)  backbone encdec; ARGS=("${BB[@]}" --method full --gradient-checkpointing --lr-full 5e-5) ;;
    L4c_encdec_lorma64) backbone encdec; ARGS=("${BB[@]}" --method lorma --rank 64 --alpha 64 --lr-peft 1e-3) ;;
    # ---- ablations of the LoRMA recipe on $BEST (one factor each)
    A_head)       backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --strategy head) ;;
    A_extract)    backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --strategy extract --extractor "$EXTRACTOR" --extract-keep 0.5) ;;
    A_overlap)    backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --overlap 128) ;;
    A_align_pos)  backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --align position) ;;
    A_chunk512)   backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --chunk 512) ;;
    A_srclm)      backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --src-lm 0.2) ;;
    A_scratch)    backbone "$BEST"; ARGS=("${BB[@]}" --method full --random-init) ;;
    A_nodapt)     backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}"); ARGS[1]=$([[ $BEST == dense ]] && echo "$DENSE_ORIG" || echo "$GQA_ORIG") ;;
    # ---- combining what the leaderboard showed works: 512-token chunks (+0.0036 for gqa), adapted backbones,
    #      full fine-tuning; one run per architecture, plus a chunk-size sweep and a longer schedule
    C1_dense_lorma_c512)  backbone dense;  ARGS=("${BB[@]}" "${LORMA[@]}" --chunk 512) ;;
    C2_gqa_full_c512)     backbone gqa;    ARGS=("${BB[@]}" --method full --gradient-checkpointing --chunk 512) ;;
    C3_dense_full_c512)   backbone dense;  ARGS=("${BB[@]}" --method full --gradient-checkpointing --chunk 512) ;;
    C4_hybrid_full_c512)  backbone hybrid; ARGS=("${BB[@]}" --method full --gradient-checkpointing --chunk 512) ;;
    C5_encdec_full_c512)  backbone encdec; ARGS=("${BB[@]}" --method full --gradient-checkpointing --lr-full 5e-5 --chunk 512) ;;
    C6_gqa_lorma_c384)    backbone gqa;    ARGS=("${BB[@]}" "${LORMA[@]}" --chunk 384) ;;
    C7_gqa_lorma_c256)    backbone gqa;    ARGS=("${BB[@]}" "${LORMA[@]}" --chunk 256) ;;
    C8_dense_lorma_c512_e5) backbone dense; ARGS=("${BB[@]}" "${LORMA[@]}" --chunk 512 --epochs 5) ;;
    C9_dense_full_c512_e5)  backbone dense; ARGS=("${BB[@]}" --method full --gradient-checkpointing --chunk 512 --epochs 5) ;;
    # ---- the weak architectures on domain-adapted backbones (gqa lost 0.025 without its adaptation)
    D1_hybrid_dapt_full_c512) backbone hybrid; ARGS=("${BB[@]}" --method full --gradient-checkpointing --chunk 512); ARGS[1]=$HYBRID_DAPT ;;
    D2_encdec_dapt_full_c512) backbone encdec; ARGS=("${BB[@]}" --method full --gradient-checkpointing --lr-full 5e-5 --chunk 512); ARGS[1]=$ENCDEC_DAPT ;;
    *) echo "unknown run: $1"; exit 1 ;;
  esac
}
one() {  # one NAME GPU
  local name=$1 gpu=$2 base=$1 seed=()
  if [[ $1 =~ ^(.+)_s([0-9]+)$ ]]; then base=${BASH_REMATCH[1]}; seed=(--seed "${BASH_REMATCH[2]}"); fi
  [[ -f $OUT/$name/submission/submission.zip ]] && { echo "[skip] $name finished"; return 0; }
  spec "$base"
  [[ -f ${ARGS[1]} ]] || { echo "[missing] $name: ${ARGS[1]}"; return 0; }
  if [[ " ${ARGS[*]} " == *" --teacher "* ]]; then
    local t; for ((i = 0; i < ${#ARGS[@]}; i++)); do [[ ${ARGS[i]} == --teacher ]] && t=${ARGS[i+1]}; done
    [[ -f $t/model.pt ]] || { echo "[missing] teacher $t/model.pt"; return 0; }
  fi
  [[ " ${ARGS[*]} " == *" --strategy extract "* && ! -f $EXTRACTOR ]] && { echo "[missing] $EXTRACTOR (run: baseline)"; return 0; }
  echo "[GPU $gpu] $name (log: $OUT/$name.log)"
  CUDA_VISIBLE_DEVICES=$gpu $PY -m phase6.run finetune "${FT[@]}" --output "$OUT/$name" "${ARGS[@]}" ${seed[@]+"${seed[@]}"} \
    > "$OUT/$name.log" 2>&1 || { echo "FAILED: $name"; tail -15 "$OUT/$name.log"; return 0; }
  grep -h "^dev BERTScore" "$OUT/$name.log" | tail -1 | sed "s/^/$name: /" | cut -c1-200
}
pair() { one "$1" "$GPU_A" & local a=$!; one "$2" "$GPU_B" & local b=$!; wait $a; wait $b; }
queue() {  # two GPUs, each working through its half of the list
  local a=() b=() i=0
  for n in "$@"; do if (( i++ % 2 == 0 )); then a+=("$n"); else b+=("$n"); fi; done
  ( for n in ${a[@]+"${a[@]}"}; do one "$n" "$GPU_A"; done ) & local pa=$!
  ( for n in ${b[@]+"${b[@]}"}; do one "$n" "$GPU_B"; done ) & local pb=$!
  wait $pa; wait $pb
}

case "${1:-}" in
  fetch)
    $PY -m phase2.fetch --output "$CACHE/raw" --token-file "${HF_TOKEN_FILE:-$CACHE/huggingface/token}" --folder Phase_6_Summ "${@:2}" ;;
  stats)      $PY -m phase6.run stats --data "$DATA" --dev-size "$DEV_SIZE" --checkpoint "$GQA" --chunk "$CHUNK" ;;
  baseline)   $PY -m phase6.run baseline --data "$DATA" --output "$OUT/B0_extractive" --dev-size "$DEV_SIZE" \
                ${BERT[@]+"${BERT[@]}"} 2>&1 | tee "$OUT/B0_extractive.log" | grep -E "^(lead|extractive|length|->)" | cut -c1-260 ;;
  one)        one "$2" "${3:-$GPU_A}" ;;
  arch)       queue L1_dense_lorma L2_gqa_lorma L4_encdec_lorma L3_hybrid_lorma L5_encdec_fid_lorma ;;
  techniques) queue T_lora T_qlora T_adalora T_houlsby T_pfeiffer T_prefix T_prompt T_full ;;
  kd)         pair K1_gqa_kd_logit K2_gqa_kd_seq ;;
  kd-small)   pair K3_dense6_kd K0_dense6_nokd ;;
  kd-spread)  pair K4_dense6s_kd K5_dense6s_nokd; one K6_dense6s_kdseq "$GPU_A" ;;
  encdec)     pair L4b_encdec_full L4c_encdec_lorma64 ;;
  combine)    queue C1_dense_lorma_c512 C2_gqa_full_c512 C3_dense_full_c512 C4_hybrid_full_c512 C5_encdec_full_c512 ;;
  chunks)     pair C6_gqa_lorma_c384 C7_gqa_lorma_c256 ;;
  longer)     pair C8_dense_lorma_c512_e5 C9_dense_full_c512_e5 ;;
  dapt)       pair D1_hybrid_dapt_full_c512 D2_encdec_dapt_full_c512 ;;
  ens)        # ens NAME RUN1 RUN2 ... : decode several finished runs together (first run's chunking) -> $OUT/NAME
    name=$2; shift 2; rs=(); for r in "$@"; do [[ -f $OUT/$r/model.pt ]] && rs+=("$OUT/$r") || echo "[missing] $r"; done
    (( ${#rs[@]} >= 2 )) || { echo "need at least two finished runs"; exit 1; }
    [[ -f $OUT/$name/submission/submission.zip ]] && { echo "[skip] $name finished"; exit 0; }
    CUDA_VISIBLE_DEVICES=${GPU:-$GPU_A} $PY -m phase6.run ensemble --data "$DATA" --runs "${rs[@]}" --output "$OUT/$name" \
      --eval-docs "$EVAL_DOCS" --gen-batch "$GEN_BATCH" --beams "$BEAMS" ${BERT[@]+"${BERT[@]}"} > "$OUT/$name.log" 2>&1 \
      || { echo "FAILED: $name"; tail -15 "$OUT/$name.log"; exit 1; }
    grep -h '^{"dev"' "$OUT/$name.log" | tail -1 | cut -c1-240 ;;
  ensembles)  # same-architecture ensembles (each still one Phase-1 model family)
    GPU=$GPU_A bash "$0" ens E_dense C3_dense_full_c512 C1_dense_lorma_c512 L1_dense_lorma &
    GPU=$GPU_B bash "$0" ens E_gqa C2_gqa_full_c512 A_chunk512 T_full &
    wait ;;
  ablations)  queue A_head A_extract A_overlap A_align_pos A_chunk512 A_srclm A_scratch A_nodapt ;;
  seeds)      pair "${S1:-L2_gqa_lorma}_s7" "${S2:-L2_gqa_lorma}_s13" ;;
  tune)
    [[ -f $OUT/$2/model.pt ]] || { echo "no $OUT/$2/model.pt"; exit 1; }
    CUDA_VISIBLE_DEVICES=${3:-$GPU_A} $PY -m phase6.run tune --data "$DATA" --run "$OUT/$2" --eval-docs "${TUNE_DOCS:-60}" \
      --gen-batch "$GEN_BATCH" ${BERT[@]+"${BERT[@]}"} ${VARIANTS:+--variants $VARIANTS} 2>&1 | tee "$OUT/$2.tune.log" \
      | grep -E '^\{"|best variant|tuned test' | cut -c1-260 ;;
  decode)     # decode NAME TAG [extra decode flags], e.g. decode L4_encdec_lorma len09 --len-scale 0.9
    CUDA_VISIBLE_DEVICES=${GPU:-$GPU_A} $PY -m phase6.run decode --data "$DATA" --run "$OUT/$2" --tag "$3" \
      --eval-docs "$EVAL_DOCS" --gen-batch "$GEN_BATCH" ${BERT[@]+"${BERT[@]}"} "${@:4}" ;;
  summary)
    $PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
rows = []
for f in sorted(Path(sys.argv[1]).glob("*/summary.json")):
    s = json.loads(f.read_text())
    if s.get("stage") == "baseline":
        for k, d in s["dev"].items():
            rows.append((d.get("selection_score", 0), f"{f.parent.name}:{k}", "-", "-", d, None, None, None, None))
        continue
    d = s.get("dev", {})
    t = f.parent / "tune.json"
    tuned = json.loads(t.read_text()) if t.exists() else None
    rows.append((d.get("selection_score", 0), f.parent.name, s.get("method", ""), s.get("strategy", ""), d,
                 s.get("trainable_parameters"), s.get("train_minutes"), s.get("peak_gpu_gib"),
                 f"{tuned['best']}:{tuned['rows'][tuned['best']]['selection_score']:.4f}" if tuned else "-"))
print(f"{'run':24s} {'method':9s} {'strategy':8s} {'BERTS-F1':>8s} {'R-1':>6s} {'R-2':>6s} {'R-L':>6s} {'words':>9s} "
      f"{'redund':>6s} {'num-pr':>6s} {'trainable':>11s} {'min':>6s} {'GiB':>5s}  tuned")
for r in sorted(rows, key=lambda x: -(x[0] or 0)):
    d = r[4]
    bs = d.get("bertscore_f1")
    print(f"{r[1]:24s} {str(r[2])[:9]:9s} {str(r[3])[:8]:8s} {(f'{bs:.4f}' if bs else '  -   '):>8s} {d.get('rouge1', 0):6.4f} "
          f"{d.get('rouge2', 0):6.4f} {d.get('rougeL', 0):6.4f} {d.get('words_pred', 0):4.0f}/{d.get('words_ref', 0):<4.0f} "
          f"{d.get('redundancy_3gram', 0):6.3f} {d.get('number_precision', 0):6.3f} "
          f"{(f'{r[5]:,}' if r[5] else '-'):>11s} {str(r[6] or '-'):>6s} {str(r[7] or '-'):>5s}  {r[8] or '-'}")
EOF
    ;;
  collect)    # the run's own submission, or the tuned one when tuning found a better decoding
    mkdir -p "$OUT/submissions"
    for d in "$OUT"/*/; do
      d=${d%/}; n=$(basename "$d"); z=$d/submission/submission.zip
      [[ -f $d/tuned/submission/submission.zip ]] && z=$d/tuned/submission/submission.zip
      [[ -f $z ]] || continue
      cp "$z" "$OUT/submissions/submission_${n}.zip"; echo "submission_${n}.zip <- ${z#$OUT/}"
    done
    echo "-> $OUT/submissions/" ;;
  check)      $PY -m phase6.run check --data "$DATA" --prediction "$2" ;;
  *) sed -n '2,25p' "$0" ;;
esac
