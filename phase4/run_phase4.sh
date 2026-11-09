#!/usr/bin/env bash
# CS781 Phase 4 (CJPE: will the appeal be accepted?) launcher for the 2x RTX 3090 server.
#
# Lives in phase4/ of the repository; run from the repository root as bash phase4/run_phase4.sh <command>:
#   bash run_phase4.sh fetch        # Phase_4_CJPE with your own token (same as Phases 2/3)
#   bash run_phase4.sh stats        # sizes, label balance, lengths, test near-duplicates of train/dev
#   bash run_phase4.sh baseline     # stage 0: TF-IDF + logistic regression (CPU, minutes)        -> B0_tfidf
#   bash run_phase4.sh round1       # stage 1: F1 dense DAPT + LoRMA || F2 Phase-2 X24 init + LoRMA
#   bash run_phase4.sh techniques   # stage 1 on $BEST: LoRA rsLoRA AdaLoRA IA3 (+ LoRMA from round1)
#   bash run_phase4.sh kd           # stage 1 KD: gqa student <- F1 || hybrid 4-layer student <- F1
#   bash run_phase4.sh arch         # stage 1: F7 encoder-decoder LoRMA || F8 gqa LoRMA (all four Phase-1 architectures)
#   bash run_phase4.sh embed RUN [GPU]       # stage 2: window embeddings with a stage-1 model
#   bash run_phase4.sh aggregate RUN         # stage 3: BiGRU+attention over them (5 seeds)  -> A_RUN
#   bash run_phase4.sh hier                  # stages 2+3 for F1 and F2 (one per GPU)
#   bash run_phase4.sh ensemble     # stage 4: average / vote of all finished runs, chosen on dev
#   bash run_phase4.sh summary      # dev scores of every run
#   bash run_phase4.sh collect      # every submission.zip, named by run, into runsP4/submissions/
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PROJECT=${PROJECT:-$HOME/cs781_phase1}
CACHE=${CACHE:-$HOME/cs781_cache}
DATA=${DATA:-$CACHE/raw/Phase_4_CJPE}
OUT2=${OUT2:-$PROJECT/runsP2}
OUT=${OUT:-$PROJECT/runsP4}
GPU_A=${GPU_A:-0}
GPU_B=${GPU_B:-1}
PY=${PY:-python}

RUNS=$PROJECT/runs
DENSE_ORIG=${DENSE_ORIG:-$RUNS/dense_gpt_full_395280384_s42_jupyter24gb_v1/best.pt}
GQA=${GQA:-$RUNS/gqa_local_full_57507840_s42_jupyter24gb_v1/best.pt}
HYBRID=${HYBRID:-$RUNS/recurrent_hybrid_full_78217216_s42_jupyter24gb_v1/best.pt}
ENCDEC=${ENCDEC:-$RUNS/encoder_decoder_full_428507136_s42_jupyter24gb_v1/best.pt}
ADAPTED=$OUT2/adapt
adapted() { local d f; for d in "$@"; do for f in "$ADAPTED"/$d/best.pt "$ADAPTED"/$d/latest.pt; do
  [[ -f $f ]] && { echo "$f"; return; }; done; done; echo "$ADAPTED/$1/best.pt"; }
DENSE_DAPT=${DENSE_DAPT:-$(adapted dense_clm denselm_clm 'dense*')}
P2_NER=${P2_NER:-$OUT2/X24_dense_coarse/model.pt}

CHUNK=${CHUNK:-512}
TAIL=${TAIL:-4}                  # windows from the end of each judgment for stage 1
EMBED=${EMBED:-8}                # windows per judgment for stages 2/3 (the last 8 x 512 tokens)
EPOCHS=${EPOCHS:-2}
BATCH=${BATCH:-16}
MAX_TRAIN=${MAX_TRAIN:-0}        # e.g. 8000 for a quick pass
BEST=${BEST:-F1_densedapt_lorma}
TEACHER=${TEACHER:-$OUT/F1_densedapt_lorma}

CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # repository root (this script lives in phase4/)
SELF="$CODE/phase4/run_phase4.sh"                 # absolute path of this launcher (it cds to $CODE below)
[[ -f $CODE/phase4/run.py && -f $CODE/phase3/model.py ]] || { echo "phase4/, phase3/, phase2/ must sit in the repository root $CODE"; exit 1; }
export PYTHONPATH="$CODE:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
cd "$CODE"
mkdir -p "$OUT"
FT=(--data "$DATA" --project-root "$PROJECT" --chunk "$CHUNK" --tail-chunks "$TAIL" --head-chunk
    --embed-chunks "$EMBED" --epochs "$EPOCHS" --batch-size "$BATCH" --max-train-docs "$MAX_TRAIN" --balance)

need() { for f in "$@"; do [[ -e "$f" ]] || { echo "MISSING: $f"; return 1; }; done; }

spec() {
  case $1 in
    F1_densedapt_lorma)  ARGS=(--checkpoint "$DENSE_DAPT" --method lorma --rank 16 --alpha 32) ;;   # best Phase-3 backbone
    F2_p2x24_lorma)      ARGS=(--init-phase2 "$P2_NER" --method lorma --rank 16 --alpha 32) ;;      # Phase-2 transfer
    F3_densedapt_lora)   ARGS=(--checkpoint "$DENSE_DAPT" --method lora --rank 16 --alpha 32) ;;
    # rsLoRA scales by alpha/sqrt(r): at r=16, alpha=32 that is 8 (4x LoRA's 2) and the run collapsed to one class.
    # Its intended regime is a high rank with a stable scale: r=64, alpha=16 -> 16/8 = 2, the same as F3's LoRA.
    F4_densedapt_rslora) ARGS=(--checkpoint "$DENSE_DAPT" --method rslora --rank 64 --alpha 16) ;;
    F5_densedapt_adalora) ARGS=(--checkpoint "$DENSE_DAPT" --method adalora --rank 16 --alpha 32) ;;
    F6_densedapt_ia3)    ARGS=(--checkpoint "$DENSE_DAPT" --method ia3) ;;
    F7_encdec_lorma)     ARGS=(--checkpoint "$ENCDEC" --method lorma --mode encoder --rank 16 --alpha 32) ;;  # 4th architecture
    F8_gqa_lorma)        ARGS=(--checkpoint "$GQA" --method lorma --rank 16 --alpha 32) ;;      # gqa without KD (KD control)
    K1_gqa_lora_kd)      ARGS=(--checkpoint "$GQA" --method lora --rank 16 --alpha 32 --teacher "$TEACHER") ;;
    K2_hybrid4_kd)       ARGS=(--checkpoint "$HYBRID" --method full --keep-layers 4 --teacher "$TEACHER") ;;
    K0_hybrid4_nokd)     ARGS=(--checkpoint "$HYBRID" --method full --keep-layers 4) ;;
    *) echo "unknown run: $1"; exit 1 ;;
  esac
}
one() {  # one NAME GPU  (NAME_s<N> = another seed)
  local name=$1 gpu=$2 base=$1 seed=()
  if [[ $1 =~ ^(.+)_s([0-9]+)$ ]]; then base=${BASH_REMATCH[1]}; seed=(--seed "${BASH_REMATCH[2]}"); fi
  [[ -f $OUT/$name/submission/submission.zip ]] && { echo "[skip] $name finished"; return 0; }
  spec "$base"
  need "${ARGS[1]}" || { echo "[missing] $name"; return 0; }
  [[ $base == K1* || $base == K2* ]] && { need "$TEACHER/model.pt" || return 0; }
  echo "[GPU $gpu] $name (log: $OUT/$name.log)"
  CUDA_VISIBLE_DEVICES=$gpu $PY -m phase4.run finetune "${FT[@]}" --output "$OUT/$name" "${ARGS[@]}" ${seed[@]+"${seed[@]}"} \
    > "$OUT/$name.log" 2>&1 || { echo "FAILED: $name"; tail -15 "$OUT/$name.log"; return 0; }
  grep -h "^dev macro-F1" "$OUT/$name.log" | tail -1 | sed "s/^/$name: /" | cut -c1-200
}
pair() { one "$1" "$GPU_A" & local a=$!; one "$2" "$GPU_B" & local b=$!; wait $a; wait $b; }
hier() {  # hier RUN GPU: stage 2 + stage 3
  local run=$1 gpu=${2:-$GPU_A}
  [[ -f $OUT/$run/model.pt ]] || { echo "[missing] $run/model.pt"; return 0; }
  [[ -f $OUT/$run/emb_test.pt ]] || CUDA_VISIBLE_DEVICES=$gpu $PY -m phase4.run embed --data "$DATA" --run "$OUT/$run" \
      --embed-chunks "$EMBED" --max-train-docs "$MAX_TRAIN" > "$OUT/$run.embed.log" 2>&1 || { echo "FAILED embed $run"; tail -5 "$OUT/$run.embed.log"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu $PY -m phase4.run aggregate --data "$DATA" --run "$OUT/$run" --output "$OUT/A_$run" --balance \
      > "$OUT/A_$run.log" 2>&1 || { echo "FAILED aggregate $run"; tail -5 "$OUT/A_$run.log"; return 0; }
  grep -h "^aggregate:" "$OUT/A_$run.log" | sed "s/^/A_$run: /" | cut -c1-200
}

case "${1:-}" in
  fetch)
    export HF_HOME=$CACHE/huggingface
    $PY -m phase2.fetch --output "$CACHE/raw" --token-file "${HF_TOKEN_FILE:-$CACHE/huggingface/token}" --folder Phase_4_CJPE "${@:2}" ;;
  fetch-all)    # every folder of the course dataset (all phases) into $CACHE/raw
    export HF_HOME=$CACHE/huggingface
    $PY -m phase2.fetch --output "$CACHE/raw" --token-file "${HF_TOKEN_FILE:-$CACHE/huggingface/token}" --all "${@:2}" ;;
  stats)     $PY -m phase4.run stats --data "$DATA" ;;
  baseline)  $PY -m phase4.run baseline --data "$DATA" --output "$OUT/B0_tfidf" 2>&1 | tee "$OUT/B0_tfidf.log" | tail -3 ;;
  one)       one "$2" "${3:-$GPU_A}" ;;
  round1)    pair F1_densedapt_lorma F2_p2x24_lorma ;;
  techniques) pair F3_densedapt_lora F4_densedapt_rslora; pair F5_densedapt_adalora F6_densedapt_ia3 ;;
  kd)        pair K1_gqa_lora_kd K2_hybrid4_kd ;;
  arch)      pair F7_encdec_lorma F8_gqa_lorma ;;        # architecture coverage: enc-dec || gqa (no KD)
  kd0)       one K0_hybrid4_nokd "$GPU_A" ;;            # KD control for K2
  embed)     hier "$2" "${3:-$GPU_A}" ;;
  aggregate) hier "$2" "${3:-$GPU_A}" ;;
  hier)      hier F1_densedapt_lorma "$GPU_A" & a=$!; hier F2_p2x24_lorma "$GPU_B" & b=$!; wait $a; wait $b ;;
  seeds)     pair "${S1:-$BEST}_s7" "${S2:-$BEST}_s13" ;;
  ensemble)
    name=${ENS_NAME:-E1_ensemble}
    runs=(); for d in "$OUT"/*/; do n=$(basename "$d"); [[ -f $d/probs.json && $n != E* ]] && runs+=("$d"); done
    [[ -n ${ENS_RUNS:-} ]] && { runs=(); for r in $ENS_RUNS; do runs+=("$OUT/$r"); done; }
    $PY -m phase4.ensemble --data "$DATA" --runs "${runs[@]}" --output "$OUT/$name" ${ENS_ARGS:-} ;;
  summary)
    $PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
rows = []
for f in sorted(Path(sys.argv[1]).glob("*/summary.json")):
    s = json.loads(f.read_text())
    rows.append((s.get("dev_macro_f1_honest", 0), f.parent.name, s.get("stage"), s.get("dev_macro_f1_at_0.5"),
                 s.get("threshold"), s.get("minutes"), (f.parent / "submission" / "submission.zip").exists()))
print(f"{'run':30s} {'stage':15s} {'dev@0.5':>8s} {'honest':>8s} {'thr':>5s} {'min':>6s} test")
for honest, name, stage, f05, t, m, sub in sorted(rows, reverse=True):
    print(f"{name:30s} {str(stage):15s} {f05 or 0:8.4f} {honest:8.4f} {t or 0.5:5.2f} {str(m):>6s} {'yes' if sub else '-'}")
EOF
    ;;
  collect)
    mkdir -p "$OUT/submissions"
    for z in "$OUT"/*/submission/submission.zip; do
      [[ -f $z ]] || continue; n=$(basename "$(dirname "$(dirname "$z")")")
      cp "$z" "$OUT/submissions/submission_${n}.zip"; echo "submission_${n}.zip"
    done
    echo "-> $OUT/submissions/" ;;
  check)     $PY -m phase4.run check --data "$DATA" --prediction "$2" ;;
  *) sed -n '2,17p' "$SELF" ;;
esac
