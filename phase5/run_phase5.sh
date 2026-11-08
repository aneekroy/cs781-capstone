#!/usr/bin/env bash
# CS781 Phase 5 (Legal Statute Identification: which IPC statutes apply to these facts?) launcher, 2x RTX 3090.
#
# Unzip next to phase2/ phase3/ phase4/ (same folder as run_phase4.sh), then:
#   bash run_phase5.sh fetch         # Phase_5_LSI with your own token (cs781_cache/huggingface/token)
#   bash run_phase5.sh stats         # sizes, statute frequencies, labels/document, train/dev copies of test documents
#   bash run_phase5.sh baseline      # TF-IDF + one-vs-rest logistic regression (CPU)                -> B0_tfidf
#   bash run_phase5.sh arch          # LoRMA on all four Phase-1 architectures: L1 dense || L2 gqa, then L3 hybrid || L4 enc-dec
#   bash run_phase5.sh techniques    # on $BEST: full, partial, QLoRA, AdaLoRA, Adapters, Prefix, Prompt   -> T_*
#   bash run_phase5.sh kd            # KD students (gqa LoRMA || 4-layer hybrid) <- $TEACHER (default L1)  -> K1, K2
#   bash run_phase5.sh ablations     # pooling (attn, max), loss (bce, wbce), section (head, tail), rare repeat  -> A_*
#   bash run_phase5.sh one NAME GPU  # any single run (NAME_s<N> = another seed)
#   bash run_phase5.sh ensemble      # probability average of finished runs, members chosen on dev     -> E1
#   bash run_phase5.sh summary | collect | check FILE
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PROJECT=${PROJECT:-$HOME/cs781_phase1}
CACHE=${CACHE:-$HOME/cs781_cache}
DATA=${DATA:-$CACHE/raw/Phase_5_LSI}
OUT2=${OUT2:-$PROJECT/runsP2}
OUT=${OUT:-$PROJECT/runsP5}
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
GQA_DAPT=${GQA_DAPT:-$(adapted gqa_clm 'gqa*')}
[[ -f $DENSE_DAPT ]] || DENSE_DAPT=$DENSE_ORIG          # domain-adapted backbones if they exist, else Phase-1
[[ -f $GQA_DAPT ]] || GQA_DAPT=$GQA

EPOCHS=${EPOCHS:-3}
CHUNK=${CHUNK:-512}
TRAIN_CHUNKS=${TRAIN_CHUNKS:-8}          # windows per document in training (ILSI facts: ~7 windows of 512)
EVAL_CHUNKS=${EVAL_CHUNKS:-16}
DOC_BATCH=${DOC_BATCH:-16}               # documents per optimizer step
DPF=${DPF:-2}                            # documents per forward (memory); 1 for full fine-tuning of dense
POOL=${POOL:-label}
LOSS=${LOSS:-asl}
MAX_TRAIN=${MAX_TRAIN:-0}                # e.g. 8000 for a quick pass
BEST=${BEST:-gqa}                        # backbone for the technique comparison: dense | gqa | hybrid | encdec
TEACHER=${TEACHER:-$OUT/L1_dense_lorma}

CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # repository root (this script lives in phase5/)
[[ -f $CODE/phase5/run.py && -f $CODE/phase3/model.py ]] || { echo "phase5/, phase3/, phase2/ must sit in the repository root $CODE"; exit 1; }
export PYTHONPATH="$CODE:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
cd "$CODE"
mkdir -p "$OUT"
FT=(--data "$DATA" --project-root "$PROJECT" --chunk "$CHUNK" --train-chunks "$TRAIN_CHUNKS" --eval-chunks "$EVAL_CHUNKS"
    --epochs "$EPOCHS" --doc-batch "$DOC_BATCH" --docs-per-forward "$DPF" --pool "$POOL" --loss "$LOSS"
    --max-train-docs "$MAX_TRAIN")
LORMA=(--method lorma --rank 16 --alpha 32)

backbone() {   # key -> BB=(checkpoint + mode)
  case $1 in
    dense)  BB=(--checkpoint "$DENSE_DAPT" --mode causal) ;;
    gqa)    BB=(--checkpoint "$GQA_DAPT" --mode causal) ;;
    hybrid) BB=(--checkpoint "$HYBRID" --mode causal) ;;
    encdec) BB=(--checkpoint "$ENCDEC" --mode encoder) ;;
    *) echo "unknown backbone $1 (dense|gqa|hybrid|encdec)"; exit 1 ;;
  esac
}
spec() {
  case $1 in
    # ---- all four Phase-1 architectures, LoRMA (compulsory), label-wise attention over every window, ASL loss
    L1_dense_lorma)   backbone dense;  ARGS=("${BB[@]}" "${LORMA[@]}") ;;
    L2_gqa_lorma)     backbone gqa;    ARGS=("${BB[@]}" "${LORMA[@]}") ;;
    L3_hybrid_lorma)  backbone hybrid; ARGS=("${BB[@]}" "${LORMA[@]}") ;;
    L4_encdec_lorma)  backbone encdec; ARGS=("${BB[@]}" "${LORMA[@]}") ;;
    # ---- the technique comparison (section 11.2.1) on $BEST
    T_full)     backbone "$BEST"; ARGS=("${BB[@]}" --method full --gradient-checkpointing --docs-per-forward 1) ;;
    T_partial)  backbone "$BEST"; ARGS=("${BB[@]}" --method partial --top-k 4) ;;
    T_qlora)    backbone "$BEST"; ARGS=("${BB[@]}" --method qlora --rank 16 --alpha 32) ;;
    T_adalora)  backbone "$BEST"; ARGS=("${BB[@]}" --method adalora --alpha 32) ;;
    T_adapter)  backbone "$BEST"; ARGS=("${BB[@]}" --method adapter --bottleneck 64) ;;
    T_prefix)   backbone "$BEST"; ARGS=("${BB[@]}" --method prefix --prefix-length 16) ;;
    T_prompt)   backbone "$BEST"; ARGS=("${BB[@]}" --method prompt --prompt-tokens 20) ;;
    # ---- knowledge distillation: smaller students learn the teacher's statute probabilities (+ gold labels)
    K1_gqa_lorma_kd)  backbone gqa;    ARGS=("${BB[@]}" "${LORMA[@]}" --teacher "$TEACHER") ;;   # 57M <- teacher
    K2_hybrid4_kd)    backbone hybrid; ARGS=("${BB[@]}" --method full --keep-layers 4 --teacher "$TEACHER") ;;
    K0_hybrid4_nokd)  backbone hybrid; ARGS=("${BB[@]}" --method full --keep-layers 4) ;;           # KD control
    # ---- ablations of the LoRMA recipe on $BEST (one factor each, for the analysis)
    A_pool_attn)  backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --pool attn) ;;
    A_pool_max)   backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --pool max) ;;
    A_loss_bce)   backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --loss bce) ;;
    A_loss_wbce)  backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --loss wbce) ;;
    A_sel_head)   backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --select head --train-chunks 2 --eval-chunks 2) ;;
    A_sel_tail)   backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --select tail --train-chunks 2 --eval-chunks 2) ;;
    A_overlap)    backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --stride 384) ;;
    A_rare)       backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --rare-repeat 3) ;;
    *) echo "unknown run: $1"; exit 1 ;;
  esac
}
one() {  # one NAME GPU
  local name=$1 gpu=$2 base=$1 seed=()
  if [[ $1 =~ ^(.+)_s([0-9]+)$ ]]; then base=${BASH_REMATCH[1]}; seed=(--seed "${BASH_REMATCH[2]}"); fi
  [[ -f $OUT/$name/submission/submission.zip ]] && { echo "[skip] $name finished"; return 0; }
  spec "$base"
  [[ -f ${ARGS[1]} ]] || { echo "[missing] $name: ${ARGS[1]}"; return 0; }
  if [[ " ${ARGS[*]} " == *" --teacher "* && ! -f $TEACHER/model.pt ]]; then echo "[missing] teacher $TEACHER/model.pt"; return 0; fi
  echo "[GPU $gpu] $name (log: $OUT/$name.log)"
  CUDA_VISIBLE_DEVICES=$gpu $PY -m phase5.run finetune "${FT[@]}" --output "$OUT/$name" "${ARGS[@]}" ${seed[@]+"${seed[@]}"} \
    > "$OUT/$name.log" 2>&1 || { echo "FAILED: $name"; tail -15 "$OUT/$name.log"; return 0; }
  grep -h "^dev macro-F1" "$OUT/$name.log" | tail -1 | sed "s/^/$name: /" | cut -c1-220
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
    export HF_HOME=$CACHE/huggingface
    $PY -m phase2.fetch --output "$CACHE/raw" --token-file "${HF_TOKEN_FILE:-$CACHE/huggingface/token}" --folder Phase_5_LSI "${@:2}" ;;
  stats)      $PY -m phase5.run stats --data "$DATA" ;;
  baseline)   $PY -m phase5.run baseline --data "$DATA" --output "$OUT/B0_tfidf" --jobs "${JOBS:-8}" 2>&1 | tee "$OUT/B0_tfidf.log" | tail -4 ;;
  one)        one "$2" "${3:-$GPU_A}" ;;
  arch)       queue L1_dense_lorma L2_gqa_lorma L3_hybrid_lorma L4_encdec_lorma ;;
  techniques) queue T_partial T_qlora T_adalora T_adapter T_prefix T_prompt T_full ;;
  kd)         pair K1_gqa_lorma_kd K2_hybrid4_kd ;;
  kd0)        one K0_hybrid4_nokd "$GPU_A" ;;
  ablations)  queue A_pool_attn A_pool_max A_loss_bce A_loss_wbce A_sel_head A_sel_tail A_overlap A_rare ;;
  seeds)      pair "${S1:-L1_dense_lorma}_s7" "${S2:-L1_dense_lorma}_s13" ;;
  ensemble)
    name=${ENS_NAME:-E1_ensemble}
    runs=(); for d in "$OUT"/*/; do n=$(basename "$d"); [[ -f $d/probs.npz && $n != E* ]] && runs+=("$d"); done
    [[ -n ${ENS_RUNS:-} ]] && { runs=(); for r in $ENS_RUNS; do runs+=("$OUT/$r"); done; }
    $PY -m phase5.ensemble --data "$DATA" --runs "${runs[@]}" --output "$OUT/$name" ${ENS_ARGS:-} 2>&1 | tee "$OUT/$name.log" | tail -3 ;;
  summary)
    $PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
rows = []
for f in sorted(Path(sys.argv[1]).glob("*/summary.json")):
    s = json.loads(f.read_text())
    t = s.get("thresholds", {})
    rows.append((s.get("dev_macro_f1_honest", 0), f.parent.name, s.get("method", s.get("stage", "")), s.get("pool", "-"),
                 s.get("loss", "-"), s.get("dev_macro_f1_at_0.5", 0), s.get("dev_micro_f1_honest", 0),
                 s.get("dev_subset_acc_honest", 0), t.get("kind", "?") + ("+1" if t.get("min_one") else ""),
                 s.get("trainable_parameters"), s.get("minutes"), (f.parent / "submission" / "submission.zip").exists()))
print(f"{'run':24s} {'method':16s} {'pool':5s} {'loss':5s} {'mF1@.5':>7s} {'macroF1':>8s} {'micro':>6s} {'subset':>6s} {'thr':12s} {'trainable':>11s} {'min':>6s} test")
for r in sorted(rows, reverse=True):
    print(f"{r[1]:24s} {str(r[2])[:16]:16s} {r[3]:5s} {r[4]:5s} {r[5]:7.4f} {r[0]:8.4f} {r[6]:6.4f} {r[7]:6.4f} {r[8]:12s} "
          f"{(f'{r[9]:,}' if r[9] else '-'):>11s} {str(r[10]):>6s} {'yes' if r[11] else '-'}")
EOF
    ;;
  collect)
    mkdir -p "$OUT/submissions"
    for z in "$OUT"/*/submission/submission.zip; do
      [[ -f $z ]] || continue; n=$(basename "$(dirname "$(dirname "$z")")")
      cp "$z" "$OUT/submissions/submission_${n}.zip"; echo "submission_${n}.zip"
    done
    echo "-> $OUT/submissions/" ;;
  check)      $PY -m phase5.run check --data "$DATA" --prediction "$2" ;;
  fetch-all)
    export HF_HOME=$CACHE/huggingface
    $PY -m phase2.fetch --output "$CACHE/raw" --token-file "${HF_TOKEN_FILE:-$CACHE/huggingface/token}" --all "${@:2}" ;;
  *) sed -n '2,15p' "$0" ;;
esac
