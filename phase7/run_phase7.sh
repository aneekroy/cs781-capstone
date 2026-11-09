#!/usr/bin/env bash
# CS781 Phase 7 (Prior Case Retrieval, IL-PCR; Micro-F1@7) launcher, 2x RTX 3090.
#
# Lives in phase7/ of the repository; run from the repository root as bash phase7/run_phase7.sh <command>:
#   bash run_phase7.sh fetch          # Phase_7_PCR with your own token
#   bash run_phase7.sh stats          # queries, pools, relevant per query, lengths, self-in-pool
#   bash run_phase7.sh lexical        # n-gram BM25 (1..5) tuned on train (CPU)                       -> B0_lexical
#   bash run_phase7.sh zeroshot       # the four Phase-1 models untrained (transfer analysis)         -> Z1..Z4
#   bash run_phase7.sh arch           # LoRMA retrievers on all four Phase-1 models                    -> L1..L4
#   bash run_phase7.sh techniques     # on $BEST: full, partial, QLoRA, IA3, prefix, prompt            -> T_*
#   bash run_phase7.sh kd             # 6-layer dense student with / without KD from $TEACHER          -> K1, K0
#   bash run_phase7.sh kd4            # 4-layer dense student with KD (size trade-off)                 -> K2
#   bash run_phase7.sh ablations      # pooling, mode, negatives, chunk size, projection, adaptation   -> A_*
#   bash run_phase7.sh lexical-all    # + citation-context queries (B1), + PRF (B2), tuned on dev (B3), + best passage (B4)
#   bash run_phase7.sh lexical-passage  # B4: best-passage BM25, n-grams up to 10, context windows up to 8
#   bash run_phase7.sh context        # retriever on citation-context queries, +-1 / +-2 sentences     -> R1_ctx1/2
#   bash run_phase7.sh round2         # continue $ROUND1 with the retriever's own hard negatives       -> R2_round2
#   bash run_phase7.sh knobs          # pairs per link, min pair similarity, tau, epochs, top passages -> R3..R9
#   bash run_phase7.sh kd-list        # list-level KD from the fused teacher (alone / + passage KD)    -> K3, K4
#   bash run_phase7.sh rerank         # passage-pair re-rankers over $FIRST and over lexical           -> X1..X4
#   bash run_phase7.sh combine NAME R1 R2 ...   # fuse runs + citation prior, later-year, neighbour citations, hub; FUSION=rrf
#   bash run_phase7.sh one NAME GPU   # any single run (NAME_s<N> = another seed)
#   bash run_phase7.sh summary | collect | check FILE
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PROJECT=${PROJECT:-$HOME/cs781_phase1}
CACHE=${CACHE:-$HOME/cs781_cache}
DATA=${DATA:-$CACHE/raw/Phase_7_PCR}
OUT2=${OUT2:-$PROJECT/runsP2}
OUT=${OUT:-$PROJECT/runsP7}
GPU_A=${GPU_A:-0}
GPU_B=${GPU_B:-1}
PY=${PY:-python}

RUNS=$PROJECT/runs
DENSE_ORIG=${DENSE_ORIG:-$RUNS/dense_gpt_full_395280384_s42_jupyter24gb_v1/best.pt}
GQA_ORIG=${GQA_ORIG:-$RUNS/gqa_local_full_57507840_s42_jupyter24gb_v1/best.pt}
HYBRID_ORIG=${HYBRID_ORIG:-$RUNS/recurrent_hybrid_full_78217216_s42_jupyter24gb_v1/best.pt}
ENCDEC_ORIG=${ENCDEC_ORIG:-$RUNS/encoder_decoder_full_428507136_s42_jupyter24gb_v1/best.pt}
ADAPTED=$OUT2/adapt
adapted() { local d f; for d in "$@"; do for f in "$ADAPTED"/$d/best.pt "$ADAPTED"/$d/latest.pt; do
  [[ -f $f ]] && { echo "$f"; return; }; done; done; echo "$ADAPTED/$1/best.pt"; }
# Phase-2 legal-domain-adapted backbones when present (decisive in Phase 6), else the Phase-1 checkpoints
DENSE=${DENSE:-$(adapted dense_clm denselm_clm 'dense*')};   [[ -f $DENSE ]] || DENSE=$DENSE_ORIG
GQA=${GQA:-$(adapted gqa_clm 'gqa*')};                        [[ -f $GQA ]] || GQA=$GQA_ORIG
HYBRID=${HYBRID:-$(adapted hybrid_clm 'hybrid*')};            [[ -f $HYBRID ]] || HYBRID=$HYBRID_ORIG
ENCDEC=${ENCDEC:-$(adapted encdec_mlm 'encdec*')};            [[ -f $ENCDEC ]] || ENCDEC=$ENCDEC_ORIG

EPOCHS=${EPOCHS:-4}
BATCH=${BATCH:-32}
BEST=${BEST:-dense}             # backbone of the technique comparison and ablations: dense | gqa | hybrid | encdec
TEACHER=${TEACHER:-$OUT/T_full}            # passage-level KD teacher
LIST_TEACHER=${LIST_TEACHER:-$OUT/T_full}  # list-level KD: its FUSED (neural + lexical) ranking teaches the student
ROUND1=${ROUND1:-$OUT/L1_dense_lorma}      # second-round negatives start from this finished LoRMA run (same backbone)
FIRST=${FIRST:-$OUT/C_all}                 # first stage of the re-rankers (a retriever, lexical or combine run)
LEX=$OUT/${LEXRUN:-B0_lexical}             # lexical configuration every neural run uses (B1/B2 once they win)

CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # repository root (this script lives in phase7/)
SELF="$CODE/phase7/run_phase7.sh"                 # absolute path of this launcher (it cds to $CODE below)
[[ -f $CODE/phase7/run.py && -f $CODE/phase6/model.py && -f $CODE/phase3/peft.py ]] || { echo "phase7/, phase6/, phase3/, phase2/ must sit in the repository root $CODE"; exit 1; }
export PYTHONPATH="$CODE:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
cd "$CODE"
mkdir -p "$OUT"
TR=(--data "$DATA" --project-root "$PROJECT" --lexical "$LEX" --epochs "$EPOCHS" --batch "$BATCH")
LORMA=(--method lorma --rank 16 --alpha 32)

backbone() {
  case $1 in
    dense)  BB=(--checkpoint "$DENSE") ;;
    gqa)    BB=(--checkpoint "$GQA") ;;
    hybrid) BB=(--checkpoint "$HYBRID") ;;
    encdec) BB=(--checkpoint "$ENCDEC") ;;
    *) echo "unknown backbone $1"; exit 1 ;;
  esac
}
spec() {
  case $1 in
    # ---- untrained Phase-1 encoders: how far do Phase-1 representations transfer?
    Z1_dense_zeroshot)  backbone dense;  ARGS=("${BB[@]}" --method frozen) ;;
    Z2_gqa_zeroshot)    backbone gqa;    ARGS=("${BB[@]}" --method frozen) ;;
    Z3_hybrid_zeroshot) backbone hybrid; ARGS=("${BB[@]}" --method frozen) ;;
    Z4_encdec_zeroshot) backbone encdec; ARGS=("${BB[@]}" --method frozen) ;;
    # ---- LoRMA (compulsory) on all four architectures
    L1_dense_lorma)     backbone dense;  ARGS=("${BB[@]}" "${LORMA[@]}") ;;
    L2_gqa_lorma)       backbone gqa;    ARGS=("${BB[@]}" "${LORMA[@]}") ;;
    L3_hybrid_lorma)    backbone hybrid; ARGS=("${BB[@]}" "${LORMA[@]}") ;;
    L4_encdec_lorma)    backbone encdec; ARGS=("${BB[@]}" "${LORMA[@]}") ;;
    # ---- techniques of section 13.2.1 on $BEST
    T_full)     backbone "$BEST"; ARGS=("${BB[@]}" --method full --gradient-checkpointing) ;;
    T_partial)  backbone "$BEST"; ARGS=("${BB[@]}" --method partial --top-k 4) ;;
    T_qlora)    backbone "$BEST"; ARGS=("${BB[@]}" --method qlora --rank 16 --alpha 32) ;;
    T_ia3)      backbone "$BEST"; ARGS=("${BB[@]}" --method ia3) ;;
    T_prefix)   backbone "$BEST"; ARGS=("${BB[@]}" --method prefix --prefix-length 16) ;;
    T_prompt)   backbone "$BEST"; ARGS=("${BB[@]}" --method prompt --prompt-tokens 16) ;;
    # ---- knowledge distillation: teacher = $TEACHER (default T_full, all layers); students keep evenly spaced layers
    K1_dense6_kd)    backbone dense; ARGS=("${BB[@]}" --method full --keep-layers 6 --lr-full 5e-5 --teacher "$TEACHER" --kd-embed 0.5) ;;
    K0_dense6_nokd)  backbone dense; ARGS=("${BB[@]}" --method full --keep-layers 6 --lr-full 5e-5) ;;
    K2_dense4_kd)    backbone dense; ARGS=("${BB[@]}" --method full --keep-layers 4 --lr-full 5e-5 --teacher "$TEACHER" --kd-embed 0.5) ;;
    # ---- ablations (one factor each, LoRMA on $BEST)
    A_bidir)       backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --mode bidirectional) ;;
    A_pool_last)   backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --pool last) ;;
    A_neg_random)  backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --negatives random) ;;
    A_neg_none)    backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --negatives none) ;;
    A_chunk100)    backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --chunk-words 100 --max-chunks 128 --max-tokens 160) ;;
    A_chunk400)    backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --chunk-words 400 --max-chunks 32 --max-tokens 600) ;;
    A_proj256)     backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --proj 256) ;;
    A_nodapt)      backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}"); ARGS[1]=$([[ $BEST == dense ]] && echo "$DENSE_ORIG" || echo "$GQA_ORIG") ;;
    # ---- query side: citation contexts (the sentences around each <CITATION> mark) instead of the whole judgment
    R1_ctx1)       backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --query-context 1) ;;
    R1_ctx2)       backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --query-context 2) ;;
    # ---- second round: continue $ROUND1 with negatives the trained retriever itself confuses (ANCE-style)
    R2_round2)     backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --init-from "$ROUND1" --negatives fused --epochs 3) ;;
    # ---- training-signal and scoring knobs (hyperparameters: report as tuning, not as experiments)
    R3_pairs6)     backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --pairs-per-link 6) ;;
    R4_minsim05)   backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --min-pair-sim 0.05) ;;
    R5_tau02)      backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --tau 0.02) ;;
    R6_tau10)      backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --tau 0.1) ;;
    R7_ep8)        backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --epochs 8) ;;
    R8_top4)       backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --top-chunks 4) ;;
    R9_top16)      backbone "$BEST"; ARGS=("${BB[@]}" "${LORMA[@]}" --top-chunks 16) ;;
    # ---- list-level KD from the fused teacher (alone, and together with passage-level KD)
    K3_dense6_listkd)  backbone dense; ARGS=("${BB[@]}" --method full --keep-layers 6 --lr-full 5e-5 --kd-list "$LIST_TEACHER") ;;
    K4_dense6_bothkd)  backbone dense; ARGS=("${BB[@]}" --method full --keep-layers 6 --lr-full 5e-5 --kd-list "$LIST_TEACHER" --teacher "$TEACHER" --kd-embed 0.5) ;;
    # ---- passage-pair re-rankers over $FIRST (CMD=rerank)
    X1_rerank_lorma)   backbone "$BEST"; CMD=rerank; ARGS=("${BB[@]}" "${LORMA[@]}" --pool last --first "$FIRST" --epochs 2 --batch 8) ;;
    X2_rerank_full)    backbone "$BEST"; CMD=rerank; ARGS=("${BB[@]}" --method full --gradient-checkpointing --pool last --first "$FIRST" --epochs 2) ;;
    X3_rerank_on_lex)  backbone "$BEST"; CMD=rerank; ARGS=("${BB[@]}" "${LORMA[@]}" --pool last --first "$LEX" --epochs 2) ;;
    X4_rerank_deep50)  backbone "$BEST"; CMD=rerank; ARGS=("${BB[@]}" "${LORMA[@]}" --pool last --first "$FIRST" --epochs 2 --rerank-depth 50 --pair-k 6) ;;
    *) echo "unknown run: $1"; exit 1 ;;
  esac
}
one() {
  local name=$1 gpu=$2 base=$1 seed=(); CMD=train
  if [[ $1 =~ ^(.+)_s([0-9]+)$ ]]; then base=${BASH_REMATCH[1]}; seed=(--seed "${BASH_REMATCH[2]}"); fi
  [[ -f $OUT/$name/submission/submission.zip ]] && { echo "[skip] $name finished"; return 0; }
  [[ -f $LEX/lexical.json ]] || { echo "[missing] $LEX/lexical.json (run: bash run_phase7.sh lexical)"; return 0; }
  spec "$base"
  [[ -f ${ARGS[1]} ]] || { echo "[missing] $name: ${ARGS[1]}"; return 0; }
  if [[ " ${ARGS[*]} " == *" --teacher "* ]]; then
    local t; for ((i = 0; i < ${#ARGS[@]}; i++)); do [[ ${ARGS[i]} == --teacher ]] && t=${ARGS[i+1]}; done
    [[ -f $t/model.pt ]] || { echo "[missing] teacher $t/model.pt"; return 0; }
  fi
  local flag dep
  for flag in --kd-list --init-from; do
    for ((i = 0; i < ${#ARGS[@]}; i++)); do [[ ${ARGS[i]} == "$flag" ]] && dep=${ARGS[i+1]} && \
      { [[ -f $dep/model.pt ]] || { echo "[missing] $flag $dep/model.pt"; return 0; }; }; done
  done
  if [[ $CMD == rerank ]]; then
    for ((i = 0; i < ${#ARGS[@]}; i++)); do [[ ${ARGS[i]} == --first ]] && dep=${ARGS[i+1]}; done
    [[ -f $dep/scores_dev.npz ]] || { echo "[missing] first stage $dep/scores_dev.npz (e.g. bash run_phase7.sh combine C_all ...)"; return 0; }
  fi
  echo "[GPU $gpu] $name (log: $OUT/$name.log)"
  CUDA_VISIBLE_DEVICES=$gpu $PY -m phase7.run "$CMD" "${TR[@]}" --output "$OUT/$name" "${ARGS[@]}" ${seed[@]+"${seed[@]}"} \
    > "$OUT/$name.log" 2>&1 || { echo "FAILED: $name"; tail -15 "$OUT/$name.log"; return 0; }
  grep -h "^dev F1@7" "$OUT/$name.log" | tail -1 | sed "s/^/$name: /" | cut -c1-220
}
pair() { one "$1" "$GPU_A" & local a=$!; one "$2" "$GPU_B" & local b=$!; wait $a; wait $b; }
queue() {
  local a=() b=() i=0
  for n in "$@"; do if (( i++ % 2 == 0 )); then a+=("$n"); else b+=("$n"); fi; done
  ( for n in ${a[@]+"${a[@]}"}; do one "$n" "$GPU_A"; done ) & local pa=$!
  ( for n in ${b[@]+"${b[@]}"}; do one "$n" "$GPU_B"; done ) & local pb=$!
  wait $pa; wait $pb
}

case "${1:-}" in
  fetch)      $PY -m phase2.fetch --output "$CACHE/raw" --token-file "${HF_TOKEN_FILE:-$CACHE/huggingface/token}" --folder Phase_7_PCR "${@:2}" ;;
  stats)      $PY -m phase7.run stats --data "$DATA" ;;
  lexical)    $PY -m phase7.run lexical --data "$DATA" --output "$OUT/B0_lexical" 2>&1 | tee "$OUT/B0_lexical.log" ;;
  lexical-context)   # + citation-context query views (sentences around each <CITATION> mark), fused with the full query
    $PY -m phase7.run lexical --data "$DATA" --output "$OUT/B1_lexical_ctx" --context-windows 0 1 2 5 2>&1 | tee "$OUT/B1_lexical_ctx.log" ;;
  lexical-prf)       # + pseudo-relevance feedback (kept only when it helps on train)
    $PY -m phase7.run lexical --data "$DATA" --output "$OUT/B2_lexical_prf" --context-windows 0 1 2 5 --prf 2>&1 | tee "$OUT/B2_lexical_prf.log" ;;
  lexical-dev)       # tuned on dev instead of train (does the train-tuned configuration transfer?)
    $PY -m phase7.run lexical --data "$DATA" --output "$OUT/B3_lexical_devtuned" --tune-on dev 2>&1 | tee "$OUT/B3_lexical_devtuned.log" ;;
  lexical-passage)   # + best-passage BM25 (150/300-token passages) and longer n-grams (6, 8, 10), with citation contexts
    $PY -m phase7.run lexical --data "$DATA" --output "$OUT/B4_lexical_passage" --grams 1 2 3 4 5 6 8 10 \
      --context-windows 1 2 3 5 8 --passage-words 150 300 2>&1 | tee "$OUT/B4_lexical_passage.log" ;;
  lexical-all)       bash "$SELF" lexical-context; bash "$SELF" lexical-prf; bash "$SELF" lexical-dev; bash "$SELF" lexical-passage ;;
  context)    pair R1_ctx1 R1_ctx2 ;;
  round2)     one R2_round2 "$GPU_A" ;;
  knobs)      queue R3_pairs6 R4_minsim05 R5_tau02 R6_tau10 R7_ep8 R8_top4 R9_top16 ;;
  kd-list)    pair K3_dense6_listkd K4_dense6_bothkd ;;
  rerank)     pair X1_rerank_lorma X3_rerank_on_lex; pair X2_rerank_full X4_rerank_deep50 ;;
  one)        one "$2" "${3:-$GPU_A}" ;;
  zeroshot)   queue Z1_dense_zeroshot Z2_gqa_zeroshot Z3_hybrid_zeroshot Z4_encdec_zeroshot ;;
  arch)       queue L1_dense_lorma L2_gqa_lorma L3_hybrid_lorma L4_encdec_lorma ;;
  techniques) queue T_full T_qlora T_partial T_ia3 T_prefix T_prompt ;;
  kd)         pair K1_dense6_kd K0_dense6_nokd ;;
  kd4)        one K2_dense4_kd "$GPU_A" ;;
  ablations)  queue A_bidir A_pool_last A_neg_random A_neg_none A_chunk100 A_chunk400 A_proj256 A_nodapt ;;
  seeds)      pair "${S1:-T_full}_s7" "${S2:-T_full}_s13" ;;
  combine)    # combine NAME RUN1 RUN2 ... (weights for each run + lexical chosen on dev)
    name=$2; shift 2; rs=(); for r in "$@"; do [[ -f $OUT/$r/scores_dev.npz ]] && rs+=("$OUT/$r") || echo "[missing] $r"; done
    (( ${#rs[@]} >= 1 )) || { echo "no finished runs"; exit 1; }
    # citation prior (train labels), later-year penalty and hubness correction: weights chosen on dev (0 = unused)
    $PY -m phase7.run combine --data "$DATA" --runs "${rs[@]}" --output "$OUT/$name" --fusion "${FUSION:-zscore}" \
      ${COMBINE_FLAGS---prior --years --hub --knn} ;;
  summary)
    $PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
rows = []
for f in sorted(Path(sys.argv[1]).glob("*/summary.json")):
    s = json.loads(f.read_text())
    n = f.parent.name
    if s.get("stage") == "lexical":
        d = s["dev"]; rows.append((d["f1"], n, "bm25", "-", d["f1"], None, d["f1"], d["precision"], d["recall"], "-", "-", "-", "-")); continue
    if s.get("stage") == "combine":
        rows.append((s["dev_f1"], n, "comb-" + s.get("fusion", "zscore")[:3], "-", None, None, s["dev_f1"], None, None,
                     "-", "-", "-", "-")); continue
    if s.get("stage") == "rerank":
        d = s["dev"]
        rows.append((d["fused"]["f1"], n, "rr-" + s["method"][:5], s["architecture"][:10], d["rerank_only"]["f1"], "pairs",
                     d["fused"]["f1"], d["fused"]["precision"], d["fused"]["recall"], s["weight_first"],
                     f"{s['parameters']['total']:,}", f"{s['parameters']['trainable']:,}",
                     f"{s.get('train_minutes', '-')}/{s.get('peak_gpu_gib', '-')}")); continue
    d = s["dev"]
    rows.append((d["fused"]["f1"], n, s["method"], s["architecture"][:10], d["neural"]["f1"], s["agg"], d["fused"]["f1"],
                 d["fused"]["precision"], d["fused"]["recall"], s["fusion_weight"], f"{s['parameters']['total']:,}",
                 f"{s['parameters']['trainable']:,}", f"{s.get('train_minutes', '-')}/{s.get('peak_gpu_gib', '-')}"))
print(f"{'run':22s} {'method':8s} {'arch':10s} {'neural':>7s} {'agg':10s} {'fused':>7s} {'P@7':>6s} {'R@7':>6s} {'lex-w':>5s} "
      f"{'params':>12s} {'trainable':>12s}  min/GiB")
for r in sorted(rows, key=lambda x: -x[0]):
    f = lambda v: f"{v:7.4f}" if isinstance(v, float) else f"{'-':>7s}"
    print(f"{r[1]:22s} {r[2]:8s} {r[3]:10s} {f(r[4])} {str(r[5] or '-'):10s} {f(r[6])} {f(r[7])[1:]} {f(r[8])[1:]} "
          f"{str(r[9]):>5s} {str(r[10]):>12s} {str(r[11]):>12s}  {r[12]}")
print("neural = Phase-1 retriever alone; fused = with the n-gram BM25 at weight lex-w (chosen on dev); "
      "rr- rows: neural = re-ranker alone over its first stage, lex-w = first-stage weight")
EOF
    ;;
  collect)    # submission_NAME.zip (fused) and submission_NAME_neural.zip (Phase-1 model alone)
    mkdir -p "$OUT/submissions"
    for d in "$OUT"/*/; do d=${d%/}; n=$(basename "$d")
      [[ -f $d/submission/submission.zip ]] && cp "$d/submission/submission.zip" "$OUT/submissions/submission_${n}.zip" && echo "submission_${n}.zip"
      [[ -f $d/submission_neural/submission.zip ]] && cp "$d/submission_neural/submission.zip" "$OUT/submissions/submission_${n}_neural.zip" && echo "submission_${n}_neural.zip"
    done; echo "-> $OUT/submissions/" ;;
  check)      $PY -m phase7.run check --data "$DATA" --prediction "$2" ;;
  *) sed -n '2,25p' "$SELF" ;;
esac
