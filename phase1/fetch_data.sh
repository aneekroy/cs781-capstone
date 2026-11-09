#!/usr/bin/env bash
# Download the official Phase-1 pretraining data (gated Hugging Face dataset) with YOUR OWN token.
#
#   bash phase1/fetch_data.sh                   # train_supreme_court.json + train_high_courts.json + the test file
#   bash phase1/fetch_data.sh --all-sources     # every Phase_1_Pretraining/Train/*.json
#   bash phase1/fetch_data.sh --check-only      # which account the token belongs to, and whether it has access
#   bash phase1/fetch_data.sh --reset-token     # forget the saved token and ask again
#   bash phase1/fetch_data.sh --revision <sha>  # a fixed dataset commit (see source_manifest.json of an earlier fetch)
#
# Token: $HF_TOKEN, else $CACHE/huggingface/token, else a hidden prompt (optionally saved there, mode 600).
# The machine-wide ~/.cache/huggingface/token is never used. The token is never printed.
#
# Environment (override as needed):
#   CACHE    download cache; files go to $CACHE/raw/Phase_1_Pretraining/{Train,Test}/
#   PROJECT  project directory; the manifest goes to $PROJECT/data/download/source_manifest.json
#   PY       python interpreter
set -euo pipefail

CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT=${PROJECT:-$HOME/cs781_phase1}
CACHE=${CACHE:-$HOME/cs781_cache}
PY=${PY:-python}

export HF_HOME=$CACHE/huggingface            # your own cache, never ~/.cache/huggingface
export HF_HUB_DISABLE_TELEMETRY=1
TOKEN_FILE=${HF_TOKEN_FILE:-$HF_HOME/token}
mkdir -p "$CACHE/raw" "$PROJECT/data/download"

$PY -c "import huggingface_hub, ijson" 2>/dev/null \
  || { echo "missing packages: pip install -r $CODE/phase1/requirements.txt"; exit 1; }

cd "$CODE"
$PY -m phase1.fetch --raw-dir "$CACHE/raw" --output "$PROJECT/data/download" --token-file "$TOKEN_FILE" "$@"

for arg in "$@"; do [[ $arg == --check-only ]] && exit 0; done
RAW=$CACHE/raw/Phase_1_Pretraining/Train
cat <<NEXT

Next step: tokenize (training-only 24k BPE, disk-backed token corpus, held-out continuations):
  cd $CODE && $PY -m phase1.data prepare \\
      --paths $RAW/train_supreme_court.json $RAW/train_high_courts.json \\
      --weights 0.6 0.4 --train-tokens 428507136 --val-tokens 5000000 --seed 42 \\
      --source-manifest $PROJECT/data/download/source_manifest.json \\
      --output $PROJECT/data/mixed_428507136_s42
(These are the settings of the submitted run: the largest measured budget, SC:HC = 0.6:0.4.)
NEXT
