"""Download the official Phase-1 pretraining data from the gated course dataset.

    bash phase1/fetch_data.sh                    # SC + HC training files and the test file (what our models used)
    bash phase1/fetch_data.sh --all-sources      # every Phase_1_Pretraining/Train/*.json
    bash phase1/fetch_data.sh --check-only       # test the token and access only, download nothing
    bash phase1/fetch_data.sh --reset-token      # forget the saved token and ask again

Which token is used (first match wins; the token itself is never printed):
  1. $HF_TOKEN
  2. --token-file (the launcher passes $CACHE/huggingface/token, your own cache)
  3. a hidden prompt, optionally saved to --token-file with permissions 600
The machine-wide ~/.cache/huggingface/token is never read. On a shared login it may belong to
someone else, which gives "not in the authorized list" (403).

The files land in <raw-dir>/Phase_1_Pretraining/{Train,Test}/. A manifest with the dataset
commit, file sizes and local paths is written to <output>/source_manifest.json. `phase1.data
prepare --source-manifest` reads it, so the prepared corpus records exactly which revision it came from.
"""
import argparse
import os
from pathlib import Path

from phase1.data import PREFIX, REPO, inspect, resolve_official
from phase2.fetch import describe, diagnose_gated, mask, resolve_token

try:
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
except ImportError:  # older huggingface_hub
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

USED = ["train_supreme_court.json", "train_high_courts.json"]   # the two sources the submitted models were trained on


def train_files(api, token, revision):
    """Every Train/*.json file name in the Phase-1 folder at this revision."""
    files = api.list_repo_files(REPO, repo_type="dataset", revision=revision, token=token)
    names = sorted(f[len(PREFIX + "Train/"):] for f in files
                   if f.startswith(PREFIX + "Train/") and f.lower().endswith(".json"))
    if not names:
        folders = sorted({f.split("/")[0] for f in files if "/" in f})
        raise SystemExit(f"No {PREFIX}Train/*.json at revision {revision}; folders present: {folders}")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", required=True, help="download root; files go to <raw-dir>/Phase_1_Pretraining/")
    parser.add_argument("--output", required=True, help="where source_manifest.json is written")
    parser.add_argument("--token-file", required=True, help="your own token file, e.g. $CACHE/huggingface/token")
    parser.add_argument("--sources", nargs="+", default=USED, help=f"training files to fetch (default: {' '.join(USED)})")
    parser.add_argument("--all-sources", action="store_true", help="every Train/*.json in the Phase-1 folder")
    parser.add_argument("--test-file", help="repository path of the test file, if the Test/ folder holds more than one")
    parser.add_argument("--revision", default="main", help="dataset branch, tag or commit (default: main)")
    parser.add_argument("--check-only", action="store_true", help="verify the token and access, download nothing")
    parser.add_argument("--reset-token", action="store_true", help="forget the saved token and ask again")
    parser.add_argument("--inspect", type=int, default=2, metavar="N", help="print N sample records per training file (0: off)")
    args = parser.parse_args()

    if args.reset_token and Path(args.token_file).is_file():
        Path(args.token_file).unlink()
        print(f"Removed {args.token_file}")

    from huggingface_hub import HfApi
    token, source = resolve_token(args.token_file)
    api = HfApi()
    name, role, warning = describe(api, token)
    print(f"token from {source}: account '{name}', type {role}, {mask(token)}")
    if warning:
        print("WARNING:", warning)

    try:
        available = train_files(api, token, args.revision)
    except GatedRepoError:
        raise SystemExit(diagnose_gated(name, warning))
    except RepositoryNotFoundError:
        raise SystemExit(f"Account '{name}' cannot see {REPO} at all (404). Log in with the course-registered account.")
    print(f"access OK: '{name}' can read {REPO}")
    print(f"training files at {args.revision}: {', '.join(available)}")
    if args.check_only:
        return

    sources = available if args.all_sources else args.sources
    unknown = [s for s in sources if s not in available]
    if unknown:
        raise SystemExit(f"Not in {PREFIX}Train/: {unknown}. Available: {available}")

    os.environ["HF_TOKEN"] = token      # this process only: resolve_official reads it from the environment
    try:
        meta = resolve_official(args.output, sources, revision=args.revision,
                                test_path=args.test_file, raw_dir=args.raw_dir)
    except OSError as error:                            # the disk check runs before anything is downloaded
        raise SystemExit(f"{error}\nNothing was downloaded. Point the download at a larger disk, e.g.\n"
                         f"  RAW=/path/with/space/raw bash phase1/fetch_data.sh\n"
                         f"or fetch fewer sources (--sources train_supreme_court.json is 1.3 GB). "
                         f"tools/data_stats.py does not need this download: it streams its Phase-1 samples.")
    print(f"\ndataset commit {meta['revision']}")
    for item, path in zip(meta["files"], meta["train_paths"] + [meta["test_path"]]):
        print(f"  {item['bytes'] / 1e6:9.1f} MB  {path}")
    print(f"manifest: {Path(args.output) / 'source_manifest.json'}")
    if args.inspect:
        inspect(meta["train_paths"], limit=args.inspect)        # the test file has prompts, not {"text": ...}


if __name__ == "__main__":
    main()
