"""Download Phase_2_LNER from the gated course dataset, with an explicit token and a clear diagnosis.

Token used (first match wins; the token itself is never printed):
  1. $HF_TOKEN
  2. --token-file (default: $HF_HOME/token, i.e. your own cs781_cache/huggingface/token)
  3. a hidden prompt, optionally saved to --token-file with permissions 600

The machine-wide ~/.cache/huggingface/token is deliberately NOT used: on a shared login it
may belong to someone else, which gives exactly "not in the authorized list" (403).

    python -m phase2.fetch --output $CACHE/raw            # download
    python -m phase2.fetch --output ... --check-only                             # only test access
"""
import argparse
import getpass
import os
import shutil
import sys
from pathlib import Path

REPO = "Exploration-Lab/CS-781-Capstone"
PROBE = "Phase_2_LNER/Train/labels.txt"
DATASETS = {
    "Phase_2_LNER": ("Phase_2_LNER/Train/labels.txt", None),
    "Phase_3_RR": ("Phase_3_RR/Train/label_vocab.json",
                   [f"Phase_3_RR/{s}/{d}_{p}.json" for d in ("CL", "IT")
                    for s, p in (("Train", "train"), ("Train", "dev"), ("Test", "test"))]
                   + ["Phase_3_RR/Train/label_vocab.json"]),
    # other official phases: only their judgment text is used in Phase 3 (unlabeled, for KD silver data)
    "Phase_4_CJPE": ("Phase_4_CJPE/Train/dev.json", ["Phase_4_CJPE/Train/train.json", "Phase_4_CJPE/Train/dev.json",
                                                      "Phase_4_CJPE/Test/test.json"]),
    "Phase_5_LSI": ("Phase_5_LSI/Train/dev.json", ["Phase_5_LSI/Train/train.json", "Phase_5_LSI/Train/dev.json",
                                                    "Phase_5_LSI/Test/test.json"]),
    "Phase_6_Summ": ("Phase_6_Summ/Train/train.json", ["Phase_6_Summ/Train/train.json", "Phase_6_Summ/Test/test.json"]),
    "Phase_7_PCR": ("Phase_7_PCR/Train/train.json", ["Phase_7_PCR/Train/train.json", "Phase_7_PCR/Train/train_candidates.json",
                                                   "Phase_7_PCR/Dev/dev.json", "Phase_7_PCR/Dev/dev_candidates.json",
                                                   "Phase_7_PCR/Test/test.json", "Phase_7_PCR/Test/test_candidates.json"]),
}
EXPECTED = ["Phase_2_LNER/Train/fold1-text.json", "Phase_2_LNER/Train/fold1-gold.json",
            "Phase_2_LNER/Train/fold2-text.json", "Phase_2_LNER/Train/fold2-gold.json",
            "Phase_2_LNER/Train/labels.txt", "Phase_2_LNER/Test/fold3-text.json"]
TOKENS_PAGE = "https://huggingface.co/settings/tokens"
DATASET_PAGE = f"https://huggingface.co/datasets/{REPO}"

try:
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
except ImportError:  # older huggingface_hub
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError


def mask(token):
    return f"{token[:3]}…{token[-4:]}" if len(token) > 10 else "…"


def save_token(token, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(token + "\n")
    os.chmod(path, 0o600)


def resolve_token(token_file, interactive=None):
    """Return (token, source). Raises SystemExit with instructions if none is available."""
    interactive = sys.stdin.isatty() if interactive is None else interactive
    env = os.environ.get("HF_TOKEN", "").strip()
    if env:
        return env, "$HF_TOKEN"
    token_file = Path(token_file)
    if token_file.is_file():
        token = token_file.read_text().strip()
        if token:
            return token, str(token_file)
    if not interactive:
        raise SystemExit(
            f"No Hugging Face token. Either\n"
            f"  export HF_TOKEN=hf_...            (this shell only), or\n"
            f"  run this command in an interactive terminal to paste it once and save it to {token_file}\n"
            f"Create a read token at {TOKENS_PAGE} while logged in as your course-registered account.")
    print(f"Paste a Hugging Face READ token for the account that has access to {REPO}")
    print(f"(create one at {TOKENS_PAGE}; input is hidden)")
    token = getpass.getpass("HF token: ").strip()
    if not token:
        raise SystemExit("No token entered.")
    answer = input(f"Save it to {token_file} (permissions 600) so later runs don't ask? [y/N] ").strip().lower()
    if answer in {"y", "yes"}:
        save_token(token, token_file)
        print(f"Saved to {token_file}")
    return token, "prompt"


def describe(api, token):
    """whoami -> (account name, human-readable token kind, warning or None)."""
    try:
        info = api.whoami(token=token)
    except HfHubHTTPError as error:
        raise SystemExit(f"The token was rejected by Hugging Face ({error.__class__.__name__}): it is invalid, "
                         f"expired or revoked. Create a new READ token at {TOKENS_PAGE}.")
    except Exception as error:  # connection/timeout/proxy errors (requests in 0.x, httpx in 1.x)
        raise SystemExit(f"Could not reach huggingface.co ({error.__class__.__name__}: {error}).\n"
                         "If this node needs a proxy, export HTTPS_PROXY (as for your Phase-1 downloads) and retry.")
    name = info.get("name", "?")
    access = (info.get("auth") or {}).get("accessToken") or {}
    role = access.get("role", "unknown")
    warning = None
    fine = access.get("fineGrained") or {}
    if role == "fineGrained" and fine.get("canReadGatedRepos") is False:
        warning = ("this fine-grained token cannot read gated repos: edit it at "
                   f"{TOKENS_PAGE} and tick 'Read access to contents of all public gated repos you can access' "
                   "(or create a classic READ token instead)")
    return name, role, warning


def diagnose_gated(name, warning):
    lines = [f"\nAccount '{name}' cannot read {REPO} (gated dataset, 403).", "What to check, in order:"]
    if warning:
        lines.append(f"  0. {warning}")
    lines += [
        f"  1. Is '{name}' the Hugging Face account you were approved with (the one that downloaded Phase 1)?",
        "     If not, use that account's token: export HF_TOKEN=hf_...  or rerun with --reset-token.",
        f"  2. If it is: open {DATASET_PAGE} logged in as '{name}' and look at the access banner",
        "     (granted / pending / not requested). If it is not granted, request access again and tell the TAs —",
        "     the organizers may have reset the access list when they published Phase 2.",
    ]
    return "\n".join(lines)


def fetch_all(output, token_file, interactive=None, only=None):
    """Every folder of the course dataset (Phase_1_*, Phase_2_*, ... whatever the organizers published)."""
    from huggingface_hub import HfApi, snapshot_download
    token, source = resolve_token(token_file, interactive)
    api = HfApi()
    name, role, warning = describe(api, token)
    print(f"token from {source}: account '{name}', type {role}, {mask(token)}")
    try:
        files = api.list_repo_files(REPO, repo_type="dataset", token=token)
    except GatedRepoError:
        raise SystemExit(diagnose_gated(name, warning))
    except RepositoryNotFoundError:
        raise SystemExit(f"Account '{name}' cannot see {REPO} at all (404).")
    available = sorted({f.split("/")[0] for f in files if "/" in f})
    print(f"{len(files)} files in {REPO}; folders available: {available}")
    if not available:
        print("files: " + ", ".join(sorted(files)[:40]))
        raise SystemExit("the dataset has no folders this account can see")
    folders = available
    if only:   # exact names, prefixes ("Phase_5" -> "Phase_5_LSI") or phase numbers ("5"), any case
        def wanted(f):
            low = f.lower()
            for o in only:
                o = o.strip().lower().rstrip("_/")
                if o.isdigit():
                    o = f"phase_{o}"
                if low == o or low.startswith(o + "_"):
                    return True
            return False
        folders = [f for f in available if wanted(f)]
        if not folders:
            raise SystemExit(f"no folder matches {only}; available: {available}")
    print(f"downloading: {folders}", flush=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sizes = folder_sizes(api, token)
    for folder in folders:
        print(f"== {folder}", flush=True)
        need = sum(n for f, n in sizes.items() if f.startswith(folder + "/")
                   and not ((output / f).is_file() and (output / f).stat().st_size == n))
        free = shutil.disk_usage(output).free
        if need + RESERVE > free:                      # never start a download that cannot finish
            print(f"  SKIPPED {folder}: needs {need / 1e9:.1f} GB, only {free / 1e9:.1f} GB free under {output} "
                  f"(keeping {RESERVE / 1e9:.0f} GB spare).\n"
                  f"  Fetch it to a larger disk with --output, or fetch only what you need"
                  + (" (Phase 1: bash phase1/fetch_data.sh fetches SC + HC only; tools/data_stats.py streams its "
                     "samples and needs no download)." if folder.startswith("Phase_1") else "."), flush=True)
            continue
        try:
            snapshot_download(REPO, repo_type="dataset", allow_patterns=[f"{folder}/*"], local_dir=output, token=token)
        except Exception as error:                     # keep going: one folder's failure should not stop the rest
            print(f"  FAILED {folder}: {error.__class__.__name__}: {error}")
            continue
        for f in sorted(x for x in files if x.startswith(folder + "/")):
            path = output / f
            print(f"  {path.stat().st_size / 1e6 if path.is_file() else float('nan'):9.2f} MB  {path}"
                  + ("" if path.is_file() else "   MISSING"))
    return output


RESERVE = 2 * 1024**3        # bytes left free after a folder download (logs, checkpoints, scratch)


def folder_sizes(api, token):
    """{repo path: bytes} for every file in the dataset (empty if the metadata call fails)."""
    try:
        info = api.dataset_info(REPO, files_metadata=True, token=token)
        return {item.rfilename: item.size or 0 for item in info.siblings}
    except Exception as error:
        print(f"  (could not read file sizes: {type(error).__name__}; downloading without a space check)")
        return {}


def fetch(output, token_file, check_only=False, interactive=None, folder="Phase_2_LNER"):
    probe, expected = DATASETS[folder]
    expected = expected or EXPECTED
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    token, source = resolve_token(token_file, interactive)
    api = HfApi()
    name, role, warning = describe(api, token)
    print(f"token from {source}: account '{name}', type {role}, {mask(token)}")
    if warning:
        print("WARNING:", warning)
    output = Path(output)
    try:
        hf_hub_download(REPO, probe, repo_type="dataset", token=token, local_dir=output)
    except GatedRepoError:
        raise SystemExit(diagnose_gated(name, warning))
    except RepositoryNotFoundError:
        raise SystemExit(f"Account '{name}' cannot see {REPO} at all (404). Check the repo name and that you "
                         f"are logged in as the course account.")
    print(f"access OK: '{name}' can read {REPO}")
    if check_only:
        return output
    snapshot_download(REPO, repo_type="dataset", allow_patterns=[f"{folder}/*"], local_dir=output, token=token)
    missing = [f for f in expected if not (output / f).is_file()]
    if missing:
        raise SystemExit(f"Download finished but files are missing: {missing}")
    for f in expected:
        print(f"  {(output / f).stat().st_size / 1e6:8.2f} MB  {output / f}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="directory that will contain Phase_2_LNER/")
    default_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    parser.add_argument("--token-file", default=str(Path(default_home) / "token"))
    parser.add_argument("--check-only", action="store_true", help="verify the token and access, download nothing big")
    parser.add_argument("--reset-token", action="store_true", help="forget the saved token and ask again")
    parser.add_argument("--folder", default="Phase_2_LNER", choices=sorted(DATASETS), help="which phase to download")
    parser.add_argument("--all", action="store_true", help="every folder in the course dataset (all phases)")
    parser.add_argument("--only", nargs="+", help="with --all: just these folders - names, prefixes (Phase_5) or numbers (5)")
    args = parser.parse_args()
    if args.reset_token and Path(args.token_file).is_file():
        Path(args.token_file).unlink()
        print(f"Removed {args.token_file}")
    if args.all:
        fetch_all(args.output, args.token_file, only=args.only)
        return
    fetch(args.output, args.token_file, args.check_only, folder=args.folder)


if __name__ == "__main__":
    main()
