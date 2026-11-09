"""Token resolution and access diagnosis for phase2.fetch (no network: the hub is mocked)."""
import os, stat, sys
from pathlib import Path
import pytest
import huggingface_hub

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2 import fetch as F


def hub_error(cls, message, status):
    """Hub errors take a required response= in huggingface_hub 1.x and none in 0.x."""
    try:
        return cls(message)
    except TypeError:
        import httpx
        return cls(message, response=httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co")))


def test_env_token_wins(tmp_path, monkeypatch):
    (tmp_path / "token").write_text("hf_file_token\n")
    monkeypatch.setenv("HF_TOKEN", "hf_env_token")
    assert F.resolve_token(tmp_path / "token", interactive=False) == ("hf_env_token", "$HF_TOKEN")


def test_file_token_and_noninteractive_error(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="export HF_TOKEN"):
        F.resolve_token(tmp_path / "token", interactive=False)
    F.save_token("hf_saved_token_1234", tmp_path / "sub" / "token")
    assert stat.S_IMODE(os.stat(tmp_path / "sub" / "token").st_mode) == 0o600
    assert F.resolve_token(tmp_path / "sub" / "token", interactive=False)[0] == "hf_saved_token_1234"


def test_prompt_saves(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(F.getpass, "getpass", lambda prompt: "hf_typed_token_5678")
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert F.resolve_token(tmp_path / "token", interactive=True) == ("hf_typed_token_5678", "prompt")
    assert (tmp_path / "token").read_text().strip() == "hf_typed_token_5678"


class FakeApi:
    def __init__(self, info=None, error=None):
        self.info, self.error = info, error

    def whoami(self, token):
        if self.error:
            raise self.error
        return self.info


def patch_hub(monkeypatch, api, download_error=None, written=()):
    calls = {"snapshot": 0}

    def download(repo, filename, repo_type, token, local_dir):
        if download_error:
            raise download_error
        Path(local_dir, filename).parent.mkdir(parents=True, exist_ok=True)
        Path(local_dir, filename).write_text("APP\n")

    def snapshot(repo, repo_type, allow_patterns, local_dir, token):
        calls["snapshot"] += 1
        for f in written:
            Path(local_dir, f).parent.mkdir(parents=True, exist_ok=True)
            Path(local_dir, f).write_text("{}")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_abcd")
    return calls


def test_gated_403_names_account_and_fine_grained_flag(tmp_path, monkeypatch):
    info = {"name": "someone_else", "auth": {"accessToken": {"role": "fineGrained",
                                                              "fineGrained": {"canReadGatedRepos": False}}}}
    patch_hub(monkeypatch, FakeApi(info), download_error=hub_error(F.GatedRepoError, "403 not in the authorized list", 403))
    with pytest.raises(SystemExit) as exc:
        F.fetch(tmp_path, tmp_path / "token")
    message = str(exc.value)
    assert "someone_else" in message and "gated repos" in message and "request access" in message


def test_invalid_token(tmp_path, monkeypatch):
    patch_hub(monkeypatch, FakeApi(error=hub_error(F.HfHubHTTPError, "401 Unauthorized", 401)))
    with pytest.raises(SystemExit, match="rejected"):
        F.fetch(tmp_path, tmp_path / "token")


def test_check_only_and_full_download(tmp_path, monkeypatch, capsys):
    info = {"name": "aneek", "auth": {"accessToken": {"role": "read"}}}
    calls = patch_hub(monkeypatch, FakeApi(info), written=F.EXPECTED)
    F.fetch(tmp_path, tmp_path / "token", check_only=True)
    assert calls["snapshot"] == 0
    F.fetch(tmp_path, tmp_path / "token")
    assert calls["snapshot"] == 1 and all((tmp_path / f).is_file() for f in F.EXPECTED)
    out = capsys.readouterr().out
    assert "access OK" in out and "hf_test_token_abcd" not in out   # token never printed


# ---- phase1.fetch: Phase-1 corpus download (hub mocked) ----
class FakeP1Api:
    FILES = ["Phase_1_Pretraining/Train/train_supreme_court.json", "Phase_1_Pretraining/Train/train_high_courts.json",
             "Phase_1_Pretraining/Train/train_acts.json", "Phase_1_Pretraining/Test/test.json", "Phase_2_LNER/Train/labels.txt"]

    def __init__(self, token=None):
        pass

    def whoami(self, token):
        return {"name": "student", "auth": {"accessToken": {"role": "read"}}}

    def list_repo_files(self, repo, repo_type, revision, token):
        return self.FILES

    def dataset_info(self, repo, revision, files_metadata):
        class Item:
            def __init__(self, name): self.rfilename, self.size = name, 10
        class Info: pass
        info = Info(); info.sha = "abc123"; info.siblings = [Item(f) for f in self.FILES]
        return info


def run_p1(monkeypatch, tmp_path, *extra):
    import json
    from phase1 import fetch as P1, data as D
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeP1Api)
    downloaded = []

    def fake_download(repo, name, repo_type, revision, token, local_dir):
        path = Path(local_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([{"id": "a", "text": "the court held"}]))
        downloaded.append(name)
        return str(path)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setenv("HF_TOKEN", "hf_env_token_0000")
    monkeypatch.setattr(sys, "argv", ["fetch", "--raw-dir", str(tmp_path / "raw"), "--output", str(tmp_path / "dl"),
                                      "--token-file", str(tmp_path / "token"), "--inspect", "0", *extra])
    P1.main()
    return downloaded


def test_phase1_fetch_default_sources(tmp_path, monkeypatch, capsys):
    got = run_p1(monkeypatch, tmp_path)
    assert got == ["Phase_1_Pretraining/Train/train_supreme_court.json",
                   "Phase_1_Pretraining/Train/train_high_courts.json", "Phase_1_Pretraining/Test/test.json"]
    assert (tmp_path / "dl" / "source_manifest.json").is_file()
    assert "hf_env_token_0000" not in capsys.readouterr().out          # the token is never printed


def test_phase1_fetch_all_and_check_only(tmp_path, monkeypatch):
    assert len(run_p1(monkeypatch, tmp_path, "--all-sources")) == 4
    assert run_p1(monkeypatch, tmp_path / "c", "--check-only") == []
    with pytest.raises(SystemExit, match="Not in"):
        run_p1(monkeypatch, tmp_path / "d", "--sources", "train_missing.json")
