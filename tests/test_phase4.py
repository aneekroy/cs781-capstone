"""CPU tests for Phase 4 (CJPE) on synthetic judgments + tiny random Phase-1 checkpoints:
    python -m pytest -q tests/test_phase4.py
"""
import json, os, random, subprocess, sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from _paths import phase1_root
PHASE1 = phase1_root()
sys.path.insert(1, str(PHASE1))

from phase4.data import (best_threshold, honest_threshold, load_split, macro_f1, near_duplicates,
                         validate_prediction, write_prediction)

FX = Path("/tmp/fx_test")
P4 = Path("/tmp/fx_p4")
FILLER = ("The learned counsel for the appellant submitted that the High Court erred in law . The respondent "
          "contended that the findings of fact were concurrent and called for no interference . We have heard "
          "the parties and perused the record carefully . ").split()


def judgment(rng, label):
    body = " ".join(rng.choice(FILLER) for _ in range(rng.randint(300, 900)))
    if label:
        end = rng.choice(["The appeal is allowed and the impugned order is set aside .",
                          "Accordingly the appeal succeeds and the judgment of the High Court is quashed ."])
    else:
        end = rng.choice(["The appeal is dismissed and the impugned order is upheld .",
                          "We find no merit in the appeal which fails and is dismissed ."])
    return body + " " + end + " " + " ".join(rng.choice(FILLER) for _ in range(rng.randint(5, 30)))


def build(root, n_train=60, n_dev=20, n_test=16, seed=0):
    rng = random.Random(seed)
    (root / "Phase_4_CJPE" / "Train").mkdir(parents=True, exist_ok=True)
    (root / "Phase_4_CJPE" / "Test").mkdir(parents=True, exist_ok=True)
    rows = lambda n, start, lab: [{"id": f"case_{start + i}", "text": judgment(rng, (i % 2) if lab else rng.randint(0, 1)),
                                   **({"label": i % 2} if lab else {})} for i in range(n)]
    json.dump(rows(n_train, 0, True), open(root / "Phase_4_CJPE/Train/train.json", "w"))
    json.dump(rows(n_dev, 1000, True), open(root / "Phase_4_CJPE/Train/dev.json", "w"))
    json.dump(rows(n_test, 2000, False), open(root / "Phase_4_CJPE/Test/test.json", "w"))
    return root / "Phase_4_CJPE"


@pytest.fixture(scope="session", autouse=True)
def fixtures():
    if not (FX / "planted.json").exists():
        subprocess.run([sys.executable, str(ROOT / "tests/make_fixture.py"), str(FX)], check=True)
    build(P4)


def run(*argv, check=True):
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{PHASE1}")
    r = subprocess.run([sys.executable, *argv], cwd=ROOT, env=env, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(r.stdout[-3000:] + r.stderr[-3000:])
    return r


def test_loaders_accept_list_dict_jsonl_and_string_labels(tmp_path):
    d = tmp_path / "Phase_4_CJPE"
    (d / "Train").mkdir(parents=True)
    (d / "Test").mkdir()
    json.dump({"a": {"text": "x y", "label": "accepted"}, "b": {"text": "z", "label": 0}}, open(d / "Train/train.json", "w"))
    (d / "Train/dev.jsonl").write_text('{"id": 7, "text": "q", "label": 1}\n')
    json.dump([{"id": "t1", "text": "w"}], open(d / "Test/test.json", "w"))
    tr, dv, te = load_split(d, "train"), load_split(d, "dev"), load_split(d, "test")
    assert tr["a"]["label"] == 1 and tr["b"]["label"] == 0 and dv["7"]["label"] == 1 and te["t1"]["label"] is None


def test_scorer_threshold_and_writer(tmp_path):
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 300), rng.integers(0, 2, 300)
    assert abs(macro_f1(y, p) - f1_score(y, p, average="macro")) < 1e-12
    prob = np.clip(y * 0.3 + rng.random(300) * 0.6, 0, 1)        # informative but shifted
    t, f = best_threshold(y, prob)
    assert f >= macro_f1(y, (prob >= 0.5).astype(int)) and 0 < t < 1
    assert 0 <= honest_threshold(y, prob) <= 1
    test = load_split(P4 / "Phase_4_CJPE", "test")
    info = write_prediction({d: 1 for d in test}, test, tmp_path / "sub")
    assert info["documents"] == len(test) and (tmp_path / "sub" / "submission.zip").exists()
    good = [{"id": d, "pred": 0} for d in test]
    assert validate_prediction(good, test)
    for bad in (good[1:], good + [{"id": "zzz", "pred": 0}], [{"id": d, "pred": 2} for d in test],
                [{"id": d, "pred": True} for d in test]):
        with pytest.raises(ValueError):
            validate_prediction(bad, test)
    docs = {"x": {"text": " ".join(f"w{i}" for i in range(400))}}
    assert near_duplicates(docs, {"y": {"text": docs["x"]["text"]}}) and not near_duplicates(docs, {"z": {"text": "a b c"}})


COMMON = ["--project-root", str(PHASE1), "--chunk", "64", "--tail-chunks", "2", "--embed-chunks", "4",
          "--batch-size", "8", "--eval-batch", "16", "--rank", "4", "--alpha", "8"]


def test_all_stages_end_to_end():
    data = str(P4 / "Phase_4_CJPE")
    out = P4 / "runs"
    run("-m", "phase4.run", "stats", "--data", data)
    b = run("-m", "phase4.run", "baseline", "--data", data, "--output", str(out / "B0"), "--c-grid", "1", "4")
    assert json.loads((out / "B0" / "summary.json").read_text())["dev_macro_f1_at_0.5"] > 0.6   # the ending is learnable
    ck = str(FX / "phase1_runs" / "dense_gpt" / "best.pt")
    run("-m", "phase4.run", "finetune", "--data", data, *COMMON, "--checkpoint", ck, "--method", "lorma",
        "--epochs", "1", "--head-chunk", "--output", str(out / "F1"))
    s = json.loads((out / "F1" / "summary.json").read_text())
    assert s["stage"] == "finetune" and (out / "F1" / "model.pt").exists() and "dev_macro_f1_honest" in s
    run("-m", "phase4.run", "embed", "--data", data, "--run", str(out / "F1"), "--embed-chunks", "4")
    run("-m", "phase4.run", "aggregate", "--data", data, "--run", str(out / "F1"), "--output", str(out / "A1"),
        "--epochs", "3", "--seeds", "2", "--hidden", "16")
    # KD: truncated student of F1; another technique on another architecture
    kd = run("-m", "phase4.run", "finetune", "--data", data, *COMMON, "--checkpoint", ck, "--method", "lorma",
             "--keep-layers", "2", "--teacher", str(out / "F1"), "--epochs", "1", "--output", str(out / "K1"))
    assert "KD: teacher" in kd.stdout
    run("-m", "phase4.run", "finetune", "--data", data, *COMMON, "--checkpoint", str(FX / "phase1_runs/gqa_local/best.pt"),
        "--method", "ia3", "--epochs", "1", "--max-train-docs", "30", "--output", str(out / "F2"))
    e = run("-m", "phase4.ensemble", "--data", data, "--runs", *(str(out / r) for r in ("B0", "F1", "A1", "K1", "F2")),
            "--output", str(out / "E1"))
    assert "ensemble (" in e.stdout
    run("-m", "phase4.run", "check", "--data", data, "--prediction", str(out / "E1" / "submission" / "prediction.json"))


def test_repeated_ids_and_test_copies_are_handled(tmp_path, monkeypatch):
    import phase4.data as D
    d = tmp_path / "P"
    (d / "Train").mkdir(parents=True)
    (d / "Test").mkdir()
    w = lambda n, seed: " ".join(f"w{(i * seed) % 997}" for i in range(n))
    (d / "Test" / "test.json").write_text(json.dumps([{"id": "t1", "text": w(400, 3)}, {"id": "t2", "text": w(400, 5)}]))
    (d / "Train" / "train.json").write_text(json.dumps([
        {"id": "a", "text": w(400, 3), "label": 1}, {"id": "b", "text": w(400, 7), "label": 0},
        {"id": "b", "text": w(400, 7), "label": 0}, {"id": "b", "text": w(400, 13), "label": 1},
        {"id": "c", "text": w(400, 5) + " extra", "label": 1}]))
    (d / "Train" / "dev.json").write_text(json.dumps([{"id": "t1", "text": w(400, 3).upper(), "label": 0},
                                                     {"id": "e", "text": w(400, 11), "label": 1}]))
    assert list(D.load_split(d, "train")) == ["b", "b#2"]          # repeat dropped, exact + near test copies dropped
    assert list(D.load_split(d, "dev")) == ["e"]
    assert list(D.load_split(d, "train", exclude_test=False)) == ["a", "b", "b#2", "c"]
    monkeypatch.setenv("CJPE_KEEP_TEST_OVERLAP", "1")
    assert len(D.load_split(d, "dev")) == 2
