"""CPU tests for Phase 5 (Legal Statute Identification) on synthetic multi-label facts + tiny Phase-1 checkpoints:
    python -m pytest -q tests/test_phase5.py
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

from phase5.data import (apply_thresholds, choose_thresholds, fit_thresholds, label_space, load_split, multi_hot,
                         scores, validate_prediction, write_prediction)

FX = Path("/tmp/fx_test")
P5 = Path("/tmp/fx_p5")
FILLER = ("the prosecution case is that on the said day the accused persons reached the village and the witnesses "
          "stated before the police that the incident took place near the house of the complainant").split()
# statute id -> its tell-tale phrase in the facts; 498 and 376 are rare
CUES = {302: "caused death of the deceased", 324: "attacked with a sharp weapon", 498: "harassed the wife for dowry",
        379: "stole the motorcycle", 376: "committed rape on the victim", 34: "in furtherance of common intention"}
RARE = {498, 376}


def facts(rng, labels):
    words = [rng.choice(FILLER) for _ in range(rng.randint(150, 500))]
    for l in labels:
        at = rng.randrange(len(words))                       # the cue can be anywhere in the facts
        words[at:at] = CUES[l].split()
    return " ".join(words)


def draw_labels(rng):
    common = [302, 324, 379, 34]
    labels = set(rng.sample(common, rng.randint(1, 2)))
    if rng.random() < 0.12:
        labels.add(rng.choice(sorted(RARE)))
    return sorted(labels)


def build(root, n_train=80, n_dev=40, n_test=20, seed=0):
    rng = random.Random(seed)
    d = root / "Phase_5_LSI"
    (d / "Train").mkdir(parents=True, exist_ok=True)
    (d / "Test").mkdir(parents=True, exist_ok=True)

    def rows(n, start, labelled):
        out = []
        for i in range(n):
            labels = draw_labels(rng)
            if labelled and i < 2:                            # make sure every statute occurs in train and dev
                labels = sorted(RARE)
            out.append({"id": f"case_{start + i}", "text": facts(rng, labels), **({"labels": labels} if labelled else {})})
        return out
    json.dump(rows(n_train, 0, True), open(d / "Train/train.json", "w"))
    json.dump(rows(n_dev, 1000, True), open(d / "Train/dev.json", "w"))
    json.dump(rows(n_test, 2000, False), open(d / "Test/test.json", "w"))
    return d


@pytest.fixture(scope="session", autouse=True)
def fixtures():
    if not (FX / "planted.json").exists():
        subprocess.run([sys.executable, str(ROOT / "tests/make_fixture.py"), str(FX)], check=True)
    build(P5)


def run(*argv, check=True):
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{PHASE1}")
    r = subprocess.run([sys.executable, *argv], cwd=ROOT, env=env, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(r.stdout[-3000:] + r.stderr[-3000:])
    return r


def test_scores_match_sklearn_and_thresholds():
    from sklearn.metrics import accuracy_score, f1_score
    rng = np.random.default_rng(0)
    y = (rng.random((300, 12)) < rng.uniform(0.02, 0.4, 12)).astype(int)
    pred = (rng.random((300, 12)) < 0.2).astype(int)
    s = scores(y, pred)
    assert abs(s["macro_f1"] - f1_score(y, pred, average="macro", zero_division=0)) < 1e-12
    assert abs(s["micro_f1"] - f1_score(y, pred, average="micro", zero_division=0)) < 1e-12
    assert abs(s["subset_accuracy"] - accuracy_score(y, pred)) < 1e-12
    scale = rng.uniform(0.15, 0.6, 12)                        # per-statute miscalibration: per-label thresholds win
    p = np.clip((y * 0.8 + rng.random((300, 12)) * 0.35) * scale, 0, 1)
    cfg, t, table = choose_thresholds(y, p, log=lambda *a: None)
    assert cfg["kind"] == "per_label" and cfg["honest_macro_f1"] > table[0]["honest_macro_f1"] + 0.2
    assert fit_thresholds(y, p, "global") < 0.5
    one = apply_thresholds(np.array([[0.1, 0.3], [0.9, 0.2]]), 0.5, min_one=True)
    assert one.tolist() == [[0, 1], [1, 0]]


def test_loader_and_writer(tmp_path):
    d = tmp_path / "Phase_5_LSI"
    (d / "Train").mkdir(parents=True)
    (d / "Test").mkdir()
    w = lambda n, s: " ".join(f"w{(i * s) % 997}" for i in range(n))
    json.dump([{"id": "a", "text": w(300, 3), "labels": ["7", 7, 3]}, {"id": "a", "text": w(300, 3), "labels": [9]},
               {"id": "b", "text": w(300, 5), "labels": [3]}, {"id": "c", "text": w(300, 11), "labels": [7]},
               {"id": "e", "text": w(300, 13), "labels": []}], open(d / "Train/train.json", "w"))
    json.dump([{"id": "x", "text": w(300, 17), "labels": [3]}], open(d / "Train/dev.json", "w"))
    json.dump([{"id": 1, "text": w(300, 11)}, {"id": "t2", "text": w(300, 19)}], open(d / "Test/test.json", "w"))
    tr = load_split(d, "train")
    assert tr["a"]["labels"] == [3, 7, 9] and "c" not in tr and "e" not in tr      # merged / test copy / no label
    labels = label_space(tr, load_split(d, "dev"))
    assert labels == [3, 7, 9]
    test = load_split(d, "test")
    info = write_prediction(np.array([[1, 0, 1], [0, 0, 0]]), test, labels, tmp_path / "sub")
    pred = json.loads((tmp_path / "sub/prediction.json").read_text())
    assert pred == [{"id": 1, "pred": [3, 9]}, {"id": "t2", "pred": []}] and info["empty"] == 1
    for bad in (pred[:1], [{"id": 1, "pred": [3, 3]}, pred[1]], [{"id": 1, "pred": [4]}, pred[1]],
                [{"id": 1, "pred": ["3"]}, pred[1]]):
        with pytest.raises(ValueError):
            validate_prediction(bad, test, labels)


def test_end_to_end_baseline_finetune_kd_ensemble():
    data = str(P5 / "Phase_5_LSI")
    out = P5 / "runs"
    st = run("-m", "phase5.run", "stats", "--data", data)
    assert '"statutes": 6' in st.stdout
    b = run("-m", "phase5.run", "baseline", "--data", data, "--output", str(out / "B0"), "--c-grid", "4", "--jobs", "1")
    s = json.loads((out / "B0" / "summary.json").read_text())
    assert s["dev_macro_f1_honest"] > 0.6, b.stdout[-2000:]           # cue phrases are learnable
    common = ["--project-root", str(PHASE1), "--chunk", "64", "--train-chunks", "4", "--eval-chunks", "8",
              "--epochs", "2", "--doc-batch", "8", "--docs-per-forward", "4", "--eval-batch", "32", "--eval-docs", "0",
              "--hidden", "32", "--rank", "4", "--alpha", "8"]
    for name, extra in (("L1", ["--checkpoint", str(FX / "phase1_runs/dense_gpt/best.pt"), "--method", "lorma"]),
                        ("L2", ["--checkpoint", str(FX / "phase1_runs/gqa_local/best.pt"), "--method", "adapter",
                                "--pool", "attn", "--loss", "wbce", "--rare-repeat", "2"]),
                        ("L3", ["--checkpoint", str(FX / "phase1_runs/encoder_decoder/best.pt"), "--mode", "encoder",
                                "--method", "partial", "--top-k", "1", "--pool", "max", "--loss", "focal",
                                "--stride", "48"]),
                        ("L4", ["--checkpoint", str(FX / "phase1_runs/recurrent_hybrid/best.pt"), "--method", "prefix",
                                "--pool", "mean", "--loss", "bce", "--select", "head"])):
        r = run("-m", "phase5.run", "finetune", "--data", data, *common, *extra, "--output", str(out / name))
        s = json.loads((out / name / "summary.json").read_text())
        assert (out / name / "submission" / "submission.zip").exists() and "dev_macro_f1_honest" in s, r.stdout[-1500:]
    kd = run("-m", "phase5.run", "finetune", "--data", data, *common, "--checkpoint",
             str(FX / "phase1_runs/dense_gpt/best.pt"), "--keep-layers", "1", "--method", "full",
             "--teacher", str(out / "L1"), "--output", str(out / "K1"))
    assert "KD: teacher" in kd.stdout and "computed" in kd.stdout
    kd2 = run("-m", "phase5.run", "finetune", "--data", data, *common, "--checkpoint",
              str(FX / "phase1_runs/gqa_local/best.pt"), "--method", "qlora", "--teacher", str(out / "L1"),
              "--output", str(out / "K2"))
    assert "0 computed" in kd2.stdout                                       # teacher logits reused from the cache
    ens = run("-m", "phase5.ensemble", "--data", data, "--runs", *[str(out / n) for n in ("B0", "L1", "L2", "L3", "L4", "K1")],
              "--output", str(out / "E1"))
    assert "ensemble of" in ens.stdout and (out / "E1" / "submission" / "submission.zip").exists()
    run("-m", "phase5.run", "check", "--data", data, "--prediction", str(out / "E1" / "submission" / "prediction.json"))
