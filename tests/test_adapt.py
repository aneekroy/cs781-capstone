"""Continued pretraining (phase2.adapt) on the synthetic fixture: CPU, a few steps."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from _paths import phase1_root
PHASE1 = phase1_root()
sys.path.insert(1, str(PHASE1))
from phase2 import adapt
from phase2.model import load_phase1, Encoder

FX = Path("/tmp/fx_test")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    if not (FX / "planted.json").exists():   # rebuilt when the fixture predates the overlap filter
        subprocess.run([sys.executable, str(ROOT / "tests/make_fixture.py"), str(FX)], check=True)
    out = tmp_path_factory.mktemp("adapt") / "data"
    meta = adapt.prepare([FX / "unlabeled.json"], [1.0], FX / "Phase_2_LNER",
                         FX / "phase1_runs/dense_gpt/tokenizer.json", out, tokens=60_000, val_tokens=3_000, val_every=10)
    return out, meta


def test_prepare_counts_and_excludes_test_text(corpus):
    out, meta = corpus
    assert meta["train_tokens"] == 60_000 and meta["val_tokens"] >= 3_000 and meta["ner_tokens"] > 0
    assert (out / "train.bin").stat().st_size == 2 * meta["train_tokens"]
    fold_tokens = 0
    tok = adapt.segment_tokenizer(__import__("tokenizers").Tokenizer.from_file(str(out / "tokenizer.json")), "punct")
    for fold in (1, 2):
        texts = json.load(open(FX / f"Phase_2_LNER/Train/fold{fold}-text.json"))
        fold_tokens += sum(len(e.ids) + 1 for e in tok.encode_batch(list(texts.values())))
    assert meta["ner_tokens"] == fold_tokens   # exactly the two training folds, no fold3 text
    assert meta["excluded_ner_overlap_documents"] == 4   # the planted NER documents (2 train, 2 test)


def test_overlap_filter_finds_only_planted_documents():
    from phase2.overlap import NerFingerprints
    fp = NerFingerprints.from_data(FX / "Phase_2_LNER")
    planted = json.load(open(FX / "planted.json"))
    records = json.load(open(FX / "unlabeled.json"))
    assert all(fp.overlaps(t) for t in planted)
    assert sum(fp.overlaps(r["text"]) for r in records) == len(planted)
    # a planted document only partly copied (first 60%) is still caught
    assert fp.overlaps(planted[0][: int(len(planted[0]) * 0.6)])


def test_kd_unlabeled_skips_ner_documents(capsys):
    from phase2.run import unlabeled_examples
    from phase2.data import segment_tokenizer
    from tokenizers import Tokenizer
    tok = segment_tokenizer(Tokenizer.from_file(str(FX / "phase1_runs/dense_gpt/tokenizer.json")), "punct")
    out = unlabeled_examples(FX / "unlabeled.json", tok, 10_000, 0, FX / "Phase_2_LNER", min_chars=500, take_prob=1.0)
    assert len(out) == 150 and "4 skipped" in capsys.readouterr().out
    unfiltered = unlabeled_examples(FX / "unlabeled.json", tok, 10_000, 0, None, min_chars=500, take_prob=1.0)
    assert len(unfiltered) == 154


def test_masking_statistics():
    ids = torch.randint(4, 500, (64, 256))
    ids[:, 0] = 2  # <eos> must never be masked
    corrupted, labels = adapt.mask_tokens(ids, 3, 500, [0, 1, 2, 3], torch.Generator().manual_seed(0))
    chosen = labels != -100
    assert abs(chosen.float().mean().item() - 0.15) < 0.01
    assert not chosen[:, 0].any()
    assert torch.equal(labels[chosen], ids[chosen])
    masked = (corrupted == 3) & chosen
    assert abs(masked.sum().item() / chosen.sum().item() - 0.8) < 0.03
    assert torch.equal(corrupted[~chosen], ids[~chosen])


def test_wsd_schedule():
    values = [adapt.wsd(s, 1000, 100) for s in range(1000)]
    assert values[0] == 0.01 and values[99] == 1.0 and values[500] == 1.0 and values[-1] < 0.12


@pytest.mark.parametrize("arch,objective", [("dense_gpt", "clm"), ("recurrent_hybrid", "clm"), ("encoder_decoder", "mlm")])
def test_train_and_reload_for_ner(corpus, tmp_path, arch, objective):
    out, _ = corpus
    ckpt = FX / f"phase1_runs/{arch}/best.pt"
    before = {k: v.clone() for k, v in torch.load(ckpt, weights_only=False)["model"].items()}
    summary = adapt.train(ckpt, out, tmp_path / "run", objective, hours=1, project_root=PHASE1, seq_len=64,
                          micro_batch=4, accumulation=1, lr=3e-3, warmup=2, eval_every=10, eval_batches=2,
                          max_steps=20)
    assert summary["best_val_loss"] < summary["start_val_loss"]       # it learned something
    assert not (tmp_path / "run/last.pt").exists()                     # optimizer state cleaned up
    lm, tok, resolved = load_phase1(tmp_path / "run/best.pt", PHASE1)  # same format as Phase-1 checkpoints
    assert resolved["adaptation"]["objective"] == objective
    after = lm.state_dict()
    changed = [k for k in after if not torch.equal(after[k], before[k])]
    if objective == "mlm":
        assert changed and not any(k.startswith("decoder.") for k in changed)   # only encoder side moves
    enc = Encoder(lm, mode="encoder" if arch == "encoder_decoder" else "causal").eval()
    assert enc(torch.randint(4, 500, (1, 32))).shape == (1, 32, enc.dim)


def test_objective_mismatch_rejected(corpus, tmp_path):
    out, _ = corpus
    with pytest.raises(ValueError, match="mlm adapts the encoder"):
        adapt.train(FX / "phase1_runs/dense_gpt/best.pt", out, tmp_path / "x", "mlm", 1, PHASE1, max_steps=1)
    with pytest.raises(ValueError, match="Use --objective mlm"):
        adapt.train(FX / "phase1_runs/encoder_decoder/best.pt", out, tmp_path / "y", "clm", 1, PHASE1, max_steps=1)
