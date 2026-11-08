"""CPU tests for Phase 6 (abstractive summarization) on synthetic judgments + tiny random Phase-1 checkpoints:
    python -m pytest -q tests/test_phase6.py
"""
import argparse, json, os, random, subprocess, sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from _paths import phase1_root
PHASE1 = phase1_root()
sys.path.insert(1, str(PHASE1))

from phase6.data import (LengthModel, assemble, budgets, chunk_targets, load_splits, prepare_doc, split_sentences,
                         validate_prediction, write_submission)
from phase6.evaluate import evaluate, lcs
from phase6.generate import Decoding, generate
from phase6.model import build_summarizer, install_masks

FX = Path("/tmp/fx_test")
P6 = Path("/tmp/fx_p6")
ARCHS = ["dense_gpt", "gqa_local", "recurrent_hybrid", "encoder_decoder"]


def judgment(rng):
    from make_fixture import document
    text = " ".join(document(rng)[0] for _ in range(rng.randint(2, 4)))
    sents = split_sentences(text)
    pick = sorted(rng.sample(range(len(sents)), max(2, len(sents) // 4)))
    summary = " ".join(("It was held that " + sents[i][0].lower() + sents[i][1:]) if k % 2 else sents[i]
                       for k, i in enumerate(pick))
    return text, summary


def build(root, n_train=40, n_test=6, seed=0):
    rng = random.Random(seed)
    d = root / "Phase_6_Summ"
    (d / "Train").mkdir(parents=True, exist_ok=True)
    (d / "Test").mkdir(parents=True, exist_ok=True)
    train = []
    for i in range(n_train):
        t, s = judgment(rng)
        train.append({"id": f"doc_{i}", "document": t, "summary": s})
    test = [{"id": f"test_{i}", "document": judgment(rng)[0]} for i in range(n_test)]
    train.append({"id": "copy", "document": test[0]["document"], "summary": "x y z."})   # a test copy: dropped
    json.dump(train, open(d / "Train/train.json", "w"))
    json.dump(test, open(d / "Test/test.json", "w"))
    return d


@pytest.fixture(scope="session", autouse=True)
def fixtures():
    if not (FX / "planted.json").exists():
        subprocess.run([sys.executable, str(ROOT / "tests/make_fixture.py"), str(FX)], check=True)
    build(P6)
    install_masks()


def run(*argv, check=True):
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{PHASE1}", BERTSCORE_MODEL="")
    r = subprocess.run([sys.executable, *argv], cwd=ROOT, env=env, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(r.stdout[-3000:] + r.stderr[-3000:])
    return r


def make(arch, method="lorma", **kw):
    a = argparse.Namespace(checkpoint=str(FX / f"phase1_runs/{arch}/best.pt"), project_root=str(PHASE1), method=method,
                           rank=4, alpha=8, top_k=1, bottleneck=8, prompt_tokens=5, prefix_length=4, prefix_hidden=16,
                           adalora_init_r=4, adalora_target_r=2, keep_layers=None, gradient_checkpointing=False, **kw)
    torch.manual_seed(0)
    model, tok, _ = build_summarizer(a)
    with torch.no_grad():                                   # non-zero adapters, so they matter in the comparisons
        for n, p in model.named_parameters():
            if p.requires_grad and (n.endswith(".U") or n.endswith(".B") or "adapter.up" in n):
                p.normal_(0, 0.05)
    return model.eval(), tok


# --------------------------------------------------------------------------- unit tests
def test_lcs_rouge_sentences_alignment_budgets():
    assert lcs("a b c d".split(), "a c d e".split()) == 3
    from phase6.generate import mbr_pick
    assert mbr_pick([[9, 9, 9], [1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 4, 6]]) == [1, 2, 3, 4]   # the consensus
    m = evaluate({"x": "the appeal is dismissed"}, {"x": "the appeal is allowed"}, use_bertscore=False)
    assert abs(m["rouge1"] - 0.75) < 1e-9 and m["selection_metric"] == "mean(rouge1, rougeL)"
    s = split_sentences("The appellant was convicted u/s. 302 I.P.C. by the court. Mr. A. B. Rao appeared. "
                        "In Civil Appeal No. 5 the High Court held so. The appeal fails.")
    assert len(s) == 4, s
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(FX / "phase1_runs/dense_gpt/tokenizer.json"))
    doc_text = " ".join(f"Sentence number {w} talks about topic{w} and nothing else at all." for w in
                        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split())
    doc = prepare_doc(tok, doc_text, 120)
    assert len(doc["chunks"]) > 2 and all(len(c["ids"]) <= 120 for c in doc["chunks"])
    targets = chunk_targets(doc, "It concerns topicmu and nothing else. Also topicalpha was discussed.")
    assert targets[-1] == ["It concerns topicmu and nothing else."] and targets[0] == ["Also topicalpha was discussed."]
    b = budgets(doc, 300)
    assert abs(sum(b) - 300) <= len(b) and budgets(doc, 300, [0.0] * 5 + [2.0] * 5)[-1] > b[-1]
    out = assemble(["The appeal is dismissed. The appeal is dismissed with costs.", "The appeal is dismissed."], 100)
    assert out == "The appeal is dismissed. The appeal is dismissed with costs."   # exact repeat dropped
    lm = LengthModel.fit({str(i): {"document": "w " * (100 * i), "summary": "s " * (25 * i)} for i in range(1, 30)})
    assert abs(lm.b - 1) < 0.01 and abs(lm.predict(800) - 200) <= 2


def test_padding_masks_generation_matches_single_rows():
    """Left-padded batched decoding (masks in attention + per-row LSTM prefill) and padded encoder batches give the
    same tokens as decoding every prompt alone, for all four architectures, greedy and beam search."""
    rng = random.Random(1)
    prompts = [[rng.randint(5, 600) for _ in range(n)] for n in (17, 40, 29)]
    for arch, method in [(a, "lorma") for a in ARCHS] + [("dense_gpt", "prefix"), ("gqa_local", "prompt"), ("dense_gpt", "prompt"), ("gqa_local", "prefix"), ("recurrent_hybrid", "prompt"), ("recurrent_hybrid", "prefix"),
                                                         ("encoder_decoder", "prefix"), ("recurrent_hybrid", "houlsby")]:
        model, _ = make(arch, method)
        for beams, mbr in ((1, 0), (3, 0), (4, 4)):
            dec = Decoding(beams=beams, no_repeat_ngram=3, batch=8, mbr=mbr)
            batched = generate(model, prompts, [6, 6, 6], [12, 12, 12], dec, torch.device("cpu"), None)
            single = [generate(model, [p], [6], [12], dec, torch.device("cpu"), None)[0] for p in prompts]
            assert batched == single, (arch, method, beams, mbr)
            assert all(6 <= len(x) <= 12 for x in batched)
    from phase6.model import Ensemble                        # mixture decoding
    a, _ = make("dense_gpt")
    b, _ = make("gqa_local", "prompt")
    dec = Decoding(beams=3, no_repeat_ngram=3, batch=8)
    assert generate(Ensemble([a, a]), prompts, [6] * 3, [12] * 3, dec, torch.device("cpu"), None) == \
        generate(a, prompts, [6] * 3, [12] * 3, dec, torch.device("cpu"), None)
    ens = Ensemble([a, b], [2.0, 1.0])
    assert generate(ens, prompts, [6] * 3, [12] * 3, dec, torch.device("cpu"), None) == \
        [generate(ens, [p], [6], [12], dec, torch.device("cpu"), None)[0] for p in prompts]
    model, _ = make("encoder_decoder")                       # FiD prompts: lists of chunks
    fid = [[prompts[0], prompts[1]], [prompts[2]]]
    dec = Decoding(beams=2, batch=8)
    assert generate(model, fid, [5, 5], [9, 9], dec, torch.device("cpu"), None) == \
        [generate(model, [f], [5], [9], dec, torch.device("cpu"), None)[0] for f in fid]


def test_teacher_forced_logits_ignore_padding():
    rng = random.Random(2)
    exs = [{"doc": str(i), "k": 0, "src": [rng.randint(5, 600) for _ in range(n)], "tgt": [rng.randint(5, 600) for _ in range(t)]}
           for i, (n, t) in enumerate([(20, 7), (45, 3), (33, 11)])]
    fid = [{**e, "src": [e["src"], e["src"][:9]] if i != 1 else [e["src"]]} for i, e in enumerate(exs)]
    for arch, batch in [(a, exs) for a in ARCHS] + [("encoder_decoder", fid)]:
        model, _ = make(arch)
        with torch.no_grad():
            joint, labels, _ = model.target_logits(batch, torch.device("cpu"))
            alone = torch.cat([model.target_logits([e], torch.device("cpu"))[0] for e in batch])
        assert labels.numel() == sum(len(e["tgt"]) + 1 for e in batch)
        assert torch.allclose(joint, alone, atol=1e-4), arch


def test_submission_writer(tmp_path):
    test = {"a": {"document": "First sentence here. Second one.", "summary": None}, "b": {"document": "Only.", "summary": None}}
    info = write_submission({"a": "A summary."}, test, tmp_path)
    pred = json.loads((tmp_path / "prediction.json").read_text())
    assert [p["id"] for p in pred] == ["a", "b"] and pred[1]["summary"] and info["filled_with_lead"] == 1
    for bad in (pred[:1], [pred[0], {"id": "b", "summary": " "}], pred + [pred[0]]):
        with pytest.raises(ValueError):
            validate_prediction(bad, test)


# --------------------------------------------------------------------------- end to end
def test_end_to_end():
    data = str(P6 / "Phase_6_Summ")
    out = P6 / "runs"
    train, dev, test = load_splits(data, 6, verbose=False)
    assert "copy" not in train and "copy" not in dev and len(test) == 6
    st = run("-m", "phase6.run", "stats", "--data", data, "--dev-size", "6", "--checkpoint",
             str(FX / "phase1_runs/dense_gpt/best.pt"), "--chunk", "96")
    assert '"tokens_per_summary_word"' in st.stdout, st.stdout
    run("-m", "phase6.run", "baseline", "--data", data, "--output", str(out / "B0"), "--dev-size", "6", "--no-bertscore")
    s = json.loads((out / "B0/summary.json").read_text())
    assert s["dev"]["extractive"]["rouge1"] > 0.3 and (out / "B0/extractor.pkl").exists()
    run("-m", "phase6.run", "check", "--data", data, "--prediction", str(out / "B0/submission/prediction.json"))
    common = ["--data", data, "--project-root", str(PHASE1), "--dev-size", "6", "--epochs", "1", "--chunk", "96",
              "--micro-tokens", "1024", "--batch-examples", "8", "--rank", "4", "--alpha", "8", "--bottleneck", "8",
              "--beams", "2", "--eval-docs", "3", "--no-bertscore", "--gen-batch", "4", "--max-target", "48",
              "--prompt-tokens", "4", "--prefix-length", "4", "--prefix-hidden", "16"]
    ck = lambda a: ["--checkpoint", str(FX / f"phase1_runs/{a}/best.pt")]
    runs = {
        "L1": ck("dense_gpt") + ["--method", "lorma", "--src-lm", "0.2"],
        "L2": ck("gqa_local") + ["--method", "houlsby", "--overlap", "16"],
        "L3": ck("recurrent_hybrid") + ["--method", "pfeiffer", "--budget", "uniform"],
        "L4": ck("encoder_decoder") + ["--method", "lorma", "--strategy", "fid", "--fid-chunks", "3"],
        "T_prefix": ck("dense_gpt") + ["--method", "prefix", "--strategy", "head"],
        "T_prompt": ck("encoder_decoder") + ["--method", "prompt"],
        "T_qlora": ck("gqa_local") + ["--method", "qlora", "--align", "position"],
        "T_adalora": ck("gqa_local") + ["--method", "adalora", "--adalora-init-r", "4", "--adalora-target-r", "2"],
        "T_extract": ck("gqa_local") + ["--method", "lora", "--strategy", "extract", "--extractor",
                                        str(out / "B0/extractor.pkl"), "--extract-keep", "0.6"],
        "T_full_scratch": ck("recurrent_hybrid") + ["--method", "full", "--random-init", "--keep-layers", "2",
                                                    "--gradient-checkpointing"],
    }
    for name, extra in runs.items():
        r = run("-m", "phase6.run", "finetune", *common, "--output", str(out / name), *extra)
        s = json.loads((out / name / "summary.json").read_text())
        assert (out / name / "submission/submission.zip").exists(), r.stdout[-2000:]
        assert s["dev"]["documents"] == 3 and s["dev"]["words_pred"] > 0, (name, s["dev"])
        run("-m", "phase6.run", "check", "--data", data, "--prediction", str(out / name / "submission/prediction.json"))
    # knowledge distillation: word-level from the dense teacher, sequence-level from the encoder-decoder teacher
    run("-m", "phase6.run", "finetune", *common, "--output", str(out / "K1"), *ck("gqa_local"), "--method", "lorma",
        "--teacher", str(out / "L1"), "--kd", "logit")
    k1 = json.loads((out / "K1/summary.json").read_text())
    assert k1["teacher"]["architecture"] == "dense_gpt" and k1["kd"] == "logit"
    for _ in range(2):                                       # second run reuses the cached teacher summaries
        r = run("-m", "phase6.run", "finetune", *common, "--output", str(out / "K2"), *ck("gqa_local"),
                "--method", "lorma", "--keep-layers", "2", "--layer-pick", "spread", "--teacher", str(out / "T_prompt"), "--kd", "seq")
    assert "(0 cached)" not in r.stdout and json.loads((out / "K2/summary.json").read_text())["teacher"]["kd"] == "seq"
    # re-decoding and the decoding variants
    run("-m", "phase6.run", "decode", "--data", data, "--run", str(out / "L2"), "--tag", "greedy", "--beams", "1",
        "--len-scale", "0.8", "--eval-docs", "3", "--no-bertscore")
    assert (out / "L2/decode_greedy/submission/submission.zip").exists()
    run("-m", "phase6.run", "tune", "--data", data, "--run", str(out / "L1"), "--variants", "default", "len1.2", "uniform",
        "--eval-docs", "3", "--no-bertscore", "--gen-batch", "4")
    t = json.loads((out / "L1/tune.json").read_text())
    assert set(t["rows"]) == {"default", "len1.2", "uniform"}
    run("-m", "phase6.run", "ensemble", "--data", data, "--runs", str(out / "L1"), str(out / "K1"), "--output",
        str(out / "E1"), "--eval-docs", "3", "--no-bertscore", "--gen-batch", "4", "--beams", "2")
    e = json.loads((out / "E1/summary.json").read_text())
    assert e["stage"] == "ensemble" and e["dev"]["documents"] == 3
    run("-m", "phase6.run", "check", "--data", data, "--prediction", str(out / "E1/submission/prediction.json"))
    r = run("-m", "phase6.run", "score", "--data", data, "--prediction", str(out / "L1/dev_pred.json"), "--no-bertscore")
    assert '"rougeL"' in r.stdout
