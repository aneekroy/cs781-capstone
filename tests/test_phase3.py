"""CPU tests for Phase 3 on synthetic data + tiny random Phase-1 checkpoints:
    python -m pytest -q tests/test_phase3.py
"""
import argparse, itertools, json, random, subprocess, sys
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

from phase3.data import (FastScorer, load_split, load_vocab, pooled_score, validate_prediction, write_prediction,
                         split_sentences)
from phase3.model import ChainCRF, merge_adapters, viterbi_np
from phase3.run import make_folds, pack
from phase3.decode import choose_decoding, tune_offsets, predict_indices

FX = Path("/tmp/fx_test")
P3 = Path("/tmp/fx_p3")
LABELS = ["Fact", "Issue", "ArgumentPetitioner", "ArgumentRespondent", "Statute", "Dissent", "PrecedentReliedUpon",
          "PrecedentNotReliedUpon", "PrecedentOverruled", "RulingByLowerCourt", "RatioOfTheDecision",
          "RulingByPresentCourt", "None"]
TEMPLATES = {
    "Fact": "The appellant filed a complaint on the said date regarding the transaction .",
    "Issue": "The question which arises for consideration is whether the order is valid .",
    "ArgumentPetitioner": "Learned counsel for the appellant submitted that the finding is perverse .",
    "ArgumentRespondent": "Learned counsel for the respondent contended that the appeal has no merit .",
    "Statute": "Section 4 of the Act provides that no enterprise shall abuse its position .",
    "PrecedentReliedUpon": "In State v. Kumar this Court held the same principle which we follow .",
    "PrecedentNotReliedUpon": "The decision cited by counsel is distinguishable on facts .",
    "RulingByLowerCourt": "The High Court dismissed the writ petition by the impugned order .",
    "RatioOfTheDecision": "We are of the view that the provision must be read strictly .",
    "RulingByPresentCourt": "The appeal is accordingly allowed and the order set aside .",
    "None": "Heard learned counsel .",
}
ORDER = ["None", "Fact", "RulingByLowerCourt", "Issue", "ArgumentPetitioner", "ArgumentRespondent", "Statute",
         "PrecedentReliedUpon", "PrecedentNotReliedUpon", "RatioOfTheDecision", "RulingByPresentCourt"]


def synthetic_doc(rng):
    sents, labs = [], []
    for label in ORDER:
        for _ in range(rng.randint(1, 4) if label not in {"Issue", "None"} else 1):
            words = TEMPLATES[label].split()
            if rng.random() < 0.3:
                words.insert(rng.randrange(len(words)), rng.choice(["further", "thus", "also", "indeed"]))
            sents.append(" ".join(words))
            labs.append(label)
    return sents, labs


def build_p3(root, n_docs=8, seed=0):
    rng = random.Random(seed)
    train, test = root / "Phase_3_RR" / "Train", root / "Phase_3_RR" / "Test"
    train.mkdir(parents=True, exist_ok=True)
    test.mkdir(parents=True, exist_ok=True)
    (train / "label_vocab.json").write_text(json.dumps({l: i for i, l in enumerate(LABELS)}))
    for domain in ("CL", "IT"):
        for split, n in (("train", n_docs), ("dev", 2)):
            rows = []
            for i in range(n):
                s, l = synthetic_doc(rng)
                rows.append({"id": f"{domain}_{split}_{i}", "text": s, "labels": l})
            (train / f"{domain}_{split}.json").write_text(json.dumps(rows))
        (test / f"{domain}_test.json").write_text(json.dumps({f"{domain}_test_{i}": synthetic_doc(rng)[0]
                                                             for i in range(2)}))
    return root / "Phase_3_RR"


@pytest.fixture(scope="session", autouse=True)
def fixtures():
    if not (FX / "planted.json").exists():
        subprocess.run([sys.executable, str(ROOT / "tests/make_fixture.py"), str(FX)], check=True)
    build_p3(P3)


def env():
    import os
    e = dict(os.environ)
    e["PYTHONPATH"] = f"{ROOT}:{PHASE1}"
    return e


def run(*argv, check=True):
    r = subprocess.run([sys.executable, *argv], cwd=ROOT, env=env(), capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(r.stdout[-3000:] + r.stderr[-3000:])
    return r


# --------------------------------------------------------------------------- scoring + format
def test_scorer_matches_sklearn():
    from sklearn.metrics import f1_score, precision_score, recall_score
    rng = np.random.default_rng(0)
    labels = LABELS
    y_true = [labels[i] for i in rng.integers(0, 11, 500)]          # Dissent / PRE-O never in gold
    y_pred = [labels[i] for i in rng.integers(0, 13, 500)]
    s = pooled_score({"a": y_true}, {"a": y_pred}, labels)
    assert abs(s["macro_f1"] - f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)) < 1e-12
    assert abs(s["macro_p"] - precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)) < 1e-12
    assert abs(s["macro_r"] - recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)) < 1e-12
    idx = {l: i for i, l in enumerate(labels)}
    fast = FastScorer([idx[l] for l in y_true], 13)([idx[l] for l in y_pred])
    assert abs(fast - s["macro_f1"]) < 1e-12


def test_vocab_formats():
    assert load_vocab({l: i for i, l in enumerate(LABELS)}) == LABELS
    assert load_vocab({str(i): l for i, l in enumerate(LABELS)}) == LABELS
    assert load_vocab(list(LABELS)) == LABELS
    with pytest.raises(ValueError):
        load_vocab(LABELS[:12])


def test_prediction_checks(tmp_path):
    data = build_p3(tmp_path / "p")
    labels = load_vocab(data / "Train" / "label_vocab.json")
    test = load_split(data, "test")
    good = {d: ["Fact"] * len(doc["sentences"]) for d, doc in test.items()}
    assert validate_prediction(good, test, labels)
    assert validate_prediction({d + ".txt": v for d, v in good.items()}, test, labels)
    bad_id = dict(good); bad_id.pop(next(iter(bad_id)))
    bad_count = dict(good); k = next(iter(bad_count)); bad_count[k] = bad_count[k][:-1]
    bad_label = dict(good); bad_label[k] = ["ARG-P"] * len(good[k])
    extra = dict(good); extra["nope"] = ["Fact"]
    for bad in (bad_id, bad_count, bad_label, extra):
        with pytest.raises(ValueError):
            validate_prediction(bad, test, labels)
    info = write_prediction(good, test, labels, tmp_path / "sub")
    assert info["documents"] == len(test) and (tmp_path / "sub" / "submission.zip").exists()


def test_cross_domain_id_collision(tmp_path):
    data = build_p3(tmp_path / "p")
    cl = json.loads((data / "Test" / "CL_test.json").read_text())
    it = json.loads((data / "Test" / "IT_test.json").read_text())
    it[next(iter(cl))] = ["x"]
    (data / "Test" / "IT_test.json").write_text(json.dumps(it))
    with pytest.raises(ValueError, match="share document ids"):
        load_split(data, "test")


# --------------------------------------------------------------------------- model pieces
def test_chain_crf_brute_force():
    torch.manual_seed(0)
    crf = ChainCRF(3)
    with torch.no_grad():
        crf.trans.normal_(); crf.start.normal_(); crf.end.normal_()
    em = torch.randn(4, 3)
    paths = list(itertools.product(range(3), repeat=4))

    def score(p):
        return (crf.start[p[0]] + crf.end[p[-1]] + sum(em[i, t] for i, t in enumerate(p))
                + sum(crf.trans[a, b] for a, b in zip(p, p[1:]))).detach()
    scores = torch.stack([score(p) for p in paths])
    logz = torch.logsumexp(scores, 0)
    tags = torch.tensor([2, 0, 1, 1])
    assert torch.allclose(crf.nll(em, tags), (logz - score(tuple(tags.tolist()))) / 4, atol=1e-5)
    marg = torch.zeros(4, 3)
    probs = (scores - logz).exp()
    for p, w in zip(paths, probs):
        for i, t in enumerate(p):
            marg[i, t] += w
    assert torch.allclose(crf.log_marginals(em).exp(), marg, atol=1e-5)
    best = paths[int(scores.argmax())]
    assert tuple(viterbi_np(em.numpy(), crf.params())) == best


def test_merge_adapters_keeps_function():
    from phase2.model import Encoder, load_phase1
    from phase2.peft import apply_method
    lm, tok, _ = load_phase1(FX / "phase1_runs" / "dense_gpt" / "best.pt", PHASE1)
    enc = Encoder(lm, mode="causal")
    apply_method(enc, "lorma", rank=4, alpha=8)
    with torch.no_grad():
        for n, p in enc.named_parameters():
            if n.endswith(".U"):
                p.normal_(0, 0.05)
    ids = torch.tensor([tok.encode(" The appellant filed a complaint .").ids])
    before = enc(ids)
    assert merge_adapters(enc) > 0
    assert torch.allclose(before, enc(ids), atol=1e-4)


def test_pack_and_folds():
    windows, spans = pack([[1, 2], [3, 4, 5], [6] * 6, [7]], window=6, packing="packed")
    assert windows == [[1, 2, 3, 4, 5], [6] * 6, [7]]
    assert spans == [(0, 0, 2), (0, 2, 5), (1, 0, 6), (2, 0, 1)]
    w2, s2 = pack([[1, 2], [3]], window=6, packing="isolated")
    assert w2 == [[1, 2], [3]] and s2 == [(0, 0, 2), (1, 0, 1)]
    docs = {f"{d}{i}": {"domain": d, "sentences": ["x"] * (i + 1)} for d in ("CL", "IT") for i in range(7)}
    folds = make_folds(docs, 3, 0)
    assert sorted(x for f in folds for x in f) == sorted(docs)
    assert all(any(x.startswith("CL") for x in f) and any(x.startswith("IT") for x in f) for f in folds)


def test_offsets_recover_rare_class():
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 3, 600)
    logp = np.log(np.full((600, 3), 0.1))
    logp[np.arange(600), gold] = np.log(0.5)
    rare = gold == 2
    logp[rare, 2] = np.log(0.3)              # the model under-scores class 2
    logp[rare, 0] = np.log(0.4)
    recs = [{"logp": logp, "emissions": logp, "domain": "CL", "crf": None}]
    scorer = FastScorer(gold, 3)
    base = scorer(predict_indices(recs, "marginal"))
    off, tuned = tune_offsets(recs, gold, 3, "marginal")
    assert tuned > base + 0.2 and off[2] > off[0]


def test_split_sentences():
    s = split_sentences("The appellant filed the appeal on time. It was dismissed by the High Court! Why? "
                        "Section 4 of the Act applies here.")
    assert s[0].startswith("The appellant") and s[-1].startswith("Section 4")


# --------------------------------------------------------------------------- end to end
COMMON = ["--data", str(P3 / "Phase_3_RR"), "--project-root", str(PHASE1), "--epochs", "2", "--eval-from", "1",
          "--folds", "2", "--segment-sentences", "8", "--segments-per-step", "2", "--window", "64",
          "--max-sent-tokens", "24", "--hidden", "32", "--rank", "4", "--alpha", "8"]


def ckpt(arch):
    return str(FX / "phase1_runs" / arch / "best.pt")


def test_cv_final_predict_check_ensemble():
    out = P3 / "runs"
    r1 = run("-m", "phase3.run", "cv", *COMMON, "--checkpoint", ckpt("dense_gpt"), "--method", "lorma",
             "--output", str(out / "R1"), "--final")
    assert "OOF pooled macro-F1" in r1.stdout
    summary = json.loads((out / "R1" / "cv_summary.json").read_text())
    assert summary["decoding"]["mode"] in {"viterbi", "marginal"} and len(summary["best_epochs"]) == 2
    data = P3 / "Phase_3_RR"
    run("-m", "phase3.run", "check", "--data", str(data), "--prediction", str(out / "R1" / "submission" / "prediction.json"))
    # sentence-only ablation, CE + class weights, transformer context on the gqa backbone
    run("-m", "phase3.run", "cv", *COMMON, "--checkpoint", ckpt("gqa_local"), "--method", "lora", "--packing",
        "isolated", "--context", "none", "--loss", "focal", "--class-weights", "sqrt", "--output", str(out / "R2"),
        "--final")
    run("-m", "phase3.run", "cv", *COMMON, "--checkpoint", ckpt("encoder_decoder"), "--method", "lorma",
        "--mode", "encoder", "--context", "transformer", "--domain-heads", "--loss", "ce", "--output", str(out / "R3"))
    # KD: truncated hybrid student, fold-matched teacher R1, teacher-labelled unlabeled text
    kd = run("-m", "phase3.run", "cv", *COMMON, "--checkpoint", ckpt("dense_gpt"), "--method", "lorma",
             "--keep-layers", "2", "--teacher", str(out / "R1"), "--kd-unlabeled", str(FX / "unlabeled.json"),
             "--kd-unlabeled-docs", "3", "--output", str(out / "K1"), "--final")
    assert "KD unlabeled" in kd.stdout
    ens = run("-m", "phase3.ensemble", "--data", str(data), "--runs", str(out / "R1"), str(out / "R2"),
              str(out / "R3"), str(out / "K1"), "--output", str(out / "E1"))
    assert "ensemble of" in ens.stdout
    vote = run("-m", "phase3.vote", "--data", str(data), "--runs", str(out / "R1"), str(out / "R2"),
               str(out / "R3"), str(out / "K1"), "--output", str(out / "V1"))
    assert "vote OOF" in vote.stdout and (out / "V1" / "ensemble_summary.json").exists()
    fixed = run("-m", "phase3.vote", "--data", str(data), "--runs", str(out / "R1"), str(out / "R2"),
                str(out / "K1"), "--output", str(out / "V2"), "--fixed")
    assert "vote of 3" in fixed.stdout
    st = run("-m", "phase3.stack", "--data", str(data), "--runs", str(out / "R1"), str(out / "R2"), str(out / "K1"),
             "--output", str(out / "ST1"), "--folds", "3")
    assert ("stack-" in st.stdout or "average of" in st.stdout) and "reference" in st.stdout and (out / "ST1" / "submission" / "submission.zip").exists()
    assert (out / "E1" / "submission" / "submission.zip").exists() or "no test.pt" in ens.stdout
    pr = run("-m", "phase3.run", "predict", "--data", str(data), "--model", str(out / "R1" / "final" / "model.pt"),
             "--split", "dev", "--output", str(out / "R1_dev"))
    assert "pooled macro-F1 on dev" in pr.stdout
    # mismatched folds are refused for KD
    bad = run("-m", "phase3.run", "cv", *COMMON, "--folds", "3", "--checkpoint", ckpt("dense_gpt"),
              "--teacher", str(out / "R1"), "--output", str(out / "K_bad"), check=False)
    assert bad.returncode != 0 and "folds differ" in (bad.stdout + bad.stderr)


def test_init_from_phase2_model(tmp_path):
    from phase2.run import build_tagger as p2_build, save_model
    a = argparse.Namespace(checkpoint=ckpt("dense_gpt"), project_root=str(PHASE1), segment="punct", mode="causal",
                           keep_layers=None, method="lorma", prompt_tokens=0, gradient_checkpointing=False, rank=4,
                           alpha=8, top_k=4, bottleneck=16, head="bilstm", crf=True, span=False, max_width=8,
                           dropout=0.1)
    tagger, _, _ = p2_build(a, ["APP", "RESP"])
    with torch.no_grad():
        for n, p in tagger.named_parameters():
            if n.endswith(".U"):
                p.normal_(0, 0.05)
    save_model(tmp_path / "model.pt", tagger, a, ["APP", "RESP"], {})
    from phase3.model import phase2_encoder
    enc, tok, info = phase2_encoder(tmp_path / "model.pt", PHASE1)
    ids = torch.tensor([tok.encode(" The appellant .").ids])
    assert torch.allclose(tagger.encoder(ids), enc(ids), atol=1e-4) and info["merged_linears"] > 0
    run("-m", "phase3.run", "cv", *COMMON, "--init-phase2", str(tmp_path / "model.pt"), "--method", "lorma",
        "--output", str(tmp_path / "R_p2"), "--protocol", "dev")
    assert json.loads((tmp_path / "R_p2" / "cv_summary.json").read_text())["protocol"] == "dev"


REAL = Path("__unset__")
REAL_FILES = {"Train/CL_train.json": "689fcdef-CL_train.json", "Train/CL_dev.json": "442a4afa-CL_dev.json",
              "Train/IT_train.json": "7ba273ce-IT_train.json", "Train/IT_dev.json": "18e10741-IT_dev.json",
              "Train/label_vocab.json": "05574c1a-label_vocab.json", "Test/CL_test.json": "4a1189fd-CL_test.json",
              "Test/IT_test.json": "1e9258bd-IT_test.json"}


@pytest.mark.skipif(not (REAL / "689fcdef-CL_train.json").exists(), reason="real Phase-3 files not attached")
def test_real_files_and_official_scorer(tmp_path):
    import shutil
    data = tmp_path / "Phase_3_RR"
    for dst, src in REAL_FILES.items():
        (data / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REAL / src, data / dst)
    labels = load_vocab(data / "Train" / "label_vocab.json")
    for split in ("train", "dev", "test"):
        load_split(data, split, labels if split != "test" else None)
    r = run("-m", "phase3.run", "truth", "--data", str(data), "--output", str(tmp_path / "truth"))
    ours = float(r.stdout.split("(our scorer):")[1].split()[0])
    evaluator = REAL / "67d9f0c0-attachment.txt"
    shutil.copy(evaluator, tmp_path / "evaluate_phase3.py")
    t = tmp_path / "truth"
    off = run(str(tmp_path / "evaluate_phase3.py"), "--truth-cl", str(t / "cl_dev_truth.json"), "--truth-it",
              str(t / "it_dev_truth.json"), "--prediction", str(t / "dev_all_fact_prediction.json"),
              "--labels", str(data / "Train" / "label_vocab.json"))
    official = float(off.stdout.split("Macro-F1:")[1].split()[0])
    assert abs(official - ours) < 1e-6
    test = load_split(data, "test")
    write_prediction({d: ["Fact"] * len(doc["sentences"]) for d, doc in test.items()}, test, labels, tmp_path / "sub")
    pred = json.loads((tmp_path / "sub" / "prediction.json").read_text())
    assert set(pred) == set(json.loads((data / "Test" / "CL_test.json").read_text())) | \
        set(json.loads((data / "Test" / "IT_test.json").read_text()))


def test_fingerprints_see_through_dataset_normalisation(tmp_path):
    from phase3.data import Fingerprints, unlabeled_documents
    rr = ("They arise out of a common judgment and order dated 2.3.2006 in ITA Number 50 of 2005 and Civil Writ "
          "Petition Number 800 of 2005 as also the Honble High Court which held that the assessee was not liable to "
          "pay the tax demanded under the provisions of the Act for the relevant assessment year in question")
    raw = rr.replace("Number 50", "No. 50").replace("Number 800", "No.800").replace("Honble", "Hon'ble")
    fp = Fingerprints([rr])
    assert fp.overlaps(raw) and not fp.overlaps(TEMPLATES["Fact"] * 5)
    other = " ".join(f"Sentence number {i} talks about the facts of a different case entirely." for i in range(30))
    path = tmp_path / "u.json"
    path.write_text(json.dumps([{"text": raw + ". " + other}, {"text": other}]))
    docs, dropped = unlabeled_documents(path, 5, 0, [rr], min_sents=5, take_prob=1.0)
    assert dropped == 1 and len(docs) == 1


# --------------------------------------------------------------------------- fine-tuning techniques
def _encoder(arch="dense_gpt", mode="causal"):
    from phase2.model import Encoder, load_phase1
    lm, tok, _ = load_phase1(FX / "phase1_runs" / arch / "best.pt", PHASE1)
    return Encoder(lm, mode=mode), tok


def _targs(**kw):
    base = dict(rank=4, alpha=8, top_k=2, bottleneck=8, prefix_length=4, prefix_hidden=16, adalora_init_r=6,
                adalora_target_r=2, prompt_tokens=3)
    base.update(kw)
    return argparse.Namespace(**base)


def test_nf4_quantisation_and_qlora_start():
    from phase3.peft import NF4Linear, apply_technique, NF4
    torch.manual_seed(0)
    lin = torch.nn.Linear(96, 40, bias=False)
    with torch.no_grad():
        lin.weight.normal_(0, 0.02)                     # NF4 assumes normally distributed weights
    q = NF4Linear(lin)
    w = lin.weight.detach()
    err = (q.weight_dequant(torch.float32) - w).pow(2).mean().sqrt() / w.pow(2).mean().sqrt()
    assert err < 0.12                                   # 4-bit: coarse but faithful
    with torch.no_grad():                               # codebook values (times block absmax) survive exactly
        lin.weight.copy_(NF4[torch.arange(96 * 40) % 16].view(40, 96) * 0.5)
    assert torch.allclose(NF4Linear(lin).weight_dequant(torch.float32), lin.weight, atol=1e-3)
    x = torch.randn(3, 96)
    assert torch.allclose(q(x), x @ q.weight_dequant(torch.float32).t(), atol=1e-5)
    assert q.packed.dtype == torch.uint8 and q.packed.numel() == 96 * 40 // 2
    enc, tok = _encoder()
    ids = torch.tensor([tok.encode(" The appellant filed a complaint .").ids])
    info, _ = apply_technique(enc, "qlora", _targs())
    assert info["base_bytes_4bit"] < info["base_bytes_bf16"] / 3
    trainable = [n for n, p in enc.named_parameters() if p.requires_grad]
    assert trainable and all(n.endswith((".A", ".B")) for n in trainable)


def test_techniques_start_at_base_and_train_only_their_parameters():
    from phase3.peft import apply_technique
    for method, suffixes in (("rslora", (".A", ".B")), ("adalora", (".A", ".E", ".B")), ("ia3", (".l",)),
                             ("lorma", (".U", ".V"))):
        enc, tok = _encoder()
        ids = torch.tensor([tok.encode(" The appellant filed a complaint before the court .").ids])
        with torch.no_grad():
            before = enc(ids)
        info, _ = apply_technique(enc, method, _targs())
        with torch.no_grad():
            after = enc(ids)
        assert torch.allclose(before, after, atol=1e-5), method          # zero-init / identity at start
        trainable = [n for n, p in enc.named_parameters() if p.requires_grad]
        assert trainable and all(n.endswith(suffixes) for n in trainable), (method, trainable[:3])
    enc, _ = _encoder()
    apply_technique(enc, "rslora", _targs(rank=4, alpha=8))
    from phase2.peft import LoRALinear
    assert all(abs(m.scale - 8 / 2) < 1e-9 for m in enc.modules() if isinstance(m, LoRALinear))   # alpha/sqrt(r)


def test_adalora_budget_shrinks_to_target():
    from phase3.peft import apply_technique, make_allocator, AdaLoRALinear
    torch.manual_seed(0)
    enc, tok = _encoder()
    a = _targs()
    apply_technique(enc, "adalora", a)
    alloc = make_allocator(enc, a, total_steps=40)
    params = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-2)
    ids = torch.tensor([tok.encode(" The appellant filed a complaint before the court .").ids])
    for step in range(1, 41):
        loss = enc(ids).pow(2).mean() + 0.1 * alloc.orth_penalty()
        opt.zero_grad(); loss.backward(); opt.step()
        alloc.step(step)
    modules = [m for m in enc.modules() if isinstance(m, AdaLoRALinear)]
    assert sum(alloc.ranks()) <= a.adalora_target_r * len(modules) + 1
    assert len(set(alloc.ranks())) > 1 or sum(alloc.ranks()) < a.adalora_init_r * len(modules)


def test_prefix_tuning_changes_output_and_keeps_causality():
    from phase3.peft import apply_technique
    for arch in ("dense_gpt", "gqa_local", "recurrent_hybrid", "encoder_decoder"):
        enc, tok = _encoder(arch, "encoder" if arch == "encoder_decoder" else "causal")
        ids = torch.tensor([tok.encode(" The appellant filed a complaint before the court on time .").ids])
        with torch.no_grad():
            before = enc(ids)
        apply_technique(enc, "prefix", _targs())
        with torch.no_grad():
            after = enc(ids)
        assert after.shape == before.shape and not torch.allclose(before, after), arch
        if arch != "encoder_decoder":
            ids2 = ids.clone(); ids2[0, -1] = (ids2[0, -1] + 1) % 500
            with torch.no_grad():
                assert torch.allclose(enc(ids)[0, :-1], enc(ids2)[0, :-1], atol=1e-5), arch   # no look-ahead
        assert all(n.startswith("prefix.") for n, p in enc.named_parameters() if p.requires_grad)


@pytest.mark.parametrize("method,arch", [("rslora", "dense_gpt"), ("qlora", "dense_gpt"), ("adalora", "dense_gpt"),
                                         ("prefix", "gqa_local"), ("prompt", "dense_gpt"), ("ia3", "recurrent_hybrid")])
def test_every_technique_end_to_end(method, arch, tmp_path):
    out = tmp_path / method
    run("-m", "phase3.run", "cv", *COMMON, "--checkpoint", ckpt(arch), "--method", method, "--protocol", "dev",
        "--prompt-tokens", "4", "--prefix-length", "4", "--prefix-hidden", "16", "--adalora-init-r", "6",
        "--adalora-target-r", "2", "--output", str(out), "--final")
    s = json.loads((out / "cv_summary.json").read_text())
    assert s["method"] == method and s["parameters"]["trainable"] > 0
    pr = run("-m", "phase3.run", "predict", "--data", str(P3 / "Phase_3_RR"), "--model", str(out / "final" / "model.pt"),
             "--split", "dev", "--output", str(tmp_path / "pred"))
    assert "pooled macro-F1 on dev" in pr.stdout


# --------------------------------------------------------------------------- rules + data mixing
def test_dissent_rule():
    from phase3.rules import apply_dissent, dissent_start
    majority = ["The informant filed the information.", "The Commission finds no case."] * 10 + \
               ["The matter is hereby closed under Section 26(2) of the Act.", "Secretary is directed to inform the parties accordingly."]
    note = ["Prasad, Member (Dissenting) The present information was filed by the informant.",
            "I have a different view.", "The OP is dominant."]
    labels = ["Fact"] * 20 + ["RulingByPresentCourt"] * 2 + ["RatioOfTheDecision"] * 3
    out = apply_dissent(labels, majority + note)
    assert out[:22] == labels[:22] and out[22:] == ["Dissent"] * 3
    closing = note + ["Secretary is directed to inform all concerned accordingly."]   # gold: Dissent to the last line
    assert apply_dissent(labels + ["RulingByPresentCourt"], majority + closing)[22:] == ["Dissent"] * 4
    minority = majority + ["Prasad (Minority) Allegations The informant is a company."] + ["x is y."] * 5
    assert dissent_start(minority) == 22
    disagree = majority + ["The informant alleged abuse.", "It is the majority view that no case exists.",
                           "I, however, do not agree with the above findings of the Majority Members.", "Reasons follow."]
    assert dissent_start(disagree) == 22                       # back to the end of the majority order
    plain = ["The court referred to the dissenting note in an earlier case."] + ["Facts."] * 30
    assert dissent_start(plain) is None and apply_dissent(["Fact"] * 31, plain) == ["Fact"] * 31


def test_block_splicing_keeps_order_and_adds_rare_blocks():
    import random
    from phase3.mixing import blocks, splice_documents, parse_weights, domain_of
    recs = {}
    for i in range(4):
        labs = ["Fact"] * 3 + ["ArgumentPetitioner"] * 2 + ["RatioOfTheDecision"] * 3 + ["RulingByPresentCourt"]
        if i == 0:
            labs = labs + ["Dissent"] * 4
        recs[f"d{i}"] = {"ids": [[i, j] for j in range(len(labs))], "labels": labs, "domain": "CL"}
    out = splice_documents(recs, 20, random.Random(0), ["Dissent"])
    assert len(out) == 20
    with_dissent = [r for r in out.values() if "Dissent" in r["labels"]]
    assert len(with_dissent) >= 10                             # rare block inserted into other documents
    for r in out.values():
        assert len(r["ids"]) == len(r["labels"])
        if "Dissent" in r["labels"]:
            assert r["labels"][-1] == "Dissent"                 # a dissent note goes to the end
        order = [b[0] for b in blocks(r["labels"])]
        assert order.index("Fact") < order.index("RatioOfTheDecision")
    assert parse_weights("Dissent=4, Issue=2.5") == {"Dissent": 4.0, "Issue": 2.5}
    assert domain_of("The Competition Commission found the opposite party in a dominant position in the relevant "
                     "market; the informant alleged a cartel. " * 5) == "CL"
    assert domain_of("The assessee claimed depreciation for the assessment year; the Income Tax officer and the "
                     "Appellate Tribunal disagreed on the capital gain. " * 5) == "IT"
    assert domain_of("The accused was convicted of murder by the Sessions Court. " * 20) is None


def test_label_weighted_segments():
    import random
    from phase3.run import make_segments
    recs = {"a": {"ids": [[1]] * 40, "labels": ["Fact"] * 30 + ["Dissent"] * 10, "domain": "CL"},
            "b": {"ids": [[1]] * 40, "labels": ["Fact"] * 40, "domain": "IT"}}
    index = {l: i for i, l in enumerate(LABELS)}
    plain = make_segments(recs, 10, random.Random(0), set(), 1, index)
    boosted = make_segments(recs, 10, random.Random(0), set(), 1, index, {"Dissent": 4}, {"IT": 2})
    count = lambda items, doc: sum(1 for d, *_ in items if d == doc)
    assert count(boosted, "a") > count(plain, "a") and count(boosted, "b") == 2 * count(plain, "b")


def test_mixing_end_to_end_and_redecode(tmp_path):
    out = tmp_path / "M"
    run("-m", "phase3.run", "cv", *COMMON, "--checkpoint", ckpt("dense_gpt"), "--method", "lorma",
        "--repeat-labels", "Issue=3,None=2", "--domain-weights", "IT=1.5", "--splice", "0.5", "--output", str(out), "--final")
    s = json.loads((out / "cv_summary.json").read_text())
    assert s["mixing"]["splice"] == 0.5 and "rules" in s["decoding"]
    r = run("-m", "phase3.redecode", "--data", str(P3 / "Phase_3_RR"), "--runs", str(out))
    assert "honest" in r.stdout and (out / "submission" / "submission.zip").exists()
    rep = run("-m", "phase3.report", str(tmp_path))
    assert "splice" in rep.stdout


def test_silver_sources_from_other_phases(tmp_path):
    from phase3.data import record_text, unlabeled_documents
    cl = ("The Competition Commission examined whether the opposite party held a dominant position in the relevant "
          "market. The informant alleged a cartel among the opposite parties. ")
    it = ("The assessee claimed depreciation for the assessment year. The Income Tax officer disallowed the claim "
          "and the Appellate Tribunal remanded it. ")
    assert record_text({"id": 1, "document": "Doc.", "summary": "Sum."}) == "Doc."
    assert record_text({"text": ["A b.", "C d."]}) == "A b. C d."
    assert record_text(["A b.", "C d."]) == "A b. C d."
    summ = tmp_path / "summ.json"
    summ.write_text(json.dumps([{"id": "s1", "document": cl * 15, "summary": "short"},
                                {"id": "s2", "document": "The accused was convicted of theft by the court. " * 30,
                                 "summary": "x"}]))
    pcr = tmp_path / "pcr.json"
    pcr.write_text(json.dumps({"q1": {"text": [x + "." for x in (it * 15).split(". ") if x.strip()]}}))
    docs, dropped = unlabeled_documents([summ, pcr], 10, 0, [], min_sents=5, take_prob=1.0, select=True)
    domains = sorted(d["domain"] for d in docs.values())
    assert domains == ["CL", "IT"] and dropped == 0


def test_offsets_are_tuned_with_rules_in_place():
    from phase3.decode import finalize_decoding, predict_indices, with_rules
    from phase3.rules import forced_indices
    index = {l: i for i, l in enumerate(LABELS)}
    majority = ["The informant filed the information under Section 19."] * 30 + \
               ["Secretary is directed to inform the parties accordingly."]
    note = ["Prasad, Member (Dissenting) The informant filed the information."] + ["I disagree here."] * 9
    forced = forced_indices(majority + note, ["dissent"], index)
    assert (forced[:31] == -1).all() and (forced[31:] == index["Dissent"]).all()
    rng = np.random.default_rng(0)
    docs, oof, gold = {}, {}, {}
    for i in range(12):                     # model never predicts Dissent; one doc in three has a dissent note
        sents = majority + (note if i % 3 == 0 else ["More facts."] * 10)
        g = ["Fact"] * 30 + ["RulingByPresentCourt"] + (["Dissent"] * 10 if i % 3 == 0 else ["Fact"] * 10)
        logp = np.log(np.full((41, 13), 0.02))
        logp[np.arange(41), [index["Fact"] if l == "Dissent" else index[l] for l in g]] = np.log(0.7)
        logp[:, index["Dissent"]] = np.log(0.05) + rng.normal(0, 0.1, 41)
        d = f"d{i}"
        docs[d] = {"sentences": sents}
        oof[d] = {"logp": logp, "emissions": logp, "domain": "CL", "crf": None}
        gold[d] = g
    config, _ = finalize_decoding(oof, gold, LABELS, docs, log=lambda *a, **k: None, modes=["marginal"])
    assert config["rules"] == ["dissent"]
    values = config.get("values")
    assert values is None or values[index["Dissent"]] <= 0.6        # no offset scattering Dissent elsewhere
    recs = with_rules(oof, docs, ["dissent"], LABELS)
    pred = predict_indices([recs["d0"]], "marginal")
    assert (pred[31:] == index["Dissent"]).all() and (pred[:30] == index["Fact"]).all()


def test_vote_plurality_and_rare_thresholds():
    from phase3.vote import vote_doc
    k = 4
    votes = [np.array([0, 1, 2]), np.array([0, 1, 3]), np.array([0, 2, 3])]
    probs = [np.full((3, k), 0.25)] * 3
    assert vote_doc(votes, probs, k).tolist() == [0, 1, 3]                  # plurality
    th = np.array([0, 0, 1, 0])                                              # label 2: one vote is enough
    assert vote_doc(votes, probs, k, th).tolist() == [0, 2, 2]
    forced = np.array([-1, -1, 1])
    assert vote_doc(votes, probs, k, th, forced).tolist() == [0, 2, 1]      # rules last
    tie = [np.array([0]), np.array([1])]
    p = [np.array([[0.4, 0.6, 0, 0]]), np.array([[0.3, 0.7, 0, 0]])]
    assert vote_doc(tie, p, k).tolist() == [1]                               # tie -> averaged probability


def test_sequence_decoding_candidates():
    import numpy as np
    from phase3.decode import (transition_prior, smoothed, crf_average, sequence_candidates, build_sequence_records,
                               viterbi_batch)
    labels = ["A", "B", "C"]
    prior = transition_prior([["A", "A", "B", "B", "C"], ["A", "B", "B", "C", "C"]], labels)
    assert prior["trans"].shape == (3, 3) and prior["trans"][0, 0] > prior["trans"][0, 2]
    rng = np.random.RandomState(0)
    rec = lambda: {"d": {"logp": np.log(rng.dirichlet([1, 1, 1], 6)), "emissions": rng.normal(size=(6, 3)),
                         "domain": "CL", "crf": {"trans": rng.normal(size=(3, 3)), "start": np.zeros(3), "end": np.zeros(3)}}}
    parts = [rec(), rec()]
    avg = {"d": {**parts[0]["d"], "crf": None}}
    ca = crf_average(parts)
    assert np.allclose(ca["d"]["emissions"], (parts[0]["d"]["emissions"] + parts[1]["d"]["emissions"]) / 2)
    kinds = [c[0]["kind"] for c in sequence_candidates(avg, ca, prior)]
    assert kinds == ["marginal", "smooth", "smooth", "smooth", "crfavg"]
    sm = build_sequence_records({"kind": "smooth", "lam": 0.5}, avg, ca, prior)
    assert np.allclose(sm["d"]["crf"]["trans"], 0.5 * prior["trans"])
    assert crf_average([avg, parts[0]]) is None              # a member without a CRF -> no crfavg candidate
    big = viterbi_batch([sm["d"]["emissions"] * 100], [sm["d"]["crf"]])[0]
    assert (big == sm["d"]["logp"].argmax(1)).all()           # strong emissions -> Viterbi = argmax


def test_r1_recipe_end_to_end():
    """Ensemble teacher (2 fold-matched runs) -> student with whole-document context (cache), BiLSTM+attention
    context, two-pass encoding, label-shift + coarse heads, similarity-ranked silver, confidence KD, silver decay,
    token dropout, splicing; then the final model reloads and predicts."""
    out = P3 / "runs_r1"
    data = P3 / "Phase_3_RR"
    base = json.loads((FX / "unlabeled.json").read_text())        # judgments long enough for the 20-sentence floor
    silver = P3 / "silver.json"
    silver.write_text(json.dumps([{"text": "\n".join(r["text"] for r in base[i:i + 4])} for i in range(0, 48, 4)]))
    run("-m", "phase3.run", "cv", *COMMON, "--checkpoint", ckpt("dense_gpt"), "--method", "lorma",
        "--output", str(out / "T1"), "--final")
    run("-m", "phase3.run", "cv", *COMMON, "--checkpoint", ckpt("gqa_local"), "--method", "lora",
        "--output", str(out / "T2"), "--final")
    r = run("-m", "phase3.run", "cv", *COMMON, "--checkpoint", ckpt("gqa_local"), "--method", "lora",
            "--teacher", str(out / "T1"), str(out / "T2"), "--kd-unlabeled", str(silver), "--kd-take-prob", "1",
            "--kd-unlabeled-docs", "2", "--kd-unlabeled-sentences", "60", "--kd-select-similar", "--kd-similar-pool",
            "3", "--kd-confidence",
            "--kd-silver-schedule", "decay", "--token-dropout", "0.1", "--context", "lstm_tf", "--doc-context",
            "--two-pass", "--aux-shift", "0.3", "--aux-coarse", "0.3", "--splice", "0.5", "--domain-heads",
            "--output", str(out / "R1"), "--final")
    assert "2 teacher(s)" in r.stdout and "OOF pooled macro-F1" in r.stdout
    assert "KD unlabeled: 6 judgments" in r.stdout                  # 3x pool, then 2 kept per fold by similarity
    sent = [json.loads(l)["unlabeled_sentences"] for l in r.stdout.splitlines() if l.startswith('{"train_docs"')]
    assert sent and all(n > 0 for n in sent)
    assert (out / "R1" / "submission" / "submission.zip").exists()
    from phase3.model import load_tagger
    tagger, _, payload = load_tagger(out / "R1" / "final" / "model.pt")
    assert tagger.two_pass and tagger.shift_head is not None and tagger.coarse_head is not None
    assert tagger.context_kind == "lstm_tf"
    pr = run("-m", "phase3.run", "predict", "--data", str(data), "--model", str(out / "R1" / "final" / "model.pt"),
             "--split", "dev", "--output", str(out / "R1_dev"))
    assert "pooled macro-F1 on dev" in pr.stdout
    # a second noisy-student round: the student joins the teachers
    r2 = run("-m", "phase3.run", "cv", *COMMON, "--checkpoint", ckpt("gqa_local"), "--method", "lora",
             "--teacher", str(out / "T1"), str(out / "R1"), "--context", "lstm_tf", "--doc-context",
             "--aux-shift", "0.3", "--output", str(out / "R2"))
    assert "2 teacher(s)" in r2.stdout
    assert "documents reused, 0 computed" in r2.stdout          # T1's posteriors come from the disk cache


def test_aux_heads_and_select_similar():
    from collections import OrderedDict
    from phase3.model import coarse_map, RoleTagger
    from phase3.data import select_similar
    mapping, groups = coarse_map(LABELS)
    assert groups[mapping[LABELS.index("ArgumentPetitioner")]] == groups[mapping[LABELS.index("ArgumentRespondent")]] == "Argument"
    assert len(groups) == 9

    class Enc(torch.nn.Module):
        dim = 8
    t = RoleTagger(Enc(), LABELS, context="lstm_tf", hidden=16, aux_shift=True, aux_coarse=True)
    em, h = t.contextualize(torch.randn(7, 16), torch.linspace(0, 0.9, 7), "CL", return_hidden=True)
    assert em.shape == (7, 13) and h.shape == (7, 32)
    tags = torch.tensor([0, 0, 2, 2, 3, 10, 10])
    loss = t.aux_loss(h, tags, h[0].detach(), torch.tensor(0), 0.5, 0.5)
    loss.backward()
    assert loss.item() > 0 and t.shift_head[0].weight.grad is not None
    ref = OrderedDict(a={"sentences": ["the competition commission found abuse of dominant position"] * 3, "domain": "CL"},
                      b={"sentences": ["the assessee claimed depreciation for the assessment year"] * 3, "domain": "IT"})
    cands = OrderedDict((f"u{i}", {"sentences": [s] * 3, "domain": None, "labels": None}) for i, s in enumerate([
        "dominant position abuse by the opposite party in the relevant market competition commission",
        "the cricket match was washed out by rain",
        "assessment year depreciation claimed by the assessee before the tribunal",
        "a recipe for bread with flour and water"]))
    kept = select_similar(cands, ref, 2)
    assert set(kept) == {"u0", "u2"} and kept["u0"]["domain"] == "CL" and kept["u2"]["domain"] == "IT"


def _fake_fa2(q, k, v, dropout_p=0.0, causal=False, window_size=(-1, -1)):
    """flash_attn_func contract on CPU: [B,L,H,D] layout, fewer KV heads (GQA), window_size=(left, right)."""
    b, l, h, d = q.shape
    g = h // k.shape[2]
    k, v = k.repeat_interleave(g, 2), v.repeat_interleave(g, 2)
    s = torch.einsum("blhd,bmhd->bhlm", q.float(), k.float()) / d ** 0.5
    i, j = torch.arange(l)[:, None], torch.arange(l)[None, :]
    allowed = torch.ones(l, l, dtype=torch.bool)
    if causal:
        allowed &= j <= i
    if window_size[0] >= 0:
        allowed &= j >= i - window_size[0]
    if window_size[1] >= 0:
        allowed &= j <= i + window_size[1]
    p = s.masked_fill(~allowed, float("-inf")).softmax(-1)
    return torch.einsum("bhlm,bmhd->blhd", p, v.float()).to(q.dtype)


def test_fast_attention_matches_phase1():
    from phase2.model import Encoder, load_phase1
    from phase3 import flash
    torch.manual_seed(0)
    cases = [("gqa_local", "causal"), ("gqa_local", "bidirectional"), ("dense_gpt", "causal"),
             ("recurrent_hybrid", "causal"), ("encoder_decoder", "encoder")]
    try:
        for arch, mode in cases:
            lm, _, _ = load_phase1(ckpt(arch), str(PHASE1))
            enc = Encoder(lm, mode=mode).eval()
            ids = torch.randint(5, 200, (2, min(enc.max_context, 160)))
            local = [b.attn.window for b in enc.blocks if getattr(getattr(b, "attn", None), "local", False)]
            if arch == "gqa_local":
                assert local and ids.size(1) > local[0]            # the sliding-window path is exercised
            flash.enable("off")
            with torch.no_grad():
                ref = enc(ids)
            for backend in ("sdpa", "fa2"):
                flash._STATE["fa2"] = _fake_fa2 if backend == "fa2" else None
                flash._STATE["allow_cpu"] = backend == "fa2"
                flash._STATE["original"] = flash._STATE["original"]
                flash.enable("sdpa" if backend == "sdpa" else "auto")
                with torch.no_grad():
                    out = enc(ids)
                assert torch.allclose(out, ref, atol=2e-5, rtol=1e-4), (arch, mode, backend, (out - ref).abs().max())
            flash.enable("off")
            enc.train()                                              # gradients flow through the patched path
            flash._STATE["fa2"], flash._STATE["allow_cpu"] = None, False
            flash.enable("sdpa")
            enc(ids[:, :40]).sum().backward()
            assert any(p.grad is not None for p in enc.parameters())
    finally:
        flash._STATE["fa2"], flash._STATE["allow_cpu"] = None, False
        flash.enable("off")


def test_encode_many_matches_encode_sentences_and_kd_cache(tmp_path):
    import argparse as ap
    from phase3.model import build_tagger
    from phase3.run import encode_many, encode_sentences
    a = ap.Namespace(checkpoint=ckpt("gqa_local"), init_phase2=None, project_root=str(PHASE1), method="lora", mode="causal",
                     rank=4, alpha=8, keep_layers=None, prompt_tokens=0, gradient_checkpointing=False, window=64,
                     packing="packed", pooling="meanlast", context="bilstm", hidden=16, context_layers=1,
                     position_buckets=20, no_position=False, domain_heads=False, dropout=0.0, loss="crf",
                     max_batch_tokens=128, infer_batch_tokens=4096, two_pass=True, bottleneck=64, top_k=4)
    tagger, tok, _ = build_tagger(a, LABELS)
    tagger.eval()
    rng = random.Random(0)
    docs = [[[rng.randrange(5, 200) for _ in range(rng.randint(3, 30))] for _ in range(rng.randint(2, 12))]
            for _ in range(5)]
    many = encode_many(tagger, docs, a, torch.device("cpu"), None, 0)
    with torch.no_grad():
        for d, v in zip(docs, many):
            ref = encode_sentences(tagger, d, a, torch.device("cpu"), None, 0)
            assert v.shape == ref.shape == (len(d), 4 * tagger.encoder.dim)
            assert torch.allclose(v, ref, atol=1e-5)


def test_speaker_rule():
    from phase3.rules import apply_speaker, speaker_side, apply_rules
    AP, AR = "ArgumentPetitioner", "ArgumentRespondent"
    sents = ["Learned counsel for the petitioner submitted that the order is bad.", "According to him the DG erred.",
             "He relied on a judgment.", "We have heard Shri X appearing on behalf of the Competition Commission of India.",
             "Shri X does not dispute the proposition canvassed by the learned counsel for the petitioner.",
             "According to him the order is only prima facie.", "He relies on SAIL.",
             "Dr. Y appeared on behalf of RJIL and supported the order passed by the Commission.", "According to him RJIL was ready.",
             "Fact one.", "Fact two.", "Fact three.", "According to him something else."]
    labels = [AP, AP, AP, "None", AP, AP, AP, AP, AP, "Fact", "Fact", "Fact", AP]
    out = apply_speaker(labels, sents, "HC_Test_doc")
    assert out[:3] == [AP] * 3 and out[3] == "None"                 # non-argument labels never change
    assert out[4:9] == [AR] * 5                                       # CCI counsel, then RJIL supporting the order
    assert out[12] == AP                                              # side forgotten after 3 non-argument sentences
    assert speaker_side(sents[4], "HC_x") is None                     # quoting the other side is not a speaker change
    # the Commission is the court in its own orders, and a cue only in High Court / COMPAT documents
    assert speaker_side("Counsel appearing for the Commission submitted.", "CCI_x") is None
    assert speaker_side("Counsel appearing for the Commission submitted.", "HC_x") == "R"
    assert speaker_side("8 Per contra, Mr. A, learned Counsel appearing for the respondent supports it.", "BHC_x") == "R"
    assert apply_rules({"d": [AP, AP]}, {"d": {"sentences": ["On the other hand, x.", "y"]}}, ["speaker"])["d"] == [AP, AP]


def test_rule_variants_and_post_rules_in_scoring():
    from phase3.decode import with_rules, predict_indices
    from phase3.rules import VARIANTS, RULES
    assert ["dissent", "speaker"] in VARIANTS and all(r in RULES for v in VARIANTS for r in v)
    labels = LABELS
    AP, AR = labels.index("ArgumentPetitioner"), labels.index("ArgumentRespondent")
    sents = ["Counsel appearing for the respondent submitted x.", "According to him y."]
    lp = np.full((2, 13), -5.0); lp[:, AP] = 0
    rec = {"d": {"logp": lp, "emissions": lp, "domain": "CL", "crf": None}}
    plain = predict_indices([rec["d"]], "marginal")
    ruled = with_rules(rec, {"d": {"sentences": sents}}, ["dissent", "speaker"], labels)
    assert list(plain) == [AP, AP] and list(predict_indices([ruled["d"]], "marginal")) == [AR, AR]


def test_ia3_keeps_attention_dtypes_equal():
    """IA3 scales keys/values with float32 parameters; under bf16 autocast the attention inputs must stay alike
    (FlashAttention-2 refuses mixed dtypes)."""
    import argparse as ap
    from phase2.model import Encoder, load_phase1
    from phase3 import flash
    from phase3.peft import apply_technique
    seen = []

    def strict_fa2(q, k, v, **kw):
        seen.append((q.dtype, k.dtype, v.dtype))
        assert q.dtype == k.dtype == v.dtype
        return _fake_fa2(q, k, v, **kw)
    lm, _, _ = load_phase1(ckpt("gqa_local"), str(PHASE1))
    enc = Encoder(lm, mode="causal")
    apply_technique(enc, "ia3", ap.Namespace(rank=4, alpha=8, prefix_length=4, prefix_hidden=16, adalora_init_r=4,
                                             adalora_target_r=2, prompt_tokens=0))
    try:
        flash._STATE["fa2"], flash._STATE["allow_cpu"] = strict_fa2, True
        flash.enable("auto")
        with torch.autocast("cpu", dtype=torch.bfloat16):
            enc(torch.randint(5, 200, (2, 40))).float().sum().backward()
        assert seen and all(d[0] == torch.bfloat16 for d in seen)
    finally:
        flash._STATE["fa2"], flash._STATE["allow_cpu"] = None, False
        flash.enable("off")
