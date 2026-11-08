"""CPU tests for Phase 7 (prior case retrieval) on synthetic cases + tiny random Phase-1 checkpoints:
    python -m pytest -q tests/test_phase7.py
"""
import argparse, json, os, random, subprocess, sys
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

from phase7.data import f1_at_k, load_split, rank_from_scores, validate_prediction, write_submission
from phase7.lexical import SplitIndex, tune

FX = Path("/tmp/fx_test")
P7 = Path("/tmp/fx_p7")
ARCHS = ["dense_gpt", "gqa_local", "recurrent_hybrid", "encoder_decoder"]


def case(rng):
    from make_fixture import document
    return " ".join(document(rng)[0] for _ in range(rng.randint(2, 3)))


def split(rng, n_queries, n_pool, prefix):
    pool = {f"{prefix}c{i:04d}": case(rng) for i in range(n_pool)}
    ids = list(pool)
    queries = []
    for i in range(n_queries):
        cited = rng.sample(ids, rng.randint(3, 9))
        quotes = []
        for c in cited:                                      # a judgment quotes the cases it cites
            words = pool[c].split()
            a = rng.randrange(max(1, len(words) - 30))
            quotes.append("It was observed that " + " ".join(words[a:a + 30]) + " <CITATION>.")
        text = case(rng) + " " + " ".join(quotes) + " " + case(rng)
        sents = [s.strip() + "." for s in text.split(".") if s.strip()]
        queries.append({"id": f"{prefix}q{i:04d}", "text": sents, "relevant_candidates": cited})
    return queries, [{"id": c, "text": [t]} for c, t in pool.items()]


def build(root, seed=0):
    rng = random.Random(seed)
    d = root / "Phase_7_PCR"
    for name, (nq, npool) in {"train": (30, 90), "dev": (8, 40), "test": (8, 40)}.items():
        folder = d / {"train": "Train", "dev": "Dev", "test": "Test"}[name]
        folder.mkdir(parents=True, exist_ok=True)
        q, p = split(rng, nq, npool, name[:2])
        if name == "test":
            for r in q:
                del r["relevant_candidates"]
        if name == "dev":                                     # a query inside its own pool must not retrieve itself
            p.append({"id": q[0]["id"], "text": q[0]["text"]})
        json.dump(q, open(folder / f"{name}.json", "w"))
        json.dump(p, open(folder / f"{name}_candidates.json", "w"))
    return d


@pytest.fixture(scope="session", autouse=True)
def fixtures():
    if not (FX / "planted.json").exists():
        subprocess.run([sys.executable, str(ROOT / "tests/make_fixture.py"), str(FX)], check=True)
    build(P7)
    from phase6.model import install_masks
    install_masks()


def run(*argv):
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{PHASE1}")
    r = subprocess.run([sys.executable, *argv], cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(r.stdout[-3000:] + r.stderr[-3000:])
    return r


def test_metric_ranking_and_submission(tmp_path):
    gold = {"a": ["1", "2", "3"], "b": ["4"] * 1 + [str(i) for i in range(10, 20)]}
    m = f1_at_k({"a": ["1", "2", "9", "8", "7", "6", "5"], "b": ["4", "10", "x", "y", "z", "u", "v"]}, gold)
    tp, p, r = 4, 4 / 14, 4 / 14
    assert m["tp"] == tp and abs(m["precision"] - p) < 1e-12 and abs(m["recall"] - r) < 1e-12
    ranked = rank_from_scores(np.array([[5.0, 9.0, 1.0, 2.0, 3.0, 4.0, 0.5, 0.1, 0.0]]), ["q"],
                              ["c0", "q", "c2", "c3", "c4", "c5", "c6", "c7", "c8"])
    assert ranked["q"] == ["c0", "c5", "c4", "c3", "c2", "c6", "c7"]          # the query itself is skipped
    q = {"q": {"text": "x", "relevant": None}}
    pool = {f"c{i}": "t" for i in range(9)}
    write_submission(ranked, q, pool, tmp_path)
    assert json.loads((tmp_path / "prediction.json").read_text()) == ranked
    for bad in ({"q": ranked["q"][:6]}, {"q": ranked["q"][:6] + ["c0"]}, {"q": ranked["q"][:6] + ["zz"]}, {}):
        with pytest.raises(ValueError):
            validate_prediction(bad, q, pool)


def test_lexical_finds_quoted_cases():
    q, p = load_split(P7 / "Phase_7_PCR", "train")
    tuned = tune(SplitIndex(q, p), grams=(1, 3), k1s=(1.2,), bs=(0.75,), max_dfs=(0.5,), log=lambda *a: None)
    assert tuned["per_n"][3]["f1"] > 0.3 and tuned["per_n"][3]["f1"] > 2 * tuned["per_n"][1]["f1"], tuned  # quotes


def test_padded_encoding_matches_single():
    from phase7.neural import build_retriever, encode
    rng = random.Random(3)
    seqs = [[rng.randint(5, 600) for _ in range(n)] for n in (11, 37, 24)]
    for arch, method, mode, pool in [(a, "lorma", "bidirectional", "mean") for a in ARCHS] + \
            [("dense_gpt", "lorma", "causal", "last"), ("gqa_local", "prefix", "bidirectional", "max"),
             ("dense_gpt", "prompt", "bidirectional", "mean"), ("recurrent_hybrid", "ia3", "causal", "mean")]:
        a = argparse.Namespace(checkpoint=str(FX / f"phase1_runs/{arch}/best.pt"), project_root=str(PHASE1),
                               method=method, rank=4, alpha=8, top_k=1, bottleneck=8, adalora_init_r=4,
                               adalora_target_r=2, prefix_length=4, prefix_hidden=16, prompt_tokens=4, mode=mode,
                               pool=pool, proj=0, keep_layers=None, layer_pick="spread", random_init=False,
                               gradient_checkpointing=False)
        torch.manual_seed(0)
        model, _, _ = build_retriever(a)
        together = encode(model, seqs, torch.device("cpu"), None, 10 ** 6)
        alone = np.concatenate([encode(model, [s], torch.device("cpu"), None, 10 ** 6) for s in seqs])
        assert np.abs(together.astype(np.float32) - alone.astype(np.float32)).max() < 2e-3, (arch, method, mode)
        assert np.allclose(np.linalg.norm(together.astype(np.float32), axis=1), 1, atol=1e-2)


def test_end_to_end():
    data = str(P7 / "Phase_7_PCR")
    out = P7 / "runs"
    r = run("-m", "phase7.run", "stats", "--data", data)
    assert '"queries_in_own_pool": 1' in r.stdout
    run("-m", "phase7.run", "lexical", "--data", data, "--output", str(out / "B0"), "--grams", "1", "2", "3")
    b0 = json.loads((out / "B0/summary.json").read_text())
    assert b0["dev"]["f1"] > 0.25 and (out / "B0/submission/submission.zip").exists()
    common = ["--data", data, "--project-root", str(PHASE1), "--lexical", str(out / "B0"), "--epochs", "1",
              "--batch", "8", "--chunk-words", "40", "--max-chunks", "16", "--max-tokens", "64", "--rank", "4",
              "--alpha", "8", "--prefix-length", "4", "--prefix-hidden", "16", "--prompt-tokens", "4",
              "--adalora-init-r", "4", "--adalora-target-r", "2", "--max-pairs", "64"]
    ck = lambda a: ["--checkpoint", str(FX / f"phase1_runs/{a}/best.pt")]
    runs = {
        "L1": ck("dense_gpt") + ["--method", "lorma"],
        "L2": ck("gqa_local") + ["--method", "lorma", "--mode", "bidirectional", "--pool", "last"],
        "L3": ck("recurrent_hybrid") + ["--method", "lorma", "--negatives", "random"],
        "L4": ck("encoder_decoder") + ["--method", "lorma", "--pool", "max"],
        "T_full": ck("dense_gpt") + ["--method", "full", "--gradient-checkpointing", "--proj", "32"],
        "T_partial": ck("gqa_local") + ["--method", "partial", "--top-k", "1", "--negatives", "none"],
        "T_qlora": ck("gqa_local") + ["--method", "qlora"],
        "T_ia3": ck("dense_gpt") + ["--method", "ia3"],
        "T_prefix": ck("dense_gpt") + ["--method", "prefix", "--mode", "bidirectional"],
        "T_prompt": ck("encoder_decoder") + ["--method", "prompt"],
        "T_adalora": ck("gqa_local") + ["--method", "adalora"],
        "Z_frozen": ck("dense_gpt") + ["--method", "frozen"],
    }
    for name, extra in runs.items():
        run("-m", "phase7.run", "train", *common, "--output", str(out / name), *extra)
        s = json.loads((out / name / "summary.json").read_text())
        assert set(s["dev"]["per_agg"]) == {"maxsim", "maxsim_top", "max", "mean"}, name
        assert s["dev"]["fused"]["f1"] >= s["dev"]["lexical"]["f1"] - 1e-9          # weight 1.0 = lexical alone
        for sub in ("submission", "submission_neural"):
            run("-m", "phase7.run", "check", "--data", data, "--prediction", str(out / name / sub / "prediction.json"))
    run("-m", "phase7.run", "train", *common, "--output", str(out / "K1"), *ck("dense_gpt"), "--method", "full",
        "--keep-layers", "2", "--teacher", str(out / "T_full"), "--kd-embed", "0.5")
    k1 = json.loads((out / "K1/summary.json").read_text())
    assert k1["teacher"]["layers"] > k1["layers"] == 2
    run("-m", "phase7.run", "combine", "--data", data, "--runs", str(out / "L1"), str(out / "L4"), "--output",
        str(out / "C1"))
    c = json.loads((out / "C1/summary.json").read_text())
    assert set(c["weights"]) == {"L1", "lexical", "L4"}
    run("-m", "phase7.run", "check", "--data", data, "--prediction", str(out / "C1/submission/prediction.json"))


def test_citation_context_and_rerank_merge():
    from phase7.data import citation_context
    from phase7.rerank import combine
    t = "Facts.\nThe court relied on <CITATION> here.\nOther text.\nMore text.\nAs held in <CITATION>, it fails.\nEnd."
    ctx, n = citation_context(t, 0)
    assert n == 2 and ctx.count("<CITATION>") == 2 and "Other" not in ctx
    assert "Other" in citation_context(t, 1)[0] and citation_context("no marks.", 2) == ("no marks.", 0)
    first = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    rr = np.full(first.shape, np.nan)
    rr[0, :3] = [0.0, 1.0, 5.0]                               # re-ranker reverses the top 3
    out = combine(first, rr, 0.0)
    assert list(np.argsort(-out[0])) == [2, 1, 0, 3, 4]      # re-ranked on top, the rest keeps first-stage order
    assert list(np.argsort(-combine(first, rr, 1.0)[0])) == [0, 1, 2, 3, 4]


def test_extensions_end_to_end():
    """Citation-context / PRF lexical tuning, context queries, second-round negatives, list-level KD from the
    fused teacher, the passage re-ranker over two first stages, z-score and RRF combination."""
    data = str(P7 / "Phase_7_PCR")
    out = P7 / "runs"
    assert (out / "L1/model.pt").exists(), "run test_end_to_end first"
    r = run("-m", "phase7.run", "lexical", "--data", data, "--output", str(out / "B1"), "--grams", "2", "4",
            "--context-windows", "0", "1", "--prf", "--passage-words", "40")
    assert "citation context" in r.stdout and "PRF" in r.stdout
    b1 = json.loads((out / "B1/lexical.json").read_text())
    assert "context" in b1 and "prf" in b1 and "passage" in b1 and "best passage" in r.stdout
    common = ["--data", data, "--project-root", str(PHASE1), "--lexical", str(out / "B1"), "--epochs", "1",
              "--batch", "8", "--chunk-words", "40", "--max-chunks", "16", "--max-tokens", "64", "--rank", "4",
              "--alpha", "8", "--max-pairs", "48"]
    dense = ["--checkpoint", str(FX / "phase1_runs/dense_gpt/best.pt")]
    runs = {
        "R_ctx": dense + ["--method", "lorma", "--query-context", "1", "--top-chunks", "4", "--min-pair-sim", "0.01"],
        "R2_round2": dense + ["--method", "lorma", "--init-from", str(out / "L1"), "--negatives", "fused"],
        "K3_listkd": dense + ["--method", "full", "--keep-layers", "2", "--kd-list", str(out / "T_full"),
                              "--list-size", "6", "--list-queries", "2"],
    }
    for name, extra in runs.items():
        run("-m", "phase7.run", "train", *common, "--output", str(out / name), *extra)
        run("-m", "phase7.run", "check", "--data", data, "--prediction", str(out / name / "submission/prediction.json"))
    s = json.loads((out / "K3_listkd/summary.json").read_text())
    assert s["list_teacher"]["run"].endswith("T_full") and s["layers"] == 2
    assert json.loads((out / "R2_round2/summary.json").read_text())["init_from"].endswith("L1")
    run("-m", "phase7.run", "combine", "--data", data, "--runs", str(out / "B1"), str(out / "R_ctx"), str(out / "L1"),
        "--output", str(out / "C2"))
    run("-m", "phase7.run", "combine", "--data", data, "--runs", str(out / "R_ctx"), str(out / "L1"), "--output",
        str(out / "C3_rrf"), "--fusion", "rrf")
    assert json.loads((out / "C3_rrf/summary.json").read_text())["fusion"] == "rrf"
    rr = ["--data", data, "--project-root", str(PHASE1), "--lexical", str(out / "B1"), "--epochs", "1", "--chunk-words",
          "40", "--max-chunks", "16", "--max-tokens", "64", "--rank", "4", "--alpha", "8", "--max-pairs", "32",
          "--groups", "4", "--rerank-depth", "10", "--pair-k", "2", "--side-tokens", "48", "--pool", "last"]
    for name, first in (("X1", "C2"), ("X2", "B0")):
        run("-m", "phase7.run", "rerank", *rr, *dense, "--method", "lorma", "--first", str(out / first),
            "--output", str(out / name))
        x = json.loads((out / name / "summary.json").read_text())
        assert x["dev"]["fused"]["f1"] >= x["dev"]["first_stage"]["f1"] - 1e-9 or x["weight_first"] < 1
        run("-m", "phase7.run", "check", "--data", data, "--prediction", str(out / name / "submission/prediction.json"))
    run("-m", "phase7.run", "combine", "--data", data, "--runs", str(out / "X1"), str(out / "R_ctx"), "--output",
        str(out / "C4"), "--prior", "--years", "--hub", "--knn")
    c4 = json.loads((out / "C4/summary.json").read_text())
    assert set(c4["features"]) == {"prior", "later_year", "neighbours"} and "hub" in c4
    run("-m", "phase7.run", "check", "--data", data, "--prediction", str(out / "C4/submission/prediction.json"))
    from phase7.data import max_year
    assert max_year("decided in 1987, citing (1954) 2 SCR 1 and AIR 1950 SC 27") == 1987 and max_year("none") == 0


def test_neighbour_citations_vote():
    from phase7.lexical import neighbor_citations
    src = {"s1": {"text": "dowry death cruelty husband section 304b evidence", "relevant": ["c1", "c2"]},
           "s2": {"text": "land acquisition compensation market value award", "relevant": ["c3"]}}
    tgt = {"t1": {"text": "the husband was charged with dowry death and cruelty under section 304b", "relevant": None}}
    f, rows = neighbor_citations(tgt, src, ["c1", "c2", "c3", "c4"], ks=(1, 2))
    assert rows == ["t1"] and f[1][0, 0] > 0 and f[1][0, 2] == 0 and f[1][0, 3] == 0   # only the nearest votes
    assert f[2][0, 2] > 0 and f[2][0, 0] > f[2][0, 2]                                  # nearer neighbour weighs more
