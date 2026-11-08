"""CPU tests on synthetic data: python -m pytest -q tests/test_phase2.py"""
import itertools, subprocess, sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from _paths import phase1_root
PHASE1 = phase1_root()
sys.path.insert(1, str(PHASE1))
from phase2.data import encode_document, spans_to_tags, tags_to_spans, tag_names, load_texts, load_gold, load_labels
from phase2.model import CRF, Encoder, load_phase1
from phase2.peft import LoRMALinear, LoRALinear, apply_method, count_parameters
from phase2.post import strict_scores, BoundaryRules, propagate, label_vote, error_breakdown

FX = Path("/tmp/fx_test")


@pytest.fixture(scope="session", autouse=True)
def fixture():
    if not (FX / "planted.json").exists():   # rebuilt when the fixture predates the overlap filter
        subprocess.run([sys.executable, str(ROOT / "tests/make_fixture.py"), str(FX)], check=True)
    if not (FX / "incl" / "Phase_2_LNER").exists():
        subprocess.run([sys.executable, str(ROOT / "tests/make_fixture.py"), str(FX / "incl"), "inclusive"], check=True)


def test_lorma_and_lora_start_at_base():
    base = torch.nn.Linear(8, 6, bias=False)
    x = torch.randn(3, 8)
    for wrapper in (LoRMALinear(base, rank=2), LoRALinear(base, rank=2, dropout=0.0)):
        assert torch.allclose(wrapper(x), base(x))
    m = LoRMALinear(base, rank=2, alpha=2)
    with torch.no_grad():
        m.U.normal_()
    expected = (torch.eye(6) + m.scale * m.U @ m.V) @ base.weight   # LoRMA+ closed form
    assert torch.allclose(m(x), x @ expected.t(), atol=1e-5)


def test_crf_matches_brute_force():
    torch.manual_seed(0)
    names = tag_names(["A", "B"])
    crf = CRF(names)
    with torch.no_grad():
        crf.trans.normal_(); crf.start.normal_(); crf.end.normal_()
    em = torch.randn(1, 3, len(names))
    trans, start, end = crf._scores()
    edge_start, edge_end = crf.start.detach(), crf.end.detach()  # nll leaves window edges unconstrained
    scores, free = {}, {}
    for path in itertools.product(range(len(names)), repeat=3):
        body = sum(em[0, i, t] for i, t in enumerate(path)) + sum(trans[a, b] for a, b in zip(path, path[1:]))
        scores[path] = (start[path[0]] + end[path[-1]] + body).detach()
        free[path] = (edge_start[path[0]] + edge_end[path[-1]] + body).detach()
    logz_free = torch.logsumexp(torch.stack(list(free.values())), 0)
    assert torch.allclose(crf.nll(em, torch.tensor([[1, 3, 0]])), (logz_free - free[(1, 3, 0)]) / 3, atol=1e-4)
    # a window that starts inside an entity (I-A E-A O) must have a finite, sane loss
    assert crf.nll(em, torch.tensor([[2, 3, 0]])) < 100
    best = max(scores, key=lambda p: float(scores[p]))
    assert tuple(crf.viterbi(em[0])) == best


def test_alignment_roundtrip_exact():
    labels = load_labels(FX / "Phase_2_LNER/Train/labels.txt")
    texts = load_texts(FX / "Phase_2_LNER/Train/fold1-text.json")
    gold, rep = load_gold(FX / "Phase_2_LNER/Train/fold1-gold.json", texts, labels)
    assert rep["end_inclusive"] is False
    _, tok, _ = load_phase1(FX / "phase1_runs/dense_gpt/best.pt", PHASE1)
    allowed = __import__("phase2.model", fromlist=["x"]).transition_mask(tag_names(labels))[0]
    for doc, spans in gold.items():
        _, offsets = encode_document(tok, texts[doc])
        tags, conflicts = spans_to_tags(offsets, spans, labels)
        assert conflicts == 0
        assert all(allowed[a, b] for a, b in zip(tags, tags[1:])), "gold tags must be valid BIOES"
        assert set(tags_to_spans(offsets, tags, labels)) == set(spans)


def test_inclusive_convention_detected():
    labels = load_labels(FX / "incl/Phase_2_LNER/Train/labels.txt")
    texts = load_texts(FX / "incl/Phase_2_LNER/Train/fold1-text.json")
    _, rep = load_gold(FX / "incl/Phase_2_LNER/Train/fold1-gold.json", texts, labels)
    assert rep["end_inclusive"] is True


@pytest.mark.parametrize("arch,mode", [("dense_gpt", "causal"), ("dense_gpt", "bidirectional"),
                                       ("gqa_local", "bidirectional"), ("encoder_decoder", "encoder"),
                                       ("recurrent_hybrid", "bidirectional")])
def test_encoder_modes_and_methods(arch, mode):
    lm, _, _ = load_phase1(FX / f"phase1_runs/{arch}/best.pt", PHASE1)
    enc = Encoder(lm, mode=mode).eval()
    ids = torch.randint(4, 600, (2, 40))
    h = enc(ids)
    assert h.shape == (2, 40, enc.dim)
    if mode == "causal":  # changing a later token must not affect earlier states
        ids2 = ids.clone(); ids2[:, 30] = 5
        assert torch.allclose(enc(ids2)[:, :30], h[:, :30], atol=1e-5)
    for method in ["lora", "lorma", "adapter", "normfit", "partial", "full", "prompt"]:
        lm, _, _ = load_phase1(FX / f"phase1_runs/{arch}/best.pt", PHASE1)
        e = Encoder(lm, mode=mode, prompt_tokens=3 if method == "prompt" else 0).eval()
        before = e(ids).detach()
        apply_method(e, method, rank=4, top_k=1)
        if method != "prompt":   # prompt tokens change attention context by design
            assert torch.allclose(e(ids), before, atol=1e-5), method  # starts at the Phase-1 function
        assert count_parameters(e)["trainable"] > 0, method


def test_postprocessing_helpers():
    text = "Ram Kumar met Ram Kumar."
    assert propagate(text, [(0, 9, "APP")], {"APP"}) == [(0, 9, "APP"), (14, 23, "APP")]
    assert label_vote(text, [(0, 9, "APP"), (14, 23, "APP"), (0, 9, "RESP")], {"APP", "RESP"})[2][2] == "APP"
    b = BoundaryRules().fit({"d": "x Ram Kumar, y"}, {"d": [(2, 11, "APP")]})
    assert b("x Ram Kumar, y", [(2, 12, "APP")]) == [(2, 11, "APP")]
    s = strict_scores({"d": [(0, 3, "A"), (4, 6, "B")]}, {"d": [(0, 3, "A"), (4, 7, "B")]}, ["A", "B"])
    assert s["per_label"]["A"]["f1"] == 1.0 and s["per_label"]["B"]["f1"] == 0.0
    err = error_breakdown({}, {"d": [(0, 3, "A"), (4, 6, "B")]}, {"d": [(0, 3, "B"), (4, 7, "B")]}, ["A", "B"])
    assert err["label_confusion"] == {"A->B": 1} and err["boundary"] == {"B": 1}


def test_punct_segmentation_fixes_merged_closing_punctuation():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from phase2.data import segment_tokenizer
    corpus = [f"Hukam Singh v. State AIR (1977 SC {i}); Eradu [(1985) 1 SCC {i % 90}]. Hence" for i in range(100, 1100)]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=500, special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    text = "relied on Shankarsan Dash v. Union [1991 (3) SCC 47]. Also State AIR (1977 SC 1063); and more"
    s1 = text.index("Shankarsan"); e1 = text.index("]") + 1          # PREC ends at ']' followed by '.'
    s2 = text.index("State AIR"); e2 = text.index(")", s2) + 1        # ends at ')' followed by ';'
    gold = [(s1, e1, "PREC"), (s2, e2, "PREC")]
    labels = ["PREC"]
    for mode, expected in (("bpe", False), ("punct", True)):
        seg = segment_tokenizer(tok, mode)
        _, offsets = encode_document(seg, text)
        tags, _ = spans_to_tags(offsets, gold, labels)
        assert (set(tags_to_spans(offsets, tags, labels)) == set(gold)) is expected, mode
    seg = segment_tokenizer(tok, "punct")
    assert seg.decode(seg.encode(text).ids) == text and tok.get_vocab() == seg.get_vocab()


def test_boundary_rules_keep_balanced_brackets_and_respect_labels():
    texts = {"d": "x (1988) 4 SCC 551 and Padmini (I) Ltd), y"}
    rules = BoundaryRules(threshold=0.5).fit(texts, {"d": [(2, 18, "CASENO"), (23, 38, "APP")]})
    caseno, app = (2, 18, "CASENO"), (23, 38, "APP")
    assert rules(texts["d"], [caseno]) == [caseno]                      # '(' kept: span contains ')'
    assert rules(texts["d"], [app]) == [app]                            # ')' kept: span contains '('
    assert rules(texts["d"], [(23, 39, "APP")]) == [app]               # unbalanced trailing ')' stripped
    assert rules(texts["d"], [(23, 39, "APP")], labels={"CASENO"}) == [(23, 39, "APP")]   # label not selected


def test_span_vote_and_cv_selection(tmp_path):
    import random
    from phase2.vote import vote, best_k, select, eligible
    labels = ["APP", "RESP", "DATE"]
    rng = random.Random(0)
    gold = {f"d{i}": [(j * 10, j * 10 + 5, labels[j % 3]) for j in range(30)] for i in range(20)}

    def noisy(drop, spurious, shift):   # each member: own false positives between gold spans
        return {d: [s for s in g if rng.random() > drop] +
                   [(s + 6, s + 7 + shift, l) for s, e, l in g if rng.random() < spurious] for d, g in gold.items()}
    good = [noisy(0.25, 0.25, shift) for shift in (1, 2, 3)]          # independent errors: a majority vote fixes most
    bad = {d: [] for d in gold}                           # useless member: selection must leave it out
    # majority vote of three beats every single member
    singles = [best_k([m], gold, labels)[0] for m in good]
    score, k = best_k(good, gold, labels)
    assert k == 2 and score > max(singles) + 0.05
    # overlapping disagreements resolved by vote count: the span with 2 votes wins over the 1-vote shifted one
    assert vote([[(0, 5, "APP")], [(0, 5, "APP")], [(2, 7, "APP")]], 1) == [(0, 5, "APP")]
    oof = {"g0": good[0], "g1": good[1], "g2": good[2], "bad": bad}
    chosen, k, s, _ = select(list(oof), oof, gold, labels, max_members=7)
    assert set(chosen) == {"g0", "g1", "g2"} and k == 2
    # KD runs: never selection members; extra test-time voters if trained with the overlap filter
    from phase2.vote import classify, vote_with_extras
    kd = tmp_path / "X8_hybrid_kd4"
    (kd / "submission").mkdir(parents=True)
    (kd / "submission" / "token_logprobs.pt").write_bytes(b"x")
    (kd / "args.json").write_text('{"teacher": "X1/model.pt"}')
    assert classify(kd)[0] == "skip"                                   # no filter line: KD text had test documents
    assert classify(kd, allow_unfiltered_kd=True)[0] == "extra"
    (tmp_path / "X8_hybrid_kd4.log").write_text("KD unlabeled: 300 documents, 12 skipped because they contain an NER document\n")
    assert classify(kd)[0] == "extra"
    span_head = tmp_path / "X4_encdec_span"
    (span_head / "submission").mkdir(parents=True)
    (span_head / "oof_predictions.pt").write_bytes(b"x")
    (span_head / "submission" / "prediction.json").write_text("{}")
    assert classify(span_head)[0] == "member"
    a, b, c = (0, 4, "APP"), (10, 14, "RESP"), (20, 24, "WIT")
    kept = vote_with_extras([[a], [a, b]], [[b, c], [c], [c]], k=2, k_all=4)
    assert a in kept and b not in kept and c not in kept               # c: 3 of 5 < 4; b: 1 member + 1 extra
    assert c in vote_with_extras([[a, c]], [[c], [c], [c]], k=2, k_all=3)
    # every KD run is refused as a selection member (its teacher was trained on both folds)
    run = tmp_path / "E4f_hybrid_kd"
    (run / "submission").mkdir(parents=True)
    (run / "oof_predictions.pt").write_bytes(b"x")
    (run / "submission" / "token_logprobs.pt").write_bytes(b"x")
    (run / "args.json").write_text('{"teacher": "X1/model.pt"}')
    assert not eligible(run)[0]
    (run / "args.json").write_text('{"teacher": null}')
    assert eligible(run)[0]


def test_augmentation_replaces_mentions_consistently():
    import random
    from phase2.data import augment_documents
    labels = load_labels(FX / "Phase_2_LNER/Train/labels.txt")
    texts = load_texts(FX / "Phase_2_LNER/Train/fold1-text.json")
    gold, _ = load_gold(FX / "Phase_2_LNER/Train/fold1-gold.json", texts, labels)
    gold = {d: [tuple(x) for x in g] for d, g in gold.items()}
    # an overlapping pair must survive with correct (shifted) offsets
    doc0 = sorted(texts)[0]
    s, e, l = sorted(gold[doc0])[3]
    gold[doc0] = gold[doc0] + [(s, e + 1, "AUTH")]
    out_t, out_g = augment_documents(texts, gold, 2, random.Random(0))
    assert len(out_t) == 3 * len(texts)
    pool = {}
    for d, spans in gold.items():
        for a, b, lab in spans:
            pool.setdefault(lab, set()).add(texts[d][a:b])
    changed = 0
    for d in texts:
        for i in range(2):
            name = f"{d}~aug{i}"
            assert len(out_g[name]) == len(set(gold[d]))
            seen = {}
            for a, b, lab in out_g[name]:
                surface = out_t[name][a:b]
                assert 0 <= a < b <= len(out_t[name])
                assert surface in pool[lab] or surface.upper() in {x.upper() for x in pool[lab]}, (lab, surface)
            orig = {texts[d][a:b]: lab for a, b, lab in gold[d] if lab == "APP"}
            apps = {out_t[name][a:b] for a, b, lab in out_g[name] if lab == "APP"}
            assert len(apps) <= len(orig)          # one appellant stays one appellant (consistent renaming)
            changed += out_t[name] != texts[d]
    assert changed >= len(texts)
    # the planted overlapping span keeps its text
    shifted = [x for x in out_g[f"{doc0}~aug0"] if x[2] == "AUTH" and out_t[f"{doc0}~aug0"][x[0]:x[1]] == texts[doc0][s:e + 1]]
    assert shifted


def test_header_windows_and_o_bias():
    from phase2.run import window_input, document_logprobs, training_windows
    from phase2.model import Tagger
    ids = list(range(10, 110))
    assert window_input(ids, 0, 30, 8) == (ids[0:30], 0)
    x, p = window_input(ids, 40, 30, 8)
    assert p == 8 and x == ids[:8] + ids[40:70]
    assert window_input(ids, 40, 30, 0) == (ids[40:70], 0)
    labels = load_labels(FX / "Phase_2_LNER/Train/labels.txt")
    lm, _, _ = load_phase1(FX / "phase1_runs/dense_gpt/best.pt", PHASE1)
    tagger = Tagger(Encoder(lm, mode="causal"), labels).eval()
    doc = list(range(5, 205))
    plain = document_logprobs(tagger, doc, 48, 24, torch.device("cpu"), None)
    head = document_logprobs(tagger, doc, 40, 24, torch.device("cpu"), None, header=8)
    assert plain.shape == head.shape == (200, len(tag_names(labels)))
    # header windows: every window after the first is fed header + body, all in one batch group
    ex = {"d": {"ids": doc, "tags": [1] * 200}}
    items = training_windows(ex, 40, 24, 1.0, set(), 1, __import__("random").Random(0), header=8)
    assert {it[3] for it in items if it[1] > 0} == {8} and {it[3] for it in items if it[1] == 0} == {0}
    # entry bonus: more entities as it grows, never longer ones (the withdrawn 'O' penalty merged them)
    from phase2.run import decode
    names = tag_names(labels)
    torch.manual_seed(0)
    scores = torch.randn(400, len(names))
    scores[:, 0] += 2.5
    stats = []
    for b in (0.0, 1.0, 3.0):
        spans = tags_to_spans([(i * 2, i * 2 + 1) for i in range(400)],
                              decode(tagger.crf.viterbi, names, scores, {l: b for l in labels}), labels)
        stats.append((len(spans), sum(e - s for s, e, _ in spans) / max(1, len(spans))))
    assert stats[0][0] < stats[1][0] < stats[2][0]
    assert stats[2][1] <= stats[0][1] + 0.5
    assert decode(tagger.crf.viterbi, names, scores, 1.5) == decode(tagger.crf.viterbi, names, scores, 0.0)  # legacy float ignored


def test_new_postprocessing_rules():
    from phase2.post import counsel_side, propagate_ci, propagate_parts, precedent_citations
    text = ("Mr. Anil Sharma, Mr. B. Rao, learned counsel for the appellant and Mr. Kapil Dev for the respondents.\n"
            "For the Appellant : Mr. Ravi Kumar")
    a = text.index("Anil Sharma"); b = text.index("B. Rao"); k = text.index("Kapil Dev"); r = text.index("Ravi Kumar")
    spans = [(a, a + 11, "R.COUNSEL"), (b, b + 6, "R.COUNSEL"), (k, k + 9, "A.COUNSEL"), (r, r + 10, "R.COUNSEL")]
    assert [l for *_, l in counsel_side(text, spans)] == ["A.COUNSEL", "A.COUNSEL", "R.COUNSEL", "A.COUNSEL"]
    text = "RAMESH KUMAR ... Appellant\nThe appellant Ramesh  Kumar was arrested. Later Ramesh was released. ramesh"
    spans = [(0, 12, "APP")]
    ci = propagate_ci(text, spans, {"APP"})
    assert (text.index("Ramesh  Kumar"), text.index("Ramesh  Kumar") + 13, "APP") in ci and len(ci) == 2
    parts = propagate_parts(text, ci, {"APP"})
    later = text.index("Later Ramesh") + 6
    assert (later, later + 6, "APP") in parts
    assert not any(s == text.rindex("ramesh") for s, *_ in parts)          # lower-case word is not a name
    text = "relied on State of U.P. v. Ram, (2003) 5 SCC 120; and also on X v. Y."
    s0 = text.index("State"); e0 = text.index(", (2003)")
    ext = precedent_citations(text, [(s0, e0, "PREC")], "extend")[0]
    assert text[ext[0]:ext[1]] == "State of U.P. v. Ram, (2003) 5 SCC 120"
    trim = precedent_citations(text, [ext], "trim")[0]
    assert text[trim[0]:trim[1]] == "State of U.P. v. Ram"


def test_cv_end_to_end_with_header_augment_and_o_bias(tmp_path):
    import json, os
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{PHASE1}")
    out = tmp_path / "run"
    cmd = [sys.executable, "-m", "phase2.run", "cv", "--data", str(FX / "Phase_2_LNER"),
           "--checkpoint", str(FX / "phase1_runs/dense_gpt/best.pt"), "--project-root", str(PHASE1),
           "--output", str(out), "--method", "lorma", "--mode", "causal", "--window", "64", "--stride", "32",
           "--header-tokens", "16", "--augment", "1", "--o-bias", "auto", "--epochs", "2", "--eval-from", "1",
           "--batch-size", "8", "--final"]
    result = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    assert "augment: 24 training documents -> 48" in result.stdout
    summary = json.loads((out / "cv_summary.json").read_text())
    assert isinstance(summary["o_bias"], dict) and len(summary["o_bias_grid"]) == 10
    payload = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
    assert payload["o_bias"] == summary["o_bias"] and payload["args"]["header_tokens"] == 16
    lp = torch.load(out / "submission/token_logprobs.pt", map_location="cpu", weights_only=False)
    assert lp["o_bias"] == summary["o_bias"]
    assert (out / "submission/submission.zip").exists()
    # predict with the saved model reproduces the header/O-penalty settings
    pred = subprocess.run([sys.executable, "-m", "phase2.run", "predict", "--model", str(out / "model.pt"),
                           "--test", str(FX / "Phase_2_LNER/Test/fold3-text.json"), "--output", str(tmp_path / "p")],
                          env=env, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert pred.returncode == 0, pred.stderr[-2000:]
    a = json.loads((out / "submission/prediction.json").read_text())
    b = json.loads((tmp_path / "p/prediction.json").read_text())
    assert a == b
    # repost with the original configuration reproduces the submission; with new rules it still writes one
    keep = subprocess.run([sys.executable, "-m", "phase2.repost", "--data", str(FX / "Phase_2_LNER"), "--run", str(out),
                           "--keep-config", "--output", str(tmp_path / "keep")], env=env, cwd=ROOT,
                          capture_output=True, text=True, timeout=300)
    assert keep.returncode == 0, keep.stderr[-2000:]
    assert json.loads((tmp_path / "keep/prediction.json").read_text()) == a
    new = subprocess.run([sys.executable, "-m", "phase2.repost", "--data", str(FX / "Phase_2_LNER"), "--run", str(out)],
                         env=env, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert new.returncode == 0, new.stderr[-2000:]
    assert (out / "repost/submission.zip").exists() and "current rules" in new.stdout


def test_reranker_fixes_role_confusion_and_drops_spurious():
    import random
    from phase2 import rerank
    from phase2.post import strict_scores
    labels = load_labels(FX / "Phase_2_LNER/Train/labels.txt")
    rng = random.Random(0)
    fold_data = []
    for fold in (1, 2):
        texts = load_texts(FX / f"Phase_2_LNER/Train/fold{fold}-text.json")
        gold, _ = load_gold(FX / f"Phase_2_LNER/Train/fold{fold}-gold.json", texts, labels)
        gold = {d: [tuple(x) for x in g] for d, g in gold.items()}
        members = []
        for _ in range(3):
            m = {}
            for d, g in gold.items():
                spans = []
                for s, e, l in g:
                    if l in ("APP", "RESP") and rng.random() < 0.5:
                        l = "RESP" if l == "APP" else "APP"          # role confusion
                    spans.append((s, e, l))
                words = [mt.span() for mt in __import__("re").finditer(r"[a-z]{6,}", texts[d])]
                spans += [(a, b, "APP") for a, b in rng.sample(words, min(5, len(words)))]   # spurious names
                m[d] = spans
            members.append(m)
        cands = {d: sorted(set(s for m in members for s in m[d])) for d in gold}
        fold_data.append((texts, cands, members, {d: set() for d in gold}, gold))
    plain_gold = {**fold_data[0][4], **fold_data[1][4]}
    plain_pred = {**fold_data[0][1], **fold_data[1][1]}
    plain = strict_scores(plain_gold, {d: __import__("phase2.post", fromlist=["x"]).remove_overlaps(v) for d, v in plain_pred.items()}, labels)["macro_f1"]
    score, preds = rerank.cross_validate(fold_data, labels)
    per = strict_scores(plain_gold, preds, labels)["per_label"]
    assert score > plain + 0.05, (plain, score)
    assert per["APP"]["f1"] > 0.8 and per["RESP"]["f1"] > 0.8, per
    # test-time path used by vote: fit on everything, then build + decide on new documents
    model = rerank.fit_all(fold_data, labels)
    texts, cands, members, rule_only, gold = fold_data[1]
    X, _, keys = rerank.build(texts, cands, members, rule_only, labels)
    out = rerank.decide(model, X, keys, labels)
    assert set(out) == {d for d in cands if cands[d]} and strict_scores(gold, out, labels)["macro_f1"] > 0.8


def test_header_parties_and_witness_markers():
    """Header styles and conventions seen in the real out-of-fold errors."""
    from phase2.post import parse_parties, mark_parties, mark_witnesses, postprocess
    sc = ("CRIMINAL APPEAL NO. 1919 of 2010\nAnversinh @ Kiransinh Fatesinh Zala ..... Appellant(s)\nVERSUS\n"
          "State of Gujarat .....Respondent(s)\nJUDGMENT\nThe State of Gujarat argued. Shreya Singhal v. Union of India.")
    assert parse_parties(sc) == {"Anversinh @ Kiransinh Fatesinh Zala": "APP", "State of Gujarat": "RESP"}
    kanoon = ("Appeal (crl.) 1613 of 2005\nPETITIONER:\nAPPASAHEB AND ANR.\nRESPONDENT:\nSTATE OF MAHARASHTRA\n"
              "DATE OF JUDGMENT: 05/01/2007\nNarayan who is uncle of appellant no.1, Appasaheb, informed him")
    assert parse_parties(kanoon) == {"APPASAHEB": "APP", "STATE OF MAHARASHTRA": "RESP"}
    a = kanoon.index("Appasaheb,")
    assert (a, a + 9, "APP") in mark_parties(kanoon, [(a, a + 9, "RESP")])          # role from the header wins
    assert mark_parties(kanoon, []) == []                         # no model support anywhere: the parse is not trusted
    delhi = ("O.M.P. (COMM) 399/2019\nUNION OF INDIA ..... Petitioner\nThrough: Ms. Pinky Anand, ASG\nversus\n"
             "M/S JAIN INDUSTRIES PVT. LTD. ..... Respondent\nThe Enforcement Directorate and Jain Industries Pvt. Ltd. disagree.")
    support = [(delhi.index("UNION"), delhi.index("UNION") + 14, "RESP"), (delhi.index("JAIN"), delhi.index("JAIN") + 4, "AUTH")]
    marked = {(delhi[s:e], l) for s, e, l in mark_parties(delhi, support)}
    assert ("Jain Industries Pvt. Ltd", "RESP") in marked and ("UNION OF INDIA", "APP") in marked
    nextline = "CIVIL APPEAL NOs. 5384-5385 OF 2014\nSEETHAKATHI TRUST MADRAS\n… Appellant\nVersus\nKRISHNAVENI\n…Respondent\n"
    assert parse_parties(nextline) == {"SEETHAKATHI TRUST MADRAS": "APP", "KRISHNAVENI": "RESP"}
    # a precedent span is never relabelled; 'only' drops APP/RESP that are not header parties
    p = sc.index("Shreya")
    body = "\n" + "x " * 1400 + "Kiransinh met Mahesh. "
    doc = sc + body
    m = doc.index("Mahesh")
    seed = [(doc.index("Anversinh"), doc.index("Anversinh") + 9, "APP"), (doc.index("State of Gujarat"), doc.index("State of Gujarat") + 5, "RESP"),
            (p, len(sc) - 1, "PREC"), (0, 8, "APP"), (m, m + 6, "APP")]
    out = mark_parties(doc, seed, only=True)
    assert (p, len(sc) - 1, "PREC") in out                     # precedents untouched
    assert (0, 8, "APP") in out                                # header spans are never dropped
    assert (m, m + 6, "APP") not in out                        # body APP that is not a party is dropped
    w = ("the complainant Jagdish (PW-5) along with his sons. The Police Patil of the village PW.3 Sandu Mohanrao Patil "
         "lodged a report. LW-17, P. Devdass, took photographs. Later Jagdish said so.")
    assert {w[s:e] for s, e, l in mark_witnesses(w, []) if l == "WIT"} == {"Jagdish", "Sandu Mohanrao Patil", "P. Devdass"}
    config = {"parties": ["on", "first"], "witness": ["on"]}
    a0 = kanoon.index("APPASAHEB")
    out = postprocess(kanoon, [(a0, a0 + 9, "RESP")], config)                   # wired into postprocess
    assert (a0, a0 + 9, "APP") in out and (a, a + 9, "APP") in out


def test_per_label_voting_beats_one_global_threshold():
    import random
    from phase2 import labelvote
    from phase2.vote import vote_all
    from phase2.post import strict_scores
    rng = random.Random(0)
    labels = ["APP", "DATE"]
    docs = [f"d{i}" for i in range(40)]
    gold = {d: [(j * 10, j * 10 + 5, labels[j % 2]) for j in range(20)] for d in docs}
    oof = {}
    for m in range(5):
        run = {}
        for d in docs:
            spans = []
            for s, e, l in gold[d]:
                if l == "APP" and rng.random() < 0.35:         # APP: each member finds a third -> union wins
                    spans.append((s, e, l))
                if l == "DATE":
                    spans.append((s, e, l))
                    if rng.random() < 0.6:                     # DATE: member-specific noise -> majority wins
                        spans.append((s + 6, s + 8, "DATE"))
                    if rng.random() < 0.6:
                        spans.append((s + 6 + m % 2, s + 9, "DATE"))
            run[d] = spans
        oof[f"m{m}"] = run
    runs = list(oof)
    config = labelvote.choose(oof, runs, gold, labels, docs)
    assert config["APP"][1] == 1 and config["DATE"][1] >= 3, config
    fold_docs = [docs[:20], docs[20:]]
    swap = labelvote.swap_score(oof, runs, gold, labels, fold_docs)
    best_global = max(strict_scores(gold, vote_all([oof[r] for r in runs], gold, k), labels)["macro_f1"] for k in range(1, 6))
    assert swap > best_global + 0.02, (swap, best_global)
    # deterministic overlap resolution: same input, same output, regardless of label order
    from phase2.post import remove_overlaps
    a, b = (0, 5, "APP"), (0, 5, "RESP")
    assert remove_overlaps([a, b], {a: 1, b: 1}) == remove_overlaps([b, a], {a: 1, b: 1}) == [a]


def test_judges_counsel_sides_and_caseno_split_on_real_patterns():
    from phase2.post import judge_names, counsel_side, split_casenos, postprocess
    cases = {
        "…RESPONDENTS J U D G M E N T INDU MALHOTRA, J. 1. The present": {"INDU MALHOTRA"},
        "CORAM : PRASANNA B. VARALE and ARUN D. UPADHYE, JJ. DATE : NOVEMBER 09, 2017.": {"PRASANNA B. VARALE", "ARUN D. UPADHYE"},
        "BENCH: BHAGWATI, NATWARLAL H. MUKHERJEA, B.K. BOSE, VIVIAN CITATION: 1954": {"BHAGWATI, NATWARLAL H.", "MUKHERJEA, B.K.", "BOSE, VIVIAN"},
        "is hereby allowed. (G.S. Ahluwalia) Judge MKB": {"G.S. Ahluwalia"},
        "HON'BLE MR. JUSTICE ASHOK BHUSHAN HON'BLE MR. JUSTICE B.R. GAVAI By Circulation": {"ASHOK BHUSHAN", "B.R. GAVAI"},
        "JUDGMENT J.P. Devadhar, J. 1. This petition": {"J.P. Devadhar"},
    }
    for text, want in cases.items():
        assert want <= judge_names(text), (text, judge_names(text))
    t = "For Appellant(s) Mr. Gagan Gupta,Adv. For Respondent(s) Mr. Ajay Bansal, AAG"
    spans = [(t.index("Gagan"), t.index("Gagan") + 11, "R.COUNSEL"), (t.index("Ajay"), t.index("Ajay") + 11, "A.COUNSEL")]
    assert [l for *_, l in counsel_side(t, spans)] == ["A.COUNSEL", "R.COUNSEL"]
    t = "We heard Mr. Akhil Sibal, learned counsel for the appellant and Mr. Vaibhav Gaggar, learned counsel for the first respondent"
    spans = [(t.index("Akhil"), t.index("Akhil") + 11, "R.COUNSEL"), (t.index("Vaibhav"), t.index("Vaibhav") + 14, "A.COUNSEL")]
    assert [l for *_, l in counsel_side(t, spans)] == ["A.COUNSEL", "R.COUNSEL"]
    t = "the C.M.P.No.15283 of 2019 AS.No.SR85894 of2019 entire amount"
    a, b = t.index("C.M.P"), t.index(" entire")
    assert [t[x:y] for x, y, _ in split_casenos(t, [(a, b, "CASENO")])] == ["C.M.P.No.15283 of 2019", "AS.No.SR85894 of2019"]
    t = "(MOUSUMI BHATTACHARYA, J.) said so. Later MOUSUMI BHATTACHARYA wrote."
    out = postprocess(t, [], {"judges": ["on"]})
    assert sum(1 for *_, l in out if l == "JUDGE") == 2


def test_merged_roles_resolve_back_and_train_end_to_end(tmp_path):
    import json, os
    from phase2.post import coarse_gold, coarse_labels, resolve_roles, strict_scores
    labels = load_labels(FX / "Phase_2_LNER/Train/labels.txt")
    assert "PARTY" in coarse_labels(labels) and "APP" not in coarse_labels(labels)
    texts = load_texts(FX / "Phase_2_LNER/Train/fold1-text.json")
    gold, _ = load_gold(FX / "Phase_2_LNER/Train/fold1-gold.json", texts, labels)
    gold = {d: [tuple(x) for x in g] for d, g in gold.items()}
    back = {d: resolve_roles(texts[d], sp) for d, sp in coarse_gold(gold).items()}
    per = strict_scores(gold, back, labels)["per_label"]
    for label in ("APP", "RESP", "A.COUNSEL", "R.COUNSEL"):
        assert per[label]["f1"] > 0.95, (label, per[label])
    # real-style cues without a header
    t = "Mr. Gaggar, learned counsel for the first respondent, and the appellant Ramesh were heard. The Directorate objected."
    sp = [(t.index("Gaggar"), t.index("Gaggar") + 6, "COUNSEL"), (t.index("Ramesh"), t.index("Ramesh") + 6, "PARTY"),
          (t.index("Directorate"), t.index("Directorate") + 11, "PARTY")]
    assert [l for *_, l in resolve_roles(t, sp)] == ["R.COUNSEL", "APP", "RESP"]
    # end to end: cv on merged roles, scored and written with the 12 labels
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{PHASE1}")
    out = tmp_path / "coarse"
    r = subprocess.run([sys.executable, "-m", "phase2.run", "cv", "--data", str(FX / "Phase_2_LNER"),
                        "--checkpoint", str(FX / "phase1_runs/dense_gpt/best.pt"), "--project-root", str(PHASE1),
                        "--output", str(out), "--method", "lorma", "--mode", "causal", "--window", "64", "--stride", "32",
                        "--coarse-roles", "--epochs", "2", "--eval-from", "1", "--batch-size", "8", "--final"],
                       env=env, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "resolved merged roles" in r.stdout
    summary = json.loads((out / "cv_summary.json").read_text())
    assert "APP" in summary["folds"][0]["per_label"] and "PARTY" not in summary["folds"][0]["per_label"]
    pred = json.loads((out / "submission/prediction.json").read_text())
    assert not any(l in ("PARTY", "COUNSEL") for v in pred.values() for *_, l in v)
    raw = json.loads((out / "submission/raw_spans.json").read_text())
    assert not any(l in ("PARTY", "COUNSEL") for v in raw.values() for *_, l in v)
    from phase2.vote import test_spans
    assert set(test_spans(out, labels)) == set(raw)


def test_header_layouts_from_real_training_documents():
    from phase2.post import parse_parties
    cases = {
        "CRIMINAL APPEAL NO._______OF 2021 (Arising out of SLP (Crl) No. 6965 OF 2019) Surendra Kumar & Anr. APPELLANT(S) "
        "VERSUS State of U.P. RESPONDENT(S) J U D G M E N T Hrishikesh Roy, J.": ("Surendra Kumar", "State of U.P"),
        "CRIMINAL APPEAL No. 822 OF 2012 RAJU @ DEVENDRA CHOUBEY …. APPELLANT VERSUS STATE OF CHHATISGARH …. RESPONDENT WITH":
            ("RAJU @ DEVENDRA CHOUBEY", "STATE OF CHHATISGARH"),
        "Case :- WRIT - A No. - 7175 of 2020 Petitioner :- Hriday Narayan Singh Respondent :- State Of U.P. And 4 Others "
        "Counsel for Petitioner :- Rajendra Rai Counsel for Respondent :- C.S.C.": ("Hriday Narayan Singh", "State Of U.P"),
        "R.F.A.No.708 OF 2006 BETWEEN: 1. Ramanna, S/o. late Channapillaiah, Aged about 63 years ...Appellants "
        "(By Sri.R. Narayana, Advocate) AND: Chaluvaiah, S/o. Doddaiah, Aged about 63 years": ("Ramanna", "Chaluvaiah"),
        "C.O. 3346 of 2018 The Kolkata Municipal Corporation Vs. Smt. Kajari Banerjee For the petitioner : Mr. Alok Kr. Ghosh":
            ("Kolkata Municipal Corporation", "Kajari Banerjee"),
        "A.P.O. NO. 3 OF 2017 With C.S. NO. 369 OF 2014 MERLIN PROJECTS LIMITED VERSUS PAWAN KUMAR AGARWAL With A.P.D.":
            ("MERLIN PROJECTS LIMITED", "PAWAN KUMAR AGARWAL"),
    }
    for text, (app, resp) in cases.items():
        parties = parse_parties(text)
        assert parties.get(app) == "APP" and parties.get(resp) == "RESP", (text[:60], parties)
        assert not any(w in n.lower() for n in parties for w in ("counsel", "appellant", "respondent")), parties


def test_greedy_selection_adds_a_pair_when_no_single_run_helps():
    import random
    from phase2.vote import select
    labels = ["APP"]
    rng = random.Random(1)
    docs = [f"d{i}" for i in range(30)]
    gold = {d: [(j * 10, j * 10 + 5, "APP") for j in range(10)] for d in docs}
    def noisy(shift):   # each member: 80% recall, own false positives
        return {d: [s for s in g if rng.random() < 0.8] + [(s[0] + 6, s[0] + 7 + shift, "APP") for s in g if rng.random() < 0.5]
                for d, g in gold.items()}
    oof = {"best": noisy(1), "b": noisy(2), "c": noisy(3)}
    chosen, k, score, _ = select(list(oof), oof, gold, labels, max_members=7, quiet=True)
    assert len(chosen) == 3 and k == 2, (chosen, k)
