"""Span-voting ensemble chosen on cross-validation (no test submissions spent on choosing).

Every finished run keeps its out-of-fold predictions (oof_predictions.pt: raw spans for every
fold-1 and fold-2 document, each predicted by the model that did not train on it). Voting on
those spans gives an honest CV score for any set of members:

  * a span (start, end, label) is kept when at least k members predict exactly it;
    overlaps are resolved in favour of more votes, then the longer span;
  * members are picked greedily: start from the best single run, add whichever run raises the
    pooled out-of-fold macro-F1 most (best k for that set), stop when nothing helps;
  * post-processing is then re-chosen on the voted out-of-fold spans, exactly as `cv` does.

At test time each member's raw spans are decoded from its saved token log-probs, voted with the
same k and post-processed with the same configuration. Members may use different tokenizers or
segmentations, because they vote on character spans, not tokens.

    python -m phase2.vote --data Phase_2_LNER --runs runsP2/X1_denseorig_lorma runsP2/A3_gqa_clm_lora ... \
        --output runsP2/V1_vote
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch

from .data import load_labels, load_texts, tag_names, tags_to_spans, write_json
from .model import CRF
from .run import decode
from .post import BoundaryRules, build_gazetteer, postprocess, remove_overlaps, strict_scores


def vote(span_lists, k):
    counts = Counter(tuple(s) for spans in span_lists for s in set(map(tuple, spans)))
    kept = [s for s, c in counts.items() if c >= k]
    return remove_overlaps(kept, {s: counts[s] for s in kept})


def vote_with_extras(member_lists, extra_lists, k, k_all):
    """Members decide as on CV (>= k of n); a span the members alone would drop is still kept when
    members and extra voters together give it >= k_all votes (the same share of a larger panel)."""
    members = Counter(tuple(s) for spans in member_lists for s in set(map(tuple, spans)))
    total = members + Counter(tuple(s) for spans in extra_lists for s in set(map(tuple, spans)))
    kept = [s for s, c in total.items() if members[s] >= k or c >= k_all]
    return remove_overlaps(kept, {s: total[s] for s in kept})


def vote_all(members, docs, k):
    return {d: vote([m.get(d, []) for m in members], k) for d in docs}


def macro(gold, pred, labels):
    return strict_scores(gold, pred, labels)["macro_f1"]


def best_k(members, gold, labels):
    """(score, k) of the best vote threshold for this member set."""
    return max((macro(gold, vote_all(members, gold, k), labels), k) for k in range(1, len(members) + 1))


FILTER_LINE = "skipped because they contain an NER document"


def classify(run, allow_unfiltered_kd=False):
    """('member' | 'extra' | 'skip', note).

    member: out-of-fold spans are honest -> takes part in selection, k, rules and the reranker.
    extra:  a KD run. Its teacher's final model saw both folds, so its out-of-fold scores are
            leaked (E4f: CV 0.80, test 0.57) and must not steer selection; its test predictions
            are honest, so it joins the test-time vote.
    skip:   no out-of-fold predictions, or a KD run trained before the NER-overlap filter (its
            teacher-labelled text contained test documents) unless allow_unfiltered_kd."""
    args = json.loads((run / "args.json").read_text()) if (run / "args.json").exists() else {}
    if not (run / "oof_predictions.pt").exists() and not args.get("teacher"):
        return "skip", "no oof_predictions.pt (CV not finished)"
    if not ((run / "submission" / "token_logprobs.pt").exists() or (run / "submission" / "prediction.json").exists()):
        return "skip", "no test predictions (run without --final)"
    if args.get("teacher"):
        log = run.parent / f"{run.name}.log"
        filtered = log.exists() and FILTER_LINE in log.read_text(errors="ignore")
        if not filtered and not allow_unfiltered_kd:
            return "skip", ("KD run from before the NER-overlap filter: its teacher-labelled text included test "
                            "documents (--allow-unfiltered-kd to use it anyway)")
        return "extra", "KD run: leaked out-of-fold scores, so it joins the test-time vote only"
    note = []
    summary = json.loads((run / "cv_summary.json").read_text()) if (run / "cv_summary.json").exists() else {}
    if isinstance(summary.get("o_bias"), (int, float)) and summary["o_bias"]:
        note.append(f"legacy 'O' penalty {summary['o_bias']} reproduced at test time")
    if not (run / "submission" / "token_logprobs.pt").exists():
        note.append("no token scores (span head): uses its post-processed test predictions")
    return "member", "; ".join(note)


def eligible(run):
    """Backwards-compatible: (usable as a selection member, reason)."""
    kind, why = classify(run)
    return kind == "member", why or kind


def select(runs, oof, gold, labels, max_members, quiet=False):
    singles = sorted(((macro(gold, oof[r], labels), r) for r in runs), reverse=True)
    for score, r in singles if not quiet else []:
        print(f"  single {r:34s} OOF raw {score:.4f}")
    chosen, (score, k) = [singles[0][1]], (singles[0][0], 1)
    trace = [(chosen[0], round(score, 4), k)]
    while len(chosen) < max_members:
        trials = []
        for r in runs:
            if r in chosen:
                continue
            s, kk = best_k([oof[m] for m in chosen + [r]], gold, labels)
            trials.append((s, kk, r))
        if not trials:
            break
        s, kk, r = max(trials)
        if s <= score + 1e-4:
            # look-ahead: a majority vote needs >= 3 members, so a pair can help where no single run does
            if len(chosen) + 2 > max_members:
                break
            ranked = [r_ for _, _, r_ in sorted(trials, reverse=True)[:8]]
            pairs = []
            for i, a in enumerate(ranked):
                for b in ranked[i + 1:]:
                    ps, pk = best_k([oof[m] for m in chosen + [a, b]], gold, labels)
                    pairs.append((ps, pk, a, b))
            if not pairs:
                break
            ps, pk, a, b = max(pairs)
            if ps <= score + 1e-4:
                break
            chosen += [a, b]
            score, k = ps, pk
            trace.append((f"{a}+{b}", round(ps, 4), pk))
            if not quiet:
                print(f"  + {a} + {b} -> OOF raw {ps:.4f} (keep spans with >= {pk} of {len(chosen)} votes)")
            continue
        chosen.append(r)
        score, k = s, kk
        trace.append((r, round(s, 4), kk))
        if not quiet:
            print(f"  + {r:34s} -> OOF raw {s:.4f} (keep spans with >= {kk} of {len(chosen)} votes)")
    return chosen, k, score, trace


def run_crf(run, labels):
    """The run's own learned CRF (transitions) from model.pt, so test spans are decoded exactly
    like its out-of-fold spans were; constraint-only CRF if model.pt is missing."""
    names = tag_names(labels)
    crf = CRF(names)
    path = run / "model.pt"
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)["state"]
        crf_state = {k[4:]: v for k, v in state.items() if k.startswith("crf.")}
        del state
        if crf_state:
            crf.load_state_dict(crf_state)
            return crf
    return CRF(names, learn=False)


def test_spans(run, labels, end_inclusive=False):
    names = tag_names(labels)
    run_args = json.loads((run / "args.json").read_text()) if (run / "args.json").exists() else {}
    if run_args.get("coarse_roles") and (run / "submission" / "raw_spans.json").exists():   # merged-role model
        payload = json.loads((run / "submission" / "raw_spans.json").read_text())
        return {d: [tuple(x) for x in spans] for d, spans in payload.items()}
    if not (run / "submission" / "token_logprobs.pt").exists():      # span head: post-processed predictions
        payload = json.loads((run / "submission" / "prediction.json").read_text())
        return {d: [(s, e + 1 if end_inclusive else e, l) for s, e, l in spans] for d, spans in payload.items()}
    part = torch.load(run / "submission" / "token_logprobs.pt", map_location="cpu", weights_only=False)
    if part["labels"] != labels:
        raise ValueError(f"{run.name}: different label set")
    crf = run_crf(run, labels)
    bias = part.get("o_bias", 0.0)
    out = {}
    with torch.no_grad():
        for doc, lp in part["logprobs"].items():
            scores = lp.float()
            if isinstance(bias, (int, float)) and bias and scores.size(0):   # legacy 'O' penalty, as in its CV
                scores = scores.clone()
                scores[:, 0] -= bias
                tags = crf.viterbi(scores)
            else:
                tags = decode(crf.viterbi, names, scores, bias)
            out[doc] = tags_to_spans(part["offsets"][doc], tags, labels)
    return out


def _write_errors(path, texts, gold, pred, labels, n=25, seed=0):
    import random
    from .errors import show
    rng = random.Random(seed)
    lines = []
    for label in labels:
        missed, spurious, confused = [], [], []
        for d, text in texts.items():
            g, p = set(map(tuple, gold.get(d, []))), set(map(tuple, pred.get(d, [])))
            pb, gb = {(a, b): l for a, b, l in p}, {(a, b): l for a, b, l in g}
            for a, b, l in g - p:
                if l == label:
                    (confused if (a, b) in pb else missed).append(
                        f"[{d}] gold {l}" + (f", predicted {pb[(a, b)]}" if (a, b) in pb else "") + ": " + show(text, a, b))
            for a, b, l in p - g:
                if l == label and (a, b) not in gb:
                    spurious.append(f"[{d}] predicted {l}: " + show(text, a, b))
        for kind, rows in (("MISSED", missed), ("SPURIOUS", spurious), ("CONFUSED", confused)):
            lines.append(f"\n=== {label} {kind}: {len(rows)} in total, {min(n, len(rows))} shown ===")
            lines += rng.sample(rows, min(n, len(rows)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    from .run import choose_postprocessing, load_split, write_submission
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="run directories (candidates)")
    p.add_argument("--output", required=True)
    p.add_argument("--max-members", type=int, default=7)
    p.add_argument("--fixed", action="store_true", help="use all given runs, only choose k (no greedy selection)")
    p.add_argument("--no-rerank", action="store_true", help="skip the second-stage span reranker")
    p.add_argument("--no-extras", action="store_true", help="leave KD runs out of the test-time vote")
    p.add_argument("--no-per-label", action="store_true", help="skip per-label member sets and thresholds")
    p.add_argument("--allow-unfiltered-kd", action="store_true",
                   help="also use KD runs trained before the NER-overlap filter (E4, A4: test text in their KD data)")
    args = p.parse_args()

    labels = load_labels(Path(args.data) / "Train" / "labels.txt")
    folds = {f: load_split(args.data, f, labels) for f in (1, 2)}
    end_inclusive = folds[1][2]["end_inclusive"]
    gold = {**folds[1][1], **folds[2][1]}

    runs, oof, extras = [], {}, []
    for path in map(Path, args.runs):
        kind, why = classify(path, args.allow_unfiltered_kd)
        if kind == "skip":
            print(f"  skip {path.name}: {why}")
            continue
        if kind == "extra":
            if args.no_extras:
                print(f"  skip {path.name}: --no-extras")
            else:
                extras.append(path.name)
                print(f"  extra voter {path.name}: {why}")
            continue
        if why:
            print(f"  member {path.name}: {why}")
        preds = torch.load(path / "oof_predictions.pt", map_location="cpu", weights_only=False)["oof"]
        if set(preds) != set(gold):
            print(f"  skip {path.name}: out-of-fold predictions do not cover both folds")
            continue
        runs.append(path.name)
        oof[path.name] = {d: [tuple(s) for s in spans] for d, spans in preds.items()}
    if not runs:
        raise SystemExit("no eligible runs")
    root = {Path(r).name: Path(r) for r in args.runs}

    if args.fixed:
        chosen = runs
        score, k = best_k([oof[r] for r in runs], gold, labels)
        trace = [("all", round(score, 4), k)]
        print(f"  fixed set of {len(runs)} -> OOF raw {score:.4f} (k={k})")
    else:
        chosen, k, score, trace = select(runs, oof, gold, labels, args.max_members)

    # per-label member sets and thresholds, if they beat the global vote under document-fold swap
    per_label, per_label_report = None, {}
    if not args.no_per_label:
        from . import labelvote
        fold_docs = [sorted(folds[1][1]), sorted(folds[2][1])]
        swap_pl = labelvote.swap_score(oof, runs, gold, labels, fold_docs, args.max_members)
        swap_global = []
        for a, b in ((0, 1), (1, 0)):
            sub = {d: gold[d] for d in fold_docs[a]}
            mem, kk_, _, _ = select(runs, {r: {d: oof[r][d] for d in fold_docs[a]} for r in runs}, sub, labels,
                                    args.max_members, quiet=True)
            swap_global.append(vote_all([oof[r] for r in mem], {d: gold[d] for d in fold_docs[b]}, kk_))
        merged = {**swap_global[0], **swap_global[1]}
        swap_gl = strict_scores(gold, merged, labels)["macro_f1"]
        per_label_report = {"swap_per_label": round(swap_pl, 4), "swap_global": round(swap_gl, 4)}
        print(f"  per-label voting: {swap_pl:.4f} vs global vote {swap_gl:.4f} (choose on one fold, score the other)")
        if swap_pl > swap_gl + 0.003:
            per_label = labelvote.choose(oof, runs, gold, labels, sorted(gold), args.max_members)
            for label, (mem, kk_, f1) in per_label.items():
                print(f"    {label:10s} k={kk_} of {len(mem)}  OOF F1 {f1:.3f}  {mem}")
            chosen = sorted({m for mem, _, _ in per_label.values() for m in mem})
            per_label_report["config"] = {l: [m, kk_] for l, (m, kk_, _) in per_label.items()}
        else:
            print("  per-label voting not used")

    def voted_oof(docs):
        if per_label:
            return labelvote.apply(per_label, {r: oof[r] for r in chosen}, list(docs))
        return vote_all([oof[r] for r in chosen], docs, k)

    # post-processing on the voted out-of-fold spans, rules fitted on the other fold as in `cv`
    fold_runs = []
    for train_fold, val_fold in ((1, 2), (2, 1)):
        texts, g, _ = folds[val_fold]
        tr_texts, tr_gold, _ = folds[train_fold]
        fold_runs.append({"raw": voted_oof(g), "texts": texts, "gold": g,
                          "boundary": BoundaryRules().fit(tr_texts, tr_gold),
                          "gazetteer": build_gazetteer(tr_texts, tr_gold)})
    config, post_trace = choose_postprocessing(fold_runs, labels)
    post_score = post_trace[-1][1]

    # diagnostics on the final out-of-fold system (voted + rules): per-label P/R/F1 and real errors
    oof_post, oof_gold, oof_text = {}, {}, {}
    for run_ in fold_runs:
        for d, raw in run_["raw"].items():
            oof_post[d] = postprocess(run_["texts"][d], raw, config, run_["boundary"], run_["gazetteer"])
            oof_gold[d], oof_text[d] = run_["gold"][d], run_["texts"][d]
    final_per_label = strict_scores(oof_gold, oof_post, labels)["per_label"]
    print("  final system, out-of-fold:   label       P      R      F1   gold  pred")
    for label in labels:
        r = final_per_label[label]
        print(f"  {'':29s}{label:10s} {r['p']:.3f}  {r['r']:.3f}  {r['f1']:.3f} {r['support']:5d} {r['pred']:5d}")
    _write_errors(Path(args.output) / "errors_final.txt", oof_text, oof_gold, oof_post,
                  ["APP", "RESP", "WIT", "A.COUNSEL", "R.COUNSEL", "JUDGE"])
    print(f"chosen {chosen} k={k} | OOF raw {score:.4f} -> post {post_score:.4f}")

    # second stage: rerank the post-processed candidates (from k-of-n voting, and from the union)
    rerank_choice, rerank_scores = None, {}
    if not args.no_rerank and not per_label:
        from . import rerank
        candidates_by_k = {}
        for kk in sorted({k, 1}):
            fold_data = []
            for run_, (train_fold, val_fold) in zip(fold_runs, ((1, 2), (2, 1))):
                voted = vote_all([oof[r] for r in chosen], run_["gold"], kk)
                cands, rule_only = {}, {}
                for d in run_["gold"]:
                    cands[d] = postprocess(run_["texts"][d], voted[d], config, run_["boundary"], run_["gazetteer"])
                    rule_only[d] = set(cands[d]) - set(voted[d])
                fold_data.append((run_["texts"], cands, [{d: oof[r][d] for d in run_["gold"]} for r in chosen],
                                  rule_only, run_["gold"]))
            fold_data = fold_data[::-1]                          # fold_runs[0] holds fold-2 documents
            rr_score, _ = rerank.cross_validate(fold_data, labels)
            rerank_scores[kk] = round(rr_score, 4)
            candidates_by_k[kk] = fold_data
            print(f"  reranker on {'union' if kk == 1 else f'{kk}-of-{len(chosen)} votes'} + rules: "
                  f"OOF {rr_score:.4f} (document-fold swap) vs {post_score:.4f} without")
        best_kk = max(rerank_scores, key=rerank_scores.get)
        if rerank_scores[best_kk] > post_score + 0.003:
            rerank_choice = best_kk
            reranker = rerank.fit_all(candidates_by_k[best_kk], labels)
            print(f"  using the reranker (candidates from k={best_kk})")
        else:
            print("  reranker does not beat plain voting on CV: not used")

    test = load_texts(Path(args.data) / "Test" / "fold3-text.json")
    member_spans = [test_spans(root[r], labels, end_inclusive) for r in chosen]
    extra_spans = [test_spans(root[r], labels, end_inclusive) for r in extras]
    texts = {**folds[1][0], **folds[2][0]}
    boundary, gazetteer = BoundaryRules().fit(texts, gold), build_gazetteer(texts, gold)
    kk = rerank_choice or k
    # extras vote with the same SHARE of votes that CV chose for the members (k of n -> k/n of n+m)
    kk_all = max(1, math.ceil(kk / len(chosen) * (len(chosen) + len(extra_spans)) - 1e-9))
    if extra_spans:
        print(f"test-time vote: {len(chosen)} members + {len(extra_spans)} KD extra voters, keep spans with >= "
              f"{kk} of {len(chosen)} member votes, or >= {kk_all} of all {len(chosen) + len(extra_spans)} votes")
    if per_label:
        voted = labelvote.apply(per_label, dict(zip(chosen, member_spans)), list(test), extra_spans)
    else:
        voted = {d: vote_with_extras([m.get(d, []) for m in member_spans], [m.get(d, []) for m in extra_spans], kk, kk_all)
                 for d in test}
    preds = {d: postprocess(test[d], voted[d], config, boundary, gazetteer) for d in test}
    if rerank_choice:
        X, _, keys = rerank.build(test, preds, member_spans, {d: set(preds[d]) - set(voted[d]) for d in test}, labels)
        reranked = rerank.decide(reranker, X, keys, labels)
        preds = {d: reranked.get(d, []) for d in test}
    out = Path(args.output)
    write_submission(preds, test, out, end_inclusive, labels)
    write_json(out / "vote_summary.json", {"members": chosen, "k": k, "oof_macro_f1_raw": score,
                                           "oof_macro_f1_post": post_trace[-1][1], "selection_trace": trace,
                                           "postprocess": config, "postprocess_trace": post_trace,
                                           "rerank_oof": rerank_scores, "rerank_used_k": rerank_choice,
                                           "extra_voters": extras, "test_vote_threshold": kk_all,
                                           "per_label": per_label_report,
                                           "oof_final_per_label": {l: {k: round(v, 4) if isinstance(v, float) else v
                                                                       for k, v in r.items()} for l, r in final_per_label.items()},
                                           "candidates": runs})
    print("Wrote", out / "submission.zip")


if __name__ == "__main__":
    main()
