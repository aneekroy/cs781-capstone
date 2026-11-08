"""CS781 Phase 6 (abstractive legal summarization) command line.

  stats     sizes, document / summary lengths, compression, length-model fit, tokens per word, chunks per document,
            where in the document the summary sentences come from, train copies of test documents
  baseline  lead and supervised-extractive references (CPU) + extractor.pkl for --strategy extract
  finetune  Phase-1 model + technique + long-document strategy (+ KD) -> model.pt, summary.json, submission.zip
  decode    re-generate a finished run with other decoding / length settings
  tune      one-factor-at-a-time decoding variants on held-out documents (+ test submission of the best)
  ensemble  decode with several finished runs at once (mixture of next-token distributions)
  score     score a prediction file against held-out references (ids of the train file)
  check     validate a prediction.json against the test set
"""
import argparse
import json
from pathlib import Path

import numpy as np


def cmd_stats(args):
    from .data import (LengthModel, chunk_targets, density_profile, load_raw, load_splits, prepare_doc,
                       split_sentences, test_overlap, words)
    full, test = load_raw(args.data, "train"), load_raw(args.data, "test")
    ov = test_overlap(full, test)
    print(f"train file: {len(full)} documents; {len(ov)} copy a test document "
          f"({sum(1 for k, _ in ov.values() if k == 'exact')} exact)")
    train, dev, test = load_splits(args.data, args.dev_size)
    rep = {}
    for name, docs in (("train", train), ("dev", dev), ("test", test)):
        dw = [words(d["document"]) for d in docs.values()]
        row = {"documents": len(docs), "doc_words_median": int(np.median(dw)), "doc_words_p90": int(np.percentile(dw, 90)),
               "doc_words_max": int(max(dw))}
        if name != "test":
            sw = [words(d["summary"]) for d in docs.values()]
            row.update({"summary_words_median": int(np.median(sw)), "summary_words_p90": int(np.percentile(sw, 90)),
                        "compression_median": round(float(np.median(np.array(sw) / np.maximum(dw, 1))), 3),
                        "summary_sentences_median": int(np.median([len(split_sentences(d["summary"]))
                                                                   for d in list(docs.values())[:500]]))})
        rep[name] = row
    lm = LengthModel.fit(train)
    pred = np.array([lm.predict(words(d["document"])) for d in dev.values()])
    gold = np.array([words(d["summary"]) for d in dev.values()])
    rep["length_model"] = {**{k: round(v, 3) for k, v in lm.state().items()},
                           "dev_median_abs_error_words": int(np.median(np.abs(pred - gold))),
                           "dev_within_25pct": round(float(np.mean(np.abs(pred - gold) <= 0.25 * gold)), 3)}
    if args.checkpoint:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(Path(args.checkpoint).parent / "tokenizer.json"))
        sample = dict(list(train.items())[:args.sample])
        dtoks = [len(tok.encode(d["document"]).ids) for d in sample.values()]
        stoks = [len(tok.encode(d["summary"]).ids) for d in sample.values()]
        swords = [words(d["summary"]) for d in sample.values()]
        prepared, targets, first = {}, {}, []
        empty = total = 0
        for d, doc in sample.items():
            prepared[d] = prepare_doc(tok, doc["document"], args.chunk)
            targets[d] = chunk_targets(prepared[d], doc["summary"])
            n = sum(words(s) for t in targets[d] for s in t)
            first.append(sum(words(s) for s in targets[d][0]) / max(1, n))
            empty += sum(1 for t in targets[d] if not t)
            total += len(targets[d])
        rep["tokens"] = {"sample": len(sample), "doc_tokens_median": int(np.median(dtoks)),
                         "doc_tokens_p90": int(np.percentile(dtoks, 90)),
                         "summary_tokens_median": int(np.median(stoks)),
                         "summary_tokens_p90": int(np.percentile(stoks, 90)),
                         "tokens_per_summary_word": round(sum(stoks) / max(1, sum(swords)), 3),
                         f"chunks_of_{args.chunk}_per_doc_median": float(np.median([len(p["chunks"]) for p in prepared.values()])),
                         "chunks_without_aligned_summary": round(empty / max(1, total), 3),
                         "summary_share_from_first_chunk": round(float(np.mean(first)), 3),
                         "summary_words_per_doc_token_by_position_decile": [round(x, 2) for x in density_profile(prepared, targets)]}
    print(json.dumps(rep, indent=1))


def cmd_score(args):
    from .data import load_raw
    from .evaluate import evaluate
    refs = {d: r for d, r in load_raw(args.data, "train").items()}
    pred = json.loads(Path(args.prediction).read_text())
    if isinstance(pred, list):
        pred = {str(p["id"]): p["summary"] for p in pred}
    ids = [d for d in pred if d in refs]
    print(json.dumps(evaluate({d: pred[d] for d in ids}, {d: refs[d]["summary"] for d in ids}, not args.no_bertscore,
                              docs={d: refs[d]["document"] for d in ids}), indent=1))


def cmd_check(args):
    from .data import load_raw, validate_prediction, words
    test = load_raw(args.data, "test")
    pred = validate_prediction(json.loads(Path(args.prediction).read_text()), test)
    w = [words(p["summary"]) for p in pred]
    print(f"OK: {len(pred)} summaries, {np.mean(w):.0f} words on average (min {min(w)}, max {max(w)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--data", required=True)
    s.add_argument("--dev-size", type=int, default=200)
    s.add_argument("--checkpoint", help="a Phase-1 best.pt: token counts and chunk alignment statistics")
    s.add_argument("--chunk", type=int, default=1024)
    s.add_argument("--sample", type=int, default=600)
    b = sub.add_parser("baseline")
    b.add_argument("--data", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--dev-size", type=int, default=200)
    b.add_argument("--max-train-docs", type=int, default=0)
    b.add_argument("--lead-only", action="store_true")
    b.add_argument("--no-bertscore", action="store_true")
    from .train import add_args, add_decode_args
    add_args(sub.add_parser("finetune"))
    for name in ("decode", "tune"):
        d = sub.add_parser(name)
        d.add_argument("--data", required=True)
        d.add_argument("--run", required=True)
        add_decode_args(d)
        d.add_argument("--attention", default="auto", choices=["auto", "fa2", "sdpa", "off"])
        if name == "decode":
            d.add_argument("--tag", required=True)
            d.add_argument("--dev-only", action="store_true")
            d.add_argument("--len-scale", type=float)
            d.add_argument("--budget", choices=["learned", "uniform"])
            d.add_argument("--dedup", type=float)
            d.add_argument("--min-ratio", type=float)
            d.add_argument("--max-ratio", type=float)
        else:
            d.add_argument("--variants", nargs="*")
            d.add_argument("--no-test", action="store_true")
    e = sub.add_parser("ensemble")
    e.add_argument("--data", required=True)
    e.add_argument("--runs", nargs="+", required=True, help="finished run directories (the first one's plan is used)")
    e.add_argument("--weights", type=float, nargs="*")
    e.add_argument("--output", required=True)
    e.add_argument("--dev-only", action="store_true")
    e.add_argument("--len-scale", type=float)
    e.add_argument("--budget", choices=["learned", "uniform"])
    e.add_argument("--dedup", type=float)
    e.add_argument("--min-ratio", type=float)
    e.add_argument("--max-ratio", type=float)
    e.add_argument("--attention", default="auto", choices=["auto", "fa2", "sdpa", "off"])
    add_decode_args(e)
    sc = sub.add_parser("score")
    sc.add_argument("--data", required=True)
    sc.add_argument("--prediction", required=True)
    sc.add_argument("--no-bertscore", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--data", required=True)
    c.add_argument("--prediction", required=True)
    args = parser.parse_args()
    if args.command == "stats":
        cmd_stats(args)
    elif args.command == "baseline":
        from .baseline import run
        run(args)
    elif args.command in ("finetune", "decode", "tune", "ensemble"):
        from phase3.flash import enable
        print(enable(args.attention), flush=True)
        from .model import install_masks
        install_masks()
        from . import train
        getattr(train, args.command)(args)
    elif args.command == "score":
        cmd_score(args)
    else:
        cmd_check(args)


if __name__ == "__main__":
    main()
