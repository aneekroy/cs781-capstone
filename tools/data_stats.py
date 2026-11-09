"""Corpus statistics for Section 3.2 of the report, from the official data only.

    python tools/data_stats.py --cache $HOME/cs781_cache --out data

Reads $CACHE/raw/Phase_N_* through each phase's own loader (so de-duplication and test-copy removal
match the experiments) and writes:
  data/data_stats.json        every number below
  data/*.pdf                  vector figures
  data/appendix_data.tex      LaTeX snippet with the figures and tables (the report \\input s it)

What it measures (the template's checklist for Section 3.2):
  Phase 1  words per document per source (sampled), vocabulary shift SC vs HC (Jensen-Shannon divergence,
           HC tokens unseen in SC, most over-represented words each way)
  Phase 2  span length per label, nested / crossing / adjacent spans, strings labelled in one place and
           unlabelled elsewhere (the boilerplate that causes false positives)
  Phase 3  role distribution in CL vs IT, sentences per document, words per sentence
  Phase 4  class balance, words per document per class
  Phase 5  labels per document (histogram), statute frequency (long tail)
  Phase 6  document vs summary length, compression, n-gram overlap with the source (extractiveness)
  Phase 7  query vs candidate length, citations per query, years of cited cases relative to the query
A phase whose data is missing is skipped with a message.

Phase 1 reads only the first --p1-docs documents of each source (SC, HC) with a streaming parser. If a
file is not under $CACHE/raw, those documents are streamed straight from the gated dataset over HTTP
(about 0.5 GB per source), so the 12 GB of training files never have to fit on disk. Streaming uses your
own token: $HF_TOKEN, else $CACHE/huggingface/token (never the machine-wide ~/.cache/huggingface one).
--p1-source local|stream|auto chooses (default auto: local file if present, else stream).
"""
import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORD = re.compile(r"[a-z]+")


def q(values, ps=(10, 50, 90)):
    v = np.asarray(values, dtype=float)
    return {f"p{p}": round(float(np.percentile(v, p)), 1) for p in ps} if len(v) else {}


def tex_escape(s):
    return (str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")
            .replace("#", "\\#").replace("$", "\\$").replace("{", "\\{").replace("}", "\\}"))


# --------------------------------------------------------------------------- Phase 1
REPO = "Exploration-Lab/CS-781-Capstone"
P1_FILES = {"SC": "Phase_1_Pretraining/Train/train_supreme_court.json",
            "HC": "Phase_1_Pretraining/Train/train_high_courts.json"}


def stream_texts(handle, limit):
    """First `limit` non-empty texts of a top-level JSON list, read incrementally from a binary handle."""
    import ijson
    for i, rec in enumerate(ijson.items(handle, "item")):
        if i >= limit:
            break
        text = rec.get("text") if isinstance(rec, dict) else None
        if isinstance(text, str) and text.strip():
            yield text


def own_token(token_file):
    """$HF_TOKEN, else your own token file, else a hidden prompt. Self-contained, so this script also runs
    as a single file outside the repository. The token is never printed."""
    import getpass
    import os
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token, "$HF_TOKEN"
    token_file = Path(token_file)
    if token_file.is_file() and token_file.read_text().strip():
        return token_file.read_text().strip(), str(token_file)
    if not sys.stdin.isatty():
        raise RuntimeError(f"Phase 1 needs a Hugging Face token to stream: export HF_TOKEN=hf_... or save it to {token_file}")
    token = getpass.getpass("Hugging Face READ token for the course dataset (hidden): ").strip()
    if not token:
        raise RuntimeError("No token entered.")
    return token, "prompt"


class HubStream:
    """Opens Phase-1 files on the Hub without downloading them; resolves the token once, on first use."""

    def __init__(self, cache):
        self.cache, self.fs = Path(cache), None

    def open(self, rel):
        if self.fs is None:
            import os
            os.environ["HF_HOME"] = str(self.cache / "huggingface")     # never the machine-wide token
            from huggingface_hub import HfFileSystem
            token, source = own_token(self.cache / "huggingface" / "token")
            print(f"phase 1: streaming from the Hub with the token from {source}")
            self.fs = HfFileSystem(token=token)
        return self.fs.open(f"datasets/{REPO}/{rel}", "rb", block_size=16 << 20)


def read_source(raw, rel, limit, mode, hub):
    """(texts, where) for one Phase-1 source: the local file if allowed and readable, else the Hub stream."""
    local = raw / rel
    if mode != "stream" and local.is_file():
        try:
            with open(local, "rb") as handle:
                return list(stream_texts(handle, limit)), str(local)
        except Exception as error:                      # e.g. a file truncated by a failed download
            if mode == "local":
                raise
            print(f"phase 1: {local} unreadable ({type(error).__name__}), streaming instead")
    if mode == "local":
        return None, None
    with hub.open(rel) as handle:
        return list(stream_texts(handle, limit)), f"hf://datasets/{REPO}/{rel}"


def phase1(raw, limit, mode="auto", hub=None):
    hub = hub or HubStream(raw.parent)
    lengths, counts, where = {}, {}, {}
    for name, rel in P1_FILES.items():
        texts, origin = read_source(raw, rel, limit, mode, hub)
        if texts is None:
            return None
        c, lens = Counter(), []
        for text in texts:
            words = WORD.findall(text.lower())
            lens.append(len(words))
            c.update(words)
        lengths[name], counts[name], where[name] = lens, c, origin
        print(f"phase 1: {name} {len(lens)} documents from {origin}")
    vocab = set(counts["SC"]) | set(counts["HC"])
    tot = {k: sum(c.values()) for k, c in counts.items()}
    p = np.array([counts["SC"][w] / tot["SC"] for w in vocab])
    r = np.array([counts["HC"][w] / tot["HC"] for w in vocab])
    m = (p + r) / 2
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0])))
    js = (kl(p, m) + kl(r, m)) / 2
    unseen = sum(n for w, n in counts["HC"].items() if w not in counts["SC"]) / tot["HC"]

    def log_odds(a, b, k=12, min_count=50):
        scores = {w: math.log((counts[a][w] + 1) / tot[a]) - math.log((counts[b][w] + 1) / tot[b])
                  for w in counts[a] if counts[a][w] >= min_count}
        return [w for w, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]]
    return {"docs_sampled": {k: len(v) for k, v in lengths.items()}, "read_from": where,
            "words_per_doc": {k: q(v) for k, v in lengths.items()},
            "js_divergence_bits": round(js, 4), "hc_tokens_unseen_in_sc": round(unseen, 4),
            "types": {k: len(c) for k, c in counts.items()},
            "over_represented_hc": log_odds("HC", "SC"), "over_represented_sc": log_odds("SC", "HC"),
            "_lengths": lengths}


# --------------------------------------------------------------------------- Phase 2
def phase2(raw):
    from phase2.data import load_gold, load_labels, load_texts
    d = raw / "Phase_2_LNER"
    if not (d / "Train" / "labels.txt").exists():
        return None
    labels = load_labels(d / "Train" / "labels.txt")
    texts, gold = {}, {}
    for fold in (1, 2):
        t = load_texts(d / "Train" / f"fold{fold}-text.json")
        g, _ = load_gold(d / "Train" / f"fold{fold}-gold.json", t, labels)
        texts.update(t)
        gold.update({k: [tuple(s) for s in v] for k, v in g.items()})
    span_words = {l: [] for l in labels}
    nested = crossing = adjacent = 0
    surfaces = Counter()
    for doc, spans in gold.items():
        spans = sorted(set(spans))
        text = texts[doc]
        for s, e, l in spans:
            span_words[l].append(len(text[s:e].split()))
            surface = text[s:e].strip()
            if len(surface) >= 4:
                surfaces[(surface, l)] += 1
        for i, (s1, e1, _) in enumerate(spans):
            for s2, e2, _ in spans[i + 1:]:
                if s2 >= e1:
                    if text[e1:s2].strip() == "":
                        adjacent += 1
                    break
                if e2 <= e1:
                    nested += 1
                else:
                    crossing += 1
    # strings that are gold somewhere but also appear unlabelled: candidate false positives
    covered = {doc: [(s, e) for s, e, _ in spans] for doc, spans in gold.items()}
    rows = []
    for (surface, label), n_lab in surfaces.most_common(400):
        pat = re.compile(r"(?<!\w)" + re.escape(surface) + r"(?!\w)")
        unl = 0
        for doc, text in texts.items():
            for m in pat.finditer(text):
                if not any(s <= m.start() < e for s, e in covered[doc]):
                    unl += 1
        if unl:
            rows.append((surface[:40], label, n_lab, unl))
    rows.sort(key=lambda r: -r[3])
    return {"documents": len(gold), "mentions": sum(len(v) for v in gold.values()),
            "span_words": {l: q(v) for l, v in span_words.items() if v},
            "span_words_mean": {l: round(float(np.mean(v)), 2) for l, v in span_words.items() if v},
            "nested": nested, "crossing": crossing, "adjacent": adjacent,
            "labelled_and_unlabelled": rows[:12], "_span_words": span_words}


# --------------------------------------------------------------------------- Phase 3
def phase3(raw):
    from phase3.data import load_split, load_vocab
    d = raw / "Phase_3_RR"
    if not (d / "Train" / "label_vocab.json").exists():
        return None
    labels = load_vocab(d / "Train" / "label_vocab.json")
    docs = {**load_split(d, "train", labels), **load_split(d, "dev", labels)}
    out = {"labels": labels, "by_domain": {}}
    for dom in ("CL", "IT"):
        ds = [x for x in docs.values() if x["domain"] == dom]
        roles = Counter(l for x in ds for l in x["labels"])
        out["by_domain"][dom] = {"documents": len(ds), "sentences": sum(len(x["labels"]) for x in ds),
                                 "sentences_per_doc": q([len(x["labels"]) for x in ds]),
                                 "words_per_sentence": q([len(s.split()) for x in ds for s in x["sentences"]]),
                                 "roles": {l: roles.get(l, 0) for l in labels}}
    return out


# --------------------------------------------------------------------------- Phases 4 and 5
def phase4(raw):
    from phase4.data import load_split
    d = raw / "Phase_4_CJPE"
    if not d.exists():
        return None
    out = {}
    for split in ("train", "dev"):
        docs = load_split(d, split, verbose=False)
        by = {0: [], 1: []}
        for x in docs.values():
            by[int(x["label"])].append(len(x["text"].split()))
        out[split] = {"documents": len(docs), "accepted_rate": round(len(by[1]) / max(1, len(docs)), 3),
                      "words_rejected": q(by[0]), "words_accepted": q(by[1])}
    return out


def phase5(raw):
    from phase5.data import load_split
    d = raw / "Phase_5_LSI"
    if not d.exists():
        cands = sorted(raw.glob("Phase_5*"))
        if not cands:
            return None
        d = cands[0]
    docs = load_split(d, "train", verbose=False)
    per_doc = [len(x["labels"]) for x in docs.values()]
    freq = Counter(l for x in docs.values() for l in x["labels"])
    ranked = sorted(freq.values(), reverse=True)
    return {"documents": len(docs), "labels_per_doc": q(per_doc), "labels_per_doc_hist": dict(sorted(Counter(per_doc).items())),
            "statutes": len(freq), "statute_frequency": q(ranked, (0, 25, 50, 75, 100)),
            "top10_share": round(sum(ranked[:10]) / sum(ranked), 3), "_ranked": ranked, "_per_doc": per_doc}


# --------------------------------------------------------------------------- Phase 6
def ngrams(words, n):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def phase6(raw):
    from phase6.data import load_raw
    d = raw / "Phase_6_Summ"
    if not d.exists():
        cands = sorted(raw.glob("Phase_6*"))
        if not cands:
            return None
        d = cands[0]
    docs = load_raw(d, "train")
    dl, sl, comp, novel = [], [], [], {n: [] for n in (1, 2, 3, 4)}
    for r in docs.values():
        dw, sw = WORD.findall(r["document"].lower()), WORD.findall(r["summary"].lower())
        if not dw or not sw:
            continue
        dl.append(len(dw)); sl.append(len(sw)); comp.append(len(sw) / len(dw))
        for n in novel:
            s = ngrams(sw, n)
            if s:
                novel[n].append(1 - len(s & ngrams(dw, n)) / len(s))
    return {"documents": len(dl), "document_words": q(dl), "summary_words": q(sl), "compression": q(comp),
            "novel_ngram_rate": {f"{n}-gram": round(float(np.mean(v)), 3) for n, v in novel.items()},
            "_dl": dl, "_sl": sl}


# --------------------------------------------------------------------------- Phase 7
def phase7(raw):
    from phase7.data import load_split, max_year
    d = raw / "Phase_7_PCR"
    if not d.exists():
        cands = sorted(raw.glob("Phase_7*"))
        if not cands:
            return None
        d = cands[0]
    queries, pool = load_split(d, "train")
    qy = {k: max_year(v["text"]) for k, v in queries.items()}
    cy = {k: max_year(t) for k, t in pool.items()}
    gaps = [qy[k] - cy[c] for k, v in queries.items() for c in (v["relevant"] or [])
            if c in cy and qy[k] and cy[c]]
    return {"queries": len(queries), "pool": len(pool),
            "query_words": q([len(v["text"].split()) for v in queries.values()]),
            "candidate_words": q([len(t.split()) for t in pool.values()]),
            "relevant_per_query": q([len(v["relevant"] or []) for v in queries.values()]),
            "query_year": q([y for y in qy.values() if y]), "candidate_year": q([y for y in cy.values() if y]),
            "cited_years_before_query": q(gaps),
            "cited_later_than_query": round(sum(g < 0 for g in gaps) / max(1, len(gaps)), 3),
            "_qw": [len(v["text"].split()) for v in queries.values()], "_cw": [len(t.split()) for t in pool.values()],
            "_gaps": gaps}


# --------------------------------------------------------------------------- figures and LaTeX
def figures(stats, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7})
    made = []

    def save(fig, name, caption):
        fig.tight_layout()
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        made.append((name, caption))

    p1, p6, p7 = stats.get("phase1"), stats.get("phase6"), stats.get("phase7")
    panels = [x for x in (p1, p6, p7) if x]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(2.9 * len(panels), 2.3))
        axes = np.atleast_1d(axes)
        bins = np.logspace(1, 5.5, 40)
        k = 0
        if p1:
            for name, lens in p1["_lengths"].items():
                axes[k].hist(lens, bins=bins, alpha=0.6, label=name)
            axes[k].set(xscale="log", xlabel="words per document", ylabel="documents", title="Phase 1: SC vs HC")
            axes[k].legend(); k += 1
        if p6:
            axes[k].hist(p6["_dl"], bins=bins, alpha=0.6, label="document")
            axes[k].hist(p6["_sl"], bins=bins, alpha=0.6, label="summary")
            axes[k].set(xscale="log", xlabel="words", title="Phase 6: documents vs summaries")
            axes[k].legend(); k += 1
        if p7:
            axes[k].hist(p7["_qw"], bins=bins, alpha=0.6, label="queries")
            axes[k].hist(p7["_cw"], bins=bins, alpha=0.6, label="candidates")
            axes[k].set(xscale="log", xlabel="words", title="Phase 7: queries vs candidates")
            axes[k].legend()
        save(fig, "data_lengths", "Document lengths (log scale): Phase-1 sources, Phase-6 documents and summaries, "
                                  "Phase-7 queries and candidates.")
    p2 = stats.get("phase2")
    if p2:
        fig, ax = plt.subplots(figsize=(6.5, 2.3))
        labs = [l for l, v in p2["_span_words"].items() if v]
        ax.boxplot([p2["_span_words"][l] for l in labs], showfliers=False)
        ax.set_xticks(range(1, len(labs) + 1), labs, rotation=30, ha="right")
        ax.set(ylabel="words per span", title="Phase 2: span length by label")
        save(fig, "data_p2_spans", "Phase-2 span length (words) by label; whiskers at 1.5 IQR, outliers hidden.")
    p3 = stats.get("phase3")
    if p3:
        fig, ax = plt.subplots(figsize=(6.5, 2.4))
        labs = p3["labels"]
        x = np.arange(len(labs))
        for i, dom in enumerate(("CL", "IT")):
            roles = p3["by_domain"][dom]["roles"]
            total = max(1, sum(roles.values()))
            ax.bar(x + (i - 0.5) * 0.4, [roles[l] / total for l in labs], width=0.4, label=dom)
        ax.set_xticks(x, labs, rotation=35, ha="right")
        ax.set(ylabel="share of sentences", title="Phase 3: role distribution by domain")
        ax.legend()
        save(fig, "data_p3_roles", "Phase-3 role distribution (share of sentences) in competition-law (CL) and "
                                   "income-tax (IT) documents.")
    p5 = stats.get("phase5")
    if p5:
        fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.3))
        a.hist(p5["_per_doc"], bins=range(1, max(p5["_per_doc"]) + 2), align="left")
        a.set(xlabel="statutes per document", ylabel="documents", title="Phase 5: labels per document", xlim=(0, 16))
        b.plot(range(1, len(p5["_ranked"]) + 1), p5["_ranked"])
        b.set(yscale="log", xlabel="statute rank", ylabel="training cases", title="Phase 5: statute frequency")
        save(fig, "data_p5_labels", "Phase-5 labels per training document and statute frequency by rank (log scale).")
    if p7 and p7["_gaps"]:
        fig, ax = plt.subplots(figsize=(3.3, 2.3))
        ax.hist(p7["_gaps"], bins=40)
        ax.set(xlabel="query year $-$ cited case year", ylabel="citations", title="Phase 7: age of cited cases")
        save(fig, "data_p7_years", "Phase-7 age of cited cases: latest year mentioned in the query minus that in the "
                                   "cited candidate.")
    return made


def latex(stats, made, out):
    t = ["% generated by tools/data_stats.py -- \\input this file in the appendix"]
    rows = []
    p1 = stats.get("phase1")
    if p1:
        rows.append(("1", f"words/doc median SC {p1['words_per_doc']['SC']['p50']:.0f}, HC {p1['words_per_doc']['HC']['p50']:.0f}; "
                          f"JS divergence {p1['js_divergence_bits']:.3f} bits; {100 * p1['hc_tokens_unseen_in_sc']:.1f}\\% of HC "
                          f"tokens unseen in SC; HC-specific: {tex_escape(', '.join(p1['over_represented_hc'][:6]))}"))
    p2 = stats.get("phase2")
    if p2:
        longest = sorted(p2["span_words_mean"].items(), key=lambda x: -x[1])[:3]
        rows.append(("2", f"{p2['mentions']} mentions in {p2['documents']} docs; nested {p2['nested']}, crossing "
                          f"{p2['crossing']}, adjacent {p2['adjacent']}; longest spans: "
                          + ", ".join(f"{tex_escape(l)} {v:.1f}" for l, v in longest) + " words"))
    p3 = stats.get("phase3")
    if p3:
        cl, it = p3["by_domain"]["CL"], p3["by_domain"]["IT"]
        rows.append(("3", f"CL {cl['documents']} docs / {cl['sentences']} sentences, IT {it['documents']} / {it['sentences']}; "
                          f"median words per sentence CL {cl['words_per_sentence']['p50']:.0f}, IT {it['words_per_sentence']['p50']:.0f}"))
    p4 = stats.get("phase4")
    if p4:
        tr = p4["train"]
        rows.append(("4", f"accepted rate {tr['accepted_rate']} (train), {p4['dev']['accepted_rate']} (dev); median words "
                          f"accepted {tr['words_accepted']['p50']:.0f}, rejected {tr['words_rejected']['p50']:.0f}"))
    p5 = stats.get("phase5")
    if p5:
        rows.append(("5", f"{p5['statutes']} statutes; labels/doc median {p5['labels_per_doc']['p50']:.0f} "
                          f"(p90 {p5['labels_per_doc']['p90']:.0f}); top-10 statutes hold {100 * p5['top10_share']:.0f}\\% of labels"))
    p6 = stats.get("phase6")
    if p6:
        nv = p6["novel_ngram_rate"]
        rows.append(("6", f"compression median {p6['compression']['p50']:.2f}; novel summary n-grams: "
                          + ", ".join(f"{k} {100 * v:.0f}\\%" for k, v in nv.items())))
    p7 = stats.get("phase7")
    if p7:
        rows.append(("7", f"median words query {p7['query_words']['p50']:.0f}, candidate {p7['candidate_words']['p50']:.0f}; "
                          f"cited cases are a median {p7['cited_years_before_query'].get('p50', 0):.0f} years older; "
                          f"{100 * p7['cited_later_than_query']:.1f}\\% mention a later year than the query"))
    if rows:
        t += ["\\begin{table}[H]", "\\centering\\footnotesize", "\\caption{Corpus statistics computed by \\texttt{tools/data\\_stats.py}.}",
              "\\label{tab:a-data}", "\\begin{tabular}{@{}l>{\\raggedright\\arraybackslash}p{14.5cm}@{}}", "\\toprule",
              "Phase & Statistics \\\\", "\\midrule"] + [f"{a} & {b} \\\\" for a, b in rows] + ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    if p2 and p2["labelled_and_unlabelled"]:
        t += ["\\begin{table}[H]", "\\centering\\footnotesize",
              "\\caption{Phase-2 strings that are entities in one place and unlabelled elsewhere (sources of false positives).}",
              "\\label{tab:a-p2fp}", "\\begin{tabular}{@{}llrr@{}}", "\\toprule", "String & Label & labelled & unlabelled \\\\", "\\midrule"]
        t += [f"{tex_escape(s)} & {tex_escape(l)} & {a} & {b} \\\\" for s, l, a, b in p2["labelled_and_unlabelled"][:10]]
        t += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    for name, caption in made:
        width = "0.5\\linewidth" if name == "data_p7_years" else "\\linewidth"
        t += ["\\begin{figure}[H]", "\\centering", f"\\includegraphics[width={width}]{{{out.name}/{name}.pdf}}",
              f"\\caption{{{caption}}}", f"\\label{{fig:{name}}}", "\\end{figure}", ""]
    (out / "appendix_data.tex").write_text("\n".join(t))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="the CACHE directory (holds raw/Phase_N_*)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--p1-docs", type=int, default=20000, help="first N documents read per Phase-1 source")
    ap.add_argument("--p1-source", choices=["auto", "local", "stream"], default="auto",
                    help="Phase 1: local files, streaming from the Hub, or local if present else stream (default)")
    ap.add_argument("--phases", default="1,2,3,4,5,6,7")
    args = ap.parse_args()
    raw, out = Path(args.cache) / "raw", Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fns = {"1": lambda: phase1(raw, args.p1_docs, args.p1_source), "2": lambda: phase2(raw), "3": lambda: phase3(raw),
           "4": lambda: phase4(raw), "5": lambda: phase5(raw), "6": lambda: phase6(raw), "7": lambda: phase7(raw)}
    stats = {}
    for n in args.phases.split(","):
        try:
            res = fns[n]()
        except Exception as error:                      # one broken phase must not stop the others
            print(f"phase {n}: failed ({type(error).__name__}: {error})")
            continue
        if res is None:
            print(f"phase {n}: data not found under {raw}, skipped")
            continue
        stats[f"phase{n}"] = res
        print(f"phase {n}: done")
    made = figures(stats, out)
    latex(stats, made, out)
    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in stats.items()}
    (out / "data_stats.json").write_text(json.dumps(clean, indent=1))
    print(json.dumps(clean, indent=1)[:4000])
    print(f"wrote {out}/data_stats.json, {len(made)} figures, {out}/appendix_data.tex")


if __name__ == "__main__":
    main()
