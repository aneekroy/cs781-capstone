"""Training curves for the report, read from the logs every phase already writes.

    python tools/plot_curves.py --project $PROJECT --out curves

Sources (nothing is retrained):
  Phase 1   runs/<run>/metrics.jsonl          train_loss every step, validation_loss every eval
  DAPT      runsP2/adapt/<name>/adapt_metrics.jsonl   loss, val_loss
  Phase 2   runsP2/<run>/cv_summary.json      folds[i].history: loss, val_macro_f1 per epoch
  Phase 3   runsP3/<run>/cv_summary.json      folds[i].history: loss, val_macro_f1 per epoch
  Phase 4   runsP4/<run>/summary.json         history: loss, dev_macro_f1 per evaluation
  Phase 5   runsP5/<run>/summary.json         history: loss, dev_macro_f1_global_t per evaluation
  Phase 6   runsP6/<run>/summary.json         history: train_loss, dev_loss per evaluation
  Phase 7   runsP7/<run>/summary.json         history: loss, dev_f1@7 per epoch

Writes one vector PDF per phase (curves/phaseN_curves.pdf) and curves/appendix_curves.tex, a ready
LaTeX snippet (\\input{curves/appendix_curves}) with one figure per PDF. Missing runs are skipped and
listed, so the script works with whatever subset of runs exists. Override the default run lists with
--p2 X1_denseorig_lorma X24_dense_coarse etc.
"""
import argparse
import glob
import json
from pathlib import Path

DEFAULTS = {
    "p1": ["dense_gpt_full_*", "gqa_local_full_*", "recurrent_hybrid_full_*", "encoder_decoder_full_*"],
    "dapt": ["dense_clm|denselm_clm", "gqa_clm", "hybrid_clm", "encdec_mlm"],   # a|b = first that exists
    "p2": ["X1_denseorig_lorma", "X24_dense_coarse", "A3_gqa_clm_lora", "A2_encdec_mlm_full", "X9_hybrid2_nokd"],
    "p3": ["T_lorma", "P1_dense_lorma_ctx", "P2_dense_lorma_sent", "T_full", "T_prompt"],
    "p4": ["F3_densedapt_lora", "F5_densedapt_adalora", "F1_densedapt_lorma", "F4_densedapt_rslora"],
    "p5": ["L1_dense_lorma", "L2_gqa_lorma", "L4_encdec_lorma"],
    "p6": ["C9_dense_full_c512_e5", "C3_dense_full_c512", "C1_dense_lorma_c512", "L5_encdec_fid_lorma"],
    "p7": ["R1_ctx2", "L1_dense_lorma", "A_neg_random", "T_full"],
}
PHASE_DIR = {"p2": "runsP2", "p3": "runsP3", "p4": "runsP4", "p5": "runsP5", "p6": "runsP6", "p7": "runsP7"}


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass                                    # a run killed mid-write leaves a partial last line
    return rows


def smooth(values, k=50):
    out, acc = [], []
    for v in values:
        acc.append(v)
        if len(acc) > k:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


def short(name):
    return name.split("_full_")[0] if "_full_" in name else name


def find(base, pattern):
    hits = sorted(glob.glob(str(base / pattern)))
    return [Path(h) for h in hits if Path(h).is_dir()]


def phase1(project, names, missing):
    series = []
    for pat in names:
        runs = [r for r in find(project / "runs", pat) if (r / "metrics.jsonl").exists()]
        if not runs:
            missing.append(f"runs/{pat}/metrics.jsonl")
            continue
        run = max(runs, key=lambda r: (r / "metrics.jsonl").stat().st_size)   # the full run, not a smoke test
        rows = read_jsonl(run / "metrics.jsonl")
        tok = [r.get("input_tokens_seen", r["step"]) / 1e6 for r in rows if "train_loss" in r]
        tr = smooth([r["train_loss"] for r in rows if "train_loss" in r])
        val = [(r.get("input_tokens_seen", r["step"]) / 1e6, r["validation_loss"]) for r in rows if "validation_loss" in r]
        series.append((short(run.name), tok, tr, val))
    return series


def history_of(path, kind):
    data = json.loads(Path(path).read_text())
    if kind == "cv":                                   # Phases 2-3: one history per fold
        return [f.get("history", []) for f in data.get("folds", []) if f.get("history")]
    return [data.get("history", [])] if data.get("history") else []


def plot_all(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7, "figure.dpi": 150})
    project, out = Path(args.project), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    missing, figures = [], []

    def save(fig, name, caption):
        fig.tight_layout()
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        figures.append((name, caption))
        print("wrote", out / f"{name}.pdf")

    # Phase 1
    s = phase1(project, args.p1, missing)
    if s:
        fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 2.4))
        for name, tok, tr, val in s:
            a.plot(tok, tr, lw=0.8, label=name)
            if val:
                b.plot(*zip(*val), marker=".", ms=3, lw=0.8, label=name)
        a.set(xlabel="training tokens (M)", ylabel="train loss (50-step mean)", title="Phase 1: training loss")
        b.set(xlabel="training tokens (M)", ylabel="validation loss", title="Phase 1: validation loss")
        b.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        save(fig, "phase1_curves", "Phase-1 language-model training and validation loss against training tokens.")

    # DAPT
    rows_by = {}
    for spec_name in args.dapt:
        options = [project / "runsP2" / "adapt" / n / "adapt_metrics.jsonl" for n in spec_name.split("|")]
        p = next((o for o in options if o.exists()), None)
        if p is not None:
            rows_by[p.parent.name] = read_jsonl(p)
        else:
            missing.append(" or ".join(str(o.relative_to(project)) for o in options))
    if rows_by:
        fig, ax = plt.subplots(figsize=(4.8, 2.4))
        for name, rows in rows_by.items():
            v = [(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]
            if v:
                ax.plot(*zip(*v), marker=".", ms=3, lw=0.8, label=name)
        ax.set(xlabel="step", ylabel="validation loss", title="DAPT (643M tokens)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        save(fig, "dapt_curves", "Continued domain pretraining: validation loss per backbone.")

    # Phases 2-7: metric (and loss) per epoch / evaluation
    spec = {
        "p2": ("cv_summary.json", "cv", "loss", "val_macro_f1", "epoch", "Phase 2: fold macro-F1 (strict)"),
        "p3": ("cv_summary.json", "cv", "loss", "val_macro_f1", "epoch", "Phase 3: fold macro-F1"),
        "p4": ("summary.json", "single", "loss", "dev_macro_f1", "step", "Phase 4: dev macro-F1"),
        "p5": ("summary.json", "single", "loss", "dev_macro_f1_global_t", "step", "Phase 5: dev macro-F1"),
        "p6": ("summary.json", "single", "train_loss", "dev_loss", "step", "Phase 6: held-out loss"),
        "p7": ("summary.json", "single", "loss", "dev_f1@7", "epoch", "Phase 7: dev F1@7"),
    }
    for key, (fname, kind, loss_key, metric_key, x_key, title) in spec.items():
        found = {}
        for run in getattr(args, key):
            p = project / PHASE_DIR[key] / run / fname
            if p.exists():
                hs = history_of(p, kind)
                if hs:
                    found[run] = hs
                    continue
            missing.append(str(p.relative_to(project)))
        if not found:
            continue
        fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 2.4))
        for i, (run, hs) in enumerate(found.items()):
            color = f"C{i}"
            for j, h in enumerate(hs):                     # one line per fold; only the first carries a label
                xs = [r.get(x_key, r.get("epoch", n)) for n, r in enumerate(h, 1)]
                lab = run if j == 0 else None
                ls = [(x, r[loss_key]) for x, r in zip(xs, h) if loss_key in r]
                ms = [(x, r[metric_key]) for x, r in zip(xs, h) if metric_key in r]
                if ls:
                    a.plot(*zip(*ls), color=color, lw=0.8, alpha=0.9 if j == 0 else 0.45, label=lab)
                if ms:
                    b.plot(*zip(*ms), color=color, marker=".", ms=3, lw=0.8, alpha=0.9 if j == 0 else 0.45, label=lab)
        a.set(xlabel=x_key, ylabel=loss_key.replace("_", " "), title=title.split(":")[0] + ": training loss")
        b.set(xlabel=x_key, ylabel=metric_key.replace("_", " "), title=title)
        b.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        n = key[1]
        save(fig, f"phase{n}_curves", f"{title.split(':')[0]}: training loss and {title.split(': ')[1]} per {x_key}"
             + (" (one line per cross-validation fold)." if kind == "cv" else "."))

    tex = ["% generated by tools/plot_curves.py -- \\input this file in the appendix"]
    for name, caption in figures:
        tex += ["\\begin{figure}[H]", "\\centering",
                f"\\includegraphics[width=\\linewidth]{{{out.name}/{name}.pdf}}",
                f"\\caption{{{caption}}}", f"\\label{{fig:{name}}}", "\\end{figure}", ""]
    (out / "appendix_curves.tex").write_text("\n".join(tex))
    print("wrote", out / "appendix_curves.tex", f"({len(figures)} figures)")
    if missing:
        print("not found (skipped):\n  " + "\n  ".join(missing))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="the Phase-1 project directory (holds runs/, runsP2/ ...)")
    ap.add_argument("--out", default="curves")
    for key, names in DEFAULTS.items():
        ap.add_argument(f"--{key}", nargs="+", default=names, help=f"runs to plot (default: {' '.join(names)})")
    plot_all(ap.parse_args())


if __name__ == "__main__":
    main()
