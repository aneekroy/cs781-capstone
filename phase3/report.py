"""Markdown comparison tables of finished phase3 runs: python -m phase3.report runsP3

  Backbone screen   S_* runs (same technique on every Phase-1/2 backbone)
  Techniques        T_* runs and KD runs (every fine-tuning technique on one backbone)
  All runs
Sorted by the honest pooled out-of-fold macro-F1 (decoder/offsets chosen on half the documents, scored on
the other half), the number to choose by. Written to <runs>/report.md.
"""
import json
import sys
from pathlib import Path


def load(out):
    rows = []
    for f in sorted(Path(out).glob("*/cv_summary.json")):
        s = json.loads(f.read_text())
        a = json.loads((f.parent / "args.json").read_text()) if (f.parent / "args.json").exists() else {}
        folds = s.get("folds") or [{}]
        # runs from older versions: method / backbone / parameters / KD teacher from args.json and the folds
        s.setdefault("method", a.get("method", "?"))
        if not s.get("backbone"):
            s["backbone"] = a.get("init_phase2") or a.get("checkpoint")
        if "teacher" not in s:
            s["teacher"] = a.get("teacher")
        if not s.get("parameters"):
            s["parameters"] = folds[0].get("parameters")
        if s.get("minutes") is None and all("minutes" in r for r in folds):
            s["minutes"] = round(sum(r["minutes"] for r in folds), 1)
        if "mixing" not in s:
            s["mixing"] = {"repeat_labels": a.get("repeat_labels"), "splice": a.get("splice"),
                           "domain_weights": a.get("domain_weights"), "kd_select_domain": a.get("kd_select_domain")}
        p = s.get("parameters") or {}
        e = s.get("encoder_parameters") or {}
        peft = s.get("peft") or {}
        mix = s.get("mixing") or {}
        tags = [t for t, on in (("label-repeat", mix.get("repeat_labels")), ("splice", mix.get("splice")),
                                ("domain-weights", mix.get("domain_weights")), ("in-domain silver", mix.get("kd_select_domain")))
                if on]
        rows.append({"run": f.parent.name, "method": s.get("method", "?") + (f" [{', '.join(tags)}]" if tags else ""),
                     "backbone": Path(str(s.get("backbone"))).parent.name, "teacher": bool(s.get("teacher")),
                     "trainable": p.get("trainable", 0), "total": p.get("total", 0),
                     "enc_trainable": e.get("trainable", 0),
                     "enc_total": e.get("total", 0) + (peft.get("base_bytes_bf16") or 0) // 2,   # + NF4 base weights
                     "base_4bit": peft.get("base_bytes_4bit"),
                     "honest": s["oof_macro_f1_honest"], "plain": s["oof_macro_f1_plain"],
                     "cl": s["oof_by_domain"]["CL"], "it": s["oof_by_domain"]["IT"], "folds": len(s["folds"]),
                     "minutes": s.get("minutes"), "test": (f.parent / "submission" / "submission.zip").exists()})
    return rows


def table(title, items):
    lines = [f"### {title}", "",
             "| run | technique | backbone | encoder: trained / total | encoder trained % | all trained (incl. "
             "context + CRF) | OOF macro-F1 (honest) | plain decoding | CL | IT | folds | GPU min | test |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|"]
    for r in sorted(items, key=lambda r: -r["honest"]):
        lines.append(f"| {r['run']} | {r['method']}{' + KD' if r['teacher'] else ''} | {r['backbone']} | "
                     + (f"{r['enc_trainable'] / 1e6:.3f}M / {r['enc_total'] / 1e6:.1f}M"
                        f"{' (base in 4-bit)' if r['base_4bit'] else ''} | {100 * r['enc_trainable'] / r['enc_total']:.2f} | "
                        if r["enc_total"] else "n/a (older run) | n/a | ") +
                     f"{r['trainable'] / 1e6:.2f}M | **{r['honest']:.4f}** | "
                     f"{r['plain']:.4f} | {r['cl']:.4f} | {r['it']:.4f} | {r['folds']} | {r['minutes']} | "
                     f"{'yes' if r['test'] else '-'} |")
    return "\n".join(lines) + "\n"


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "runsP3")
    rows = load(out)
    groups = [("Backbone screen (same technique on every backbone: LoRMA r16)", [r for r in rows if r["run"].startswith("S_")]),
              ("Fine-tuning techniques on one backbone", [r for r in rows if r["run"].startswith("T_") or r["teacher"]]),
              ("All runs", rows)]
    text = "\n".join(table(t, items) for t, items in groups if items) or "no finished runs yet\n"
    (out / "report.md").write_text(text)
    print(text)
    for title, items in groups[:2]:
        if items:
            b = max(items, key=lambda r: r["honest"])
            print(f"best in '{title}': {b['run']} ({b['method']} on {b['backbone']}) OOF {b['honest']:.4f}")


if __name__ == "__main__":
    main()
