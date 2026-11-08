"""Stage 4: ensemble of finished Phase-4 runs (baseline, fine-tuned, aggregated) by probability averaging or
majority vote, members chosen greedily on dev, threshold on dev with the honest half-swap check.

  python -m phase4.ensemble --data Phase_4_CJPE --runs runsP4/B0 runsP4/F1 runsP4/A1 ... --output runsP4/E1
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .data import best_threshold, honest_threshold, load_split, macro_f1, write_prediction


def combine(members, split, ids, how, thresholds=None):
    p = np.array([[m[split][d] for d in ids] for m in members])
    if how == "vote":
        t = np.array([thresholds[i] for i in range(len(members))])[:, None]
        return (p >= t).mean(0)                       # share of members voting "accepted"
    return p.mean(0)


def score(members, ids, y, how, thresholds):
    return honest_threshold(y, combine(members, "dev", ids, how, thresholds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fixed", action="store_true")
    ap.add_argument("--max-members", type=int, default=7)
    args = ap.parse_args()
    dev, test = load_split(args.data, "dev"), load_split(args.data, "test")
    dev_ids, test_ids = list(dev), list(test)
    y = np.array([dev[d]["label"] for d in dev_ids])
    members = []
    for r in args.runs:
        r = Path(r)
        if not (r / "probs.json").exists():
            print(f"[skip] {r.name}: no probs.json")
            continue
        probs = json.loads((r / "probs.json").read_text())
        if set(probs["dev"]) != set(dev_ids) or set(probs["test"]) != set(test_ids):
            print(f"[skip] {r.name}: document ids differ")
            continue
        summary = json.loads((r / "summary.json").read_text()) if (r / "summary.json").exists() else {}
        members.append({"name": r.name, "dev": probs["dev"], "test": probs["test"],
                        "threshold": summary.get("threshold", 0.5)})
        print(f"{r.name:32s} dev honest {honest_threshold(y, np.array([probs['dev'][d] for d in dev_ids])):.4f}")
    if not members:
        raise SystemExit("no usable runs")
    results = []
    for how in ("average", "vote"):
        th = [m["threshold"] for m in members]
        if args.fixed:
            chosen = list(range(len(members)))
        else:
            chosen = [max(range(len(members)), key=lambda i: score([members[i]], dev_ids, y, how, [th[i]]))]
            best = score([members[i] for i in chosen], dev_ids, y, how, [th[i] for i in chosen])
            while len(chosen) < args.max_members:
                trials = [(score([members[i] for i in chosen + [j]], dev_ids, y, how, [th[i] for i in chosen + [j]]), j)
                          for j in range(len(members)) if j not in chosen]
                if not trials:
                    break
                s, j = max(trials)
                if s <= best + 1e-4:
                    break
                chosen.append(j)
                best = s
        group = [members[i] for i in chosen]
        gth = [th[i] for i in chosen]
        p_dev = combine(group, "dev", dev_ids, how, gth)
        honest = honest_threshold(y, p_dev)
        results.append((honest, how, chosen, p_dev))
        print(f"{how:8s} members {[members[i]['name'] for i in chosen]}  dev honest {honest:.4f}")
    honest, how, chosen, p_dev = max(results, key=lambda r: r[0])
    group, gth = [members[i] for i in chosen], [members[i]["threshold"] for i in chosen]
    t, tuned = best_threshold(y, p_dev)
    f05 = macro_f1(y, (p_dev >= 0.5).astype(int))
    if honest <= f05 + 0.003:
        t = 0.5
    p_test = combine(group, "test", test_ids, how, gth)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    probs = {"dev": dict(zip(dev_ids, map(float, p_dev))), "test": dict(zip(test_ids, map(float, p_test)))}
    (out / "probs.json").write_text(json.dumps(probs))
    info = write_prediction({d: int(probs["test"][d] >= t) for d in test}, test, out / "submission", probs["test"])
    summary = {"stage": "ensemble", "combine": how, "members": [m["name"] for m in group],
               "dev_macro_f1_at_0.5": round(f05, 5), "dev_macro_f1_tuned": round(tuned, 5),
               "dev_macro_f1_honest": round(honest if t != 0.5 else f05, 5), "threshold": round(t, 3),
               "test": info}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"ensemble ({how}) of {len(group)}: dev honest {summary['dev_macro_f1_honest']:.4f}  threshold {t:.2f}  {info}")


if __name__ == "__main__":
    main()
