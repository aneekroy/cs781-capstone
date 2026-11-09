"""Dry-run or concurrently execute the four-architecture manifest."""
import argparse
import csv
import json
from pathlib import Path
import shlex
import sys
from .common import read_json
from .train import resolve, check_resume
from .scheduler import run_jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    matrix = sub.add_parser("matrix")
    matrix.add_argument("--data", required=True)
    matrix.add_argument("--output", required=True)
    matrix.add_argument("--configs", nargs="+", default=[f"configs/{name}.json" for name in
                              ["dense_gpt", "gqa_local", "encoder_decoder", "recurrent_hybrid"]])
    matrix.add_argument("--execute", action="store_true", help="Otherwise only print a dry-run")
    matrix.add_argument("--resume", action="store_true")
    matrix.add_argument("--gpus", nargs="*")
    matrix.add_argument("--allow-code-change", action="store_true")
    matrix.add_argument("--max-hours-per-run", type=float)
    ledger = sub.add_parser("ledger")
    ledger.add_argument("--runs", required=True)
    ledger.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "matrix":
        run_ids = set()
        commands = []
        for config in args.configs:
            row = resolve(config, args.data)
            if not 50_000_000 <= row["parameters"] <= 300_000_000:
                raise ValueError(f"Out-of-range competition model: {row['run_id']} ({row['parameters']:,})")
            if row["run_id"] in run_ids:
                raise ValueError(f"Duplicate run_id: {row['run_id']}")
            run_ids.add(row["run_id"])
            output = Path(args.output) / row["run_id"]
            status = read_json(output / "status.json") if (output / "status.json").exists() else {}
            if (output / "resolved_config.json").exists() and ((output / "last.pt").exists() or status.get("status") == "completed"):
                previous = read_json(output / "resolved_config.json")
                check_resume(previous, row, args.allow_code_change)
            if status.get("status") == "completed" and args.resume:
                print(f"SKIP completed run {row['run_id']}")
                continue
            if output.exists() and any(output.iterdir()) and (output / 'last.pt').exists() and not args.resume:
                raise FileExistsError(f"Run exists: {output}; use --resume or a new output root")
            command = [sys.executable, "-m", "phase1.train", "--config", config,
                       "--data", args.data, "--output", str(output)]
            if args.resume and (output / "last.pt").exists():
                command.append("--resume")
            if args.allow_code_change:
                command.append("--allow-code-change")
            if args.max_hours_per_run is not None:
                command += ["--max-hours", str(args.max_hours_per_run)]
            print({"run_id": row["run_id"], "parameters": row["parameters"],
                   "optimizer_steps": row["total_steps"], "command": shlex.join(command)}, flush=True)
            commands.append((command, output))
        # Resolve every manifest row before launching the first expensive run.
        if args.execute:
            jobs = [{'id': output.name, 'argv': command,
                     'log': str(output.parent / (output.name + '.console.log'))}
                    for command, output in commands]
            run_jobs(jobs, args.gpus, status_path=Path(args.output)/'scheduler_status.json')
    else:
        rows = []
        for path in sorted(Path(args.runs).glob("*/resolved_config.json")):
            config = read_json(path)
            root = path.parent
            status = read_json(root / "status.json") if (root / "status.json").exists() else {}
            selection = read_json(root / "selection.json") if (root / "selection.json").exists() else {}
            local = read_json(root / "local_metrics.json") if (root / "local_metrics.json").exists() else {}
            row = {"run_id": config["run_id"], "status": status.get("status", "unknown"),
                   "architecture": config["model"]["architecture"], "parameters": config["parameters"],
                   "seed": config["training"]["seed"], "input_tokens_seen": status.get("input_tokens_seen"),
                   "target_tokens_seen": status.get("target_tokens_seen"),
                   "best_validation_loss": status.get("best_validation_loss"),
                   "conditional_nll": local.get("conditional_nll"),
                   "rouge_l_whitespace_f1": local.get("rouge_l_whitespace_f1"),
                   "official_bertscore": selection.get("official_bertscore"),
                   "selected_checkpoint": selection.get("checkpoint"), "selected_decoding": json.dumps(selection.get("decoding")),
                   "train_seconds": status.get("train_seconds"), "checkpoint": status.get("best_checkpoint"),
                   "data_fingerprint": config["data_fingerprint"], "code_fingerprint": config["code_fingerprint"]}
            rows.append(row)
        if not rows:
            raise ValueError("No runs found")
        if len({row["run_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate run IDs in ledger")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} ledger rows to {args.output}")


if __name__ == "__main__":
    main()
