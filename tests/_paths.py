"""Locate the phase1/ package for the tests: $CS781_PROJECT, this repo, or a sibling cs781_phase1/."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def phase1_root():
    candidates = [os.environ.get("CS781_PROJECT"), ROOT, ROOT.parent / "cs781_phase1",
                  "$HOME/cs781_phase1"]
    for candidate in candidates:
        if candidate and (Path(candidate) / "phase1" / "models.py").is_file():
            return Path(candidate)
    raise RuntimeError("phase1/ not found; set CS781_PROJECT=/path/to/cs781_phase1")
