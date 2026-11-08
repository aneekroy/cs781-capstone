"""Execute the supplied Phase 1 scorer unchanged and retain its exact output."""
import argparse
import math
from pathlib import Path
import re
import subprocess
import sys
from .common import file_hash, read_json, write_json

DEFAULT_SCORER = Path(__file__).resolve().parent.parent / 'evaluators' / 'evaluate_phase1.py'


def parse_scores(stdout):
    scores = {}
    for label, key in [('BERTScore F1', 'official_bertscore'), ('ROUGE-L F1', 'official_rouge_l'), ('BLEU-4', 'official_bleu4')]:
        match = re.search(r'^'+re.escape(label)+r':\s*([-+\deE.]+)\s*$', stdout, re.M)
        if not match or not math.isfinite(float(match.group(1))):
            raise ValueError(f'Official scorer did not emit a finite {label}')
        scores[key] = float(match.group(1))
    return scores


def score(prediction, reference, output, scorer=DEFAULT_SCORER):
    scorer = Path(scorer).resolve()
    argv = [sys.executable, str(scorer), '--truth', str(reference), '--prediction', str(prediction)]
    result = subprocess.run(argv, capture_output=True, text=True)
    evidence = {'argv': argv, 'scorer_sha256': file_hash(scorer),
                'prediction_sha256': file_hash(prediction), 'reference_sha256': file_hash(reference),
                'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    write_json(output, evidence)
    if result.returncode:
        raise RuntimeError(f'Official evaluator exited {result.returncode}; see {output}: {result.stderr[-2000:]}')
    evidence.update(parse_scores(result.stdout))
    write_json(output, evidence)
    return evidence


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ['prediction', 'reference', 'output']:
        ap.add_argument('--'+key, required=True)
    ap.add_argument('--scorer', default=str(DEFAULT_SCORER))
    a = ap.parse_args()
    print(score(a.prediction, a.reference, a.output, a.scorer))


if __name__ == '__main__':
    main()
