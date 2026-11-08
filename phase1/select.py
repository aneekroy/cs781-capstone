"""Select a checkpoint/decoder with official held-out BERTScore, retaining evidence."""
import argparse
from dataclasses import asdict
from pathlib import Path
from .common import file_hash, fingerprint, read_json, write_json
from .generate import DecodingConfig, generate
from .official_eval import DEFAULT_SCORER, score

DECODERS = [
    DecodingConfig(temperature=0.0, repetition_penalty=1.15),
    DecodingConfig(temperature=0.1, top_p=0.95, repetition_penalty=1.1),
    DecodingConfig(temperature=0.7, top_p=0.9, repetition_penalty=1.2, no_repeat_ngram_size=3),
]


def select(run_dir, examples_path, scorer=DEFAULT_SCORER, batch_size=4, screen_examples=16):
    run_dir = Path(run_dir)
    examples = read_json(examples_path)
    if not examples or any(not r.get('reference') for r in examples.values()):
        raise ValueError('Selection requires held-out prompts and references')
    # Balanced screening across sources. Every architecture uses this same order.
    groups = {}
    for identifier, record in sorted(examples.items()):
        groups.setdefault(str(record.get('source_id', 0)), []).append((identifier, record))
    ordered = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                ordered.append(group.pop(0))
    if screen_examples < 1:
        raise ValueError('Screen example count must be positive')
    identity = {'checkpoints': {name:file_hash(run_dir/name) for name in ['best.pt','latest.pt']
                               if (run_dir/name).exists()},
                'examples':file_hash(examples_path), 'scorer':file_hash(scorer),
                'decoders':[asdict(d) for d in DECODERS], 'screen_examples':screen_examples,
                'batch_size':batch_size}
    selection_path = run_dir/'selection.json'
    if selection_path.exists() and read_json(selection_path).get('identity') == identity:
        return read_json(selection_path)
    root = run_dir / 'selection' / fingerprint(identity)[:20]
    root.mkdir(parents=True, exist_ok=True)
    screen = dict(ordered[:min(screen_examples, len(examples))])
    write_json(root/'screen_prompts.json', screen)
    write_json(root/'screen_truth.json', {i: {'text': r['reference']} for i, r in screen.items()})
    write_json(root/'full_truth.json', {i: {'text': r['reference']} for i, r in examples.items()})
    checkpoints, steps = [], set()
    import torch
    for name in ['best.pt', 'latest.pt']:
        path = run_dir/name
        if path.exists():
            state = torch.load(path, map_location='cpu', weights_only=False)
            step = state['step']
            del state
            if step not in steps:
                checkpoints.append(path)
                steps.add(step)
    if not checkpoints:
        raise FileNotFoundError('No best/latest model-only checkpoint')
    candidates, errors = [], []
    for checkpoint in checkpoints:
        for index, decoder in enumerate(DECODERS):
            output = root/f'{checkpoint.stem}_decoder{index}'
            try:
                generate(checkpoint, root/'screen_prompts.json', output, decoder,
                         resume=output.exists(), batch_size=batch_size)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                evidence = score(output/'prediction.json', root/'screen_truth.json', output/'official_metrics.json', scorer)
                row = {'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': file_hash(checkpoint),
                       'decoding': asdict(decoder), 'official_bertscore': evidence['official_bertscore'],
                       'evidence': str((output/'official_metrics.json').resolve())}
                candidates.append(row)
            except Exception as error:
                errors.append({'checkpoint': str(checkpoint), 'decoder': index, 'error': str(error)})
            write_json(root/'candidates.json', {'scores': candidates, 'errors': errors})
    if not candidates:
        raise RuntimeError(f'All selection candidates failed; inspect {root}/candidates.json')
    # Screening is the controlled comparison. Confirm its winner on the full set;
    # do not claim all candidates received a full-set evaluation.
    winner = max(candidates, key=lambda row: row['official_bertscore'])
    full_dir = root/'full_winner'
    decoder = DecodingConfig(**winner['decoding'])
    generate(winner['checkpoint'], examples_path, full_dir, decoder,
             resume=full_dir.exists(), batch_size=batch_size)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    evidence = score(full_dir/'prediction.json', root/'full_truth.json', full_dir/'official_metrics.json', scorer)
    selection = dict(winner, identity=identity, screen_bertscore=winner['official_bertscore'],
                     official_bertscore=evidence['official_bertscore'],
                     selection_examples=len(screen), confirmation_examples=len(examples),
                     scorer_sha256=file_hash(scorer), examples_sha256=file_hash(examples_path),
                     evidence=str((full_dir/'official_metrics.json').resolve()),
                     criterion='highest official BERTScore on balanced screening set; winner confirmed on full held-out set')
    write_json(run_dir/'selection.json', selection)
    return selection


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    ap.add_argument('--examples', required=True)
    ap.add_argument('--scorer', default=str(DEFAULT_SCORER))
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--screen-examples', type=int, default=16)
    a = ap.parse_args()
    print(select(a.run, a.examples, a.scorer, a.batch_size, a.screen_examples))


if __name__ == '__main__':
    main()
