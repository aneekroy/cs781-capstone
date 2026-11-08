"""Measure prompt lengths and throughput; derive budgets and disk projections."""
import argparse
import math
import shutil
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from .common import directory_bytes, read_json, write_json
from .submission import load_test


def prompt_profile(test_path, tokenizer_path, reserve_tokens=250, max_context=2048):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    test = load_test(test_path)
    lengths = np.array([len(e.ids) for e in tokenizer.encode_batch([r['text'].rstrip() for r in test.values()])])
    p99, maximum = int(np.quantile(lengths, .99, method='higher')), int(lengths.max())
    if maximum+reserve_tokens > max_context:
        raise ValueError(f'Longest prompt {maximum} + reserve {reserve_tokens} exceeds context {max_context}; revise tokenizer/context before training')
    sequence = 1024 if p99+reserve_tokens <= 1024 else max_context
    return {'examples': len(test), 'p50': int(np.median(lengths)), 'p99': p99, 'max': maximum,
            'reserve_tokens': reserve_tokens, 'sequence_length': sequence,
            # Learned absolute positions need exposure beyond the short sequence.
            'long_sequence_length': max_context if sequence < max_context else None,
            'long_sequence_probability': 0.1 if sequence < max_context else 0.0,
            'source_length_range': [max(1, int(np.quantile(lengths, .05))), int(np.quantile(lengths, .95))]}


def measured_budget(run_dir, hours, safety=0.85):
    if hours <= 0 or not 0 < safety <= 1:
        raise ValueError('Hours must be positive and safety in (0,1]')
    run_dir = Path(run_dir)
    status = read_json(run_dir/'status.json')
    config = read_json(run_dir/'resolved_config.json')
    if status['step'] < 50:
        raise ValueError('Measure at least 50 optimizer steps before budgeting')
    wall_rate = status['input_tokens_seen']/status.get('wall_seconds', status['session_seconds'])
    train_rate = status['input_tokens_seen']/status['train_seconds']
    tokens_per_step = config['tokens_per_step']
    budget = max(tokens_per_step, int(min(wall_rate, train_rate)*hours*3600*safety)//tokens_per_step*tokens_per_step)
    return {'token_budget': budget, 'wall_tokens_per_second': wall_rate,
            'train_tokens_per_second': train_rate, 'hours': hours, 'safety': safety,
            'profile_step': status['step'], 'training_targets_per_input': status['target_tokens_seen']/status['input_tokens_seen']}


def disk_check(project, parameters, train_tokens, val_tokens, active_jobs=2, limit_bytes=None, token_bytes=2):
    project = Path(project)
    # Existing pilot logs/models plus final best/latest model-only + full Adam
    # checkpoints. Two simultaneous atomic full-checkpoint writes need headroom.
    additional = sum(p*20 for p in parameters)+sum(sorted(parameters, reverse=True)[:active_jobs])*12
    if token_bytes not in {2, 4} or active_jobs < 1:
        raise ValueError('Invalid token storage dtype or active job count')
    additional += (train_tokens+val_tokens)*token_bytes + 512*1024**2
    current = directory_bytes(project)
    free = shutil.disk_usage(project).free
    result = {'current_saved_bytes': current, 'additional_peak_estimate_bytes': additional,
              'projected_saved_peak_bytes': current+additional, 'saved_limit_bytes': limit_bytes,
              'filesystem_free_bytes': free, 'active_jobs': active_jobs}
    if (limit_bytes is not None and current+additional > limit_bytes) or additional > free:
        raise OSError(f'Insufficient projected disk headroom: {result}; reduce corpus/budgets or retained checkpoints')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--test', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    result = prompt_profile(a.test, a.tokenizer)
    write_json(a.output, result)
    print(result)


if __name__ == '__main__':
    main()
