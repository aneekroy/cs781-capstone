"""One isolated child per GPU, reusable for training, profiling and inference."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from .common import read_json, write_json


def visible_gpu_ids():
    import torch
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        return [x.strip() for x in visible.split(',') if x.strip() and x.strip() != '-1']
    return [str(i) for i in range(torch.cuda.device_count())]


def run_jobs(jobs, gpus=None, cwd=None, status_path=None):
    """Keep completed work/logs on failure; each child sees its own cuda:0.

    GPU identifiers are physical IDs (or UUIDs), including the inherited visible
    mapping. An empty GPU list runs a single CPU worker. No shell is invoked.
    """
    gpus = visible_gpu_ids() if gpus is None else [str(g) for g in gpus]
    if len(set(gpus)) != len(gpus):
        raise ValueError('GPU assignments must be unique')
    if len({job['id'] for job in jobs}) != len(jobs):
        raise ValueError('Job IDs must be unique')
    for job in jobs:
        if not job.get('argv') or any(not isinstance(x, str) for x in job['argv']):
            raise ValueError('Each job needs a nonempty argv list of strings')
    pending, active, rows = list(jobs), {}, {}
    slots = gpus or ['']
    last_notice = time.monotonic()
    def persist():
        if status_path:
            write_json(status_path, rows)
    try:
        while pending or active:
            for gpu in slots:
                if pending and gpu not in active:
                    job = pending.pop(0)
                    log_path = Path(job['log'])
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log = log_path.open('a', encoding='utf-8')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2')
                    try:
                        process = subprocess.Popen(job['argv'], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
                    except Exception:
                        log.close()
                        raise
                    active[gpu] = (process, job, log, time.monotonic())
                    rows[job['id']] = {'status': 'running', 'gpu': gpu or 'CPU', 'log': str(log_path), 'argv': job['argv']}
                    print(f"START {job['id']} on {gpu or 'CPU'}; log {log_path}", flush=True)
                    persist()
            for gpu, (process, job, log, tick) in list(active.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    rows[job['id']].update(status='completed' if code == 0 else 'failed',
                                           returncode=code, seconds=time.monotonic()-tick)
                    del active[gpu]
                    print(f"END {job['id']}: returncode={code}", flush=True)
                    persist()
            if time.monotonic()-last_notice >= 30:
                print('Running:', [job['id'] for _, job, _, _ in active.values()], 'queued:', len(pending), flush=True)
                last_notice = time.monotonic()
            if pending or active:
                time.sleep(0.2)
    except BaseException:
        for process, job, log, _ in active.values():
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
            rows[job['id']]['status'] = 'interrupted'
        persist()
        raise
    failed = [name for name, row in rows.items() if row['status'] != 'completed']
    if failed:
        raise RuntimeError(f'Jobs failed: {failed}; inspect per-job logs; successful outputs are preserved')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--jobs-json', required=True)
    ap.add_argument('--gpus', nargs='*')
    ap.add_argument('--cwd')
    ap.add_argument('--status', required=True)
    a = ap.parse_args()
    run_jobs(read_json(a.jobs_json), a.gpus, a.cwd, a.status)


if __name__ == '__main__':
    main()
