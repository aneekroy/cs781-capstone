"""Hardware validation and batch sizing for the two-GPU Jupyter workflow."""
from pathlib import Path
from .scheduler import visible_gpu_ids


def gpu_preflight(count=2, minimum_vram_gib=22):
    import torch
    from .train import precision
    if not torch.cuda.is_available() or torch.cuda.device_count() < count:
        raise RuntimeError(f'This workflow requires {count} visible CUDA GPUs. Check the notebook kernel and CUDA_VISIBLE_DEVICES.')
    visible = visible_gpu_ids()
    if len(visible) < count:
        raise RuntimeError('GPU visibility mapping does not contain enough usable devices')
    rows = []
    for logical in range(count):
        device = torch.device(f'cuda:{logical}')
        props = torch.cuda.get_device_properties(device)
        total = props.total_memory / 1024**3
        if total < minimum_vram_gib:
            raise RuntimeError(f'GPU {logical} has {total:.1f} GiB; this configuration expects about 24 GB per GPU')
        mode, _ = precision(device)
        free, _ = torch.cuda.mem_get_info(device)
        rows.append({'logical_device': logical, 'worker_gpu_id': visible[logical],
                     'name': props.name, 'total_vram_gib': round(total, 2),
                     'free_vram_gib': round(free/1024**3, 2), 'precision': mode})
    return rows


def accumulation(sequence_length, micro_batch_size=2, tokens_per_update=32768,
                 long_sequence_length=None):
    if min(sequence_length, micro_batch_size, tokens_per_update) < 1:
        raise ValueError('Batch dimensions must be positive')
    for length in [sequence_length, long_sequence_length]:
        if length is not None and tokens_per_update % (length*micro_batch_size):
            raise ValueError('Effective token count must divide both short and long micro-batches')
    return tokens_per_update // (sequence_length*micro_batch_size)
