"""Small provenance, serialization, and reproducibility utilities."""
import hashlib
import json
import os
import random
import uuid
import warnings
from pathlib import Path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=unique_object)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def document_hash(text):
    # Whitespace-normalized exact duplicates share the split across sources.
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def is_validation(text, seed=42, fraction=0.02):
    key = hashlib.sha256(f"{seed}:{document_hash(text)}".encode()).digest()
    return int.from_bytes(key[:8], "big") / 2**64 < fraction


def quarantine_directory(path, reason):
    """Preserve incomplete work under a new name before restarting."""
    path = Path(path)
    if not path.exists():
        return None
    saved = path.with_name(path.name + '.incomplete-' + uuid.uuid4().hex[:10])
    path.rename(saved)
    warnings.warn(f"{reason}; preserved incomplete files at {saved}", RuntimeWarning)
    return saved


def directory_bytes(path):
    path = Path(path)
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.exists() else 0
