"""Stream official legal JSON and create reproducible disk-backed token corpora."""
import argparse
import itertools
import json
import math
import os
import shutil
import sqlite3
from pathlib import Path
import warnings
import ijson
import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from .common import document_hash, file_hash, is_validation, read_json, write_json, quarantine_directory

SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]
REPO = "Exploration-Lab/CS-781-Capstone"
PREFIX = "Phase_1_Pretraining/"


def records(path):
    """Accept the starter's top-level list, an ID-keyed object, or JSONL."""
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if line.strip():
                    record = json.loads(line)
                    if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                        raise ValueError(f"Invalid JSONL record {index}: expected string text")
                    yield record
        return
    with path.open("rb") as handle:
        first = b""
        while not first.strip():
            first = handle.read(1)
            if not first:
                raise ValueError(f"Empty JSON file: {path}")
        handle.seek(0)
        if first == b"[":
            iterator = ijson.items(handle, "item")
        elif first == b"{":
            iterator = (value for _, value in ijson.kvitems(handle, ""))
        else:
            raise ValueError(f"Unsupported JSON container in {path}")
        for index, record in enumerate(iterator):
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"{path}, record {index}: expected {{'text': string}}; inspect real schema before training")
            if record["text"].strip():
                yield record


def resolve_official(output_dir, sources, revision="main", test_path=None, raw_dir=None):
    """Resolve commit and real filenames first; never guess a test filename."""
    from huggingface_hub import HfApi, hf_hub_download
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = api.dataset_info(REPO, revision=revision, files_metadata=True)
    sizes = {item.rfilename: item.size or 0 for item in info.siblings}
    train_files = [PREFIX + "Train/" + item for item in sources]
    missing = set(train_files) - set(sizes)
    if missing:
        raise ValueError(f"Training files absent at {info.sha}: {sorted(missing)}")
    candidates = sorted(name for name in sizes if name.startswith(PREFIX + "Test/") and name.lower().endswith(".json"))
    if test_path is None:
        if len(candidates) != 1:
            raise ValueError(f"Choose --test-file from actual repository paths: {candidates}")
        test_path = candidates[0]
    elif test_path not in candidates:
        raise ValueError(f"Test path absent; available: {candidates}")
    selected = train_files + [test_path]
    download_dir = Path(raw_dir) if raw_dir is not None else output / "raw"
    download_dir.mkdir(parents=True, exist_ok=True)
    needed = sum(sizes[name] for name in selected if not (download_dir / name).exists())
    reserve = 4 * 1024**3  # token files, checkpoints, optimizer, and scratch
    if needed + reserve > shutil.disk_usage(download_dir).free:
        raise OSError(f"Selected raw files need {needed / 1024**3:.2f} GiB plus a 4 GiB reserve; reduce sources or attach local data")
    paths = [hf_hub_download(REPO, name, repo_type="dataset", revision=info.sha,
                            token=os.environ.get("HF_TOKEN"), local_dir=download_dir)
             for name in selected]
    metadata = {"repo": REPO, "revision": info.sha,
                "files": [{"path": name, "bytes": sizes[name]} for name in selected],
                "train_paths": paths[:-1], "test_path": paths[-1]}
    write_json(output / "source_manifest.json", metadata)
    return metadata


def inspect(paths, limit=3):
    for path in paths:
        print(f"FILE {path}")
        for record in itertools.islice(records(path), limit):
            print({"keys": list(record), "words": len(record["text"].split()),
                   "text_preview": record["text"][:240]})


def train_tokenizer(paths, destination, seed=42, val_fraction=0.02,
                    vocab_size=24000, max_chars=20_000_000):
    if vocab_size < 260:
        raise ValueError("Byte BPE needs at least 256 bytes plus four special tokens")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                   special_tokens=SPECIAL,
                                   initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

    def text_iterator():
        total = 0
        iterators = [iter(records(path)) for path in paths]
        # Round-robin avoids training the tokenizer entirely on the first source.
        while iterators and total < max_chars:
            active = []
            for iterator in iterators:
                try:
                    text = next(iterator)["text"]
                except StopIteration:
                    continue
                active.append(iterator)
                if not is_validation(text, seed, val_fraction):
                    part = text[:max_chars-total]
                    total += len(part)
                    yield part
                if total >= max_chars:
                    break
            iterators = active
        if not total:
            raise ValueError("No training-only tokenizer text available")

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    tokenizer.save(str(destination))
    return tokenizer


def quotas(total, weights):
    if any(weight <= 0 for weight in weights):
        raise ValueError("All source weights must be positive")
    norm = sum(weights)
    result = [int(total * weight / norm) for weight in weights]
    result[-1] += total - sum(result)
    return result


def record_batches(path, size=32, max_chars=2_000_000):
    batch, chars = [], 0
    for record in records(path):
        batch.append(record)
        chars += len(record['text'])
        if len(batch) >= size or chars >= max_chars:
            yield batch
            batch, chars = [], 0
    if batch:
        yield batch


def _prepare(paths, output_dir, train_tokens=50_000_000, val_tokens=1_000_000,
            weights=None, seed=42, val_fraction=0.02, vocab_size=24000,
            tokenizer_chars=20_000_000, generation_examples=64, source_manifest=None,
            tokenizer_path=None, tokenize_batch_size=32):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite preprocessing: {output}; reuse it or choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    if not paths or not 0 < val_fraction < 1 or min(train_tokens, val_tokens) <= 0:
        raise ValueError("Need sources, positive token budgets, and val_fraction in (0,1)")
    if tokenize_batch_size < 1 or generation_examples < 0 or tokenizer_chars < 1:
        raise ValueError('Invalid tokenizer batch/sample settings')
    paths = [str(Path(path).resolve()) for path in paths]
    weights = weights or [1] * len(paths)
    if len(weights) != len(paths):
        raise ValueError("Source/weight count mismatch")
    if tokenizer_path is None:
        tokenizer = train_tokenizer(paths, output / "tokenizer.json", seed, val_fraction,
                                    vocab_size, tokenizer_chars)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        if any(tokenizer.token_to_id(token) is None for token in SPECIAL):
            raise ValueError('Reused tokenizer must define pad, bos, eos and unk special tokens')
        shutil.copyfile(tokenizer_path, output / 'tokenizer.json')
    dtype = np.dtype("uint16" if tokenizer.get_vocab_size() <= 65536 else "uint32")
    eos = tokenizer.token_to_id("<eos>")
    train_quota, val_quota = quotas(train_tokens, weights), quotas(val_tokens, weights)
    db = sqlite3.connect(output / "documents.sqlite")
    db.execute("CREATE TABLE documents (hash TEXT PRIMARY KEY, split TEXT, source INTEGER, tokens INTEGER)")
    counts = {"train": 0, "validation": 0, "duplicates": 0, "documents": 0}
    per_source, examples = [], {}
    example_quotas = quotas(generation_examples, [1] * len(paths))
    reservoirs = [[] for _ in paths]
    eligible = [0 for _ in paths]
    try:
        with (output / "train.bin").open("wb") as train_handle, (output / "validation.bin").open("wb") as val_handle:
            for index, path in enumerate(paths):
                used = {"train": 0, "validation": 0}
                target = {"train": train_quota[index], "validation": val_quota[index]}
                reservoir_rng = np.random.default_rng(seed + 10000 + index)
                source_documents = {'train': 0, 'validation': 0}
                for batch in record_batches(path, tokenize_batch_size):
                    selected = [record for record in batch if used['validation' if is_validation(record['text'], seed, val_fraction) else 'train'] < target['validation' if is_validation(record['text'], seed, val_fraction) else 'train']]
                    encodings = tokenizer.encode_batch([record['text'] for record in selected])
                    for record, encoding in zip(selected, encodings):
                        text = record["text"]
                        key = document_hash(text)
                        split = "validation" if is_validation(text, seed, val_fraction) else "train"
                        if used[split] >= target[split]:
                            continue
                        try:
                            db.execute("INSERT INTO documents VALUES (?,?,?,?)", (key, split, index, 0))
                        except sqlite3.IntegrityError:
                            counts["duplicates"] += 1
                            continue
                        words = text.split() if split == "validation" else None
                        if words is not None and len(words) >= 640:
                            eligible[index] += 1
                            start = int(key[:8], 16) % (len(words)-640+1)
                            example = (key, {"text": " ".join(words[start:start+512]),
                                             "reference": " ".join(words[start+512:start+640]),
                                             "source": Path(path).name, 'source_id': index})
                            if len(reservoirs[index]) < example_quotas[index]:
                                reservoirs[index].append(example)
                            elif example_quotas[index]:
                                slot = int(reservoir_rng.integers(0, eligible[index]))
                                if slot < example_quotas[index]:
                                    reservoirs[index][slot] = example
                        tokens = (encoding.ids + [eos])[:target[split]-used[split]]
                        np.asarray(tokens, dtype=dtype).tofile(val_handle if split == "validation" else train_handle)
                        used[split] += len(tokens)
                        counts[split] += len(tokens)
                        counts["documents"] += 1
                        source_documents[split] += 1
                        db.execute("UPDATE documents SET tokens=? WHERE hash=?", (len(tokens), key))
                        if counts["documents"] % 1000 == 0:
                            db.commit()
                            print({"source": index, "tokens": dict(counts)}, flush=True)
                    if all(used[part] >= target[part] for part in used):
                        break
                per_source.append({"path": path, "bytes": Path(path).stat().st_size,
                                   "requested": target, "observed": used,
                                   'documents': source_documents,
                                   'eligible_generation_documents': eligible[index],
                                   'generation_examples': len(reservoirs[index])})
        db.commit()
        if not counts["train"] or not counts["validation"]:
            raise ValueError("Empty training/validation split; add data or change split fraction")
    finally:
        db.close()
    examples = dict(pair for reservoir in reservoirs for pair in reservoir)
    metadata = {"seed": seed, "val_fraction": val_fraction, "split_unit": "whitespace-normalized document hash",
                "vocab_size": tokenizer.get_vocab_size(), "dtype": dtype.name,
                "counts": counts, "sources": per_source,
                "official_source_manifest": source_manifest,
                "tokenizer_sha256": file_hash(output / "tokenizer.json"),
                "train_sha256": file_hash(output / "train.bin"),
                "validation_sha256": file_hash(output / "validation.bin"),
                'generation_sampling': 'balanced per-source reservoir over eligible held-out documents',
                'tokenize_batch_size': tokenize_batch_size,
                "generation_examples": len(examples), "complete": True}
    write_json(output / "validation_examples.json", examples)
    write_json(output / "metadata.json", metadata)
    for source in per_source:
        if source["requested"] != source["observed"]:
            print("WARNING: exhausted source before requested budgets:", source)
    if not examples:
        print("WARNING: no held-out 640-word document; generation evaluation unavailable")
    return metadata


def prepare(paths, output_dir, train_tokens=50_000_000, val_tokens=1_000_000,
            weights=None, seed=42, val_fraction=0.02, vocab_size=24000,
            tokenizer_chars=20_000_000, generation_examples=64, source_manifest=None,
            tokenizer_path=None, tokenize_batch_size=32, reuse=False,
            recover_incomplete=True):
    """Publish only completed preprocessing; preserve interrupted stages."""
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    arguments = dict(train_tokens=train_tokens, val_tokens=val_tokens, weights=weights,
                     seed=seed, val_fraction=val_fraction, vocab_size=vocab_size,
                     tokenizer_chars=tokenizer_chars, generation_examples=generation_examples,
                     source_manifest=source_manifest,
                     tokenizer_path=str(Path(tokenizer_path).resolve()) if tokenizer_path else None,
                     tokenize_batch_size=tokenize_batch_size)
    request = {'paths': [str(Path(p).resolve()) for p in paths],
               'source_files': [{'path':str(Path(p).resolve()), 'bytes':Path(p).stat().st_size,
                                 'mtime_ns':Path(p).stat().st_mtime_ns} for p in paths],
               'reused_tokenizer_sha256':file_hash(tokenizer_path) if tokenizer_path else None, **arguments}
    metadata_path = output / 'metadata.json'
    if metadata_path.exists() and read_json(metadata_path).get('complete'):
        if reuse and read_json(output / 'preparation_request.json') == request:
            return read_json(metadata_path)
        raise FileExistsError(f'Completed preprocessing exists: {output}; reuse matching settings or choose a new directory')
    if output.exists() and any(output.iterdir()):
        if not recover_incomplete:
            raise FileExistsError(f'Incomplete preprocessing at {output}')
        quarantine_directory(output, 'Restarting incomplete preprocessing')
    stage = output.with_name(output.name + '.partial')
    if stage.exists():
        if not recover_incomplete:
            raise FileExistsError(f'Incomplete preprocessing stage at {stage}')
        quarantine_directory(stage, 'Restarting interrupted preprocessing stage')
    try:
        result = _prepare(paths, stage, **arguments)
        write_json(stage / 'preparation_request.json', request)
        # The final directory never exposes half-written token files.
        os.replace(stage, output)
        return result
    except Exception as error:
        if stage.exists():
            write_json(stage / 'failure.json', {'error': str(error), 'status': 'incomplete'})
        raise


class TokenCorpus:
    def __init__(self, directory, split="train"):
        directory = Path(directory)
        self.metadata = read_json(directory / "metadata.json")
        if not self.metadata.get("complete"):
            raise ValueError("Incomplete preprocessing")
        self.tokens = np.memmap(directory / f"{split}.bin", dtype=self.metadata["dtype"], mode="r")

    def sample(self, rng, batch_size, length):
        if len(self.tokens) < length:
            raise ValueError(f"Token split ({len(self.tokens)}) is shorter than a block ({length})")
        starts = rng.integers(0, len(self.tokens)-length+1, size=batch_size)
        return np.stack([self.tokens[start:start+length].astype(np.int64) for start in starts])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--output", required=True)
    fetch.add_argument("--sources", nargs="+", default=["train_supreme_court.json"])
    fetch.add_argument("--revision", default="main")
    fetch.add_argument("--test-file")
    fetch.add_argument('--raw-dir', help='Temporary raw download location, e.g. /tmp/cs781_raw')
    show = sub.add_parser("inspect")
    show.add_argument("paths", nargs="+")
    prep = sub.add_parser("prepare")
    prep.add_argument("--paths", nargs="+", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--train-tokens", type=int, default=50_000_000)
    prep.add_argument("--val-tokens", type=int, default=1_000_000)
    prep.add_argument("--weights", type=float, nargs="+")
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--val-fraction", type=float, default=0.02)
    prep.add_argument("--vocab-size", type=int, default=24000)
    prep.add_argument("--tokenizer-chars", type=int, default=20_000_000)
    prep.add_argument("--generation-examples", type=int, default=64)
    prep.add_argument("--source-manifest")
    prep.add_argument('--tokenizer-path', help='Reuse your course/pilot tokenizer with the same document split')
    prep.add_argument('--tokenize-batch-size', type=int, default=32)
    prep.add_argument('--reuse', action='store_true')
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "fetch":
        args["output_dir"] = args.pop("output")
        args["test_path"] = args.pop("test_file")
        print(resolve_official(**args))
    elif command == "inspect":
        inspect(args["paths"])
    else:
        args["output_dir"] = args.pop("output")
        if args["source_manifest"]:
            args["source_manifest"] = read_json(args["source_manifest"])
        print(prepare(**args))


if __name__ == "__main__":
    main()
