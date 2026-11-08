"""Strict Phase 1 schema, ID, word-count and ZIP checks (stdlib only)."""
import argparse
import json
import zipfile
import warnings
from pathlib import Path
from .common import read_json, unique_object, write_json


def load_test(path, expected_prompt_words=512, strict_prompt_words=False):
    data = read_json(path)
    if not isinstance(data, dict) or not data:
        raise ValueError("Expected a nonempty object mapping official IDs to {'text': prompt}")
    for identifier, record in data.items():
        if not isinstance(identifier, str) or not isinstance(record, dict) or not isinstance(record.get("text"), str):
            raise ValueError(f"Invalid test record: {identifier!r}")
        if not record['text'].strip():
            raise ValueError(f"Empty test prompt: {identifier!r}")
        if expected_prompt_words is not None and len(record["text"].split()) != expected_prompt_words:
            message = f"Test ID {identifier!r} has {len(record['text'].split())} whitespace words; expected {expected_prompt_words}. Full prompt is retained."
            if strict_prompt_words:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning)
    return data


def validate(predictions, test, words=128):
    if not isinstance(predictions, dict):
        raise ValueError("Phase 1 predictions must be a JSON object")
    if set(predictions) != set(test):
        raise ValueError(f"ID mismatch: missing={sorted(set(test)-set(predictions))[:10]}, unknown={sorted(set(predictions)-set(test))[:10]}")
    for identifier, record in predictions.items():
        if not isinstance(record, dict) or set(record) != {"text"} or not isinstance(record["text"], str):
            raise ValueError(f"Invalid prediction schema for ID {identifier}")
        if len(record["text"].split()) != words:
            raise ValueError(f"Prediction {identifier}: expected exactly {words} words, got {len(record['text'].split())}")
        if "\x00" in record["text"]:
            raise ValueError(f"NUL character in prediction {identifier}")
        prompt = test[identifier]["text"].strip()
        if prompt and record["text"].strip().startswith(prompt):
            raise ValueError(f"Prediction {identifier} contains the original prompt")
    return {"valid": True, "examples": len(predictions), "words_per_prediction": words}


def package(predictions, test, output_dir):
    result = validate(predictions, test)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "prediction.json", predictions)
    archive = output / "submission.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        handle.write(output / "prediction.json", arcname="prediction.json")
    validate_archive(archive, test)
    write_json(output / "validation.json", result)
    return archive


def validate_archive(archive, test):
    with zipfile.ZipFile(archive) as handle:
        if handle.namelist() != ["prediction.json"]:
            raise ValueError("ZIP must contain exactly prediction.json at its root")
        data = json.loads(handle.read("prediction.json").decode("utf-8"), object_pairs_hook=unique_object)
    return validate(data, test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--predictions")
    parser.add_argument("--zip")
    parser.add_argument("--prompt-words", type=int, default=512)
    args = parser.parse_args()
    if bool(args.predictions) == bool(args.zip):
        parser.error("Specify exactly one of --predictions or --zip")
    test = load_test(args.test, args.prompt_words)
    print(validate_archive(args.zip, test) if args.zip else validate(read_json(args.predictions), test))


if __name__ == "__main__":
    main()
