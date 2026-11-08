"""Comparable held-out continuation loss plus lexical diagnostics.

No pretrained BERT evaluator is downloaded here. Use the supplied official
scorer for BERTScore; lexical scores do not reproduce the primary metric.
"""
import argparse
from pathlib import Path
import math
import torch
from torch.nn import functional as F
from .common import read_json, write_json
from .generate import load_checkpoint
from .submission import validate
from .train import amp_context, precision


def rouge_l(prediction, reference):
    a, b = prediction.split(), reference.split()
    row = [0] * (len(b)+1)
    for word in a:
        nxt = [0]
        for j, other in enumerate(b):
            nxt.append(row[j]+1 if word == other else max(row[j+1], nxt[-1]))
        row = nxt
    return 2 * row[-1] / (len(a)+len(b)) if a or b else 1.0


@torch.inference_mode()
def conditional_nll(model, tokenizer, examples, dtype=None):
    """Score exactly the same reference tokens conditioned on each 512-word prompt.

    Prompt and continuation are encoded separately, consistently for all four
    architectures. This boundary convention is logged and not a Codabench score.
    """
    device = next(model.parameters()).device
    model.eval()
    summed, count = 0.0, 0
    for record in examples.values():
        prompt = tokenizer.encode(record["text"].rstrip()).ids
        target = tokenizer.encode(" " + record["reference"].lstrip()).ids
        if not prompt or not target:
            raise ValueError("Empty prompt/reference token sequence")
        source = torch.tensor([prompt], device=device)
        labels = torch.tensor([target], device=device)
        with amp_context(device, dtype):
            if model.config.architecture == "encoder_decoder":
                inputs = torch.tensor([[tokenizer.token_to_id("<bos>")] + target[:-1]], device=device)
                logits, _ = model(inputs, source_ids=source)
            else:
                inputs = torch.tensor([prompt + target[:-1]], device=device)
                logits, _ = model(inputs)
                logits = logits[:, len(prompt)-1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), reduction="sum")
        summed += loss.item()
        count += len(target)
    mean = summed/count
    return {"conditional_nll": mean, "conditional_perplexity": math.exp(min(mean, 700)),
            "reference_tokens": count, "examples": len(examples),
            "boundary": "prompt.rstrip() and space-prefixed reference encoded separately"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--predictions")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.checkpoint and not args.predictions:
        parser.error("Provide checkpoint and/or generated predictions")
    examples = read_json(args.examples)
    if not examples or any(not isinstance(row.get("reference"), str) for row in examples.values()):
        raise ValueError("Need nonempty held-out examples with reference fields")
    metrics = {}
    if args.checkpoint:
        model, tokenizer, _ = load_checkpoint(args.checkpoint)
        _, dtype = precision(next(model.parameters()).device)
        metrics.update(conditional_nll(model, tokenizer, examples, dtype))
    if args.predictions:
        predictions = read_json(args.predictions)
        validate(predictions, examples)
        scores = [rouge_l(predictions[key]["text"], row["reference"]) for key, row in examples.items()]
        metrics["rouge_l_whitespace_f1"] = sum(scores) / len(scores)
        metrics["repeated_trigram_fraction"] = sum(
            1 - len(set(zip(words, words[1:], words[2:]))) / max(1, len(words)-2)
            for words in [row["text"].split() for row in predictions.values()]) / len(predictions)
    metrics["official_bertscore"] = None
    metrics["official_metric_note"] = "Run the supplied course scorer; lexical diagnostics are not official BERTScore"
    write_json(args.output, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
