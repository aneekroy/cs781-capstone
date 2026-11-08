"""Cached length-bucketed generation, deterministic short-output retries and resume."""
import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from .common import file_hash, fingerprint, read_json, write_json
from .models import build_model
from .submission import load_test, package
from .train import amp_context, precision


class ShortGenerationError(ValueError):
    """The decoder exhausted its budget or stalled before a complete continuation."""


@dataclass
class DecodingConfig:
    words: int = 128
    max_new_tokens: int = 768
    temperature: float = 0.3
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 0  # Token n-grams; tune against held-out BERTScore.
    stall_tokens: int = 64
    seed: int = 42

    def validate(self):
        if self.words <= 0 or self.max_new_tokens <= 0 or self.temperature < 0:
            raise ValueError('Invalid generation word/token/temperature setting')
        if not 0 < self.top_p <= 1 or self.top_k < 0 or self.repetition_penalty < 1:
            raise ValueError('Invalid top-p/top-k/repetition setting')
        if self.no_repeat_ngram_size < 0 or self.stall_tokens < 1:
            raise ValueError('Invalid n-gram/stall setting')


def select_token(logits, history, forbidden_ids, config, generator=None):
    logits = logits.float().clone()
    if history and config.repetition_penalty != 1:
        ids = torch.tensor(sorted(set(history)), device=logits.device)
        logits[ids] = torch.where(logits[ids] < 0, logits[ids]*config.repetition_penalty,
                                  logits[ids]/config.repetition_penalty)
    logits[forbidden_ids] = -torch.inf
    n = config.no_repeat_ngram_size
    if n and len(history) >= n-1:
        prefix = tuple(history[-(n-1):]) if n > 1 else ()
        blocked = {history[i+n-1] for i in range(len(history)-n+1)
                   if tuple(history[i:i+n-1]) == prefix}
        if blocked:
            logits[list(blocked)] = -torch.inf
    if not torch.isfinite(logits).any():
        raise FloatingPointError('No finite next-token logit after generation filters')
    if config.temperature == 0:
        return int(logits.argmax())
    logits /= config.temperature
    if config.top_k:
        threshold = logits.topk(min(config.top_k, logits.numel())).values[-1]
        logits[logits < threshold] = -torch.inf
    values, indices = logits.sort(descending=True)
    probabilities = values.softmax(-1)
    values[probabilities.cumsum(-1)-probabilities > config.top_p] = -torch.inf
    sample = torch.multinomial(values.softmax(-1), 1, generator=generator)
    return int(indices[sample])


@torch.inference_mode()
def continuation_batch(model, tokenizer, prompts, config, device, dtype=None, seeds=None):
    """Identical token lengths prevent padding entering attention or LSTM state.

    Return successes and ShortGenerationError objects per row. Other errors
    propagate. Finished rows stay in the cache until their peers finish.
    """
    config.validate()
    model.eval()
    encoded = tokenizer.encode_batch([p.rstrip() for p in prompts])
    lengths = [len(item.ids) for item in encoded]
    if not lengths or not min(lengths) or len(set(lengths)) != 1:
        raise ValueError('A batch requires nonempty prompts of identical token length')
    length = lengths[0]
    if length > model.config.max_context:
        raise ValueError(f'Prompt uses {length} tokens; context is {model.config.max_context}; no silent truncation')
    seq2seq = model.config.architecture == 'encoder_decoder'
    cap = min(config.max_new_tokens, model.config.max_context-1 if seq2seq else model.config.max_context-length)
    if cap < 1:
        raise ValueError('No context capacity for continuation')
    source = torch.tensor([item.ids for item in encoded], dtype=torch.long, device=device)
    memory = None
    with amp_context(device, dtype):
        if seq2seq:
            memory = model.encode(source)
            logits, cache = model(torch.full((len(prompts), 1), tokenizer.token_to_id('<bos>'),
                                             dtype=torch.long, device=device), memory=memory, use_cache=True)
        else:
            logits, cache = model(source, use_cache=True)
    forbidden = [tokenizer.token_to_id(t) for t in ['<pad>', '<bos>', '<eos>', '<unk>']]
    forbidden = [t for t in forbidden if t is not None]
    generators = [torch.Generator(device=device).manual_seed(int(s) % (2**63-1))
                  for s in (seeds or [config.seed]*len(prompts))]
    generated = [[] for _ in prompts]
    results = [None]*len(prompts)
    progress = [(0, 0) for _ in prompts]
    for index in range(cap):
        next_tokens = []
        for row in range(len(prompts)):
            if results[row] is not None:
                next_tokens.append(tokenizer.token_to_id('<bos>'))
                continue
            token = select_token(logits[row, -1], generated[row], forbidden, config, generators[row])
            next_tokens.append(token)
            generated[row].append(token)
            text = tokenizer.decode(generated[row], skip_special_tokens=True)
            words = text.split()
            if len(words) > config.words or (len(words) == config.words and text[-1:].isspace()):
                results[row] = (' '.join(words[:config.words]),
                    {'generated_tokens': len(generated[row]), 'prompt_tokens': length, 'cap_tokens': cap})
            else:
                count, last_progress = progress[row]
                if len(words) > count:
                    progress[row] = (len(words), index)
                elif index-last_progress >= config.stall_tokens:
                    results[row] = ShortGenerationError(f'Decoder stalled at {len(words)} words')
        if all(result is not None for result in results):
            break
        if index+1 < cap:
            with amp_context(device, dtype):
                logits, cache = model(torch.tensor(next_tokens, device=device)[:, None], memory=memory,
                                      cache=cache, use_cache=True)
    for row in range(len(prompts)):
        if results[row] is None:
            results[row] = ShortGenerationError(f'Generated only {len(tokenizer.decode(generated[row]).split())} words '
                f'without a boundary after word {config.words} in {cap} tokens')
    return results


def continuation(model, tokenizer, prompt, config=None, device=None, dtype=None):
    config = config or DecodingConfig()
    result = continuation_batch(model, tokenizer, [prompt], config,
                                device or next(model.parameters()).device, dtype)[0]
    if isinstance(result, Exception):
        raise result
    return result


def retry_short(model, tokenizer, prompt, config, device, dtype, seed, initial_error):
    errors = [str(initial_error)]
    retry = replace(config, temperature=0.8, top_p=0.9, repetition_penalty=max(1.2, config.repetition_penalty))
    for attempt in range(1, 3):
        result = continuation_batch(model, tokenizer, [prompt], retry, device, dtype, [seed+attempt])[0]
        if not isinstance(result, ShortGenerationError):
            text, info = result
            info.update(fallback=True, fallback_attempt=attempt, fallback_decoding=asdict(retry), prior_errors=errors)
            return text, info
        errors.append(str(result))
    raise ShortGenerationError('All decoding attempts were short: ' + '; '.join(errors))


def load_checkpoint(checkpoint_path, tokenizer_path=None, device=None):
    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = Path(checkpoint_path)
    tokenizer_path = Path(tokenizer_path or checkpoint_path.parent / 'tokenizer.json')
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != state['resolved']['model']['vocab_size']:
        raise ValueError('Checkpoint and tokenizer vocabulary sizes differ')
    expected = state['resolved'].get('tokenizer_sha256')
    if expected and file_hash(tokenizer_path) != expected:
        raise ValueError('Checkpoint and tokenizer identities differ')
    model = build_model(state['resolved']['model'])
    model.load_state_dict(state['model'])
    model.to(device).eval()
    for key in ['model', 'optimizer', 'scaler']:
        state.pop(key, None)
    return model, tokenizer, state


def generate(checkpoint_path, test_path, output_dir, config=None, tokenizer_path=None,
             prompt_words=512, precision_mode='auto', resume=False, batch_size=4,
             benchmark_examples=None):
    config = config or DecodingConfig()
    config.validate()
    if config.words != 128 or batch_size < 1:
        raise ValueError('Submissions require 128 words and a positive batch size')
    full_test = load_test(test_path, prompt_words)
    if benchmark_examples is not None and benchmark_examples < 1:
        raise ValueError('Benchmark example count must be positive')
    test = dict(list(full_test.items())[:benchmark_examples]) if benchmark_examples else full_test
    start = time.perf_counter()
    model, tokenizer, state = load_checkpoint(checkpoint_path, tokenizer_path)
    device = next(model.parameters()).device
    precision_name, dtype = precision(device, precision_mode)
    output = Path(output_dir)
    identity = {'checkpoint_sha256': file_hash(checkpoint_path), 'test_sha256': file_hash(test_path),
                'tokenizer_sha256': file_hash(tokenizer_path or Path(checkpoint_path).parent / 'tokenizer.json'),
                'decoding': asdict(config), 'precision': precision_name,
                'batch_size': batch_size, 'benchmark_examples': benchmark_examples}
    predictions, diagnostics = {}, {}
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError(f'Generation directory exists: {output}; use --resume or a new path')
        if read_json(output / 'generation_identity.json') != identity:
            raise ValueError('Generation checkpoint, tokenizer, test or decoding changed')
        if (output / 'partial_predictions.json').exists():
            predictions = read_json(output / 'partial_predictions.json')
            diagnostics = read_json(output / 'generation_diagnostics.json')
            if not set(predictions) <= set(test):
                raise ValueError('Unknown IDs in partial predictions')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'generation_identity.json', identity)
    buckets = defaultdict(list)
    pending = [(i, r) for i, r in test.items() if i not in predictions]
    for (identifier, record), encoded in zip(pending, tokenizer.encode_batch([r['text'].rstrip() for _, r in pending])):
        buckets[len(encoded.ids)].append((identifier, record))
    def save():
        write_json(output / 'partial_predictions.json', predictions)
        write_json(output / 'generation_diagnostics.json', diagnostics)
    processed = 0
    decode_start = time.perf_counter()
    for bucket in buckets.values():
        for offset in range(0, len(bucket), batch_size):
            items = bucket[offset:offset+batch_size]
            seeds = [config.seed+int(fingerprint(i)[:8], 16) for i, _ in items]
            try:
                results = continuation_batch(model, tokenizer, [r['text'] for _, r in items], config, device, dtype, seeds)
                for (identifier, _), result in zip(items, results):
                    if not isinstance(result, ShortGenerationError):
                        text, info = result
                        predictions[identifier], diagnostics[identifier] = {'text': text}, info
                save()
                for (identifier, record), result, id_seed in zip(items, results, seeds):
                    if isinstance(result, ShortGenerationError):
                        try:
                            text, info = retry_short(model, tokenizer, record['text'], config, device, dtype, id_seed, result)
                            predictions[identifier], diagnostics[identifier] = {'text': text}, info
                        except ShortGenerationError as error:
                            if not benchmark_examples:
                                raise
                            diagnostics[identifier] = {'short_output_error': str(error)}
                        save()
            except Exception as error:
                save()
                write_json(output / 'generation_error.json', {'ids': [i for i, _ in items], 'error': str(error)})
                raise
            processed += len(items)
            print(f'Generated {len(predictions)}/{len(test)}; session {time.perf_counter()-start:.1f}s', flush=True)
    archive = None
    if benchmark_examples:
        write_json(output / 'benchmark_predictions.json', predictions)
    else:
        archive = package(predictions, test, output)
    decode_seconds = time.perf_counter()-decode_start
    write_json(output / 'generation_summary.json', {'checkpoint_step': state['step'],
                'run_id': state['resolved']['run_id'], 'session_seconds': time.perf_counter()-start,
                'decode_seconds': decode_seconds, 'session_examples': processed,
                'examples': len(predictions), 'benchmark': bool(benchmark_examples),
                'projected_full_decode_seconds': decode_seconds/max(processed, 1)*len(full_test),
                'fallback_examples': sum(bool(d.get('fallback')) for d in diagnostics.values()),
                'short_examples': sum('short_output_error' in d for d in diagnostics.values()),
                'archive': str(archive.resolve()) if archive else None})
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'test', 'output']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--tokenizer')
    parser.add_argument('--prompt-words', type=int, default=512)
    parser.add_argument('--precision', default='auto', choices=['auto', 'fp32', 'bf16', 'fp16'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--benchmark-examples', type=int)
    for key in ['temperature', 'top_p', 'top_k', 'repetition_penalty', 'max_new_tokens', 'seed',
                'no_repeat_ngram_size', 'stall_tokens']:
        default = getattr(DecodingConfig(), key)
        parser.add_argument('--'+key.replace('_', '-'), type=type(default), default=default)
    args = vars(parser.parse_args())
    keys = set(asdict(DecodingConfig()))-{'words'}
    config = DecodingConfig(**{key: args.pop(key) for key in keys})
    generate(args.pop('checkpoint'), args.pop('test'), args.pop('output'), config,
             args.pop('tokenizer'), args.pop('prompt_words'), args.pop('precision'), **args)


if __name__ == '__main__':
    main()
