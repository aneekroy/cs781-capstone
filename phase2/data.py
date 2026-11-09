"""Loading, annotation-convention detection, statistics and offset-exact alignment.

Everything is done in ORIGINAL character offsets of the untouched document text.
The text is never normalized before tokenization, so predicted offsets index the
exact string the evaluator uses.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

TAG_SCHEME = ("B", "I", "E", "S")


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=1)


def load_labels(path):
    labels = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate labels in labels.txt")
    return labels


def load_texts(path):
    data = read_json(path)
    if isinstance(data, list):
        data = {str(item.get("id", item.get("doc_id"))): item for item in data}
    texts = {}
    for key, value in data.items():
        text = value if isinstance(value, str) else value.get("text")
        if not isinstance(text, str):
            raise ValueError(f"Document {key!r}: cannot find text")
        texts[str(key)] = text
    return texts


def _entries(value):
    if isinstance(value, dict):
        for key in ("entities", "spans", "annotations", "labels"):
            if key in value:
                return value[key]
    return value


def _parse(entry):
    if isinstance(entry, dict):
        start = entry.get("start", entry.get("begin", entry.get("start_offset")))
        end = entry.get("end", entry.get("end_offset"))
        label = entry.get("label", entry.get("type", entry.get("entity")))
        text = entry.get("text", entry.get("entity_text"))
    else:
        start, end, label = entry[0], entry[1], entry[2]
        text = entry[3] if len(entry) > 3 else None
    return int(start), int(end), str(label), text


def load_gold(path, texts, labels):
    """Return ({doc: [(start, end_exclusive, label)]}, convention report).

    The end-offset convention (exclusive vs inclusive) is inferred from the
    entity_text field when present, so predictions can be written back in the
    SAME convention the evaluator compares against.
    """
    data = read_json(path)
    if isinstance(data, list):
        data = {str(item.get("id", item.get("doc_id"))): item for item in data}
    raw = {str(k): [_parse(e) for e in _entries(v)] for k, v in data.items()}
    votes = Counter()
    for doc, items in raw.items():
        text = texts.get(doc)
        if text is None:
            raise ValueError(f"Gold document {doc} missing from texts")
        for start, end, _, surface in items:
            if surface is None:
                continue
            if text[start:end] == surface:
                votes["exclusive"] += 1
            elif text[start:end + 1] == surface:
                votes["inclusive"] += 1
            elif text[start:end].strip() == surface.strip():
                votes["exclusive_ws"] += 1
            else:
                votes["mismatch"] += 1
    inclusive = votes["inclusive"] > votes["exclusive"] + votes["exclusive_ws"]
    gold, unknown = {}, Counter()
    for doc, items in raw.items():
        spans = set()
        for start, end, label, _ in items:
            if label not in labels:
                unknown[label] += 1
                continue
            spans.add((start, end + 1 if inclusive else end, label))
        gold[doc] = sorted(spans)
    report = {"end_inclusive": inclusive, "convention_votes": dict(votes),
              "unknown_labels": dict(unknown), "documents": len(gold),
              "entities": sum(len(v) for v in gold.values())}
    if votes["mismatch"]:
        print(f"WARNING: {votes['mismatch']} gold spans do not match their entity_text; inspect them.")
    return gold, report


def dataset_stats(texts, gold, labels, tokenizer=None):
    per_label = defaultdict(lambda: {"count": 0, "chars": [], "docs": set(), "last_char": Counter(),
                                     "first_char": Counter()})
    overlaps = nested = 0
    mention_cov = defaultdict(list)
    for doc, spans in gold.items():
        text = texts[doc]
        ordered = sorted(spans)
        for (s1, e1, l1), (s2, e2, l2) in zip(ordered, ordered[1:]):
            if s2 < e1:
                overlaps += 1
                nested += e2 <= e1
        for start, end, label in spans:
            row = per_label[label]
            row["count"] += 1
            row["chars"].append(end - start)
            row["docs"].add(doc)
            row["last_char"][text[end - 1]] += 1
            row["first_char"][text[start]] += 1
        # Are ALL exact mentions of an annotated string annotated? (decides propagation)
        by_label = defaultdict(set)
        for start, end, label in spans:
            by_label[label].add(text[start:end])
        annotated = {(s, e) for s, e, _ in spans}
        for label, strings in by_label.items():
            for surface in strings:
                if len(surface) < 4:
                    continue
                occurrences, position = [], text.find(surface)
                while position != -1:
                    occurrences.append((position, position + len(surface)))
                    position = text.find(surface, position + 1)
                if occurrences:
                    mention_cov[label].append(sum(o in annotated for o in occurrences) / len(occurrences))
    table = {}
    for label in labels:
        row = per_label.get(label)
        if not row:
            table[label] = {"count": 0}
            continue
        chars = sorted(row["chars"])
        table[label] = {"count": row["count"], "docs": len(row["docs"]),
                        "chars_median": chars[len(chars) // 2], "chars_max": chars[-1],
                        "top_last_chars": row["last_char"].most_common(4),
                        "top_first_chars": row["first_char"].most_common(4),
                        "mention_coverage": round(sum(mention_cov[label]) / len(mention_cov[label]), 3)
                        if mention_cov[label] else None}
    stats = {"documents": len(gold), "entities": sum(len(v) for v in gold.values()),
             "overlapping_pairs": overlaps, "nested_pairs": nested, "labels": table,
             "doc_chars_median": sorted(len(texts[d]) for d in gold)[len(gold) // 2] if gold else 0}
    if tokenizer is not None:
        lengths = sorted(len(tokenizer.encode(texts[d]).ids) for d in gold)
        stats["doc_tokens_median"] = lengths[len(lengths) // 2]
        stats["doc_tokens_max"] = lengths[-1]
    return stats


# --------------------------------------------------------------------------- alignment
def tag_names(labels):
    return ["O"] + [f"{p}-{label}" for label in labels for p in TAG_SCHEME]


SEGMENTS = ("punct", "bpe")


def segment_tokenizer(tokenizer, mode="punct"):
    """Same vocabulary, different pre-tokenization.

    punct  isolate every punctuation character before byte-level BPE, so merged tokens
           such as '].' or ');' cannot straddle an entity boundary (they cap strict F1:
           a span ending at ']' would otherwise end mid-token).
    bpe    the Phase-1 tokenizer unchanged.
    """
    if mode == "bpe":
        return tokenizer
    if mode != "punct":
        raise ValueError(f"Unknown segmentation {mode!r}; choose from {SEGMENTS}")
    from tokenizers import Tokenizer, pre_tokenizers
    copy = Tokenizer.from_str(tokenizer.to_str())
    inner = tokenizer.pre_tokenizer
    steps = [pre_tokenizers.Punctuation(behavior="isolated")] + ([inner] if inner is not None else [])
    copy.pre_tokenizer = pre_tokenizers.Sequence(steps)
    return copy


def encode_document(tokenizer, text):
    """Token ids plus whitespace-trimmed character offsets in the original text.

    Byte-level BPE offsets for tokens like 'Ġword' start at the space; trimming
    them is what makes predicted span starts land on the first real character.
    """
    encoding = tokenizer.encode(text)
    trimmed = []
    for start, end in encoding.offsets:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        trimmed.append((start, end))
    return encoding.ids, trimmed


def spans_to_tags(offsets, spans, labels):
    """BIOES tags per token. Tokens overlapping a span (by non-space chars) belong to it."""
    index = {name: i for i, name in enumerate(tag_names(labels))}
    tags = [0] * len(offsets)
    conflicts = 0
    for start, end, label in spans:
        members = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
        if not members:
            conflicts += 1
            continue
        # whitespace-only tokens INSIDE an entity are I- (keeps BIOES sequences valid)
        members = list(range(members[0], members[-1] + 1))
        if any(tags[i] for i in members):
            conflicts += 1
            continue
        if len(members) == 1:
            tags[members[0]] = index[f"S-{label}"]
        else:
            tags[members[0]] = index[f"B-{label}"]
            for i in members[1:-1]:
                tags[i] = index[f"I-{label}"]
            tags[members[-1]] = index[f"E-{label}"]
    return tags, conflicts


def tags_to_spans(offsets, tags, labels):
    """Lenient BIOES decoding (an I/E without B starts a span) -> char spans."""
    names = tag_names(labels)
    spans, current = [], None
    for i, tag in enumerate(tags):
        name = names[tag]
        if offsets[i][1] <= offsets[i][0]:
            continue  # whitespace-only token: does not open/close spans
        if name == "O":
            if current:
                spans.append(current)
            current = None
            continue
        prefix, label = name.split("-", 1)
        if prefix == "S":
            if current:
                spans.append(current)
            spans.append([offsets[i][0], offsets[i][1], label])
            current = None
        elif prefix == "B" or current is None or current[2] != label:
            if current:
                spans.append(current)
            current = [offsets[i][0], offsets[i][1], label]
            if prefix == "E":
                spans.append(current)
                current = None
        else:
            current[1] = offsets[i][1]
            if prefix == "E":
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return [tuple(s) for s in spans]


def alignment_ceiling(texts, gold, tokenizer, labels, refine=None):
    """Exact-match rate of gold -> token tags -> spans. Anything below 100% is lost F1."""
    hit, total, misses = 0, 0, Counter()
    examples = []
    for doc, spans in gold.items():
        _, offsets = encode_document(tokenizer, texts[doc])
        tags, _ = spans_to_tags(offsets, spans, labels)
        decoded = set(tags_to_spans(offsets, tags, labels))
        if refine is not None:
            decoded = set(refine(texts[doc], decoded))
        for span in spans:
            total += 1
            if span in decoded:
                hit += 1
            else:
                misses[span[2]] += 1
                if len(examples) < 10:
                    examples.append((doc, span, texts[doc][max(0, span[0] - 5):span[1] + 5]))
    return {"exact_rate": hit / max(1, total), "missed_by_label": dict(misses), "examples": examples}


def make_windows(n_tokens, window, stride):
    if n_tokens <= window:
        return [0]
    starts = list(range(0, n_tokens - window + 1, stride))
    if starts[-1] + window < n_tokens:
        starts.append(n_tokens - window)
    return starts


# --------------------------------------------------------------------------- augmentation
def _match_case(replacement, original):
    if original.isupper():
        return replacement.upper()
    if original.islower():
        return replacement.lower()
    return replacement


def augment_documents(texts, gold, copies, rng):
    """Mention replacement (Dai & Adel, 2020): `copies` new versions of every training document in
    which each annotated mention is replaced by another mention of the same label taken from the
    training gold. One document keeps one replacement per (surface, label), so a party renamed in
    the header is renamed the same way in the body. Mentions that overlap another mention are kept
    as they are (only their offsets move). Returns texts/gold including the originals; new
    documents are named '<doc>~aug<i>'."""
    pool = {}
    for doc, spans in gold.items():
        for s, e, label in spans:
            surface = texts[doc][s:e]
            if surface.strip():
                pool.setdefault(label, set()).add(surface)
    pool = {label: sorted(v) for label, v in pool.items()}
    out_texts, out_gold = dict(texts), {d: list(g) for d, g in gold.items()}
    for doc, spans in gold.items():
        text = texts[doc]
        spans = sorted(set(map(tuple, spans)))
        free = [sp for sp in spans if not any(o != sp and o[0] < sp[1] and sp[0] < o[1] for o in spans)]
        free_set = set(free)
        for copy_index in range(copies):
            mapping, pieces, moved, pos, delta = {}, [], [], 0, 0
            for s, e, label in free:
                surface = text[s:e]
                key = (surface, label)
                if key not in mapping:
                    choices = [c for c in pool.get(label, []) if c != surface]
                    mapping[key] = _match_case(rng.choice(choices), surface) if choices else surface
                new = mapping[key]
                pieces += [text[pos:s], new]
                moved.append((s, e, label, s + delta, s + delta + len(new)))
                delta += len(new) - (e - s)
                pos = e
            pieces.append(text[pos:])
            new_text = "".join(pieces)
            new_spans = [(a, b, label) for _, _, label, a, b in moved]
            for s, e, label in spans:             # overlapping mentions: unchanged text, shifted offsets
                if (s, e, label) in free_set:
                    continue
                shift = sum(nb - na - (oe - os_) for os_, oe, _, na, nb in moved if oe <= s)
                new_spans.append((s + shift, e + shift, label))
            name = f"{doc}~aug{copy_index}"
            out_texts[name], out_gold[name] = new_text, sorted(new_spans)
    return out_texts, out_gold
