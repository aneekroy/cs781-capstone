"""Detect NER documents inside the official Phase-1 corpus (word 12-gram fingerprints).

Why: the KD student and continued pretraining read official Phase-1 documents. If an NER
document (training fold or test) is among them, (a) KD cross-validation leaks: the teacher was
trained on the validation fold and labels that same text for the student, and (b) the "never
use NER test text" promise breaks. Overlapping documents are therefore skipped there.

Fingerprints: every word 12-gram of every NER document (lower-cased, whitespace-split). A corpus
document is checked at every STRIDE-th position; it "overlaps" when at least MIN_HITS sampled
12-grams are NER 12-grams, i.e. it shares roughly MIN_HITS * STRIDE words of running text with
some NER document. Short boilerplate shared by many judgments stays far below that.

    python -m phase2.overlap --ner Phase_2_LNER --raw train_supreme_court.json [train_high_courts.json]
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

N = 12
STRIDE = 8
MIN_HITS = 20


def _words(text):
    return text.lower().split()


def _grams(words, step=1):
    for i in range(0, len(words) - N + 1, step):
        yield hash(tuple(words[i:i + N]))


class NerFingerprints:
    def __init__(self, texts_by_doc):
        """texts_by_doc: {(split, doc_id): text}."""
        self.owner = {}
        self.size = Counter()
        for key, text in texts_by_doc.items():
            for g in _grams(_words(text)):
                self.owner.setdefault(g, key)
                self.size[key] += 1

    @classmethod
    def from_data(cls, ner_dir, splits=("fold1", "fold2", "fold3")):
        from .data import load_texts
        texts = {}
        for split in splits:
            sub = "Test" if split == "fold3" else "Train"
            path = Path(ner_dir) / sub / f"{split}-text.json"
            if path.exists():
                texts.update({(split, doc): text for doc, text in load_texts(path).items()})
        return cls(texts)

    def hits(self, text):
        """{ner key: sampled hits} for one corpus document."""
        found = Counter()
        for g in _grams(_words(text), STRIDE):
            key = self.owner.get(g)
            if key is not None:
                found[key] += 1
        return found

    def overlaps(self, text):
        found = self.hits(text)
        return bool(found) and max(found.values()) >= MIN_HITS


def report(ner_dir, raw_paths):
    from .run import _stream_records
    fp = NerFingerprints.from_data(ner_dir)
    best = defaultdict(float)          # ner key -> best coverage by a single corpus document
    flagged = Counter()
    started = time.time()
    for path in raw_paths:
        for record in _stream_records(path):
            text = record.get("text", "") if isinstance(record, dict) else str(record)
            found = fp.hits(text)
            if not found or max(found.values()) < MIN_HITS:
                continue
            flagged[Path(path).name] += 1
            for key, count in found.items():
                best[key] = max(best[key], min(1.0, count * STRIDE / max(1, fp.size[key])))
        print(f"scanned {Path(path).name} ({(time.time() - started) / 60:.1f} min)", flush=True)
    splits = Counter(split for split, _ in fp.size)
    present = Counter(split for (split, _), cov in best.items() if cov >= 0.5)
    summary = {"ner_documents": dict(splits), "found_in_corpus_(>=50%_of_text)": dict(present),
               "corpus_documents_flagged": dict(flagged)}
    print(json.dumps(summary, indent=1))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ner", required=True, help="Phase_2_LNER directory")
    parser.add_argument("--raw", nargs="+", required=True, help="official Phase-1 train_*.json files")
    args = parser.parse_args()
    report(args.ner, args.raw)


if __name__ == "__main__":
    main()
