"""Synthetic Phase-2 data + tiny random Phase-1 checkpoints (all four architectures).

Mechanics only: the documents are templated legal text, the checkpoints are random.
"""
import json, random, sys
from pathlib import Path
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

LABELS = ["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT", "AUTH", "COURT", "STAT", "PREC", "DATE", "CASENO"]
NAMES = ["Ram Kumar", "Sita Devi", "Mohan Lal", "Rakesh Sharma", "Anil Gupta", "Priya Singh", "Vijay Nair", "Lakshmi Iyer"]
STATES = ["State of Uttar Pradesh", "Union of India", "State of Bihar", "Municipal Corporation of Delhi"]
JUDGES = ["R.M. Lodha", "A.K. Sikri", "S.A. Bobde", "Indu Malhotra"]
STATS = ["Section 302 of the Indian Penal Code", "Section 482 of the Code of Criminal Procedure",
         "Article 226 of the Constitution", "Section 4 of the Land Acquisition Act, 1894"]
PRECS = ["Bachan Singh v. State of Punjab, (1980) 2 SCC 684", "Maneka Gandhi v. Union of India, (1978) 1 SCC 248",
         "K.S. Puttaswamy v. Union of India, (2017) 10 SCC 1"]
FILLER = ("The learned counsel submitted that the impugned order suffers from infirmity . "
          "We have heard the parties at length and perused the record . The evidence on record does not support "
          "the finding recorded by the court below . ").split(" ")


def document(rng):
    parts, spans = [], []

    def add(text, label=None):
        start = sum(len(p) for p in parts)
        parts.append(text)
        if label:
            spans.append([start, start + len(text), label, text])
    add("REPORTABLE\nIN THE ")
    add("SUPREME COURT\n  OF INDIA", "COURT")
    add("\n")
    add(f"Criminal Appeal No. {rng.randint(1, 999)} of {rng.randint(1990, 2020)}", "CASENO")
    app, resp = rng.choice(NAMES), rng.choice(STATES)
    add("\n")
    add(app, "APP")
    add(" ... Appellant\nVersus\n")
    add(resp, "RESP")
    add(" ... Respondent\n\nJUDGMENT\n\n")
    add(rng.choice(JUDGES), "JUDGE")
    add(", J.\n\n1. This appeal arises from the judgment dated ")
    add(f"{rng.randint(1, 28)}.{rng.randint(1, 12):02d}.{rng.randint(1990, 2020)}", "DATE")
    add(". Mr. ")
    add(rng.choice(NAMES), "A.COUNSEL")
    add(",  learned counsel for the appellant, and Mr. ")   # double space on purpose
    add(rng.choice(NAMES), "R.COUNSEL")
    add(", learned counsel for the respondent, were heard. ")
    for _ in range(rng.randint(6, 14)):
        add(" ".join(rng.choice(FILLER) for _ in range(rng.randint(15, 40))) + " ")
        roll = rng.random()
        if roll < 0.3:
            add("The appellant was convicted under ")
            add(rng.choice(STATS), "STAT")
            add(". ")
        elif roll < 0.5:
            add("Reliance was placed on ")
            add(rng.choice(PRECS), "PREC")
            add(". ")
        elif roll < 0.6:
            add("The witness ")
            add(rng.choice(NAMES), "WIT")
            add(" deposed before the ")
            add("Collector", "AUTH")
            add(". ")
        elif roll < 0.7:
            add("The appellant ")
            add(app, "APP")
            add(" was arrested on ")
            add(f"{rng.randint(1, 28)} March, {rng.randint(1990, 2020)}", "DATE")
            add(". ")
    add("\nAppeal dismissed.\n")
    return "".join(parts), spans


def main(root, n_docs=24, inclusive=False):
    root = Path(root)
    rng = random.Random(0)
    data = root / "Phase_2_LNER"
    (data / "Train").mkdir(parents=True, exist_ok=True)
    (data / "Test").mkdir(parents=True, exist_ok=True)
    (data / "Train" / "labels.txt").write_text("\n".join(LABELS) + "\n")
    corpus = []
    for fold in (1, 2, 3):
        texts, gold = {}, {}
        for i in range(n_docs if fold < 3 else 6):
            text, spans = document(rng)
            doc = str(1000000 + fold * 1000 + i)
            texts[doc] = text
            gold[doc] = [[s, e - 1 if inclusive else e, l, t] for s, e, l, t in spans]
            corpus.append(text)
        split = "Train" if fold < 3 else "Test"
        json.dump(texts, open(data / split / f"fold{fold}-text.json", "w"))
        if fold < 3:
            json.dump(gold, open(data / "Train" / f"fold{fold}-gold.json", "w"))
        else:
            json.dump(gold, open(root / "fold3-gold-private.json", "w"))
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=700, special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _paths import phase1_root
    sys.path.insert(0, str(phase1_root()))
    from phase1.models import build_model
    # Unlabeled corpus: fresh documents, plus 4 NER documents planted in it (2 training, 2 test) the
    # way the official Phase-1 corpus contains NER judgments; the overlap filter must drop exactly those.
    other = random.Random(1)
    fresh = [document(other)[0] for _ in range(150)]
    planted = corpus[:2] + corpus[-2:]
    json.dump([{"text": t} for t in fresh[:50] + planted[:2] + fresh[50:100] + planted[2:] + fresh[100:]],
              open(root / "unlabeled.json", "w"))
    (root / "planted.json").write_text(json.dumps(planted))
    for arch in ["dense_gpt", "gqa_local", "encoder_decoder", "recurrent_hybrid"]:
        cfg = {"architecture": arch, "vocab_size": tok.get_vocab_size(), "dim": 64, "layers": 4, "heads": 4,
               "kv_heads": 2, "ff_dim": 128, "max_context": 1024, "encoder_layers": 3, "decoder_layers": 2,
               "local_window": 32, "query_chunk": 16, "global_every": 2, "dropout": 0.0, "gradient_checkpointing": False}
        model = build_model(cfg)
        out = root / "phase1_runs" / arch
        out.mkdir(parents=True, exist_ok=True)
        tok.save(str(out / "tokenizer.json"))
        torch.save({"model": model.state_dict(), "resolved": {"model": cfg, "run_id": arch}, "step": 0}, out / "best.pt")
    print("fixture ready at", root)


if __name__ == "__main__":
    main(sys.argv[1], inclusive=len(sys.argv) > 2 and sys.argv[2] == "inclusive")
