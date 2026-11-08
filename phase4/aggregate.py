"""Stage 3: document model over the stage-2 chunk embeddings (hierarchical, as in the ILDC baselines).

Input per document: its last --embed-chunks window vectors (mean-pooled hidden state + stage-1 logit, in
reading order, the head window first when present). A projection, a BiGRU over the windows, attention
pooling, and a linear output give the document logit. Training is cheap (embeddings are cached), so several
seeds are trained and averaged; early stopping and the decision threshold use dev (threshold kept only if
it wins the honest half-swap check).
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import best_threshold, honest_threshold, load_split, macro_f1, write_prediction


class Aggregator(nn.Module):
    def __init__(self, dim, hidden=256, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, hidden), nn.GELU())
        self.rnn = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hidden, 1)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, 1))

    def forward(self, x, mask):
        h, _ = self.rnn(self.proj(self.norm(x)))
        scores = self.att(h).squeeze(-1).masked_fill(~mask, -1e4)
        weights = scores.softmax(-1).unsqueeze(-1)
        pooled = (weights * h).sum(1)
        lengths = mask.sum(1).clamp_min(1)
        last = h[torch.arange(h.size(0), device=h.device), lengths - 1]   # the window nearest the decision
        return self.out(torch.cat((pooled, last), -1)).squeeze(-1)


def pad_batch(seqs, device):
    n = max(s.size(0) for s in seqs)
    x = torch.zeros(len(seqs), n, seqs[0].size(1))
    mask = torch.zeros(len(seqs), n, dtype=torch.bool)
    for i, s in enumerate(seqs):
        x[i, :s.size(0)] = s.float()
        mask[i, :s.size(0)] = True
    return x.to(device), mask.to(device)


@torch.no_grad()
def predict(model, vectors, ids, device, batch=128):
    model.eval()
    out = {}
    for b in range(0, len(ids), batch):
        chunk = ids[b:b + batch]
        x, mask = pad_batch([vectors[d] for d in chunk], device)
        for d, p in zip(chunk, torch.sigmoid(model(x, mask)).cpu().tolist()):
            out[d] = p
    return out


def train_one(train, dev, seed, args, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dim = next(iter(train["vectors"].values())).size(1)
    model = Aggregator(dim, args.hidden, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    ids = sorted(train["vectors"])
    y = torch.tensor([float(train["labels"][d]) for d in ids])
    pos_weight = torch.tensor(float((y == 0).sum() / max(1, (y == 1).sum())), device=device) if args.balance else None
    dev_ids = sorted(dev["vectors"])
    y_dev = np.array([dev["labels"][d] for d in dev_ids])
    best, stale = (-1.0, None, None), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(ids))
        for b in range(0, len(order), args.batch_size):
            idx = order[b:b + args.batch_size]
            x, mask = pad_batch([train["vectors"][ids[i]] for i in idx], device)
            loss = F.binary_cross_entropy_with_logits(model(x, mask), y[idx].to(device), pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        p = predict(model, dev["vectors"], dev_ids, device)
        score = macro_f1(y_dev, (np.array([p[d] for d in dev_ids]) >= 0.5).astype(int))
        if score > best[0]:
            best, stale = (score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch), 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best[1])
    return model, best[0], best[2]


def run(args):
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1 = Path(args.run)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    emb = {s: torch.load(stage1 / f"emb_{s}.pt", map_location="cpu", weights_only=False) for s in ("train", "dev", "test")}
    dev_ids, test_ids = sorted(emb["dev"]["vectors"]), sorted(emb["test"]["vectors"])
    y_dev = np.array([emb["dev"]["labels"][d] for d in dev_ids])
    p_dev, p_test, rows = np.zeros(len(dev_ids)), np.zeros(len(test_ids)), []
    for seed in range(args.seeds):
        model, score, epoch = train_one(emb["train"], emb["dev"], seed, args, device)
        pd = predict(model, emb["dev"]["vectors"], dev_ids, device)
        pt = predict(model, emb["test"]["vectors"], test_ids, device)
        p_dev += np.array([pd[d] for d in dev_ids]) / args.seeds
        p_test += np.array([pt[d] for d in test_ids]) / args.seeds
        rows.append({"seed": seed, "dev_macro_f1": round(score, 4), "best_epoch": epoch})
        print(json.dumps(rows[-1]), flush=True)
    f05 = macro_f1(y_dev, (p_dev >= 0.5).astype(int))
    threshold, tuned = best_threshold(y_dev, p_dev)
    honest = honest_threshold(y_dev, p_dev)
    use = honest > f05 + 0.003
    t = threshold if use else 0.5
    test_docs = load_split(args.data, "test")
    probs = {"dev": dict(zip(dev_ids, map(float, p_dev))), "test": dict(zip(test_ids, map(float, p_test)))}
    (out / "probs.json").write_text(json.dumps(probs))
    info = write_prediction({d: int(probs["test"][d] >= t) for d in test_docs}, test_docs, out / "submission",
                            probs["test"])
    summary = {"stage": "aggregate", "stage1": str(stage1), "seeds": rows, "dev_macro_f1_at_0.5": round(f05, 5),
               "dev_macro_f1_tuned": round(tuned, 5), "dev_macro_f1_honest": round(honest if use else f05, 5),
               "threshold": round(t, 3), "minutes": round((time.time() - started) / 60, 1), "test": info}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"aggregate: dev macro-F1 {f05:.4f} @0.5 | honest {honest:.4f} -> threshold {t:.2f}  {info}", flush=True)
