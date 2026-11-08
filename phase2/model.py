"""Backbone wrappers around Phase-1 models, NER heads, CRF and biaffine spans."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def load_phase1(checkpoint_path, project_root=None):
    """Load a Phase-1 checkpoint (best.pt/latest.pt) + its tokenizer.json, on CPU."""
    import sys
    if project_root and str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from tokenizers import Tokenizer
    from phase1.models import build_model
    checkpoint_path = Path(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(state["resolved"]["model"])
    model.load_state_dict(state["model"])
    tokenizer = Tokenizer.from_file(str(checkpoint_path.parent / "tokenizer.json"))
    return model, tokenizer, state["resolved"]


class Encoder(nn.Module):
    """Token representations from any of the four Phase-1 architectures.

    mode:
      causal         decoder-only model as trained (each token sees only the left)
      bidirectional  same weights with the causal mask removed (LLM2Vec-style);
                     LSTM layers of the hybrid stay left-to-right
      encoder        encoder stack of the encoder-decoder model (bidirectional)
    """
    def __init__(self, lm, mode="bidirectional", keep_layers=None, prompt_tokens=0,
                 gradient_checkpointing=False):
        super().__init__()
        config = lm.config
        self.architecture = config.architecture
        self.seq2seq = config.architecture == "encoder_decoder"
        if self.seq2seq:
            mode = "encoder"
        elif mode == "encoder":
            raise ValueError("mode=encoder needs the encoder_decoder checkpoint")
        self.mode = mode
        self.dim = config.dim
        self.max_context = config.max_context
        self.embedding = lm.embedding
        self.position = lm.position
        blocks = lm.encoder if self.seq2seq else lm.blocks
        if keep_layers:
            blocks = nn.ModuleList(list(blocks)[:keep_layers])  # student = truncated Phase-1 model
        self.blocks = blocks
        self.final_norm = lm.encoder_norm if self.seq2seq else lm.norm
        self.prompt = nn.Parameter(torch.randn(prompt_tokens, config.dim) * 0.02) if prompt_tokens else None
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, ids):
        x = self.embedding(ids)
        prompt = 0
        if self.prompt is not None:
            prompt = self.prompt.size(0)
            x = torch.cat((self.prompt.unsqueeze(0).expand(x.size(0), -1, -1).to(x.dtype), x), 1)
        if x.size(1) > self.max_context:
            raise ValueError("Window plus prompt exceeds the model context")
        if self.position is not None:
            x = x + self.position(torch.arange(x.size(1), device=x.device))
        causal = self.mode == "causal"
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(lambda y, b=block: b(y, causal=causal)[0], x, use_reentrant=False)
            else:
                x, _ = block(x, causal=causal)
        x = self.final_norm(x)
        return x[:, prompt:]


# --------------------------------------------------------------------------- CRF
def transition_mask(tag_names):
    """Allowed BIOES transitions (True = allowed)."""
    k = len(tag_names)
    allowed = torch.zeros(k, k, dtype=torch.bool)
    start = torch.zeros(k, dtype=torch.bool)
    end = torch.zeros(k, dtype=torch.bool)
    for i, a in enumerate(tag_names):
        pa, la = a.split("-", 1) if a != "O" else ("O", None)
        start[i] = pa in {"O", "B", "S"}
        end[i] = pa in {"O", "E", "S"}
        for j, b in enumerate(tag_names):
            pb, lb = b.split("-", 1) if b != "O" else ("O", None)
            if pa in {"O", "E", "S"}:
                allowed[i, j] = pb in {"O", "B", "S"}
            else:  # B or I
                allowed[i, j] = pb in {"I", "E"} and la == lb
    return allowed, start, end


class CRF(nn.Module):
    def __init__(self, tag_names, learn=True):
        super().__init__()
        allowed, start, end = transition_mask(tag_names)
        self.register_buffer("allowed", allowed)
        self.register_buffer("start_ok", start)
        self.register_buffer("end_ok", end)
        k = len(tag_names)
        self.trans = nn.Parameter(torch.zeros(k, k), requires_grad=learn)
        self.start = nn.Parameter(torch.zeros(k), requires_grad=learn)
        self.end = nn.Parameter(torch.zeros(k), requires_grad=learn)

    def _scores(self):
        neg = -1e4
        return (self.trans.masked_fill(~self.allowed, neg), self.start.masked_fill(~self.start_ok, neg),
                self.end.masked_fill(~self.end_ok, neg))

    def nll(self, emissions, tags):
        """emissions [B,T,K] (float32), tags [B,T] -> NLL per token (mean over batch).

        Training windows can cut through an entity (a window may begin with I-/E- or
        end with B-/I-), so start/end constraints are NOT applied here; they are
        applied at document-level decoding where edges are real. Transition
        constraints always apply."""
        trans, _, _ = self._scores()
        start, end = self.start, self.end
        b, t, _ = emissions.shape
        score = start[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for i in range(1, t):
            score = score + trans[tags[:, i - 1], tags[:, i]] + emissions[:, i].gather(1, tags[:, i:i + 1]).squeeze(1)
        score = score + end[tags[:, -1]]
        alpha = start.unsqueeze(0) + emissions[:, 0]
        for i in range(1, t):
            alpha = torch.logsumexp(alpha.unsqueeze(2) + trans.unsqueeze(0), dim=1) + emissions[:, i]
        partition = torch.logsumexp(alpha + end.unsqueeze(0), dim=1)
        return (partition - score).mean() / t

    @torch.no_grad()
    def viterbi(self, emissions, entry_bonus=None):
        """emissions [T,K] -> best tag list (document level: edge constraints apply).

        entry_bonus [K] (optional): added to the score of ENTERING a tag from "O" or at the document
        start. Used on B-/S- tags it rewards starting an entity, so the decoder finds more
        entities without merging neighbours (a per-token bonus on every entity tag, or a penalty
        on "O", would favour long merged spans instead)."""
        trans, start, end = self._scores()
        if entry_bonus is not None:
            trans = trans.clone()
            trans[0] = trans[0] + entry_bonus.to(trans)
            start = start + entry_bonus.to(start)
        score = start + emissions[0]
        back = []
        for i in range(1, emissions.size(0)):
            total = score.unsqueeze(1) + trans
            score, index = total.max(0)
            score = score + emissions[i]
            back.append(index)
        score = score + end
        best = [int(score.argmax())]
        for index in reversed(back):
            best.append(int(index[best[-1]]))
        return best[::-1]


# --------------------------------------------------------------------------- heads
class TokenHead(nn.Module):
    def __init__(self, dim, num_tags, kind="linear", lstm_hidden=256, dropout=0.1):
        super().__init__()
        self.kind = kind
        self.drop = nn.Dropout(dropout)
        if kind == "bilstm":
            self.lstm = nn.LSTM(dim, lstm_hidden, batch_first=True, bidirectional=True)
            self.out = nn.Linear(2 * lstm_hidden, num_tags)
        elif kind == "linear":
            self.out = nn.Linear(dim, num_tags)
        else:
            raise ValueError(kind)

    def forward(self, h):
        h = self.drop(h)
        if self.kind == "bilstm":
            with torch.autocast(h.device.type, enabled=False):
                h, _ = self.lstm(h.float())
            h = self.drop(h)
        return self.out(h.float())


class BiaffineSpanHead(nn.Module):
    """score[b,i,j,c] for spans tokens i..j (inclusive), c=0 means 'no entity'."""
    def __init__(self, dim, num_labels, hidden=150, max_width=48, dropout=0.1):
        super().__init__()
        self.max_width = max_width
        self.start = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, hidden), nn.GELU())
        self.end = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, hidden), nn.GELU())
        self.U = nn.Parameter(torch.zeros(num_labels + 1, hidden + 1, hidden + 1))
        self.width = nn.Embedding(max_width, num_labels + 1)
        nn.init.xavier_uniform_(self.U)

    def forward(self, h):
        h = h.float()
        s = self.start(h)
        e = self.end(h)
        ones = torch.ones_like(s[..., :1])
        s, e = torch.cat((s, ones), -1), torch.cat((e, ones), -1)
        scores = torch.einsum("bid,cde,bje->bijc", s, self.U, e)
        t = h.size(1)
        width = torch.arange(t, device=h.device)
        width = width.view(1, t) - width.view(t, 1)  # j - i
        valid = (width >= 0) & (width < self.max_width)
        scores = scores + self.width(width.clamp(0, self.max_width - 1)).unsqueeze(0)
        return scores, valid


class Tagger(nn.Module):
    def __init__(self, encoder, labels, head="bilstm", crf=True, span=False, max_width=48, dropout=0.1):
        super().__init__()
        from .data import tag_names
        self.encoder = encoder
        self.labels = list(labels)
        self.tag_names = tag_names(labels)
        self.span = span
        if span:
            self.head = BiaffineSpanHead(encoder.dim, len(labels), max_width=max_width, dropout=dropout)
        else:
            self.head = TokenHead(encoder.dim, len(self.tag_names), head, dropout=dropout)
        # Without learned transitions the CRF still enforces valid BIOES at decode time.
        self.crf = CRF(self.tag_names, learn=crf and not span)
        self.use_crf = crf and not span

    def emissions(self, ids):
        return self.head(self.encoder(ids))

    def token_loss(self, emissions, tags, class_weights=None):
        if self.use_crf:
            return self.crf.nll(emissions.float(), tags)
        return F.cross_entropy(emissions.reshape(-1, emissions.size(-1)).float(), tags.reshape(-1),
                               weight=class_weights)

    def decode_token(self, scores):
        """scores [T,K] document-level (CRF emissions, or log-probs without CRF)."""
        return self.crf.viterbi(scores)
