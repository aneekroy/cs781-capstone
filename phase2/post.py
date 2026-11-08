"""Strict metrics and post-processing: boundary refinement, document consistency,
mention propagation and pattern rules. Every component is switched on per label
only if it improves cross-validated strict F1 (see run.choose_postprocessing)."""
import re
from collections import Counter, defaultdict

PUNCT = set(".,;:()[]\"'`’‘“”-–—")


def strict_scores(gold, pred, labels):
    """gold/pred: {doc: iterable of (s,e,label)}. Returns per-label and macro P/R/F1."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for doc in gold:
        g = set(map(tuple, gold[doc]))
        p = set(map(tuple, pred.get(doc, [])))
        for span in p & g:
            tp[span[2]] += 1
        for span in p - g:
            fp[span[2]] += 1
        for span in g - p:
            fn[span[2]] += 1
    rows = {}
    for label in labels:
        p = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        r = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        rows[label] = {"p": p, "r": r, "f1": f, "support": tp[label] + fn[label], "pred": tp[label] + fp[label]}
    present = [l for l in labels if rows[l]["support"] or rows[l]["pred"]]
    macro = {k: sum(rows[l][k] for l in present) / max(1, len(present)) for k in ("p", "r", "f1")}
    all12 = {k: sum(rows[l][k] for l in labels) / len(labels) for k in ("p", "r", "f1")}
    return {"macro_f1": macro["f1"], "macro_p": macro["p"], "macro_r": macro["r"],
            "macro_f1_all_labels": all12["f1"], "per_label": rows}


def error_breakdown(texts, gold, pred, labels):
    """Classify every miss as boundary error, label confusion, or missed/spurious."""
    out = {"boundary": Counter(), "label_confusion": Counter(), "missed": Counter(), "spurious": Counter()}
    for doc in gold:
        g = set(map(tuple, gold[doc]))
        p = set(map(tuple, pred.get(doc, [])))
        for s, e, l in g - p:
            overlap = [q for q in p if q[0] < e and q[1] > s]
            if any(q[0] == s and q[1] == e for q in overlap):
                other = next(q for q in overlap if q[0] == s and q[1] == e)
                out["label_confusion"][f"{l}->{other[2]}"] += 1
            elif any(q[2] == l for q in overlap):
                out["boundary"][l] += 1
            else:
                out["missed"][l] += 1
        for s, e, l in p - g:
            if not any(q[0] < e and q[1] > s for q in g):
                out["spurious"][l] += 1
    return {k: dict(v.most_common()) for k, v in out.items()}


PAIRS = {")": "(", "]": "[", "}": "{"}
OPENERS = {v: k for k, v in PAIRS.items()}


class BoundaryRules:
    """Per label, strip leading/trailing punctuation that gold spans (almost) never include.

    Fitted on training gold only. A bracket is only stripped when it is unbalanced inside the
    span: '(1988) 4 SCC 551' keeps its '(', 'X (I) Ltd' is untouched, 'X (I) Ltd)' loses the last ')'.
    Which labels it is applied to is decided by cross-validation (choose_postprocessing).
    """
    def __init__(self, threshold=0.01):
        self.threshold = threshold
        self.strip_end = {}
        self.strip_start = {}

    def fit(self, texts, gold):
        ends, starts, totals = defaultdict(Counter), defaultdict(Counter), Counter()
        for doc, spans in gold.items():
            text = texts[doc]
            for s, e, label in spans:
                totals[label] += 1
                ends[label][text[e - 1]] += 1
                starts[label][text[s]] += 1
        for label in totals:
            self.strip_end[label] = "".join(c for c in PUNCT if ends[label][c] / totals[label] < self.threshold)
            self.strip_start[label] = "".join(c for c in PUNCT if starts[label][c] / totals[label] < self.threshold)
        return self

    def __call__(self, text, spans, labels=None):
        out = []
        for s, e, label in spans:
            if labels is not None and label not in labels:
                out.append((s, e, label))
                continue
            strip_end = self.strip_end.get(label, "")
            strip_start = self.strip_start.get(label, "")
            while e > s:
                c = text[e - 1]
                inner = text[s:e]
                balanced = c in PAIRS and inner.count(PAIRS[c]) >= inner.count(c)   # closer has its opener
                if c.isspace() or (c in strip_end and not balanced):
                    e -= 1
                else:
                    break
            while s < e:
                c = text[s]
                inner = text[s:e]
                balanced = c in OPENERS and inner.count(OPENERS[c]) >= inner.count(c)  # opener has its closer
                if c.isspace() or (c in strip_start and not balanced):
                    s += 1
                else:
                    break
            if e > s:
                out.append((s, e, label))
        return out

    def state(self):
        return {"threshold": self.threshold, "strip_end": self.strip_end, "strip_start": self.strip_start}

    @classmethod
    def from_state(cls, state):
        rules = cls(state["threshold"])
        rules.strip_end, rules.strip_start = state["strip_end"], state["strip_start"]
        return rules


def remove_overlaps(spans, scores=None):
    """Keep longest (or highest scoring) of overlapping spans; exact duplicates removed."""
    # full key -> the same result in every process (set order of strings varies with PYTHONHASHSEED)
    order = sorted(set(spans), key=lambda x: (-(scores or {}).get(x, 0), -(x[1] - x[0]), x[0], x[1], str(x[2])))
    kept = []
    for span in order:
        if all(span[1] <= k[0] or span[0] >= k[1] for k in kept):
            kept.append(span)
    return sorted(kept)


def label_vote(text, spans, labels_to_vote):
    """Same surface string -> majority label within a document (role consistency)."""
    votes = defaultdict(Counter)
    for s, e, label in spans:
        if label in labels_to_vote:
            votes[text[s:e]][label] += 1
    out = []
    for s, e, label in spans:
        if label in labels_to_vote and votes[text[s:e]]:
            label = votes[text[s:e]].most_common(1)[0][0]
        out.append((s, e, label))
    return out


def propagate(text, spans, labels_to_propagate, min_chars=4):
    """Annotate every other exact, word-bounded occurrence of predicted strings."""
    occupied = [(s, e) for s, e, _ in spans]
    new = list(spans)
    seen = set()
    for s, e, label in spans:
        surface = text[s:e]
        if label not in labels_to_propagate or len(surface) < min_chars or (surface, label) in seen:
            continue
        seen.add((surface, label))
        for match in re.finditer(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", text):
            a, b = match.span()
            if all(b <= x or a >= y for x, y in occupied):
                new.append((a, b, label))
                occupied.append((a, b))
    return sorted(set(new))


def propagate_ci(text, spans, labels_to_propagate, min_chars=4):
    """Like propagate, but case-insensitive and whitespace-tolerant ("RAMESH KUMAR" in the header
    also marks "Ramesh  Kumar" in the body)."""
    occupied = [(s, e) for s, e, _ in spans]
    new = list(spans)
    seen = set()
    for s, e, label in spans:
        words = text[s:e].split()
        key = (" ".join(words).lower(), label)
        if label not in labels_to_propagate or len(key[0]) < min_chars or key in seen:
            continue
        seen.add(key)
        pattern = r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"(?!\w)"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            a, b = match.span()
            if all(b <= x or a >= y for x, y in occupied):
                new.append((a, b, label))
                occupied.append((a, b))
    return sorted(set(new))


NAME_STOP = {"shri", "smt", "kumari", "late", "state", "union", "india", "government", "and", "others", "another",
             "anr", "ors", "the", "of", "son", "wife", "daughter", "through", "managing", "director", "company",
             "limited", "ltd", "private", "pvt", "corporation", "bank", "board", "municipal", "district", "officer",
             "commissioner", "collector", "secretary", "inspector", "police", "station", "court", "high", "supreme"}


def propagate_parts(text, spans, labels_to_propagate, min_chars=4):
    """Single-word mentions of predicted multi-word names (first name or surname), when that word
    belongs to one label only in this document. Only capitalised occurrences are marked."""
    owner = {}
    for s, e, label in spans:
        if label not in labels_to_propagate:
            continue
        words = re.findall(r"[A-Za-z]+", text[s:e])
        if len(words) < 2:
            continue
        for w in words:
            if len(w) >= min_chars and w.lower() not in NAME_STOP:
                owner.setdefault(w.lower(), set()).add(label)
    occupied = [(s, e) for s, e, _ in spans]
    new = list(spans)
    for word, owners in owner.items():
        if len(owners) != 1:
            continue
        label = next(iter(owners))
        for match in re.finditer(r"(?<![\w.])" + re.escape(word) + r"(?!\w)", text, re.IGNORECASE):
            a, b = match.span()
            if not text[a].isupper():
                continue
            if all(b <= x or a >= y for x, y in occupied):
                new.append((a, b, label))
                occupied.append((a, b))
    return sorted(set(new))


ROLES_RX = r"(appellants?|petitioners?|applicants?|plaintiffs?|complainants?|respondents?|defendants?|opposite\s+part(?:y|ies)|non-applicants?)"
SIDE_CUE = re.compile(r"\b(?:for|on\s+behalf\s+of)\s+(?:the\s+)?(?:[\w-]+\s+){0,2}?" + ROLES_RX + r"\b", re.IGNORECASE)
HEADER_CUE = re.compile(r"(?:\bfor\s+(?:the\s+)?|\bcounsel\s+for\s+(?:the\s+)?)?" + ROLES_RX + r"\s*(?:\(s\)|nos?\.?\s*[\d,\s&to]*)?\s*(?::-?|-\s|through\s*:)", re.IGNORECASE)
HEADER_CUE_FOR = re.compile(r"\bfor\s+(?:the\s+)?" + ROLES_RX + r"\s*(?:\(s\))?\s*:?\s*(?=(?:mr|ms|mrs|sri|shri|smt|dr)\b)",
                            re.IGNORECASE)
A_SIDE = ("appellant", "petitioner", "applicant", "plaintiff", "complainant")


def _side(word):
    word = word.lower()
    if word.startswith("non-applicant") or word.startswith(("respondent", "defendant", "opposite")):
        return "R.COUNSEL"
    return "A.COUNSEL" if word.startswith(A_SIDE) else None


def counsel_side(text, spans, window=160, back=300):
    """Re-label A.COUNSEL / R.COUNSEL from the side cue that governs the name:
    1. a header-style cue BEFORE it ("For the Appellant(s) :", "Counsel for Petitioner :-",
       "Respondent Through:") - the last such cue within 300 characters;
    2. otherwise an inline cue AFTER it in the same clause ("..., learned counsel for the first
       respondent", "appearing on behalf of the petitioner"), or "for the <party>" where the party
       is named in the header (its side comes from there)."""
    parties = {" ".join(n.split()).lower(): role for n, role in parse_parties(text).items()}
    party_rx = None
    if parties:
        alts = "|".join(r"\s+".join(map(re.escape, n.split())) for n in sorted(parties, key=len, reverse=True))
        party_rx = re.compile(r"\b(?:for|on\s+behalf\s+of)\s+(?:the\s+)?(" + alts + r")\b", re.IGNORECASE)
    out = []
    for s, e, label in spans:
        if label in ("A.COUNSEL", "R.COUNSEL"):
            side = None
            before = sorted(list(HEADER_CUE.finditer(text, max(0, s - back), s)) +
                            list(HEADER_CUE_FOR.finditer(text, max(0, s - back), s)), key=lambda m: m.start())
            if before:
                side = _side(before[-1].group(1))
            if side is None:
                limit = min(len(text), e + window)
                stop = re.search(r"\n\s*\n|;|\.\s+[A-Z]|\b(?:for|on behalf of)\s+(?:the\s+)?(?:[\w-]+\s+){0,2}?" + ROLES_RX + r"\s*(?::|through)",
                                 text[e:limit], re.IGNORECASE)
                end = e + stop.start() if stop and not SIDE_CUE.match(text, e + stop.start()) else limit
                after = SIDE_CUE.search(text, e, end) or SIDE_CUE.search(text, e, limit)
                cand = [(after.start(), _side(after.group(1)))] if after else []
                if party_rx:
                    pm = party_rx.search(text, e, limit)
                    if pm:
                        role = parties.get(" ".join(pm.group(1).split()).lower())
                        cand.append((pm.start(), "A.COUNSEL" if role == "APP" else "R.COUNSEL"))
                if cand:
                    side = min(cand)[1]
            if side:
                label = side
        out.append((s, e, label))
    return out


# --------------------------------------------------------------------------- judges
JUDGE_STOP = {"judgment", "judgement", "order", "oral", "coram", "bench", "before", "hon'ble", "honble", "hon", "the",
              "mr", "ms", "mrs", "dr", "justice", "per", "and", "sri", "shri", "smt", "date", "dated", "chief", "acting",
              "reportable", "court", "state", "act", "section", "learned", "sir", "lord", "j", "jj", "c", "cj", "with",
              "delivered", "by", "of", "in", "as", "was", "his", "lordship", "then", "speaking", "for", "through",
              "respondent", "respondents", "appellant", "appellants", "petitioner", "petitioners", "applicant",
              "applicants", "versus", "vs", "plaintiff", "defendant", "government", "union", "india", "upon",
              "circulation", "registrar", "dy", "deputy", "additional", "district", "sessions", "magistrate"}
NAME_TOKEN = re.compile(r"(?:[A-Z]\.\s?)+[A-Z][A-Za-z'\-]+|(?:[A-Z]\.\s?){1,3}(?=[\s,)])|[A-Z][A-Za-z'\-]+")


def _clean_judge(raw):
    tokens = [t for t in re.findall(r"(?:[A-Z]\.\s?)*[A-Z][A-Za-z'\-]*\.?", raw)]
    while tokens and tokens[0].rstrip(".").lower().replace(" ", "") in JUDGE_STOP:
        tokens.pop(0)
    # one case style per name: "B.R. GAVAI By Circulation" -> "B.R. GAVAI"
    words = [t for t in tokens if re.search(r"[A-Za-z]{2,}", t)]
    if words:
        upper = words[0].isupper()
        for i, t in enumerate(tokens):
            if re.search(r"[A-Za-z]{2,}", t) and t.isupper() != upper:
                tokens = tokens[:i]
                break
    while tokens and re.fullmatch(r"(?:[A-Z]\.\s?)+", tokens[-1].strip()) and len(tokens) > 1:
        tokens.pop()                                       # "S.M.SUBRAMANIAM C.M.P." -> drop trailing initials
    name = " ".join(t.strip() for t in tokens).strip(" ,.")
    name = re.sub(r"\s+", " ", name)
    words = re.findall(r"[A-Za-z]{3,}", name)
    if not words or len(name) < 3 or len(name.split()) > 5 or any(w.lower() in JUDGE_STOP for w in words):
        return None
    return name


def _names_before(text, pos, multi):
    """Name(s) written right before a ', J.' / ' JJ.' style suffix at pos."""
    chunk = text[max(0, pos - 120):pos]
    chunk = re.split(r"[:;()\[\]\n]|\.\s{2,}|[a-z]\.\s|\b(?:JUDGMENT|ORDER|CORAM|BENCH|PER|by|of|and\s+the)\b", chunk)[-1]
    parts = re.split(r"\s*(?:,|\band\b|&)\s*", chunk) if multi else [chunk]
    names = []
    for part in parts:
        part = part.strip(" ,")
        m = re.search(r"((?:(?:[A-Z]\.\s?)*[A-Z][A-Za-z'\-]+\.?\s?){1,4})$", part)
        name = _clean_judge(m.group(1)) if m else None
        if name:
            names.append(name)
    return names if multi else names[-1:]


SUFFIX = re.compile(r"\s*,?\s*\(?\b(JJ|J|C\.\s?J)\b\)?\s*\.?(?=[\s,;:)\]\-–]|$)")
JUSTICE = re.compile(r"\bJustice\s+((?:(?:[A-Z]\.\s?)*[A-Z][A-Za-z'\-]+\.?\s?){1,4})", re.IGNORECASE)
SIGNOFF = re.compile(r"\(\s*([^()]{3,60}?)\s*\)\s*(?:Judge|JUDGE)\b|\bJ\.\s*[(\[]\s*([^()\[\]]{3,60}?)\s*[)\]]")
KANOON_BENCH = re.compile(r"BENCH\s*:\s*(.{3,300}?)(?=\s*(?:BENCH\s*:|CITATION|JUDGMENT|ACT\s*:|HEADNOTE|$))", re.DOTALL)
KANOON_NAME = re.compile(r"[A-Z][A-Z'\-]+(?:\s[A-Z][A-Z'\-]+)?,\s(?:(?:[A-Z]\.\s?)+(?:[A-Z][A-Z'\-]+(?=\s*\(J\)))?"
                         r"|[A-Z][A-Z'\-]+(?:\s[A-Z][A-Z'\-]+)?(?:\s?[A-Z]\.)?)")


def judge_names(text):
    names = set()
    for m in SUFFIX.finditer(text):
        if m.start() == 0 or not text[m.start() - 1].isalpha() and text[m.start() - 1] not in " .":
            continue
        names.update(_names_before(text, m.start(), multi=m.group(1) == "JJ"))
    for m in JUSTICE.finditer(text):
        raw = re.split(r"\s+(?:AND|and|&|HON'BLE|Hon'ble|on|in|who|has|had|was|is|speaking|said|observed)\b", m.group(1))[0]
        name = _clean_judge(raw)
        if name:
            names.add(name)
    for m in SIGNOFF.finditer(text):
        name = _clean_judge(m.group(1) or m.group(2))
        if name:
            names.add(name)
    for region in KANOON_BENCH.finditer(text[:6000]):
        body = region.group(1)
        found = [x.group(0).strip() for x in KANOON_NAME.finditer(body)]
        if found:
            names.update(n.rstrip(" (") for n in found)
        else:
            for part in re.split(r"\s*(?:&|,|\band\b)\s*", body):
                name = _clean_judge(part)
                if name:
                    names.add(name)
    return {n for n in names if len(n) >= 3}


def mark_judges(text, spans):
    """Judges named by ', J.' / 'JJ.' / 'Justice X' / '(X) JUDGE' / Indian-Kanoon 'BENCH:' lists:
    every mention becomes JUDGE (overrides person-role guesses, never precedents/statutes)."""
    names = judge_names(text)
    if not names:
        return spans
    return _mark(text, spans, {n: "JUDGE" for n in names},
                 keep_labels=("PREC", "STAT", "CASENO", "DATE", "COURT", "APP", "RESP"))


# --------------------------------------------------------------------------- case numbers
YEAR_END = re.compile(r"(?:\bof\s*\d{4}|/\d{4})\b", re.IGNORECASE)


def split_casenos(text, spans):
    """A CASENO span holding two numbers ('C.M.P.No.15283 of 2019 AS.No.SR85894 of2019') is split
    after each 'of YYYY' / '/YYYY'."""
    out = []
    for s, e, label in spans:
        if label == "CASENO":
            ends = [s + m.end() for m in YEAR_END.finditer(text[s:e])]
            if len(ends) >= 2 and ends[0] < e:
                start = s
                for cut in ends + [e]:
                    piece = text[start:cut]
                    a = start + len(piece) - len(piece.lstrip(" ,&"))
                    b = start + len(piece.rstrip(" ,&"))
                    if b - a >= 4:
                        out.append((a, b, label))
                    start = cut
                continue
        out.append((s, e, label))
    return sorted(set(out))


CITATION = (r"(?:\(\d{4}\)\s*\d+\s*S\.?C\.?C\.?\s*(?:\(\w+\)\s*)?\d+"
            r"|\d{4}\s*\(\d+\)\s*S\.?C\.?C\.?\s*\d+"
            r"|A\.?I\.?R\.?\s*\d{4}\s*[A-Z][\w.]*\s*\d+"
            r"|\[\d{4}\]\s*\d*\s*S\.?C\.?R\.?\s*\d+"
            r"|\(\d{4}\)\s*\d+\s*S\.?C\.?R\.?\s*\d+"
            r"|JT\s*\d{4}\s*\(\d+\)\s*S\.?C\.?\s*\d+"
            r"|\d{4}\s*Cri\.?\s*L\.?\s*J\.?\s*\d+)")
CITATION_AFTER = re.compile(r"\s*[,:]?\s*" + CITATION + r"(?:\s*[;,=]\s*" + CITATION + r")*")
CITATION_TAIL = re.compile(r"\s*[,:]?\s*" + CITATION + r"(?:\s*[;,=]\s*" + CITATION + r")*\s*$")


def precedent_citations(text, spans, mode):
    """extend: PREC spans swallow a citation that directly follows them;
    trim: a citation at the end of a PREC span is cut off (whichever the annotation does)."""
    out = []
    for s, e, label in spans:
        if label == "PREC":
            if mode == "extend":
                match = CITATION_AFTER.match(text, e)
                if match and match.end() > e:
                    e = match.end()
            elif mode == "trim":
                match = CITATION_TAIL.search(text[s:e])
                if match and match.start() > 0:
                    e = s + match.start()
        out.append((s, e, label))
    return out


# --------------------------------------------------------------------------- parties and witnesses
# In this annotation APP / RESP are document-level ROLES: the parties named in the header (people,
# companies, the State, a Directorate ...) and every later mention of them, including a first name
# alone. WIT is a person who testified ("PW-5", "PW.3", "LW-17") and every mention of that person.
ROLE_WORD = r"(appellants?|petitioners?|applicants?|plaintiffs?|complainants?|respondents?|defendants?|opposite\s+part(?:y|ies))"
A_ROLES = ("appellant", "petitioner", "applicant", "plaintiff", "complainant")
TRAILING_ROLE = re.compile(r"(?:[.…·]{2,}|…|\s\.\s|\s-{1,2}\s|:)\s*" + ROLE_WORD + r"\s*(?:\(s\)|nos?\.?[\d,\s&and]*)?", re.IGNORECASE)
# "Surendra Kumar & Anr. APPELLANT(S) VERSUS ..." / "... Panjwani Appellant(s) versus ...": no dots before the role
TRAILING_BARE = re.compile(r"\s(APPELLANTS?|PETITIONERS?|APPLICANTS?|RESPONDENTS?)(?:\s*\(S\))?(?=\s|$)"
                           r"|\s(Appellants?|Petitioners?|Applicants?|Respondents?)\s*\(s\)")
# "APPLICANT : X, S/o ..." / "Petitioner :- X Respondent :- Y" / "APPELLANTS: - 1. X" / "BETWEEN: 1. X ... AND: Y"
LEADING_ROLE = re.compile(r"\b(APPLICANTS?|APPELLANTS?|PETITIONERS?|RESPONDENTS?|Applicants?|Appellants?|Petitioners?|"
                          r"Respondents?|BETWEEN|AND)\s*(?:\(S\)|\(s\))?\s*:\s*-?\s*(?:\d+\s*[.\]]\s*)?")
NAME_END = re.compile(r",|\(|\[|\.{2,}|…|\s(?:S/o|W/o|D/o|wd/o|d/o|s/o|Aged|aged|R/o|Residing|REP\b|Rep\.?\b|"
                      r"REPRESENTED|THROUGH|Through|HAVING|A\s+COMPANY|DEPARTMENT\s+OF|Counsel|COUNSEL|VERSUS|Versus|versus|"
                      r"Vs\.?|VS\.?|v/s|DATE\s+OF|Respondents?\s*:|RESPONDENTS?\s*:|Petitioners?\s*:|PETITIONERS?\s*:|"
                      r"AND\s*:|For\s|FOR\s|With\s|WITH\s|Mr\.|Ms\.|Shri\s|Dated|JUDGMENT|ORDER|Heard|CORAM|Coram|"
                      r"Gwalior|Present|Before|-{3,})")
# "C.O. 3346 of 2018 The Kolkata Municipal Corporation Vs. Smt. Kajari Banerjee For the petitioner ..."
SHORT_TITLE = re.compile(r"(?:\b[Oo][Ff]\s*\d{4}|/\d{4})\)?\s*(?:\([^)]{0,40}\)\s*)?(.{3,120}?)\s+(?:Vs\.?|VS\.?|vs\.?|VERSUS|Versus|v/s)\s+(.{3,160})")
SEPARATOR = re.compile(r"\bversus\b|\bvs\.?(?=\s)|\bv/s\.?|\bv\.(?=\s)|\bjudgment\b|\border\b|\bwith\b|"
                       r"\bno\.?\s*[\d/().\-]+\s*(?:of|/)\s*\d{2,4}|\bof\s+\d{4}\b|\n\s*\n|\bpetitioner\s*:|\brespondent\s*:",
                       re.IGNORECASE)
LABELLED = re.compile(r"\bPETITIONER\s*:\s*(.{2,200}?)\s+(?:(?:Vs\.?|V/s|Versus)\s+)?RESPONDENT\s*:\s*(.{2,200}?)\s+"
                      r"(?:DATE OF JUDGMENT|BENCH|$)", re.IGNORECASE | re.DOTALL)
PARTY_TAIL = re.compile(r"(?:\s*[,&]?\s*(?:and|&)\s+(?:\d+\s+)?(?:others|another|ors\.?|anr\.?)|\s*[,&]\s*(?:ors|anr)\.?|"
                        r"\s*,?\s*etc\.?|\s*\(dead\).*|\s*through.*|\s*rep(?:\.|resented)?\s+by.*|\s*\.+)\s*$", re.IGNORECASE)
PARTY_HEAD = re.compile(r"^(?:\d+\s*[.)]\s*|\[\d+\]\s*|m/s\.?\s*|smt\.?\s+|shri\s+|sri\s+|sh\.\s*|mr\.?\s+|ms\.?\s+|"
                        r"mrs\.?\s+|dr\.?\s+|the\s+)+", re.IGNORECASE)
COMPANY = re.compile(r"\b(?:ltd|limited|pvt|private|corporation|company|co\.|board|bank|trust|society|association|"
                     r"union|state|government|directorate|department|commission|authority|council|industries|works|"
                     r"municipal|india)\b", re.IGNORECASE)


def _clean_party(raw):
    name = " ".join(raw.split())
    name = name.split(" @ ")[0] if " @ " in name and len(name) > 60 else name
    for _ in range(3):
        name = PARTY_TAIL.sub("", name).strip(" ,;:-")
        name = PARTY_HEAD.sub("", name).strip(" ,;:-")
    name = name.lstrip(")]}.,;:-–— ").rstrip(" .…·").strip()
    name = re.split(r"(?:\b[Oo][Ff]\s*\d{4}|/\d{4})\)?\s*", name)[-1].strip()   # last case number wins
    if not (2 <= len(name) <= 120) or not re.search(r"[A-Za-z]{2}", name) or len(name.split()) > 14:
        return None
    if re.search(r"\b(?:appellants?|respondents?|petitioners?|applicants?|counsel|advocates?|learned|judgment|"
                 r"jurisdiction|versus)\b", name, re.IGNORECASE):
        return None
    return name


def parse_parties(text, limit=4000):
    """{party name: 'APP'|'RESP'} from the document header."""
    head = text[:limit]
    parties = {}
    for m in LABELLED.finditer(head):
        for raw, role in ((m.group(1), "APP"), (m.group(2), "RESP")):
            name = _clean_party(raw)
            if name:
                parties.setdefault(name, role)
    for m in TRAILING_ROLE.finditer(head):
        end = len(head[:m.start()].rstrip())               # name ends before the dots (maybe a line earlier)
        start = max(0, end - 220)
        for sep in SEPARATOR.finditer(head, start, end):
            start = max(start, sep.end())
        start = max(start, head.rfind("\n", 0, end) + 1)
        name = _clean_party(head[start:end])
        if name:
            parties.setdefault(name, "APP" if m.group(1).lower().startswith(A_ROLES) else "RESP")
    for m in TRAILING_BARE.finditer(head):
        word = (m.group(1) or m.group(2)).lower()
        end = m.start()
        start = max(0, end - 160)
        for sep in SEPARATOR.finditer(head, start, end):
            start = max(start, sep.end())
        name = _clean_party(head[start:end])
        if name:
            parties.setdefault(name, "APP" if word.startswith(A_ROLES) else "RESP")
    for m in LEADING_ROLE.finditer(head):
        if re.search(r"(?:for|counsel\s+for)\s+(?:the\s+)?$", head[max(0, m.start() - 16):m.start()], re.IGNORECASE):
            continue                                    # "Counsel for Petitioner :- <counsel>", not a party
        word = m.group(1).lower()
        if word in ("between", "and") and not re.search(r"BETWEEN\s*:", head[:m.start() + 10]):
            continue                                    # "AND:" only inside a "BETWEEN: ... AND:" header
        rest = head[m.end():m.end() + 200]
        cut = NAME_END.search(rest)
        name = _clean_party(rest[:cut.start()] if cut else rest[:120])
        if name:
            role = "APP" if word.startswith(A_ROLES) or word == "between" else "RESP"
            parties.setdefault(name, role)
    if not parties:
        m = SHORT_TITLE.search(head[:1500])
        if m:
            first = _clean_party(m.group(1))
            rest = m.group(2)
            cut = NAME_END.search(rest)
            second = _clean_party(rest[:cut.start()] if cut else rest[:120])
            if first:
                parties.setdefault(first, "APP")
            if second:
                parties.setdefault(second, "RESP")
    return parties


def _variants(name, first_names):
    out = {name}
    for alias in name.split(" @ "):
        alias = alias.strip()
        if len(alias) >= 3:
            out.add(alias)
    if first_names and not COMPANY.search(name):
        words = re.findall(r"[A-Za-z]+", name)
        if len(words) >= 2 and len(words[0]) >= 4 and words[0].lower() not in NAME_STOP:
            out.add(words[0])
    return out


def _mark(text, spans, names_to_label, keep_labels=("PREC", "STAT", "CASENO", "DATE", "COURT", "JUDGE")):
    """Label every occurrence of each name; overrides other labels except those in keep_labels."""
    spans = list(spans)
    for name, label in sorted(names_to_label.items(), key=lambda kv: -len(kv[0])):
        pattern = r"(?<![\w.])" + r"\s+".join(map(re.escape, name.split())) + r"(?!\w)"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            a, b = match.span()
            blocking = [x for x in spans if x[0] < b and a < x[1]]
            if any(x[2] in keep_labels for x in blocking):
                continue
            if any((x[0], x[1]) != (a, b) and x[2] == label and x[0] <= a and b <= x[1] for x in blocking):
                continue                                   # already inside a longer span of the same role
            spans = [x for x in spans if x not in blocking] + [(a, b, label)]
    return spans


def _occurs(text, name):
    pattern = r"(?<![\w.])" + r"\s+".join(map(re.escape, name.split())) + r"(?!\w)"
    return [m.span() for m in re.finditer(pattern, text, re.IGNORECASE)]


HEADER_CHARS = 2500


def mark_parties(text, spans, first_names=False, only=False):
    """Header parties -> every mention gets the party's role.

    Safeguards (test headers come in layouts the parser can misread):
      * a parsed party is used only if the model also predicted an entity (any label) on at least
        one of its mentions - a misparse ("CRIMINAL APPELLATE JURISDICTION") has no model support;
      * only=True drops APP/RESP spans that match no party, but only when trusted parties were
        found on BOTH sides, and never inside the header (first 2500 characters)."""
    parties = parse_parties(text)
    if not parties:
        return spans
    trusted = {}
    for name, role in parties.items():
        if any(a < y and x < b for a, b in _occurs(text, name) for x, y, _ in spans):
            trusted[name] = role
    if not trusted:
        return spans
    names = {}
    for name, role in trusted.items():
        for v in _variants(name, first_names):
            names.setdefault(v, role)
    spans = _mark(text, spans, names)
    if only and {"APP", "RESP"} <= set(trusted.values()):
        lowered = {" ".join(n.split()).lower() for n in names}
        spans = [x for x in spans if x[2] not in ("APP", "RESP") or x[0] < HEADER_CHARS
                 or " ".join(text[x[0]:x[1]].split()).lower() in lowered]
    return spans


WITNESS_MARK = r"(?:P\.?\s?W|L\.?\s?W|C\.?\s?W)\.?\s*[-.:]?\s*\d+"
NAME = r"((?:(?:Smt|Shri|Sri|Dr|Mr|Ms|Mrs|Sh)\.?\s+)?(?:[A-Z]\.\s*)*[A-Z][a-zA-Z]+(?:\s+(?:[A-Z]\.\s*)*[A-Z][a-zA-Z]+){0,3})"
WITNESS_AFTER = re.compile(WITNESS_MARK + r"\s*,?\s*(?:namely\s+)?" + NAME)
WITNESS_BEFORE = re.compile(NAME + r"\s*(?:\(|,|-|–)\s*" + WITNESS_MARK)


def mark_witnesses(text, spans):
    """Names next to PW/LW/CW markers are witnesses; every mention of them becomes WIT
    (party mentions stay parties)."""
    names = {}
    for rx in (WITNESS_AFTER, WITNESS_BEFORE):
        for m in rx.finditer(text):
            name = PARTY_HEAD.sub("", m.group(1)).strip()
            if len(name) >= 4 and name.split()[0].lower() not in NAME_STOP and not COMPANY.search(name):
                names[name] = "WIT"
    if not names:
        return spans
    return _mark(text, spans, names, keep_labels=("PREC", "STAT", "CASENO", "DATE", "COURT", "JUDGE", "APP", "RESP",
                                                  "A.COUNSEL", "R.COUNSEL"))


MONTHS = (r"(?:January|February|March|April|May|June|July|August|September|October|November|December"
          r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?")
PATTERNS = {
    "DATE": [r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?" + MONTHS + r",?\s+\d{4}\b",
             MONTHS + r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
             r"\b\d{1,2}[./-]\d{1,2}[./-](?:\d{4}|\d{2})\b"],
    "CASENO": [r"\b(?:Civil|Criminal|Crl\.|Cr\.|Writ|Special Leave|S\.?L\.?P\.?|Transfer|Review|Contempt)"
               r"[\w\s.()]{0,40}?(?:Petition|Appeal|Case)s?\s*(?:\(\w+\)\s*)?No[s]?\.?\s*[\d\-–/ ,&and]+?\s*of\s+\d{4}"],
}


def pattern_candidates(text, label, gazetteer=None):
    out = set()
    for pattern in PATTERNS.get(label, []):
        for match in re.finditer(pattern, text):
            out.add((match.start(), match.end(), label))
    for surface in (gazetteer or {}).get(label, ()):
        for match in re.finditer(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", text):
            out.add((match.start(), match.end(), label))
    return out


def build_gazetteer(texts, gold, labels=("STAT", "COURT", "AUTH"), min_count=2, min_chars=6):
    counts = defaultdict(Counter)
    for doc, spans in gold.items():
        for s, e, label in spans:
            if label in labels and e - s >= min_chars:
                counts[label][texts[doc][s:e]] += 1
    return {label: sorted(s for s, c in counter.items() if c >= min_count) for label, counter in counts.items()}


def postprocess(text, spans, config, boundary=None, gazetteer=None):
    spans = [tuple(s) for s in spans]
    chosen = config.get("boundary")
    if boundary is not None and chosen:
        spans = boundary(text, spans, None if chosen is True else set(chosen))   # True = all labels (old configs)
    if config.get("vote"):
        spans = label_vote(text, spans, set(config["vote"]))
    if config.get("counsel"):
        spans = counsel_side(text, spans)
    if config.get("parties"):
        modes = set(config["parties"])
        spans = mark_parties(text, spans, first_names="first" in modes, only="only" in modes)
    if config.get("witness"):
        spans = mark_witnesses(text, spans)
    if config.get("judges"):
        spans = mark_judges(text, spans)
    if config.get("caseno"):
        spans = split_casenos(text, spans)
    if config.get("propagate"):
        spans = propagate(text, spans, set(config["propagate"]))
    if config.get("propagate_ci"):
        spans = propagate_ci(text, spans, set(config["propagate_ci"]))
    if config.get("parts"):
        spans = propagate_parts(text, spans, set(config["parts"]))
    for mode in config.get("prec", []):
        spans = precedent_citations(text, spans, mode)
    for label in config.get("rules", []):
        extra = [c for c in pattern_candidates(text, label, gazetteer)
                 if all(c[1] <= s or c[0] >= e for s, e, _ in spans)]
        spans = spans + extra
    return remove_overlaps(spans)


# --------------------------------------------------------------------------- merged roles
# The tagger can be trained on PARTY (APP+RESP) and COUNSEL (A.COUNSEL+R.COUNSEL): what a party or
# counsel mention looks like is local, which side it belongs to is decided by the header and by
# cue words far from most mentions. resolve_roles() turns PARTY / COUNSEL back into the 12 labels.
COARSE = {"APP": "PARTY", "RESP": "PARTY", "A.COUNSEL": "COUNSEL", "R.COUNSEL": "COUNSEL"}
PUBLIC_BODY = re.compile(r"\b(?:state|union\s+of\s+india|government|govt|corporation|board|directorate|commission|"
                         r"custodian|authority|department|municipal|officer|collector|police|commissioner|ministry|"
                         r"bureau|cbi|revenue|council|tribunal|registrar)\b", re.IGNORECASE)
A_CUE = re.compile(r"appellants?|petitioners?|applicants?|plaintiffs?|complainants?", re.IGNORECASE)
R_CUE = re.compile(r"respondents?|defendants?|opposite\s+part(?:y|ies)|non-applicants?", re.IGNORECASE)


def coarse_labels(labels):
    out = []
    for label in labels:
        c = COARSE.get(label, label)
        if c not in out:
            out.append(c)
    return out


def coarse_gold(gold):
    return {d: [(s, e, COARSE.get(l, l)) for s, e, l in spans] for d, spans in gold.items()}


def _norm(x):
    return " ".join(re.sub(r"[^\w\s]", " ", x).split()).lower()


def _cue_role(text, s, e, width):
    left, right = text[max(0, s - width):s], text[e:e + width]
    best = None
    for rx, role in ((A_CUE, "APP"), (R_CUE, "RESP")):
        for m in rx.finditer(left):
            d = len(left) - m.end()
            best = min(best, (d, role)) if best else (d, role)
        for m in rx.finditer(right):
            d = m.start()
            best = min(best, (d, role)) if best else (d, role)
    return best[1] if best else None


def resolve_roles(text, spans):
    """PARTY -> APP / RESP and COUNSEL -> A.COUNSEL / R.COUNSEL, in this order of evidence:
    the header parties (exact, contained or shared distinctive word), cue words around the
    mention, public bodies (respondents far more often than appellants), else APP."""
    parties = {_norm(n): r for n, r in parse_parties(text).items() if _norm(n)}
    words = {}
    for n, r in parties.items():
        for w in re.findall(r"[a-z]{4,}", n):
            if w not in NAME_STOP:
                words.setdefault(w, set()).add(r)
    out = []
    for s, e, label in spans:
        if label == "PARTY":
            surface = _norm(text[s:e])
            role = parties.get(surface)
            if role is None and surface:
                hits = {r for n, r in parties.items() if surface in n or n in surface}
                role = hits.pop() if len(hits) == 1 else None
            if role is None:
                hits = set()
                for w in re.findall(r"[a-z]{4,}", surface):
                    hits |= words.get(w, set())
                role = hits.pop() if len(hits) == 1 else None
            if role is None:
                role = _cue_role(text, s, e, 25)                 # "the appellant X", "X, the respondent"
            if role is None and PUBLIC_BODY.search(text[s:e]):
                role = "RESP"                                     # public bodies are usually respondents
            if role is None:
                role = _cue_role(text, s, e, 60) or "APP"
            label = role
        elif label == "COUNSEL":
            label = "A.COUNSEL"                         # side decided by counsel_side just below
        out.append((s, e, label))
    if any(l == "COUNSEL" for *_, l in spans):
        resolved = counsel_side(text, out)
        # counsel with no cue at all: the first-listed counsel in a document is usually the appellant's
        out = resolved
    return out
