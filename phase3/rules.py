"""Document-structure rules applied after decoding (each kept only if it raises the honest OOF score).

Dissent. In this corpus a dissent is a separate note appended to a CCI order after the majority's
operative part ("... Secretary is directed to inform the parties accordingly."), headed by the dissenting
member ("Prasad, Member (Dissenting)", "Prasad (Minority)"), and it runs to the end of the document. A
sentence-level tagger rarely predicts it (284 training sentences in 2 documents), yet one test document
has such a note and Dissent is 1/13 of the score. The rule:
  * header found ("(dissenting)", "(minority)", "dissent note" as a heading) -> Dissent from there on;
  * otherwise a first-person disagreement with the majority ("I ... do not agree with ... the Majority")
    in the second half of an order -> Dissent from the sentence after the last closing line of the
    majority order before it ("Secretary is directed ...", "... closed under Section 26(2)");
  * everything from the start to the last sentence is Dissent, including the note's own closing line
    ("Secretary is directed to inform all concerned accordingly."): both training dissents are labelled
    that way to their final sentence.
"""
import re

HEADER = re.compile(r"\(\s*(?:dissent(?:ing)?|minority)\s*\)|^\s*(?:per\s+)?[A-Z][\w.]*(?:\s+[A-Z][\w.]*){0,3},?\s+"
                    r"(?:member|chairperson)\s*[,:-]?\s*\(?\s*dissent", re.I)
DISAGREE = re.compile(r"\b(?:I|me)\b[^.]{0,60}\b(?:do|does|did|could|can|would)\s*n[o']?t\s+(?:agree|concur)\b[^.]{0,80}"
                      r"\bmajority\b|\bI\s+(?:respectfully\s+)?dissent\b|\bunable to (?:agree|concur)\b[^.]{0,80}\bmajority\b",
                      re.I)
CLOSING = re.compile(r"secretary is directed|is hereby closed|deems it fit to close|closed under section 26\(2\)|"
                     r"communicate the decision|inform (?:the parties|all concerned|the informant)", re.I)


def dissent_start(sentences, min_position=0.25):
    n = len(sentences)
    for i, s in enumerate(sentences):
        if i >= min_position * n and HEADER.search(s):
            return i
    for i, s in enumerate(sentences):
        if i >= 0.4 * n and DISAGREE.search(s):
            start = i
            for j in range(i - 1, max(-1, i - 200), -1):     # back to the end of the majority order
                if CLOSING.search(sentences[j]):
                    start = j + 1
                    break
            return start
    return None


def apply_dissent(labels, sentences):
    start = dissent_start(sentences)
    if start is None:
        return list(labels)
    out = list(labels)
    for i in range(start, len(out)):
        out[i] = "Dissent"
    return out


# --------------------------------------------------------------------------- speaker tracking
# Who is arguing is stated once ("Shri Aney ... appearing on behalf of the Competition Commission",
# "Dr. Sathe ... supported the order passed by the Commission") and then carried by "According to him ..."
# for dozens of sentences. Taggers lose it: on the largest test judgment every model labels a 57-sentence
# block of the respondents' (CCI's and RJIL's) arguments ArgumentPetitioner. This rule tracks the speaker:
# a side cue sets the current side, every sentence the MODEL already labelled as an argument takes that side,
# and the side is forgotten after 3 non-argument sentences. It never creates or removes argument labels.
# Cue precision on the 90 training documents (side of the next argument sentence): petitioner 90%,
# respondent 88%. Parties whose side varies (the Revenue in tax appeals; the Commission in its own orders and
# in Supreme Court appeals) are not used as cues; the Commission / CCI / DG count as respondent only in High
# Court writs and COMPAT appeals against CCI orders.
#   speaker      explicit counsel introductions + "supported the impugned order" + "per contra / on the
#                other hand" (on gold labels it changes 5% of argument sentences; on labels where every
#                respondent argument was called petitioner it lifts ArgumentRespondent F1 0 -> 0.49)
#   speaker_all  also "<party> has submitted / contended / alleged ..." (more recovery, 0 -> 0.60, but changes
#                12% of correct labels)
# Like every rule, a run keeps one only if it raises the honest OOF score of that run's own predictions.
AP, AR = "ArgumentPetitioner", "ArgumentRespondent"
_RSIDE = r"(?:respondents?|respondent\s*(?:no\.?|number)\s*\d+|opposite\s+part(?:y|ies)|OPs?(?:\s*-?\s*\d+)?)"
_RSIDE_CL = r"(?:respondents?|respondent\s*(?:no\.?|number)\s*\d+|opposite\s+part(?:y|ies)|OPs?(?:\s*-?\s*\d+)?|(?:competition\s+)?commission(?:\s+of\s+india)?|CCI|DG|director\s+general)"
_PSIDE = r"(?:petitioners?|appellants?|applicants?|informants?|complainants?|plaintiffs?)"
_VERBS = (r"(?:alleged|submitted|contended|argued|urged|stated|replied|denied|objected|countered|refuted|pleaded|"
          r"prayed|canvassed|asserted|maintained|claimed)")
_INTRO = (r"\b(?:(?:counsel|advocates?|solicitor(?:\s+general)?|A\.?S\.?G\.?|D\.?R\.?|appearing|appeared|represent(?:ing|ed|s))"
          r"\b[^.;]{0,40}?\b(?:for|on\s+behalf\s+of)|on\s+behalf\s+of)\s+(?:the\s+)?")
_FLIP = re.compile(r"^\W*(?:\d+\s*)?(?:per\s+contra|on\s+the\s+other\s+hand|in\s+rebuttal|countering)\b", re.I)
_SUPPORT = re.compile(r"\bsupported\s+the\s+(?:impugned\s+)?(?:order|judgment|decision|finding)", re.I)
_PATTERNS = {}


def _patterns(doc_id, broad):
    key = (doc_id.split("_")[0] in ("HC", "COM"), broad)
    if key not in _PATTERNS:
        r_party = _RSIDE_CL if key[0] else _RSIDE
        r = [re.compile(_INTRO + r_party + r"\b", re.I), _SUPPORT]
        p = [re.compile(_INTRO + _PSIDE + r"\b", re.I)]
        if broad:
            r.append(re.compile(rf"\b{r_party}\b[^.]{{0,40}}\b(?:has\s+|have\s+|had\s+)?{_VERBS}\b", re.I))
            p += [re.compile(rf"\b{_PSIDE}\b[^.]{{0,40}}\b(?:has\s+|have\s+|had\s+)?{_VERBS}\b", re.I),
                  re.compile(r"\b(?:it\s+is|it\s+has\s+been|has\s+been)\s+alleged\b", re.I)]
        _PATTERNS[key] = (r, p)
    return _PATTERNS[key]


_OBJECT = re.compile(
    r"(?:\b(?:of|to|with|against)\s+(?:the\s+)?(?:learned\s+)?(?:senior\s+)?(?:(?:counsel|advocates?)\s*)?"
    r"|\b(?:canvassed|advanced|raised|made|urged|put\s+forth|relied\s+upon|cited|referred\s+to|argued|submitted)\s+by\s+"
    r"(?:the\s+)?(?:learned\s+)?(?:senior\s+)?(?:(?:counsel|advocates?)\s*)?"
    r"|\b(?:contentions?|arguments?|submissions?|pleas?|propositions?|case|stand|grievances?)(?:\s+\w+){0,2}\s*)$", re.I)


def _speaks(patterns, sentence):
    """A cue counts only when its counsel is the speaker: '... canvassed by the learned counsel for the
    petitioner' or 'the argument of the counsel for the petitioners' quotes the other side."""
    for x in patterns:
        for m in x.finditer(sentence):
            if not _OBJECT.search(sentence[max(0, m.start() - 60):m.start()]):
                return True
    return False


def speaker_side(sentence, doc_id="", broad=False):
    """'R' / 'P' / 'flip' / None for one sentence."""
    r, p = _patterns(doc_id, broad)
    hit_r, hit_p = _speaks(r, sentence), _speaks(p, sentence)
    if hit_r != hit_p:
        return "R" if hit_r else "P"
    if not hit_r and _FLIP.search(sentence):
        return "flip"
    return None


def apply_speaker(labels, sentences, doc_id="", broad=False, reset_gap=3):
    out, state, gap = list(labels), None, 0
    for i, sentence in enumerate(sentences):
        side = speaker_side(sentence, doc_id, broad)
        if side == "flip":
            side = {"P": "R", "R": "P"}.get(state)
        if side:
            state, gap = side, 0
        if out[i] in (AP, AR):
            gap = 0
            if state:
                out[i] = AP if state == "P" else AR
        else:
            gap += 1
            if gap >= reset_gap:
                state = None
    return out


RULES = {"dissent": lambda labels, sentences, doc_id="": apply_dissent(labels, sentences),
         "speaker": lambda labels, sentences, doc_id="": apply_speaker(labels, sentences, doc_id, broad=False),
         "speaker_all": lambda labels, sentences, doc_id="": apply_speaker(labels, sentences, doc_id, broad=True)}
TEXT_RULES = {"dissent"}                 # depend only on the text: imposed while offsets are tuned (forced labels)
VARIANTS = [[], ["dissent"], ["dissent", "speaker"], ["dissent", "speaker_all"]]    # tried by rules="auto"


def apply_rules(pred, docs, rules):
    """pred {doc: [labels]}, docs {doc: {"sentences": [...]}} -> new dict (rules in the given order)."""
    out = {}
    for doc, labels in pred.items():
        for name in rules:
            labels = RULES[name](labels, docs[doc]["sentences"], doc)
        out[doc] = labels
    return out


_CODE = {None: 0, "P": 1, "R": 2, "flip": 3}


def _speaker_on_indices(idx, cues, ap, ar, reset_gap):
    """apply_speaker on label indices with precomputed cue codes (0 none, 1 P, 2 R, 3 flip)."""
    out = idx.copy()
    state, gap = 0, 0
    for i in range(len(idx)):
        side = cues[i]
        if side == 3:
            side = 2 if state == 1 else (1 if state == 2 else 0)
        if side:
            state, gap = side, 0
        if out[i] == ap or out[i] == ar:
            gap = 0
            if state:
                out[i] = ap if state == 1 else ar
        else:
            gap += 1
            if gap >= reset_gap:
                state = 0
    return out


try:                                       # decoding re-runs the rule thousands of times while tuning offsets
    from numba import njit
    _speaker_on_indices = njit(cache=True)(_speaker_on_indices)
except ImportError:
    pass


def post_rule(rules, sentences, doc_id, labels):
    """index-array function applying the label-dependent rules (speaker*), or None. The text cues are found once
    per document; each call only runs the state machine over the label indices."""
    import numpy as np
    names = [r for r in rules if r not in TEXT_RULES]
    if not names:
        return None
    unknown = [r for r in names if r not in ("speaker", "speaker_all")]
    if unknown:
        raise ValueError(f"no index form for rules {unknown}")
    ap, ar = labels.index(AP), labels.index(AR)
    cue_sets = [np.array([_CODE[speaker_side(s, doc_id, broad=name == "speaker_all")] for s in sentences], dtype=np.int64)
                for name in names]

    def run(idx):
        idx = np.asarray(idx, dtype=np.int64)
        for cues in cue_sets:
            idx = _speaker_on_indices(idx, cues, ap, ar, 3)
        return idx
    return run


def forced_indices(sentences, rules, index):
    """Per-sentence label index a rule set imposes, -1 where it leaves the model's choice. Rules here
    depend only on the text, so offsets can be tuned on top of them (decode.choose_decoding)."""
    import numpy as np
    out = np.full(len(sentences), -1, dtype=np.int64)
    for name in rules:
        if name not in TEXT_RULES:
            continue
        for i, label in enumerate(RULES[name]([None] * len(sentences), sentences)):
            if label is not None:
                out[i] = index[label]
    return out
