"""Label-free write policy: candidate extraction + gate (RQ1 / Exp1).

Signals per candidate (rare unit in the LLM hypothesis):
  support : best fuzzy-match ratio of its reading against the kana-CTC hypothesis
            (acoustic anchor; `1 - support` is the CTC/LLM disagreement signal)
  conf    : mean CTC confidence over the matched window
  repeat  : number of distinct utterances in the session containing it
All decisions (accepted and rejected) are logged with their features, so threshold sweeps for the
precision/recall curve can be done offline (scripts/sweep_write_gate.py).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from ..asr.base import CTCResult
from ..matching.fuzzy import best_match
from ..text.lexicon import RareTagger
from ..text.normalize import normalize_text


@dataclass
class Candidate:
    surface: str
    reading: str        # dictionary (MeCab) reading
    ctc_reading: str    # kana of the best-matching CTC window (acoustic reading)
    context: str
    utt_id: str
    support: float
    conf: float


@dataclass
class Decision:
    surface: str
    reading: str
    contexts: list
    n_utt: int
    n_occ: int
    support: float
    conf: float
    accepted: bool
    reason: str

    def to_dict(self):
        return asdict(self)


class CandidateExtractor:
    def __init__(self, rare: RareTagger, context_chars: int = 8, slack: int = 1, min_len_fuzzy: int = 3):
        self.rare, self.cc, self.slack, self.mlf = rare, context_chars, slack, min_len_fuzzy

    def extract(self, hyp: str, ctc: CTCResult | None, utt_id: str) -> list[Candidate]:
        text = normalize_text(hyp)
        out = []
        for sp in self.rare.rare_spans(text):
            sup = conf = 0.0
            ctc_reading = ""
            if ctc is not None and ctc.kana and sp.reading:
                m = best_match(sp.reading, ctc.kana, self.slack, self.mlf)
                sup, conf, ctc_reading = m.score, ctc.window_conf(m.start, m.end), ctc.kana[m.start:m.end]
            ctx = text[max(0, sp.start - self.cc): sp.end + self.cc]
            out.append(Candidate(sp.surface, sp.reading, ctc_reading, ctx, utt_id, sup, conf))
        return out


class WriteGate:
    MODES = ("none", "text_freq", "glossary", "acoustic", "label")

    def __init__(self, mode: str = "acoustic", use_support: bool = True, use_conf: bool = True,
                 use_repeat: bool = True, min_support: float = 0.85, min_conf: float = 0.6,
                 min_repeat: int = 2, agg: str = "mean", reading_source: str = "ctc", **_):
        assert mode in self.MODES, mode
        self.mode, self.use_support, self.use_conf, self.use_repeat = mode, use_support, use_conf, use_repeat
        self.min_support, self.min_conf, self.min_repeat = min_support, min_conf, min_repeat
        self.agg, self.reading_source = agg, reading_source

    def _agg(self, xs):
        return max(xs) if self.agg == "max" else sum(xs) / len(xs)

    def decide(self, cands: list[Candidate], session_rare_refs: set[str] | None = None) -> list[Decision]:
        by: dict[str, list[Candidate]] = {}
        for c in cands:
            by.setdefault(c.surface, []).append(c)
        out = []
        for surf, cs in by.items():
            n_utt = len({c.utt_id for c in cs})
            sup, conf = self._agg([c.support for c in cs]), self._agg([c.conf for c in cs])
            if self.reading_source == "ctc":
                rs = Counter(c.ctc_reading for c in cs if c.ctc_reading and c.support >= self.min_support)
                reading = rs.most_common(1)[0][0] if rs else cs[0].reading
            else:
                reading = cs[0].reading
            contexts = list(dict.fromkeys(c.context for c in cs))
            ok, why = self._accept(surf, n_utt, sup, conf, session_rare_refs)
            out.append(Decision(surf, reading, contexts, n_utt, len(cs), sup, conf, ok, why))
        return out

    def _accept(self, surf, n_utt, sup, conf, refs):
        m = self.mode
        if m == "none":
            return False, "disabled"
        if m in ("text_freq", "glossary"):
            return (n_utt >= self.min_repeat), "freq"
        if m == "label":
            if refs is None:
                raise ValueError("label gate needs session reference rare words")
            return (surf in refs), "label"
        fails = []
        if self.use_support and sup < self.min_support:
            fails.append("support")
        if self.use_conf and conf < self.min_conf:
            fails.append("conf")
        if self.use_repeat and n_utt < self.min_repeat:
            fails.append("repeat")
        return (not fails), ("ok" if not fails else "fail:" + "+".join(fails))
