"""Read policy: which memory/static entries are offered to the decoder for an utterance.

verify=True  -> acoustic verification: an entry is offered only if its reading is found in the
                kana-CTC hypothesis (fuzzy ratio >= min_score); ranked by match score.
verify=False -> text-only retrieval: top-k by memory priority (frequency / weight).
representation (Exp3): surface | surface_reading | surface_reading_context
  'surface' stores no reading, so verification falls back to the MeCab reading of the surface.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..asr.base import CTCResult, Hotword
from ..matching.fuzzy import best_match
from ..text.lexicon import Tokenizer
from .store import MemoryEntry


@dataclass
class Offered:
    hotword: Hotword
    score: float
    source: str      # memory | static | oracle | glossary


class Reader:
    def __init__(self, tokenizer: Tokenizer, verify: bool = True, verify_static: bool = True,
                 min_score: float = 0.8, top_k: int | None = 30, representation: str = "surface_reading",
                 slack: int = 1, min_len_fuzzy: int = 3, **_):
        self.tok, self.verify, self.verify_static = tokenizer, verify, verify_static
        self.min_score, self.top_k, self.rep = min_score, top_k, representation
        self.slack, self.mlf = slack, min_len_fuzzy

    def reading_of(self, surface: str, stored: str | None) -> str:
        if self.rep == "surface" or not stored:
            return self.tok.reading(surface)
        return stored

    def match_score(self, reading: str, ctc: CTCResult | None) -> float:
        if ctc is None or not ctc.kana or not reading:
            return 0.0
        return best_match(reading, ctc.kana, self.slack, self.mlf).score

    def _hotword(self, surface, reading, contexts) -> Hotword:
        if self.rep == "surface":
            return Hotword(surface)
        ctx = contexts[0] if (self.rep == "surface_reading_context" and contexts) else None
        return Hotword(surface, reading or None, ctx)

    def select(self, memory: list[MemoryEntry], static: list[Hotword], ctc: CTCResult | None,
               extra: list[tuple[Hotword, str]] | None = None) -> list[Offered]:
        """`extra`: unverified, always-offered lists (oracle / in-session glossary) with their source tag."""
        items = [(e.surface, e.reading, e.contexts, "memory", e.priority(), self.verify) for e in memory]
        items += [(h.surface, h.reading, [h.context] if h.context else [], "static", 0.0, self.verify_static)
                  for h in static]
        seen, verified, unverified = set(), [], []
        for surf, reading, ctxs, src, prio, need_verify in items:
            if surf in seen:
                continue
            seen.add(surf)
            r = self.reading_of(surf, reading)
            if need_verify:
                s = self.match_score(r, ctc)
                if s >= self.min_score:
                    verified.append(Offered(self._hotword(surf, r, ctxs), s, src))
            else:
                unverified.append(Offered(self._hotword(surf, r, ctxs), prio, src))
        verified.sort(key=lambda o: -o.score)
        unverified.sort(key=lambda o: -o.score)
        out = verified + unverified
        if self.top_k:
            out = out[: self.top_k]
        for h, src in extra or []:
            if h.surface not in seen:
                seen.add(h.surface)
                out.append(Offered(h, 1.0, src))
        return out
