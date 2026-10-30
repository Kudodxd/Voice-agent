"""Tokenisation, readings and the rare-word definition (shared by eval and candidate extraction)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .normalize import is_hira_only, kana_key


@dataclass
class Span:
    surface: str
    reading: str
    start: int
    end: int
    pos: str = ""


class Tokenizer:
    """fugashi + UniDic (unidic-lite by default). Readings are hiragana via `kana_key`."""

    def __init__(self, mecab_args: str = ""):
        import fugashi
        self.tagger = fugashi.Tagger(mecab_args) if mecab_args else fugashi.Tagger()
        self.reading = lru_cache(maxsize=200_000)(self._reading)

    def tokenize(self, text: str) -> list[Span]:
        out, cur = [], 0
        for w in self.tagger(text):
            s = w.surface
            i = text.find(s, cur)
            i = cur if i < 0 else i
            f = w.feature
            pos1 = getattr(f, "pos1", None) or ""
            pos2 = getattr(f, "pos2", None) or ""
            kana = getattr(f, "kana", None)
            reading = kana_key(kana) if kana and kana != "*" else kana_key(s)
            out.append(Span(s, reading, i, i + len(s), f"{pos1}-{pos2}"))
            cur = i + len(s)
        return out

    def _reading(self, text: str) -> str:
        return "".join(sp.reading for sp in self.tokenize(text))


class RareTagger:
    """Content units (optionally merged noun compounds) not in the common-vocab set = rare words.

    `common` is the top-N most frequent units of the CSJ *train* text (scripts/build_rare_vocab.py),
    built with the same unit definition, so eval and extraction stay consistent.
    """

    def __init__(self, tokenizer: Tokenizer, common: set[str] | None = None, content_pos=("名詞",),
                 exclude_pos2=("数詞",), merge_pos=("名詞", "接頭辞", "接尾辞"), merge_compounds=True,
                 min_chars=2, exclude_hiragana_only=True):
        self.tok = tokenizer
        self.common = common or set()
        self.content_pos = tuple(content_pos)
        self.exclude_pos2 = tuple(exclude_pos2)
        self.merge_pos = tuple(merge_pos)
        self.merge_compounds = merge_compounds
        self.min_chars = min_chars
        self.exclude_hiragana_only = exclude_hiragana_only

    def _is_content(self, sp: Span) -> bool:
        p1, _, p2 = sp.pos.partition("-")
        return p1 in self.content_pos and p2 not in self.exclude_pos2

    def units(self, text: str) -> list[Span]:
        toks = self.tok.tokenize(text)
        units: list[Span] = []
        if not self.merge_compounds:
            return [t for t in toks if self._is_content(t)]
        group: list[Span] = []

        def flush():
            if group and any(self._is_content(t) for t in group):
                # strip leading suffixes / trailing prefixes
                g = list(group)
                while g and g[0].pos.startswith("接尾辞"):
                    g.pop(0)
                while g and g[-1].pos.startswith("接頭辞"):
                    g.pop()
                if g and any(self._is_content(t) for t in g):
                    units.append(Span("".join(t.surface for t in g), "".join(t.reading for t in g),
                                      g[0].start, g[-1].end, "compound" if len(g) > 1 else g[0].pos))
            group.clear()

        for t in toks:
            p1 = t.pos.partition("-")[0]
            mergeable = p1 in self.merge_pos and not t.pos.endswith(self.exclude_pos2)
            if mergeable and (not group or group[-1].end == t.start):
                group.append(t)
            else:
                flush()
                if mergeable:
                    group.append(t)
        flush()
        return units

    def is_rare(self, surface: str) -> bool:
        if len(surface) < self.min_chars or surface in self.common:
            return False
        return not (self.exclude_hiragana_only and is_hira_only(surface))

    def rare_spans(self, text: str) -> list[Span]:
        return [u for u in self.units(text) if self.is_rare(u.surface)]


def count_units(tagger: RareTagger, texts) -> tuple[Counter, dict[str, Counter]]:
    counts: Counter = Counter()
    readings: dict[str, Counter] = {}
    for t in texts:
        for u in tagger.units(t):
            counts[u.surface] += 1
            readings.setdefault(u.surface, Counter())[u.reading] += 1
    return counts, readings


def load_common(path: str | Path | None) -> set[str]:
    if not path or not Path(path).exists():
        return set()
    return {l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()}


def build_rare_tagger(cfg: dict, tokenizer: Tokenizer | None = None) -> RareTagger:
    rc = cfg["rare"]
    tok = tokenizer or Tokenizer(cfg.get("tokenizer", {}).get("mecab_args", ""))
    return RareTagger(tok, load_common(rc.get("common_path")),
                      content_pos=rc.get("content_pos", ["名詞"]),
                      exclude_pos2=rc.get("exclude_pos2", ["数詞"]),
                      merge_pos=rc.get("merge_pos", ["名詞", "接頭辞", "接尾辞"]),
                      merge_compounds=rc.get("merge_compounds", True),
                      min_chars=rc.get("min_chars", 2),
                      exclude_hiragana_only=rc.get("exclude_hiragana_only", True))
