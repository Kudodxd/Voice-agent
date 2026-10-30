"""Fuzzy substring search over unsegmented kana (no word boundaries).

best `Levenshtein.ratio` over sliding windows of length len(needle)±slack in the haystack.
"""
from __future__ import annotations

from dataclasses import dataclass

import Levenshtein


@dataclass
class Match:
    score: float
    start: int
    end: int


def best_match(needle: str, hay: str, slack: int = 1, min_len_fuzzy: int = 3) -> Match:
    if not needle or not hay:
        return Match(0.0, 0, 0)
    i = hay.find(needle)
    if i >= 0:
        return Match(1.0, i, i + len(needle))
    if len(needle) < min_len_fuzzy:  # short readings: exact only (fuzzy is meaningless)
        return Match(0.0, 0, 0)
    n, H = len(needle), len(hay)
    best = Match(0.0, 0, 0)
    for L in range(max(1, n - slack), n + slack + 1):
        if L > H:
            r = Levenshtein.ratio(needle, hay)
            if r > best.score:
                best = Match(r, 0, H)
            continue
        for s in range(H - L + 1):
            r = Levenshtein.ratio(needle, hay[s:s + L])
            if r > best.score:
                best = Match(r, s, s + L)
    return best
