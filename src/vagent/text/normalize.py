from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")


def strip_pos(text: str, drop_pos: tuple[str, ...] | list[str] | None = None) -> str:
    """Kaldi-CSJ style `えー+感動詞 今日+名詞` -> `えー今日` (joined, no spaces).

    Tokens whose POS starts with any of `drop_pos` (e.g. 感動詞) are removed. Tokens without
    a `+POS` suffix are kept as-is. Check your own `text` format before relying on this.
    """
    out = []
    for tok in text.split():
        if "+" in tok:
            w, _, pos = tok.partition("+")
            if drop_pos and any(pos.startswith(p) for p in drop_pos):
                continue
        else:
            w = tok
        out.append(w)
    return "".join(out)


def _is_punct(c: str) -> bool:
    if c == "ー":
        return False
    cat = unicodedata.category(c)
    return cat.startswith("P") or cat.startswith("S")


def normalize_text(s: str, remove_punct: bool = True) -> str:
    """Scoring/matching normalisation: NFKC, drop whitespace and (optionally) punctuation/symbols."""
    s = unicodedata.normalize("NFKC", s)
    s = _WS.sub("", s)
    if remove_punct:
        s = "".join(c for c in s if not _is_punct(c))
    return s


def kata2hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in s)


def hira2kata(s: str) -> str:
    return "".join(chr(ord(c) + 0x60) if 0x3041 <= ord(c) <= 0x3096 else c for c in s)


def is_hira(c: str) -> bool:
    return 0x3041 <= ord(c) <= 0x3096 or c == "ー"


def kana_key(s: str) -> str:
    """Reading key used for all kana matching: NFKC -> hiragana, keep only hiragana + ー."""
    s = kata2hira(unicodedata.normalize("NFKC", s))
    return "".join(c for c in s if is_hira(c))


def is_hira_only(s: str) -> bool:
    return bool(s) and all(is_hira(c) for c in s)
