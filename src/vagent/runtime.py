"""Shared builders used by the scripts."""
from __future__ import annotations

import csv
from pathlib import Path

from .asr.base import Hotword
from .asr.factory import build_ctc, build_decoder
from .text.lexicon import RareTagger, Tokenizer, build_rare_tagger


def build_text(cfg: dict) -> tuple[Tokenizer, RareTagger]:
    tok = Tokenizer(cfg.get("tokenizer", {}).get("mecab_args", ""))
    return tok, build_rare_tagger(cfg, tok)


def build_backends(cfg: dict, tok: Tokenizer, rare: RareTagger):
    b = cfg["backends"]
    cache = cfg.get("paths", {}).get("cache_dir")
    return build_ctc(b.get("ctc", {"type": "none"}), tok, cache), build_decoder(b["decoder"], rare, cache)


def load_static_list(path: str | None, tok: Tokenizer) -> list[Hotword]:
    """TSV: surface[<TAB>reading]. Missing readings are filled with the MeCab reading."""
    if not path or not Path(path).exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if row and row[0].strip() and not row[0].startswith("#"):
                s = row[0].strip()
                r = row[1].strip() if len(row) > 1 and row[1].strip() else tok.reading(s)
                out.append(Hotword(s, r))
    return out
