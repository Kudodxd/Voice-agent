from __future__ import annotations

import importlib
from pathlib import Path

from .base import CTCBackend, Decoder
from .cache import CachedCTC, CachedDecoder


def load_object(spec: str):
    mod, _, name = spec.partition(":")
    return getattr(importlib.import_module(mod), name)


def build_ctc(bcfg: dict, tokenizer, cache_dir: str | None) -> CTCBackend | None:
    t = bcfg.get("type", "none")
    kw = bcfg.get("kwargs", {}) or {}
    if t == "none":
        return None
    if t == "mock":
        from .mock import MockCTC
        inner = MockCTC(tokenizer, **kw)
    elif t == "hf":
        from .ctc_hf import HFKanaCTC
        inner = HFKanaCTC(**kw)
    elif t == "external":
        inner = load_object(bcfg["class"])(**kw)
    else:
        raise ValueError(t)
    tag = bcfg.get("cache_tag")
    return CachedCTC(inner, Path(cache_dir) / f"ctc_{tag}.jsonl") if cache_dir and tag else inner


def build_decoder(bcfg: dict, rare_tagger, cache_dir: str | None) -> Decoder:
    t = bcfg["type"]
    kw = bcfg.get("kwargs", {}) or {}
    if t == "mock":
        from .mock import MockDecoder
        inner = MockDecoder(rare_tagger, **kw)
    elif t == "whisper_prompt":
        from .whisper_prompt import WhisperPromptDecoder
        inner = WhisperPromptDecoder(**kw)
    elif t == "external":
        inner = load_object(bcfg["class"])(**kw)
    else:
        raise ValueError(t)
    tag = bcfg.get("cache_tag")
    return CachedDecoder(inner, Path(cache_dir) / f"dec_{tag}.jsonl") if cache_dir and tag else inner
