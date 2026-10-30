"""JSONL caches so that CTC (hotword-independent) and decoder outputs are reused across systems."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .base import CTCBackend, CTCResult, Decoder


def _load(path: Path) -> dict:
    d = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                d[rec["key"]] = rec["val"]
    return d


class _Cache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.d = _load(self.path)
        self.f = open(self.path, "a", encoding="utf-8")

    def put(self, key, val):
        self.d[key] = val
        self.f.write(json.dumps({"key": key, "val": val}, ensure_ascii=False) + "\n")
        self.f.flush()


class CachedCTC(CTCBackend):
    def __init__(self, inner: CTCBackend, path):
        self.inner, self.c = inner, _Cache(path)

    def transcribe(self, utt) -> CTCResult:
        if utt.utt_id not in self.c.d:
            self.c.put(utt.utt_id, asdict(self.inner.transcribe(utt)))
        return CTCResult(**self.c.d[utt.utt_id])


class CachedDecoder(Decoder):
    def __init__(self, inner: Decoder, path):
        self.inner, self.c = inner, _Cache(path)

    def transcribe(self, utt, hotwords, ctc=None) -> str:
        sig = json.dumps([[h.surface, h.reading, h.context] for h in hotwords], ensure_ascii=False)
        key = f"{utt.utt_id}|{hashlib.sha1(sig.encode()).hexdigest()[:16]}"
        if key not in self.c.d:
            self.c.put(key, self.inner.transcribe(utt, hotwords, ctc))
        return self.c.d[key]
