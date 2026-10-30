"""Greedy kana CTC with per-char confidence from any HF `AutoModelForCTC` (e.g. WavLM + CTC head)."""
from __future__ import annotations

import unicodedata

from ..text.normalize import kana_key
from .audio import load_audio
from .base import CTCBackend, CTCResult


class HFKanaCTC(CTCBackend):
    def __init__(self, model: str, device: str = "cuda", blank_id: int | None = None):
        import torch
        from transformers import AutoModelForCTC, AutoProcessor
        self.torch = torch
        self.device = device
        self.proc = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForCTC.from_pretrained(model).to(device).eval()
        tok = self.proc.tokenizer
        self.blank = tok.pad_token_id if blank_id is None else blank_id
        self.skip = set(tok.all_special_tokens) | {getattr(tok, "word_delimiter_token", "|") or "|"}
        self.id2tok = {i: t for t, i in tok.get_vocab().items()}

    def transcribe(self, utt) -> CTCResult:
        torch = self.torch
        wav = load_audio(utt)
        inputs = self.proc(wav, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).logits[0]
        p, ids = logits.float().softmax(-1).max(-1)
        p, ids = p.tolist(), ids.tolist()
        kana, conf = [], []
        i = 0
        while i < len(ids):  # collapse runs of identical ids
            j = i
            while j + 1 < len(ids) and ids[j + 1] == ids[i]:
                j += 1
            if ids[i] != self.blank:
                tok = self.id2tok.get(ids[i], "")
                if tok and tok not in self.skip:
                    c = sum(p[i:j + 1]) / (j - i + 1)
                    for ch in kana_key(unicodedata.normalize("NFKC", tok)):
                        kana.append(ch)
                        conf.append(c)
            i = j + 1
        return CTCResult("".join(kana), conf)
