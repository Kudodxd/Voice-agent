"""Backend interfaces. Plug your own WavLM-CTC / speech-LLM in by implementing these two classes."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..data.manifest import Utterance


@dataclass
class CTCResult:
    kana: str                      # hiragana-normalised greedy CTC output
    conf: list[float] = field(default_factory=list)  # one confidence per char of `kana`

    def window_conf(self, s: int, e: int) -> float:
        w = self.conf[s:e]
        return sum(w) / len(w) if w else 0.0


@dataclass(frozen=True)
class Hotword:
    surface: str
    reading: str | None = None
    context: str | None = None


class CTCBackend:
    def transcribe(self, utt: Utterance) -> CTCResult:
        raise NotImplementedError


class Decoder:
    """Hotword-conditioned decoder (speech LLM with prompt / concat embedding, Whisper prompt, ...)."""

    def transcribe(self, utt: Utterance, hotwords: list[Hotword], ctc: CTCResult | None) -> str:
        raise NotImplementedError


def format_hotwords(hws: list[Hotword], style: str = "list", sep: str = "、") -> str:
    """list: 表記 / with_reading: 表記(よみ) / with_context: 表記(よみ)「…文脈…」"""
    items = []
    for h in hws:
        s = h.surface
        if style in ("with_reading", "with_context") and h.reading:
            s += f"({h.reading})"
        if style == "with_context" and h.context:
            s += f"「{h.context}」"
        items.append(s)
    return sep.join(items)
