"""Template for plugging in your own pipeline (WavLM-CTC filtering + LLM with hotword prompt /
concat embedding). Point `backends.decoder.class` to e.g. `mypkg.speechllm:MySpeechLLMDecoder`.

The agent hands you the already selected (and, for verified reads, acoustically checked) hotwords.
Keep your own CTC filtering inside the decoder if you want the original pipeline unchanged; in
that case set `agent.read.verify: false` for B1 so filtering is not applied twice.
"""
from __future__ import annotations

from .base import CTCResult, Decoder, Hotword, format_hotwords


class MySpeechLLMDecoder(Decoder):
    def __init__(self, ckpt: str, device: str = "cuda", prompt_style: str = "with_reading"):
        self.prompt_style = prompt_style
        # self.model = load_your_model(ckpt).to(device).eval()
        raise NotImplementedError("load your Whisper/WavLM encoder + projector + LLM here")

    def transcribe(self, utt, hotwords: list[Hotword], ctc: CTCResult | None) -> str:
        prompt = format_hotwords(hotwords, self.prompt_style)  # e.g. inserted into the LLM prompt
        # audio = vagent.asr.audio.load_audio(utt)
        # return self.model.generate(audio, hotword_prompt=prompt, hotword_embeds=...)
        raise NotImplementedError
