"""Whisper with hotwords in the decoder prompt (training-free prompt biasing baseline)."""
from __future__ import annotations

from .audio import load_audio
from .base import Decoder, format_hotwords


class WhisperPromptDecoder(Decoder):
    def __init__(self, model: str = "openai/whisper-medium", device: str = "cuda", dtype: str = "float16",
                 language: str = "ja", num_beams: int = 1, prompt_style: str = "list",
                 max_prompt_tokens: int = 200, max_new_tokens: int = 220):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.torch = torch
        self.device, self.dtype = device, getattr(torch, dtype)
        self.proc = WhisperProcessor.from_pretrained(model)
        self.model = WhisperForConditionalGeneration.from_pretrained(model, torch_dtype=self.dtype).to(device).eval()
        self.gen = dict(language=language, task="transcribe", num_beams=num_beams, max_new_tokens=max_new_tokens)
        self.prompt_style, self.max_prompt_tokens = prompt_style, max_prompt_tokens

    def transcribe(self, utt, hotwords, ctc=None) -> str:
        torch = self.torch
        feats = self.proc(load_audio(utt), sampling_rate=16000, return_tensors="pt").input_features
        kwargs = dict(self.gen)
        prompt = None
        if hotwords:
            prompt = format_hotwords(hotwords, self.prompt_style)
            pids = self.proc.get_prompt_ids(prompt, return_tensors="pt")
            if pids.shape[-1] > self.max_prompt_tokens:
                pids = pids[: self.max_prompt_tokens]
            kwargs["prompt_ids"] = pids.to(self.device)
        with torch.inference_mode():
            out = self.model.generate(feats.to(self.device, self.dtype), **kwargs)
        text = self.proc.batch_decode(out, skip_special_tokens=True)[0].strip()
        if prompt and text.startswith(prompt):  # some transformers versions echo the prompt
            text = text[len(prompt):].strip()
        return text
