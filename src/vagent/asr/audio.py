from __future__ import annotations

import numpy as np

from ..data.manifest import Utterance


def load_audio(utt: Utterance, sr: int = 16000) -> np.ndarray:
    import soundfile as sf
    if not utt.wav:
        raise ValueError(f"{utt.utt_id}: no wav path in manifest")
    file_sr = sf.info(utt.wav).samplerate
    start = int(round((utt.start or 0.0) * file_sr))
    stop = int(round(utt.end * file_sr)) if utt.end is not None else None
    wav, _ = sf.read(utt.wav, start=start, stop=stop, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if file_sr != sr:
        import torch
        import torchaudio.functional as AF
        wav = AF.resample(torch.from_numpy(wav), file_sr, sr).numpy()
    return wav
