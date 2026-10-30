"""Reference-based mock backends. ONLY for sanity-checking the pipeline end to end (no audio needed).

Simulated phenomena: kana CTC noise with low confidence on corrupted chars; rare words
misrecognised either as phonetic katakana (acoustically right, wrong surface -> dangerous for
memory) or as a garbled variant (acoustically wrong); offered hotwords fix rare words;
irrelevant hotwords occasionally get inserted (over-biasing).
"""
from __future__ import annotations

import hashlib
import random

from ..text.lexicon import RareTagger, Tokenizer
from ..text.normalize import hira2kata, normalize_text
from .base import CTCBackend, CTCResult, Decoder

_HIRA = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわん"
_GARBLE = "会解回界化下価課科可期機記気器生性成制製"


def _rng(*keys) -> random.Random:
    h = hashlib.sha1("|".join(map(str, keys)).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


class MockCTC(CTCBackend):
    def __init__(self, tokenizer: Tokenizer, err_rate: float = 0.1, seed: int = 0):
        self.tok, self.err, self.seed = tokenizer, err_rate, seed

    def transcribe(self, utt) -> CTCResult:
        r = _rng(self.seed, "ctc", utt.utt_id)
        kana, conf = [], []
        for ch in self.tok.reading(normalize_text(utt.text)):
            if r.random() < self.err:
                kana.append(r.choice(_HIRA)); conf.append(r.uniform(0.2, 0.6))
            else:
                kana.append(ch); conf.append(r.uniform(0.7, 1.0))
        return CTCResult("".join(kana), conf)


class MockDecoder(Decoder):
    def __init__(self, rare: RareTagger, p_base: float = 0.3, p_fix: float = 0.9, p_false: float = 0.03,
                 p_katakana: float = 0.5, seed: int = 0):
        self.rare, self.p_base, self.p_fix, self.p_false, self.p_kata, self.seed = rare, p_base, p_fix, p_false, p_katakana, seed

    def transcribe(self, utt, hotwords, ctc=None) -> str:
        offered = {h.surface for h in hotwords}
        r = _rng(self.seed, "dec", utt.utt_id, "、".join(sorted(offered)))
        ref = normalize_text(utt.text)
        out, cur = [], 0
        for sp in self.rare.rare_spans(ref):
            out.append(ref[cur:sp.start])
            p_ok = self.p_fix if sp.surface in offered else self.p_base
            # error type is a property of the word (deterministic), not of the trial
            kata = _rng(self.seed, "type", sp.surface).random() < self.p_kata
            if r.random() < p_ok:
                out.append(sp.surface)
            elif kata:
                out.append(hira2kata(sp.reading))
            else:
                g = list(sp.surface)
                g[_rng(self.seed, "pos", sp.surface).randrange(len(g))] = _rng(self.seed, "ch", sp.surface).choice(_GARBLE)
                out.append("".join(g))
            cur = sp.end
        out.append(ref[cur:])
        hyp = "".join(out)
        for h in hotwords:  # over-biasing: irrelevant hotword inserted
            if h.surface not in ref and r.random() < self.p_false:
                pos = r.randrange(len(hyp) + 1)
                hyp = hyp[:pos] + h.surface + hyp[pos:]
        return hyp
