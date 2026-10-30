"""VoiceAgent: per-session loop  CTC -> read(memory, verify) -> decode -> extract -> gate -> write."""
from __future__ import annotations

from .asr.base import CTCBackend, Decoder, Hotword
from .data.manifest import Utterance
from .memory.read_policy import Offered, Reader
from .memory.store import MemoryStore
from .memory.write_policy import CandidateExtractor, Decision, WriteGate
from .text.lexicon import RareTagger, Tokenizer
from .text.normalize import normalize_text


class VoiceAgent:
    def __init__(self, scfg: dict, ctc: CTCBackend | None, decoder: Decoder, tokenizer: Tokenizer,
                 rare: RareTagger, static_hws: list[Hotword] | None = None):
        r, w, m = scfg["read"], scfg["write"], scfg["memory"]
        self.ctc, self.decoder = ctc, decoder
        self.read_enabled = r.get("enabled", True)
        self.read_source = r.get("source", "memory")          # memory | oracle_session
        self.reader = Reader(tokenizer, **r)
        self.gate = WriteGate(**w)
        self.write_enabled = w.get("enabled", True) and self.gate.mode != "none"
        self.glossary_max = w.get("glossary_max", 50)
        self.confirm_th = w.get("confirm_threshold", r.get("min_score", 0.8))
        self.extractor = CandidateExtractor(rare, w.get("context_chars", 8), r.get("slack", 1))
        self.persist = m.get("persist_across_sessions", True)
        self.mem_cfg = m
        self.static = list(static_hws or []) if scfg.get("use_static", False) else []
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.memory = MemoryStore(**self.mem_cfg)

    def inject(self, entries: list[dict], step: int = -1) -> None:
        """Controlled poisoning (Exp2): wrong entries inserted before session 0 with plausible stats."""
        for e in entries:
            self.memory.upsert(e["surface"], e.get("reading", ""), [], step, n_occ=e.get("n_occ", 2),
                               support=e.get("support", 0.9), conf=e.get("conf", 0.8), source="poison",
                               meta={k: v for k, v in e.items() if k not in ("surface", "reading")})

    # ------------------------------------------------------------------ helpers
    def _record(self, u: Utterance, hyp: str, ctc, offered: list[Offered], **extra) -> dict:
        return {"utt_id": u.utt_id, "talk_id": u.talk_id, "ref": u.text, "hyp": hyp,
                "ctc_kana": ctc.kana if ctc else None,
                "offered": [{"surface": o.hotword.surface, "reading": o.hotword.reading,
                             "score": round(o.score, 4), "source": o.source} for o in offered], **extra}

    def _feedback(self, offered: list[Offered], hyp: str, ctc, step: int) -> None:
        for o in offered:
            if o.source == "memory":
                self.memory.mark_used(o.hotword.surface, step)
        if ctc is None:
            return
        h = normalize_text(hyp)
        for e in self.memory.values():
            if e.surface in h:
                s = self.reader.match_score(self.reader.reading_of(e.surface, e.reading), ctc)
                self.memory.feedback(e.surface, s >= self.confirm_th)

    def _commit(self, decisions: list[Decision], step: int) -> None:
        if self.write_enabled:
            for d in decisions:
                if d.accepted:
                    self.memory.upsert(d.surface, d.reading, d.contexts, step, d.n_occ, d.support, d.conf)
        self.memory.end_session(step)
        if not self.persist:
            self.memory.clear()

    # ------------------------------------------------------------------ main
    def process_session(self, utts: list[Utterance], step: int, ref_rare: set[str] | None = None,
                        oracle: list[Hotword] | None = None) -> tuple[list[dict], list[Decision]]:
        if self.gate.mode == "glossary":
            return self._glossary_session(utts, step)
        mem = self.memory.values() if (self.read_enabled and self.read_source == "memory") else []
        extra = [(h, "oracle") for h in oracle or []] if self.read_source == "oracle_session" else None
        records, cands = [], []
        for u in utts:
            ctc = self.ctc.transcribe(u) if self.ctc else None
            offered = self.reader.select(mem, self.static, ctc, extra) if self.read_enabled else []
            hyp = self.decoder.transcribe(u, [o.hotword for o in offered], ctc)
            self._feedback(offered, hyp, ctc, step)
            cands += self.extractor.extract(hyp, ctc, u.utt_id)
            records.append(self._record(u, hyp, ctc, offered))
        decisions = self.gate.decide(cands, ref_rare) if self.write_enabled else []
        self._commit(decisions, step)
        return records, decisions

    def _glossary_session(self, utts, step):
        """B2 (ASR-GLOSSARY-like): pass-1 frequent rare terms become pass-2 hotwords; no persistence."""
        ctcs, hyp1, cands = {}, {}, []
        for u in utts:
            ctcs[u.utt_id] = ctc = self.ctc.transcribe(u) if self.ctc else None
            off = self.reader.select([], self.static, ctc) if self.read_enabled else []
            hyp1[u.utt_id] = self.decoder.transcribe(u, [o.hotword for o in off], ctc)
            cands += self.extractor.extract(hyp1[u.utt_id], ctc, u.utt_id)
        decisions = self.gate.decide(cands)
        acc = sorted((d for d in decisions if d.accepted), key=lambda d: -d.n_utt)[: self.glossary_max]
        gl = [(Hotword(d.surface, None if self.reader.rep == "surface" else d.reading), "glossary") for d in acc]
        records = []
        for u in utts:
            ctc = ctcs[u.utt_id]
            off = self.reader.select([], self.static, ctc, gl)
            hyp = self.decoder.transcribe(u, [o.hotword for o in off], ctc)
            records.append(self._record(u, hyp, ctc, off, hyp_pass1=hyp1[u.utt_id]))
        self.memory.end_session(step)
        self.memory.clear()
        return records, decisions
