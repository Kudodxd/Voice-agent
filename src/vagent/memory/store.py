"""Cross-session rare-word memory with capacity / forgetting policies."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class MemoryEntry:
    surface: str
    reading: str = ""
    contexts: list[str] = field(default_factory=list)
    source: str = "auto"          # auto | poison  (analysis only; the agent never reads it)
    created_step: int = 0
    last_seen_step: int = 0       # last session in which it was (re)written
    last_used_step: int = -1      # last session in which it was offered to the decoder
    n_sessions: int = 0
    n_occ: int = 0
    support: float = 0.0          # running mean of CTC support at write time
    conf: float = 0.0             # running mean of CTC confidence at write time
    weight: float = 1.0
    confirm: int = 0              # emitted in a hypothesis AND acoustically supported
    contradict: int = 0           # emitted in a hypothesis but NOT acoustically supported
    meta: dict = field(default_factory=dict)

    def evidence(self, lam: float = 1.0) -> float:
        return self.support * math.log1p(self.n_occ) + self.confirm - lam * self.contradict

    def priority(self) -> float:
        return self.weight * (1.0 + math.log1p(self.n_occ))


class MemoryStore:
    def __init__(self, policy: str = "unbounded", max_size: int | None = None, decay_gamma: float = 0.8,
                 min_weight: float = 0.2, prune_on_contradiction: bool = False, contradiction_margin: int = 2,
                 evidence_lambda: float = 1.0, max_contexts: int = 3, **_):
        assert policy in ("unbounded", "lru", "decay", "confidence"), policy
        self.policy, self.max_size = policy, max_size
        self.gamma, self.min_weight = decay_gamma, min_weight
        self.prune_contra, self.margin = prune_on_contradiction, contradiction_margin
        self.lam, self.max_contexts = evidence_lambda, max_contexts
        self.entries: dict[str, MemoryEntry] = {}
        self.events: list[dict] = []

    def __len__(self):
        return len(self.entries)

    def values(self):
        return list(self.entries.values())

    def clear(self):
        self.entries.clear()

    def upsert(self, surface: str, reading: str, contexts: list[str], step: int, n_occ: int,
               support: float, conf: float, source: str = "auto", meta: dict | None = None) -> None:
        e = self.entries.get(surface)
        if e is None:
            e = MemoryEntry(surface, reading, [], source, created_step=step, last_seen_step=step, meta=meta or {})
            self.entries[surface] = e
            self.events.append({"step": step, "op": "add", "surface": surface, "source": source})
        else:
            self.events.append({"step": step, "op": "update", "surface": surface})
            e.weight += 1.0
        tot = e.n_occ + n_occ
        if tot:
            e.support = (e.support * e.n_occ + support * n_occ) / tot
            e.conf = (e.conf * e.n_occ + conf * n_occ) / tot
        e.n_occ = tot
        e.n_sessions += 1
        e.last_seen_step = step
        if reading:
            e.reading = reading
        for c in contexts:
            if c not in e.contexts and len(e.contexts) < self.max_contexts:
                e.contexts.append(c)

    def mark_used(self, surface: str, step: int) -> None:
        if surface in self.entries:
            self.entries[surface].last_used_step = step

    def feedback(self, surface: str, confirmed: bool) -> None:
        e = self.entries.get(surface)
        if e is None:
            return
        if confirmed:
            e.confirm += 1
            e.weight += 0.5
        else:
            e.contradict += 1

    def _drop(self, surface, step, reason):
        self.entries.pop(surface, None)
        self.events.append({"step": step, "op": "prune", "surface": surface, "reason": reason})

    def end_session(self, step: int) -> None:
        if self.prune_contra:
            for e in self.values():
                if e.contradict - e.confirm >= self.margin:
                    self._drop(e.surface, step, "contradiction")
        if self.policy == "decay":
            for e in self.values():
                if e.last_seen_step != step and e.last_used_step != step:
                    e.weight *= self.gamma
                if e.weight < self.min_weight:
                    self._drop(e.surface, step, "decay")
        if self.max_size and len(self) > self.max_size and self.policy != "unbounded":
            key = {"lru": lambda e: (max(e.last_used_step, e.last_seen_step), e.n_occ),
                   "decay": lambda e: e.weight,
                   "confidence": lambda e: e.evidence(self.lam)}[self.policy]
            for e in sorted(self.values(), key=key)[: len(self) - self.max_size]:
                self._drop(e.surface, step, f"capacity:{self.policy}")

    def snapshot(self) -> list[dict]:
        return [asdict(e) for e in self.entries.values()]

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=0), encoding="utf-8")
