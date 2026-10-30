"""CER split into rare-span CER (B-CER) and the rest (U-CER), term P/R, harmful change rate.

Japanese has no word boundaries, so errors are attributed per reference *character*:
substitutions/deletions to the ref char they hit; an insertion counts as rare only when both
neighbouring ref chars are inside the same kind of rare span, otherwise as U.
Term counts use non-overlapping substring counts on normalised strings (simple; note that a
rare word that is a substring of another is counted in both).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import Levenshtein

from ..text.lexicon import Span


def ref_error_flags(ref: str, hyp: str) -> tuple[list[bool], list[int]]:
    wrong, ins = [False] * len(ref), []
    for op, s, _ in Levenshtein.editops(ref, hyp):
        if op == "insert":
            ins.append(s)
        else:
            wrong[s] = True
    return wrong, ins


def rare_mask(n: int, spans: list[Span]) -> list[bool]:
    m = [False] * n
    for sp in spans:
        for i in range(sp.start, min(sp.end, n)):
            m[i] = True
    return m


@dataclass
class UttStats:
    n_ref: int = 0
    n_err: int = 0
    n_rare: int = 0
    err_rare: int = 0
    err_u: int = 0
    rec_tp: int = 0
    rec_fn: int = 0
    prec_tp: int = 0
    prec_fp: int = 0
    induced_fp: int = 0       # occurrences of offered hotwords beyond their ref count
    base_correct: int = 0
    harmful: int = 0          # ref chars correct in baseline, wrong in this system

    def __add__(self, o: "UttStats") -> "UttStats":
        return UttStats(*[getattr(self, f.name) + getattr(o, f.name) for f in fields(self)])


def utt_stats(ref: str, hyp: str, rare_spans: list[Span], offered: set[str] | None = None,
              base_hyp: str | None = None) -> UttStats:
    """`ref`/`hyp` must already be normalised; `rare_spans` are spans of the normalised ref."""
    st = UttStats(n_ref=len(ref))
    wrong, ins = ref_error_flags(ref, hyp)
    st.n_err = sum(wrong) + len(ins)
    mask = rare_mask(len(ref), rare_spans)
    st.n_rare = sum(mask)
    st.err_rare = sum(1 for i, w in enumerate(wrong) if w and mask[i])
    st.err_rare += sum(1 for s in ins if 0 < s < len(ref) and mask[s - 1] and mask[s])
    st.err_u = st.n_err - st.err_rare
    ref_terms = {sp.surface for sp in rare_spans}
    for w in ref_terms:
        cr, ch = ref.count(w), hyp.count(w)
        st.rec_tp += min(cr, ch)
        st.rec_fn += max(0, cr - ch)
    for w in ref_terms | (offered or set()):
        cr, ch = ref.count(w), hyp.count(w)
        st.prec_tp += min(cr, ch)
        st.prec_fp += max(0, ch - cr)
    for w in offered or set():
        st.induced_fp += max(0, hyp.count(w) - ref.count(w))
    if base_hyp is not None:
        wb, _ = ref_error_flags(ref, base_hyp)
        st.base_correct = sum(not x for x in wb)
        st.harmful = sum(1 for i in range(len(ref)) if not wb[i] and wrong[i])
    return st


def _div(a, b):
    return a / b if b else math.nan


def summarize(st: UttStats) -> dict:
    p, r = _div(st.prec_tp, st.prec_tp + st.prec_fp), _div(st.rec_tp, st.rec_tp + st.rec_fn)
    return {
        "CER": _div(st.n_err, st.n_ref), "B-CER": _div(st.err_rare, st.n_rare),
        "U-CER": _div(st.err_u, st.n_ref - st.n_rare), "term_R": r, "term_P": p,
        "term_F1": _div(2 * p * r, p + r) if not (math.isnan(p) or math.isnan(r)) else math.nan,
        "harmful": _div(st.harmful, st.base_correct), "induced_fp_per_1k": _div(1000 * st.induced_fp, st.n_ref),
        "n_ref": st.n_ref, "n_rare": st.n_rare,
    }


def memory_quality(snapshot: list[dict], refs_so_far: str, next_rare: set[str] | None) -> dict:
    surfs = {e["surface"] for e in snapshot}
    correct = sum(1 for s in surfs if s in refs_so_far)
    return {
        "mem_size": len(surfs), "mem_correct": correct, "mem_wrong": len(surfs) - correct,
        "mem_precision": _div(correct, len(surfs)),
        "mem_poison": sum(1 for e in snapshot if e.get("source") == "poison"),
        "coverage_next": _div(len(surfs & next_rare), len(next_rare)) if next_rare else math.nan,
    }
