"""Export a run directory to the project website (site/index.html).

  python scripts/export_site_data.py --config configs/toy.yaml --exps main poison --out site/data/results.js

Collects runs/<exp>/eval/*.csv (overall, by_step, memory_by_step, bootstrap, write-gate sweeps), the
resolved system configs, logged write-gate candidates (perm0, for the interactive gate), a session
replay trace (system vs baseline, perm0: utterances, offered hotwords, gate decisions, memory after
each session) and a map of the codebase. Output is `window.VAGENT_DATA = {...};`, so the page also
works when opened straight from disk. Run evaluate.py (and sweep_write_gate.py) first.
"""
import _common  # noqa: F401
import argparse
import ast
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import Levenshtein

from vagent.config import load_config, system_config
from vagent.data.manifest import group_by_talk, read_manifest
from vagent.eval.metrics import UttStats, summarize, utt_stats
from vagent.runtime import build_text
from vagent.text.normalize import normalize_text

ROOT = Path(__file__).resolve().parents[1]
TRACE_METRICS = ("CER", "B-CER", "U-CER", "term_R", "term_P")


def clean(v):
    """JSON-safe, compact: NaN/inf -> null, floats rounded."""
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else round(v, 5)
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    return v


def num(s: str):
    try:
        f = float(s)
    except ValueError:
        return s
    return int(f) if f.is_integer() and "." not in s and "e" not in s.lower() else f


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [{k: num(v) for k, v in row.items()} for row in csv.DictReader(f)]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def exp_block(exp_root: Path) -> dict:
    ev = exp_root / "eval"
    return {
        "systems": sorted(p.name for p in exp_root.iterdir() if (p / "system.yaml").exists()),
        "overall": read_csv(ev / "overall.csv"),
        "by_step": read_csv(ev / "by_step.csv"),
        "memory_by_step": read_csv(ev / "memory_by_step.csv"),
        "bootstrap": read_csv(ev / "bootstrap.csv"),
        "sweeps": {p.stem.removeprefix("write_gate_sweep_"): read_csv(p)
                   for p in sorted(ev.glob("write_gate_sweep_*.csv"))},
    }


def gate_candidates(sys_root: Path, rare) -> dict:
    """Every logged candidate of perm0 with the same correctness labels as sweep_write_gate.py."""
    cands, n_ref = [], 0
    for d in sorted(sys_root.glob("*/perm0")):
        refs, rr = defaultdict(str), defaultdict(set)
        for r in read_jsonl(d / "utts.jsonl"):
            ref = normalize_text(r["ref"])
            refs[r["step"]] += ref
            rr[r["step"]] |= {sp.surface for sp in rare.rare_spans(ref)}
        n_ref += sum(len(v) for v in rr.values())
        for c in read_jsonl(d / "decisions.jsonl"):
            cands.append({"seq": d.parent.name, "step": c["step"], "surface": c["surface"],
                          "support": c["support"], "conf": c["conf"], "n_utt": c["n_utt"],
                          "correct": c["surface"] in refs[c["step"]],
                          "in_ref_rare": c["surface"] in rr[c["step"]]})
    return {"exp": sys_root.parent.name, "system": sys_root.name, "n_ref": n_ref, "candidates": cands}


def diff_ops(ref: str, hyp: str) -> list:
    return [[op[0], i1, i2, j1, j2] for op, i1, i2, j1, j2 in Levenshtein.opcodes(ref, hyp) if op != "equal"]


def trace_seq(seq_dir: Path, base_dir: Path, rare, lam: float, max_utts: int) -> dict:
    recs = read_jsonl(seq_dir / "utts.jsonl")
    base = {r["utt_id"]: normalize_text(r["hyp"]) for r in read_jsonl(base_dir / "utts.jsonl")}
    decs, events = defaultdict(list), defaultdict(list)
    for d in read_jsonl(seq_dir / "decisions.jsonl"):
        decs[d["step"]].append(d)
    for e in read_jsonl(seq_dir / "memory_events.jsonl"):
        events[e["step"]].append(e)
    by_step = defaultdict(list)
    for r in recs:
        by_step[r["step"]].append(r)

    def snapshot(name, refs_so_far):
        p = seq_dir / "memory" / name
        if not p.exists():
            return None
        out = []
        for e in json.loads(p.read_text(encoding="utf-8")):
            ev = e["support"] * math.log1p(e["n_occ"]) + e["confirm"] - lam * e["contradict"]
            out.append({k: e[k] for k in ("surface", "reading", "source", "created_step", "last_used_step",
                                          "n_sessions", "n_occ", "support", "conf", "weight", "confirm",
                                          "contradict")} | {"evidence": ev, "correct": e["surface"] in refs_so_far})
        return out

    steps, refs_so_far = [], ""
    for t in sorted(by_step):
        rs = by_step[t]
        sys_st, base_st, utts = UttStats(), UttStats(), []
        step_ref = ""
        for r in rs:
            ref, hyp = normalize_text(r["ref"]), normalize_text(r["hyp"])
            b = base.get(r["utt_id"], hyp)
            spans = rare.rare_spans(ref)
            offered = {o["surface"] for o in r["offered"]}
            sys_st += utt_stats(ref, hyp, spans, offered)
            base_st += utt_stats(ref, b, spans)
            step_ref += ref
            utts.append({"utt_id": r["utt_id"], "ref": ref, "hyp": hyp, "base": b, "ctc": r.get("ctc_kana"),
                         "offered": r["offered"], "spans": [[s.start, s.end] for s in spans],
                         "ops": diff_ops(ref, hyp), "base_ops": diff_ops(ref, b)})
        refs_so_far += step_ref
        # keep the informative utterances when a talk is long (rare words, hotwords, or a changed hyp)
        if len(utts) > max_utts:
            keep = {u["utt_id"] for u in sorted(utts, key=lambda u: -(bool(u["spans"]) + bool(u["offered"])
                                                                      + (u["hyp"] != u["base"])))[:max_utts]}
            utts = [u for u in utts if u["utt_id"] in keep]
        ss, bs = summarize(sys_st), summarize(base_st)
        steps.append({
            "step": t, "talk_id": rs[0]["talk_id"], "n_utts": len(rs), "utts": utts,
            "summary": {"system": {k: ss[k] for k in TRACE_METRICS}, "baseline": {k: bs[k] for k in TRACE_METRICS}},
            "decisions": [{k: d[k] for k in ("surface", "reading", "n_utt", "n_occ", "support", "conf",
                                              "accepted", "reason")} | {"correct": d["surface"] in step_ref}
                          for d in decs[t]],
            "events": events[t],
            "memory": snapshot(f"step_{t:03d}.json", refs_so_far) or [],
        })
    return {"seq_id": seq_dir.parent.name, "init_memory": snapshot("step_init.json", ""), "steps": steps}


def trace_block(exp_root: Path, system: str, baseline: str, rare, lam: float, max_seqs: int, max_utts: int):
    sys_root = exp_root / system
    if not sys_root.exists() or not (exp_root / baseline).exists():
        return None
    seqs = [trace_seq(d, exp_root / baseline / d.parent.name / "perm0", rare, lam, max_utts)
            for d in sorted(sys_root.glob("*/perm0"))[:max_seqs]]
    return {"exp": exp_root.name, "system": system, "baseline": baseline, "seqs": seqs}


def code_map() -> list[dict]:
    out = []
    for p in sorted(list((ROOT / "src" / "vagent").rglob("*.py")) + list((ROOT / "scripts").glob("*.py"))):
        src = p.read_text(encoding="utf-8")
        if not src.strip():
            continue
        try:
            doc = (ast.get_docstring(ast.parse(src)) or "").strip()
        except SyntaxError:
            doc = ""
        doc = " ".join(doc.split("\n\n")[0].split())  # first paragraph, one line
        out.append({"path": str(p.relative_to(ROOT)), "lines": src.count("\n"), "doc": doc})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exps", nargs="+", default=["main"])
    ap.add_argument("--trace-exp", default=None, help="default: first of --exps")
    ap.add_argument("--trace-system", default="P")
    ap.add_argument("--trace-baseline", default="B0")
    ap.add_argument("--trace-max-seqs", type=int, default=3)
    ap.add_argument("--trace-max-utts", type=int, default=24, help="per session")
    ap.add_argument("--gate-system", default=None, help="default: P-noRepeat if it was run, else P")
    ap.add_argument("--go-nogo", default=None, help="csv from go_nogo.py --out (default: <work_dir>/go_nogo.csv)")
    ap.add_argument("--out", default="site/data/results.js")
    a = ap.parse_args()

    cfg = load_config(a.config)
    _, rare = build_text(cfg)
    runs = Path(cfg["paths"]["runs_dir"])
    exps = {e: exp_block(runs / e) for e in a.exps if (runs / e / "eval").exists()}
    if not exps:
        raise SystemExit(f"no evaluated experiments under {runs} (run scripts/evaluate.py first)")
    trace_exp = a.trace_exp or next(iter(exps))
    gate_sys = a.gate_system or ("P-noRepeat" if (runs / trace_exp / "P-noRepeat").exists() else "P")
    lam = cfg["agent"]["memory"].get("evidence_lambda", 1.0)
    go_nogo = Path(a.go_nogo) if a.go_nogo else Path(cfg["paths"]["work_dir"]) / "go_nogo.csv"
    talks = group_by_talk(read_manifest(cfg["data"]["eval_manifest"], cfg["data"].get("text_field", "text")))
    b = cfg["backends"]

    data = {
        "meta": {
            "config": a.config, "runs_dir": str(runs), "exps": list(exps),
            "backends": {"ctc": b.get("ctc", {}).get("type", "none"), "decoder": b["decoder"]["type"]},
            "toy": "mock" in (b.get("ctc", {}).get("type"), b["decoder"]["type"]),
            "n_talks": len(talks), "n_utts": sum(len(v) for v in talks.values()),
            "rare_top_n": cfg["rare"].get("top_n"),
        },
        "defaults": {"read": cfg["agent"]["read"], "write": cfg["agent"]["write"], "memory": cfg["agent"]["memory"]},
        "systems": {n: system_config(cfg, n) for n in cfg.get("systems", {})},
        "exps": exps,
        "gate": gate_candidates(runs / trace_exp / gate_sys, rare) if (runs / trace_exp / gate_sys).exists() else None,
        "trace": trace_block(runs / trace_exp, a.trace_system, a.trace_baseline, rare, lam,
                             a.trace_max_seqs, a.trace_max_utts),
        "go_nogo": read_csv(go_nogo),
        "code": code_map(),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.VAGENT_DATA = " + json.dumps(clean(data), ensure_ascii=False, separators=(",", ":"))
                   + ";\n", encoding="utf-8")
    print(f"exps={list(exps)} gate={gate_sys} trace={a.trace_system} vs {a.trace_baseline} "
          f"-> {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
