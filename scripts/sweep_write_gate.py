"""Exp1 (offline): precision/recall of the write gate over thresholds, from logged decisions.

A candidate is *correct* if its surface occurs in the references of that session (normalised).
recall = correct accepted types / rare-word types in the session references.
Run on a system that logs all candidates (any memory system). Offline sweeps ignore the feedback
loop (memory -> later decoding); use them to choose operating points, then confirm with real runs.
"""
import _common  # noqa: F401
import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

from vagent.config import load_config
from vagent.runtime import build_text
from vagent.text.normalize import normalize_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--system", default="P")
    ap.add_argument("--support", nargs="+", type=float, default=[0.0, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0])
    ap.add_argument("--conf", nargs="+", type=float, default=[0.0, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--repeat", nargs="+", type=int, default=[1, 2, 3])
    a = ap.parse_args()
    cfg = load_config(a.config)
    _, rare = build_text(cfg)
    root = Path(cfg["paths"]["runs_dir"]) / a.exp / a.system
    cands, n_ref = [], 0
    for d in sorted(root.glob("*/perm0")):
        refs, rr = defaultdict(str), defaultdict(set)
        for l in (d / "utts.jsonl").open(encoding="utf-8"):
            r = json.loads(l)
            ref = normalize_text(r["ref"])
            refs[r["step"]] += ref
            rr[r["step"]] |= {sp.surface for sp in rare.rare_spans(ref)}
        n_ref += sum(len(v) for v in rr.values())
        for l in (d / "decisions.jsonl").open(encoding="utf-8"):
            c = json.loads(l)
            c["correct"] = c["surface"] in refs[c["step"]]
            c["in_ref_rare"] = c["surface"] in rr[c["step"]]
            cands.append(c)
    rows = []
    for s, c, k in itertools.product(a.support, a.conf, a.repeat):
        acc = [x for x in cands if x["support"] >= s and x["conf"] >= c and x["n_utt"] >= k]
        tp = sum(x["correct"] for x in acc)
        tp_r = sum(x["in_ref_rare"] for x in acc)
        rows.append({"min_support": s, "min_conf": c, "min_repeat": k, "n_accepted": len(acc),
                     "precision": tp / len(acc) if acc else float("nan"), "recall": tp_r / n_ref if n_ref else 0.0})
    out = root.parent / "eval"; out.mkdir(exist_ok=True)
    p = out / f"write_gate_sweep_{a.system}.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    best = sorted((r for r in rows if r["n_accepted"]), key=lambda r: (-r["precision"], -r["recall"]))[:5]
    print(f"{len(cands)} logged candidates, {n_ref} reference rare types (perm0)")
    for r in best:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"-> {p}")


if __name__ == "__main__":
    main()
