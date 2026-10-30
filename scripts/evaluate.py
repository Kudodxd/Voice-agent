"""Evaluate a run directory: overall table, per-step curves, memory quality, paired bootstrap.

  python scripts/evaluate.py --config configs/base.yaml --exp main --baseline B0 --compare P
"""
import _common  # noqa: F401
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from vagent.config import load_config
from vagent.eval.metrics import UttStats, memory_quality, summarize, utt_stats
from vagent.eval.stats import paired_cluster_bootstrap
from vagent.runtime import build_text
from vagent.text.normalize import normalize_text


def write_csv(rows, path):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def nanmean(v):
    v = [x for x in v if not math.isnan(x)]
    return sum(v) / len(v) if v else math.nan


def fmt(v):
    return "-" if isinstance(v, float) and math.isnan(v) else (f"{v:.4f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--baseline", default="B0", help="system used for harmful-change rate")
    ap.add_argument("--compare", default="P", help="bootstrap: this system vs every other")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    cfg = load_config(a.config)
    _, rare = build_text(cfg)
    root = Path(cfg["paths"]["runs_dir"]) / a.exp
    systems = sorted(p.name for p in root.iterdir() if (p / "system.yaml").exists())
    span_cache = {}

    def spans_of(ref_norm):
        if ref_norm not in span_cache:
            span_cache[ref_norm] = rare.rare_spans(ref_norm)
        return span_cache[ref_norm]

    base_hyp = {}
    if a.baseline in systems:
        for f in (root / a.baseline).glob("*/perm*/utts.jsonl"):
            for line in f.open(encoding="utf-8"):
                r = json.loads(line)
                base_hyp.setdefault(r["utt_id"], normalize_text(r["hyp"]))

    overall, by_step, mem_rows, sess_tot = {}, [], [], defaultdict(dict)
    for s in systems:
        tot = UttStats()
        step_acc = defaultdict(UttStats)
        mem_acc = defaultdict(lambda: defaultdict(list))
        for f in sorted((root / s).glob("*/perm*/utts.jsonl")):
            recs = [json.loads(l) for l in f.open(encoding="utf-8")]
            steps = defaultdict(list)
            for r in recs:
                ref, hyp = normalize_text(r["ref"]), normalize_text(r["hyp"])
                st = utt_stats(ref, hyp, spans_of(ref), {o["surface"] for o in r["offered"]}, base_hyp.get(r["utt_id"]))
                tot += st; step_acc[r["step"]] += st
                steps[r["step"]].append((ref, st))
                key = (r["seq_id"], r["perm"], r["step"])
                cell = sess_tot[s].setdefault(key, [0, 0, 0, 0])
                cell[0] += st.n_err; cell[1] += st.n_ref; cell[2] += st.err_rare; cell[3] += st.n_rare
            # memory quality after each session
            refs_so_far = ""
            for t in sorted(steps):
                refs_so_far += "".join(ref for ref, _ in steps[t])
                snap_p = f.parent / "memory" / f"step_{t:03d}.json"
                if not snap_p.exists():
                    continue
                nxt = {sp.surface for ref, _ in steps.get(t + 1, []) for sp in spans_of(ref)} or None
                q = memory_quality(json.loads(snap_p.read_text(encoding="utf-8")), refs_so_far, nxt)
                for k, v in q.items():
                    mem_acc[t][k].append(v)
        overall[s] = summarize(tot)
        for t in sorted(step_acc):
            by_step.append({"system": s, "step": t, **summarize(step_acc[t])})
        for t in sorted(mem_acc):
            mem_rows.append({"system": s, "step": t, **{k: nanmean(v) for k, v in mem_acc[t].items()}})

    out = root / "eval"; out.mkdir(exist_ok=True)
    write_csv([{"system": s, **m} for s, m in overall.items()], out / "overall.csv")
    write_csv(by_step, out / "by_step.csv")
    write_csv(mem_rows, out / "memory_by_step.csv")

    cols = ["CER", "B-CER", "U-CER", "term_R", "term_P", "term_F1", "harmful", "induced_fp_per_1k"]
    print("| system | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1))
    for s, m in overall.items():
        print(f"| {s} | " + " | ".join(fmt(m[c]) for c in cols) + " |")

    if a.compare in sess_tot:
        rows = []
        A = sess_tot[a.compare]
        for s in systems:
            if s == a.compare:
                continue
            keys = sorted(set(A) & set(sess_tot[s]))
            if not keys:
                continue
            B = sess_tot[s]
            for name, ie, i_n in (("CER", 0, 1), ("B-CER", 2, 3)):
                res = paired_cluster_bootstrap([B[k][ie] for k in keys], [B[k][i_n] for k in keys],
                                               [A[k][ie] for k in keys], [A[k][i_n] for k in keys], a.n_boot)
                rows.append({"compare": f"{a.compare} - {s}", "metric": name, **res})
        write_csv(rows, out / "bootstrap.csv")
        print(f"\nbootstrap ({a.compare} minus other; negative = {a.compare} better):")
        for r in rows:
            print(f"  {r['compare']:>12} {r['metric']:>6}: {r['delta']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] p={r['p']:.3f}")
    print(f"\n-> {out}/(overall|by_step|memory_by_step|bootstrap).csv")


if __name__ == "__main__":
    main()
