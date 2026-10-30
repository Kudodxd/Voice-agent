"""Go/no-go: rare-word recurrence across sessions (reference only, no ASR).

For session t of a sequence: R_t = rare word types in its references.
  type_recur(t)  = |R_t ∩ (R_0 ∪ ... ∪ R_{t-1})| / |R_t|
  token_recur(t) = share of rare *tokens* of session t whose type appeared before
This is the ceiling of any cross-session memory. Compare groupings against --group-by random.
"""
import _common  # noqa: F401
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from vagent.config import load_config
from vagent.data.manifest import group_by_talk, read_manifest
from vagent.data.sequences import load_sequences
from vagent.runtime import build_text
from vagent.text.normalize import normalize_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sequences", nargs="+", required=True)
    ap.add_argument("--n-perm", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    _, rare = build_text(cfg)
    talks = group_by_talk(read_manifest(cfg["data"]["eval_manifest"], cfg["data"].get("text_field", "text")))
    cache = {}

    def rare_tokens(tid):
        if tid not in cache:
            cache[tid] = [sp.surface for u in talks[tid] for sp in rare.rare_spans(normalize_text(u.text))]
        return cache[tid]

    rows = []
    for sp in a.sequences:
        acc = defaultdict(lambda: [0, 0, 0, 0, 0])  # step -> [type_hit, type_n, tok_hit, tok_n, count]
        for seq in load_sequences(sp):
            for order in seq["perms"][: a.n_perm]:
                seen = set()
                for t, tid in enumerate(order):
                    toks = rare_tokens(tid)
                    types = set(toks)
                    s = acc[t]
                    s[0] += len(types & seen); s[1] += len(types)
                    s[2] += sum(1 for w in toks if w in seen); s[3] += len(toks); s[4] += 1
                    seen |= types
        name = Path(sp).stem
        for t in sorted(acc):
            th, tn, kh, kn, c = acc[t]
            rows.append({"sequences": name, "step": t, "n": c, "rare_types_mean": tn / c,
                         "type_recur": th / tn if tn else 0, "token_recur": kh / kn if kn else 0})
        later = [acc[t] for t in acc if t >= 1]
        th, tn = sum(x[0] for x in later), sum(x[1] for x in later)
        kh, kn = sum(x[2] for x in later), sum(x[3] for x in later)
        print(f"[{name}] steps>=1: type_recur={th / max(tn, 1):.3f}  token_recur={kh / max(kn, 1):.3f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
