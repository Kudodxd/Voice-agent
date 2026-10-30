"""Build session sequences from the eval manifest.

  --group-by speaker|domain   needs data.metadata TSV (talk_id, speaker, domain, ...)
  --group-by prefix           first --prefix-len chars of the talk ID (heuristic! verify what the
                              CSJ ID prefix encodes in the CSJ documentation before using it)
  --group-by random           control condition for the go/no-go recurrence check
"""
import _common  # noqa: F401
import argparse

from vagent.config import load_config
from vagent.data.manifest import read_manifest
from vagent.data.sequences import build_sequences, load_metadata, save_sequences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--group-by", required=True, choices=["speaker", "domain", "prefix", "random"])
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=5)
    ap.add_argument("--prefix-len", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    talks = sorted({u.talk_id for u in read_manifest(cfg["data"]["eval_manifest"])})
    meta = load_metadata(cfg["data"].get("metadata"))
    seqs = build_sequences(talks, a.group_by, meta, a.min_len, a.max_len, a.n_perm, a.seed, a.prefix_len)
    save_sequences(seqs, a.out)
    print(f"{len(seqs)} sequences ({sum(len(s['talks']) for s in seqs)} talks) -> {a.out}")


if __name__ == "__main__":
    main()
