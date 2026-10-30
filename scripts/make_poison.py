"""Controlled poisoning set (Exp2): k wrong entries per sequence, injected before session 0.

homophone : same reading as a target rare word of the sequence, different surface (from train vocab)
near      : target reading with one mora substituted, written in katakana (near-sound non-word)
Targets are rare words that occur in >= 2 sessions of the sequence (so the poison can actually hurt).
"""
import _common  # noqa: F401
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from vagent.config import load_config
from vagent.data.manifest import group_by_talk, read_manifest
from vagent.data.sequences import load_sequences
from vagent.runtime import build_text
from vagent.text.normalize import hira2kata, normalize_text

_SUB = {"か": "が", "き": "ぎ", "く": "ぐ", "け": "げ", "こ": "ご", "さ": "ざ", "し": "じ", "す": "ず",
        "せ": "ぜ", "そ": "ぞ", "た": "だ", "ち": "ぢ", "て": "で", "と": "ど", "は": "ば", "ひ": "び",
        "ふ": "ぶ", "へ": "べ", "ほ": "ぼ", "ま": "な", "み": "に", "む": "ぬ", "め": "ね", "も": "の",
        "ら": "な", "り": "に", "る": "ぬ", "れ": "ね", "ろ": "の", "う": "お", "い": "え", "あ": "お"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--types", nargs="+", default=["homophone", "near"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    tok, rare = build_text(cfg)
    rng = random.Random(a.seed)
    by_reading: dict[str, set] = {}
    vp = Path(cfg["rare"]["vocab_path"])
    for line in vp.read_text(encoding="utf-8").splitlines():
        w, r, _ = line.split("\t")
        by_reading.setdefault(r, set()).add(w)
    talks = group_by_talk(read_manifest(cfg["data"]["eval_manifest"], cfg["data"].get("text_field", "text")))
    out = {}
    for seq in load_sequences(a.sequences):
        sess = Counter()
        reading = {}
        for tid in seq["talks"]:
            ws = {}
            for u in talks[tid]:
                for sp in rare.rare_spans(normalize_text(u.text)):
                    ws[sp.surface] = sp.reading
            sess.update(ws.keys()); reading.update(ws)
        targets = [w for w, c in sess.most_common() if c >= 2]
        rng.shuffle(targets)
        entries = []
        for w in targets:
            if len(entries) >= a.k:
                break
            r = reading[w]
            pref = a.types[len(entries) % len(a.types)]
            for t in [pref] + [x for x in a.types if x != pref]:  # fall back to another type
                e = None
                if t == "homophone":
                    alts = sorted(by_reading.get(r, set()) - {w})
                    if alts:
                        e = {"surface": rng.choice(alts), "reading": r}
                else:
                    pos = [i for i, c in enumerate(r) if c in _SUB]
                    if pos:
                        i = rng.choice(pos)
                        r2 = r[:i] + _SUB[r[i]] + r[i + 1:]
                        e = {"surface": hira2kata(r2), "reading": r2}
                if e:
                    entries.append({**e, "type": t, "target": w})
                    break
        out[seq["seq_id"]] = entries
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"poison for {len(out)} seqs, {sum(map(len, out.values()))} entries -> {a.out}")


if __name__ == "__main__":
    main()
