"""Kaldi/ESPnet-style CSJ data dir (wav.scp, segments, text[, utt2spk]) -> JSONL manifest.

  python scripts/prepare_csj.py --data-dir /path/espnet/egs2/csj/asr1/data/eval1 --out work/csj/manifests/eval1.jsonl

`text` field: POS suffixes (`word+POS`) stripped if present. `text_nofiller`: additionally drops
tokens whose POS starts with --drop-pos (default 感動詞). Inspect your text file first: if it has no
`+POS` tags, `text_nofiller` equals `text` and filler handling must be done another way.
"""
import _common  # noqa: F401
import argparse
from pathlib import Path

from vagent.data.manifest import Utterance, write_manifest
from vagent.text.normalize import strip_pos


def read_kv(path: Path) -> dict[str, str]:
    d = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                k, _, v = line.strip().partition(" ")
                d[k] = v.strip()
    return d


def wav_path(v: str) -> str:
    if v.endswith("|"):  # piped command: take the first token that looks like an audio file
        for tok in v.split():
            if tok.lower().endswith((".wav", ".flac", ".sph")):
                return tok
        raise ValueError(f"cannot find audio path in: {v}")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-pos", nargs="*", default=["感動詞"])
    a = ap.parse_args()
    d = Path(a.data_dir)
    wavs = {k: wav_path(v) for k, v in read_kv(d / "wav.scp").items()}
    segs = {k: v.split() for k, v in read_kv(d / "segments").items()}
    texts = read_kv(d / "text")
    spk = read_kv(d / "utt2spk")
    utts = []
    for uid, raw in texts.items():
        if uid in segs:
            rec, st, en = segs[uid][0], float(segs[uid][1]), float(segs[uid][2])
        else:
            rec, st, en = uid, None, None
        utts.append(Utterance(uid, rec, strip_pos(raw), wavs.get(rec), st, en, spk.get(uid),
                              {"text_nofiller": strip_pos(raw, a.drop_pos), "text_raw": raw}))
    write_manifest(utts, a.out)
    print(f"{len(utts)} utts, {len({u.talk_id for u in utts})} talks -> {a.out}")


if __name__ == "__main__":
    main()
