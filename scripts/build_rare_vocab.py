"""Count content units on the TRAIN manifest; write vocab.tsv and the top-N common list.

rare word := content unit (merged noun compound by default) not in the top-N train units.
"""
import _common  # noqa: F401
import argparse
from pathlib import Path

from vagent.config import load_config
from vagent.data.manifest import read_manifest
from vagent.runtime import build_text
from vagent.text.normalize import normalize_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("-o", "--override", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.override)
    common_path = cfg["rare"]["common_path"]
    cfg["rare"]["common_path"] = None  # counting must not depend on an old common list
    _, rare = build_text(cfg)
    from vagent.text.lexicon import count_units
    utts = read_manifest(cfg["data"]["train_manifest"], cfg["data"].get("text_field", "text"))
    counts, readings = count_units(rare, (normalize_text(u.text) for u in utts))
    rc = cfg["rare"]
    rc["common_path"] = common_path
    vp = Path(rc["vocab_path"]); vp.parent.mkdir(parents=True, exist_ok=True)
    with open(vp, "w", encoding="utf-8") as f:
        for w, c in counts.most_common():
            f.write(f"{w}\t{readings[w].most_common(1)[0][0]}\t{c}\n")
    common = [w for w, _ in counts.most_common(rc["top_n"])]
    Path(rc["common_path"]).write_text("\n".join(common) + "\n", encoding="utf-8")
    print(f"{len(counts)} unit types from {len(utts)} utts; top-{rc['top_n']} -> {rc['common_path']}")


if __name__ == "__main__":
    main()
