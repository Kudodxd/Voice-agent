"""Session sequences: ordered lists of talks processed with a persistent memory."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path


def load_metadata(path: str | Path | None) -> dict[str, dict]:
    """TSV with header; must contain `talk_id`, e.g. talk_id<TAB>speaker<TAB>domain."""
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return {row["talk_id"]: row for row in csv.DictReader(f, delimiter="\t")}


def _group(talk_ids, group_by, metadata, prefix_len, max_len, rng):
    groups: dict[str, list[str]] = {}
    if group_by == "random":
        ids = list(talk_ids)
        rng.shuffle(ids)
        for i in range(0, len(ids), max_len):
            groups[f"rand{i // max_len:03d}"] = ids[i:i + max_len]
        return groups
    for t in talk_ids:
        if group_by == "prefix":
            key = t[:prefix_len]
        elif group_by in ("speaker", "domain"):
            if t not in metadata or not metadata[t].get(group_by):
                continue
            key = metadata[t][group_by]
        else:
            raise ValueError(group_by)
        groups.setdefault(key, []).append(t)
    return groups


def build_sequences(talk_ids, group_by: str, metadata: dict | None = None, min_len: int = 3,
                    max_len: int = 10, n_perm: int = 5, seed: int = 0, prefix_len: int = 3) -> list[dict]:
    rng = random.Random(seed)
    groups = _group(sorted(talk_ids), group_by, metadata or {}, prefix_len, max_len, rng)
    seqs = []
    for key in sorted(groups):
        talks = sorted(groups[key])
        for ci, i in enumerate(range(0, len(talks), max_len)):
            chunk = talks[i:i + max_len]
            if len(chunk) < min_len:
                continue
            perms = [list(chunk)]  # perm 0 = natural (ID) order
            while len(perms) < n_perm:
                p = list(chunk)
                rng.shuffle(p)
                perms.append(p)
            seqs.append({"seq_id": f"{group_by}-{key}-{ci}", "group_by": group_by, "key": key,
                         "talks": chunk, "perms": perms})
    return seqs


def save_sequences(seqs: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(seqs, ensure_ascii=False, indent=1), encoding="utf-8")


def load_sequences(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
