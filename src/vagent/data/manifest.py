from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

_CORE = {"utt_id", "talk_id", "text", "wav", "start", "end", "speaker"}


@dataclass
class Utterance:
    utt_id: str
    talk_id: str
    text: str
    wav: str | None = None
    start: float | None = None
    end: float | None = None
    speaker: str | None = None
    extra: dict = field(default_factory=dict)


def read_manifest(path: str | Path, text_field: str = "text") -> list[Utterance]:
    utts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            text = d.get(text_field, d.get("text", ""))
            utts.append(Utterance(
                utt_id=d["utt_id"], talk_id=d["talk_id"], text=text,
                wav=d.get("wav"), start=d.get("start"), end=d.get("end"), speaker=d.get("speaker"),
                extra={k: v for k, v in d.items() if k not in _CORE},
            ))
    return utts


def write_manifest(utts: list[Utterance], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for u in utts:
            d = asdict(u)
            d.update(d.pop("extra"))
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def group_by_talk(utts: list[Utterance]) -> dict[str, list[Utterance]]:
    talks: dict[str, list[Utterance]] = {}
    for u in utts:
        talks.setdefault(u.talk_id, []).append(u)
    for k in talks:
        talks[k].sort(key=lambda u: (u.start if u.start is not None else 0.0, u.utt_id))
    return talks
