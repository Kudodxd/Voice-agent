"""Run systems over session sequences with a persistent memory (Exp1-4).

  python scripts/run_longitudinal.py --config configs/base.yaml --exp main \
      --sequences work/csj/seq_domain.json --systems B0 B1 B2 B3 B4 B5 P --n-perm 5

Output: runs/<exp>/<system>/<seq_id>/perm<k>/{utts.jsonl, decisions.jsonl, memory/step_XXX.json,
memory_events.jsonl, DONE}. Finished (system, seq, perm) cells are skipped, so runs can be resumed.
"""
import _common  # noqa: F401
import argparse
import json
from pathlib import Path

import yaml

from vagent.agent import VoiceAgent
from vagent.asr.base import Hotword
from vagent.config import load_config, system_config
from vagent.data.manifest import group_by_talk, read_manifest
from vagent.data.sequences import load_sequences
from vagent.runtime import build_backends, build_text, load_static_list
from vagent.text.normalize import normalize_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--n-perm", type=int, default=1)
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--poison", default=None, help="json from make_poison.py (memory systems only)")
    ap.add_argument("-o", "--override", nargs="*", default=[])
    a = ap.parse_args()

    cfg = load_config(a.config, a.override)
    tok, rare = build_text(cfg)
    ctc, dec = build_backends(cfg, tok, rare)
    static = load_static_list(cfg["data"].get("static_list"), tok)
    talks = group_by_talk(read_manifest(cfg["data"]["eval_manifest"], cfg["data"].get("text_field", "text")))
    seqs = load_sequences(a.sequences)[: a.max_seqs]
    poison = json.loads(Path(a.poison).read_text(encoding="utf-8")) if a.poison else {}
    root = Path(cfg["paths"]["runs_dir"]) / a.exp
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    ref_rare_cache: dict[str, list] = {}

    def session_rare(tid):
        if tid not in ref_rare_cache:
            ref_rare_cache[tid] = [sp for u in talks[tid] for sp in rare.rare_spans(normalize_text(u.text))]
        return ref_rare_cache[tid]

    for name in a.systems:
        scfg = system_config(cfg, name)
        (root / name).mkdir(exist_ok=True)
        (root / name / "system.yaml").write_text(yaml.safe_dump(scfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        agent = VoiceAgent(scfg, ctc, dec, tok, rare, static)
        need_label = agent.gate.mode == "label"
        need_oracle = agent.read_source == "oracle_session"
        for seq in seqs:
            for p, order in enumerate(seq["perms"][: a.n_perm]):
                out = root / name / seq["seq_id"] / f"perm{p}"
                if (out / "DONE").exists():
                    continue
                out.mkdir(parents=True, exist_ok=True)
                for f in ("utts.jsonl", "decisions.jsonl"):
                    (out / f).unlink(missing_ok=True)
                agent.reset()
                if poison.get(seq["seq_id"]) and agent.persist and agent.write_enabled:
                    agent.inject(poison[seq["seq_id"]])
                    agent.memory.save(out / "memory" / "step_init.json")
                for t, tid in enumerate(order):
                    spans = session_rare(tid) if (need_label or need_oracle) else []
                    ref_rare = {s.surface for s in spans} if need_label else None
                    oracle = list({s.surface: Hotword(s.surface, s.reading) for s in spans}.values()) if need_oracle else None
                    recs, decs = agent.process_session(talks[tid], t, ref_rare, oracle)
                    with open(out / "utts.jsonl", "a", encoding="utf-8") as f:
                        for r in recs:
                            f.write(json.dumps({"seq_id": seq["seq_id"], "perm": p, "step": t, **r}, ensure_ascii=False) + "\n")
                    with open(out / "decisions.jsonl", "a", encoding="utf-8") as f:
                        for d in decs:
                            f.write(json.dumps({"step": t, "talk_id": tid, **d.to_dict()}, ensure_ascii=False) + "\n")
                    agent.memory.save(out / "memory" / f"step_{t:03d}.json")
                with open(out / "memory_events.jsonl", "w", encoding="utf-8") as f:
                    for e in agent.memory.events:
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")
                (out / "DONE").touch()
                print(f"[{name}] {seq['seq_id']} perm{p}: {len(order)} sessions, |mem|={len(agent.memory)}")


if __name__ == "__main__":
    main()
