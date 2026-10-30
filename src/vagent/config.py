from __future__ import annotations

import copy
from pathlib import Path

import yaml


def deep_merge(base: dict, over: dict | None) -> dict:
    """Recursive merge; a dict containing `_replace_: true` replaces the base dict instead."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and v.get("_replace_"):
            out[k] = {kk: copy.deepcopy(vv) for kk, vv in v.items() if kk != "_replace_"}
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | Path) -> dict:
    """Load YAML; a top-level `_base_: other.yaml` is merged underneath (relative path)."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("_base_", None)
    if base:
        cfg = deep_merge(load_yaml(path.parent / base), cfg)
    return cfg


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """CLI overrides: `agent.read.top_k=20`, values parsed as YAML."""
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    return apply_overrides(load_yaml(path), overrides)


def system_config(cfg: dict, name: str) -> dict:
    """Resolve a system spec: cfg['agent'] defaults <- (optional `_extends_` chain) <- spec."""
    systems = cfg.get("systems", {})
    if name not in systems:
        raise KeyError(f"unknown system '{name}'. available: {sorted(systems)}")
    spec = dict(systems[name] or {})
    parent = spec.pop("_extends_", None)
    base = system_config(cfg, parent) if parent else cfg["agent"]
    return deep_merge(base, spec)
