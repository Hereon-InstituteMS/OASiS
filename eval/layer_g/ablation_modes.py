"""Layer G ablation modes — control what catalog content the LLM sees.

Three modes:
  - "full"           : catalog as currently shipped, no modification.
  - "stripped"       : all pitfalls removed; only template + signatures remain.
  - "minus:<token>"  : pitfalls whose body contains <token> are removed;
                      other pitfalls for the same physics stay.

Used by claude_runner.build_catalog_context() to construct the system-prompt
catalog payload before each Claude call.
"""
from __future__ import annotations
import copy
from typing import Any


def apply_ablation(knowledge: dict[str, Any], mode: str) -> dict[str, Any]:
    """Return a copy of `knowledge` with pitfalls modulated per `mode`.

    `knowledge` is the dict returned by `backend.get_knowledge(physics)` —
    contains keys like "pitfalls", "common_errors", "template", etc.
    """
    if mode == "full":
        return knowledge

    out = copy.deepcopy(knowledge)

    if mode == "stripped":
        out["pitfalls"] = []
        out["common_errors"] = []
        return out

    if mode.startswith("minus:"):
        token = mode[len("minus:"):]
        if not token:
            raise ValueError("minus: ablation requires a non-empty token")
        pitfalls = out.get("pitfalls", []) or []
        out["pitfalls"] = [
            p for p in pitfalls
            if not _pitfall_matches(p, token)
        ]
        ce = out.get("common_errors", []) or []
        out["common_errors"] = [
            e for e in ce
            if not _pitfall_matches(e, token)
        ]
        return out

    raise ValueError(f"unknown ablation mode: {mode!r}")


def _pitfall_matches(pitfall: Any, token: str) -> bool:
    """Match if `token` appears in any string field of the pitfall."""
    token_l = token.lower()
    if isinstance(pitfall, str):
        return token_l in pitfall.lower()
    if isinstance(pitfall, dict):
        for v in pitfall.values():
            if isinstance(v, str) and token_l in v.lower():
                return True
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, str) and token_l in item.lower():
                        return True
        return False
    return False


def count_pitfalls(knowledge: dict[str, Any]) -> int:
    """For logging — how many pitfalls remain after ablation?"""
    return len(knowledge.get("pitfalls", []) or [])
