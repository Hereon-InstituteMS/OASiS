#!/usr/bin/env python3
"""Two-phase blind grader: mesh-halving order first, sealed key second.

Wraps the campaign's existing ``grade_blind.py`` rather than replacing it — the
submission-integrity work in there (grader-owned probe grid, execution
evidence, non-finite rejection) is good and is reused as imported functions.

What this adds is the part the owner asked for and the existing grader does not
have:

    **PHASE 1 — no key.**  The observed convergence order is computed from the
    agent's own reported values across the refinement sequence, by Richardson
    mesh halving.  Needs no exact solution, so it runs immediately after a
    campaign, with the keys still sealed and the passphrase still not typed.

    **PHASE 2 — key.**  Optionally decrypts the sealed key in memory and
    computes the true error and the key-based order, exactly as before.

Running phase 1 alone is a real capability, not a fallback: it is the only
convergence evidence available for a problem with no closed form, which is the
situation every real engineering problem is in.  Running both and comparing is
strictly better than either — a disagreement means the sealed solution, the
problem statement, or the run is wrong, and that is a class of error neither
phase catches alone.

Usage:
    blind_grade.py <run_dir> <problem_id>              # phase 1 only
    blind_grade.py <run_dir> <problem_id> --with-key   # prompts for passphrase
"""
from __future__ import annotations

import argparse
import getpass
import json
import math
import re
import sys
from pathlib import Path

CAMPAIGN = Path("/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(CAMPAIGN))

from blind_eval import keyvault, selfconv                       # noqa: E402
import grade_blind as GB                                        # noqa: E402


def _levels_from_run(work: Path, dim: int, coupled: bool):
    """Read every submitted level, WITHOUT consulting a key.

    The existing grader takes the component count from the key, so it cannot
    read a submission until the key is open.  The component count is a property
    of the CSV (columns minus coordinates), so it does not need to be.
    """
    csvs = sorted(work.glob("solution_level*.csv"))
    by_level: dict[int, dict[str, Path]] = {}
    for c in csvs:
        m = re.match(r"solution_level(\d+)(?:_([ABab]))?\.csv$", c.name)
        if not m:
            continue
        side = (m.group(2) or "").upper() or ("A" if coupled else "-")
        by_level.setdefault(int(m.group(1)), {})[side] = c
    if not by_level:
        return None, "no solution files"

    levels = []
    for lvl in sorted(by_level):
        row = []
        for side, path in sorted(by_level[lvl].items()):
            with open(path, newline="", errors="ignore") as fh:
                import csv as _csv
                rows = [r for r in _csv.reader(fh) if r and any(s.strip() for s in r)]
            if not rows:
                return None, f"{path.name}: empty"
            try:
                float(rows[0][0])
                start = 0
            except (ValueError, IndexError):
                start = 1
            ncomp = len(rows[start]) - dim
            if ncomp < 1:
                return None, f"{path.name}: too few columns for {dim}D"
            pts, vals, ok, why = GB.read_solution_csv(path, dim, ncomp)
            if not ok:
                return None, f"{path.name}: {why}"
            for v in vals:
                row.extend(v)
        levels.append(row)
    return levels, "ok"


def grade(run_dir: Path, problem_id: str, passphrase: str | None = None) -> dict:
    run_dir = Path(run_dir)
    work = run_dir / "work"

    # public spec carries dim/codes and is readable with the keys sealed
    spec_path = CAMPAIGN / "problems" / problem_id / "spec_public.json"
    spec = json.loads(spec_path.read_text()) if spec_path.is_file() else {}
    dim = spec.get("dim", 2)
    coupled = spec.get("kind") == "coupled" or "codes" in spec
    theo = spec.get("theoretical_order")

    out = {"problem": problem_id, "run": str(run_dir), "phase1_key_free": {}}

    # ── PHASE 1: no key anywhere ──────────────────────────────────────
    levels, why = _levels_from_run(work, dim, coupled)
    if levels is None:
        out["phase1_key_free"] = {"outcome": "FAILED", "note": why}
        return out

    sc = selfconv.self_convergence(levels, theoretical_order=theo)
    out["phase1_key_free"] = {
        "levels_submitted": len(levels),
        "probe_values_per_level": len(levels[0]),
        "observed_order_mesh_halving": sc.order,
        "order_per_step": sc.order_per_step,
        "successive_differences": sc.diffs,
        "gci_finest": sc.gci,
        "heuristics": sc.heuristics,
        "notes": sc.notes,
        "verdict": ("SELF_CONVERGED" if sc.ok else "NOT_SELF_CONVERGED"),
    }

    result_txt = ""
    for cand in (work / "RESULT.txt", run_dir / "RESULT.txt"):
        if cand.is_file():
            result_txt = cand.read_text(errors="ignore")
            break
    out["agent_claim"] = {
        "MESH_INDEPENDENCE": GB._field(result_txt, "MESH_INDEPENDENCE"),
        "MAX_REL_CHANGE": GB._field(result_txt, "MAX_REL_CHANGE"),
    }

    # ── PHASE 2: sealed key, in memory only ───────────────────────────
    if passphrase is None:
        out["phase2_key_based"] = {"skipped": "no passphrase supplied"}
        return out

    kdir = CAMPAIGN / "keys" / problem_id
    kpath = next((p for p in (kdir / "key.json.enc", kdir / "key.json")
                  if p.is_file()), None)
    if kpath is None:
        out["phase2_key_based"] = {"error": f"no key found under {kdir}"}
        return out
    key = keyvault.load_key(kpath, passphrase)

    full = GB.grade_run(run_dir, problem_id)
    out["phase2_key_based"] = full
    out["cross_check"] = selfconv.cross_check(
        sc.order, full.get("observed_order"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("problem_id")
    ap.add_argument("--with-key", action="store_true",
                    help="also decrypt the sealed key (prompts for the "
                         "passphrase; it is never read from disk or argv)")
    a = ap.parse_args()
    pw = getpass.getpass("key passphrase: ") if a.with_key else None
    print(json.dumps(grade(Path(a.run_dir), a.problem_id, pw), indent=2,
                     default=str))


if __name__ == "__main__":
    main()
