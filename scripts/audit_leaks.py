#!/usr/bin/env python3
"""Leak audit over run trajectories — now including campaign 3.

Three defects in the previous version, all of which made it report clean:

1. **It never reached campaign 3.**  Its globs were ``single/T*_seed*/work/``
   and ``matrix/T*_seed*/work/``.  Campaign-3 runs live under
   ``campaign3_blind/runs/<cell>/work/``, so the blind campaign — the one whose
   whole claim is that nothing leaked — was the one campaign never audited.

2. **``BAD_BOTH`` was dead code.**  The line read
   ``for m in set(BAD_BOTH.findall(txt)) if False else []: pass``.  The regex
   was compiled, never evaluated, and the two things it alone covered
   (``paper_experiments/runs``, archive directories) were only partly recovered
   by a later ad-hoc ``re.search``.

3. **It could not see a campaign-3 leak even where it looked.**  A blind
   campaign's leak is not "the agent read a sibling run"; it is "the agent read
   an answer".  That needs the sealed key, a symbolic comparison, and a sweep
   for the builders and the key directory.

What counts as a leak
---------------------
``BARE`` arm      any access to OASiS curated sources — the bare arm is defined
                  by not having them.
``both arms``     reading another run's results, an archive, the key directory,
                  a builder that holds hidden fields, or any file under
                  ``keys/``.
``blind cells``   the sealed solution appearing in the agent's context, compared
                  SYMBOLICALLY rather than as a substring: a builder writes
                  ``x * (1 - x) * ...`` where the key stores ``x*(1-x)*...``,
                  so a string search finds nothing and reports safety.

Usage:
    audit_leaks.py --root <campaign dir> [--keys <dir>] [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

BAD_BARE = re.compile(
    r"_installed_api|open-fem-agent/src/(backends|tools)|knowledge\.py")

# Reading another run's answers, an archive of them, or the key custody tree.
BAD_BOTH = re.compile(
    r"paper_experiments/runs|_v1_frozen|_calibration_archive|_pre_fix6|"
    r"campaign3_blind/keys|/keys/[A-Z]\d+/key\.json|"
    r"build_problems\.py|build_extra\.py|build_coupled\.py|"
    r"single/T\d+_\w+_(?:BARE|MCP)_seed\d+/(?:work/)?result")

SIBLING = re.compile(r"(?:single|matrix|runs)/[A-Za-z0-9_]+_seed\d+")

TRAJECTORY_GLOBS = (
    "single/T*_seed*/work/trajectory.txt",
    "matrix/T*_seed*/work/trajectory.txt",
    "campaign3_blind/runs/*/work/trajectory.txt",
    "campaign3_blind/runs/*/work/trajectory_live.txt",
    "runs/*/work/trajectory.txt",
    "runs/*/work/trajectory_live.txt",
)


def _load_keys(keys_dir: Path, passphrase: str | None):
    """Sealed solutions, if they can be opened. Absence is reported, not hidden."""
    if not keys_dir or not keys_dir.is_dir():
        return {}, "no keys directory supplied: symbolic rule disabled"
    try:
        from blind_eval import keyvault
    except Exception as exc:                                   # pragma: no cover
        return {}, f"keyvault unavailable ({exc}); symbolic rule disabled"
    if keyvault.is_sealed(keys_dir):
        return {}, "keys are sealed: symbolic rule disabled (unseal to audit)"
    out = {}
    for kp in sorted(list(keys_dir.rglob("key.json"))
                     + list(keys_dir.rglob("key.json.enc"))):
        try:
            k = keyvault.load_key(kp, passphrase)
        except Exception:
            continue
        ex = k.get("exact_solution")
        vals = []
        for g in (ex.values() if isinstance(ex, dict) else [ex]):
            vals += list(g) if isinstance(g, list) else [g]
        out[kp.parent.name] = [v for v in vals if isinstance(v, str)]
    return out, f"{len(out)} key(s) opened for the symbolic rule"


def _symbolic_hits(text: str, exacts: list) -> list:
    """Does the trajectory contain an expression equal to a sealed solution?"""
    if not exacts:
        return []
    try:
        from blind_eval.leakgate import _candidate_expressions, SYMS
        import sympy as sp
    except Exception:                                          # pragma: no cover
        return []
    targets = []
    for e in exacts:
        try:
            targets.append(sp.sympify(e, locals=SYMS))
        except Exception:
            continue
    hits = []
    for txt, expr in _candidate_expressions(text[:400_000]):
        for tgt in targets:
            try:
                if sp.simplify(expr - tgt) == 0:
                    hits.append(f"sealed solution appears as {txt[:70]!r}")
                    break
            except Exception:
                continue
        if hits:
            break
    return hits


def audit(root: Path, keys_dir: Path | None = None,
          passphrase: str | None = None) -> dict:
    root = Path(root)
    keys, key_note = _load_keys(keys_dir, passphrase)
    paths = []
    for g in TRAJECTORY_GLOBS:
        paths += glob.glob(str(root / g))
    paths = sorted(set(paths))

    hits = []
    for T in paths:
        p = Path(T)
        try:
            txt = p.read_text(errors="ignore")
        except OSError:
            continue
        run_dir = p.parent.parent
        own = run_dir.name
        arm = "BARE" if "_BARE_" in own else "MCP"
        findings = []
        if arm == "BARE" and BAD_BARE.search(txt):
            findings.append("OASiS-source access")
        for m in sorted(set(BAD_BOTH.findall(txt) or [])):
            findings.append(f"forbidden path: {m}" if isinstance(m, str)
                            else "forbidden path")
        if BAD_BOTH.search(txt) and not findings:
            findings.append(f"forbidden path: "
                            f"{BAD_BOTH.search(txt).group(0)[:60]}")
        for m in SIBLING.finditer(txt):
            if m.group(0).rsplit("/", 1)[-1] != own:
                findings.append(f"sibling-run access: {m.group(0)}")
                break
        pid = own.split("_")[0]
        if pid in keys:
            findings += _symbolic_hits(txt, keys[pid])
        if findings:
            hits.append({"run": str(run_dir), "trajectory": T,
                         "ledger": str(run_dir / "ledger.json"),
                         "findings": sorted(set(findings))})
    return {"root": str(root), "audited": len(paths),
            "key_note": key_note, "tainted": hits}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--keys", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    pw = os.environ.get("OASIS_KEY_PASSPHRASE")
    rep = audit(Path(a.root), Path(a.keys) if a.keys else None, pw)
    txt = json.dumps(rep, indent=1)
    if a.json:
        Path(a.json).write_text(txt)
    print(txt)
    if rep["audited"] == 0:
        print("WARNING: no trajectories matched — an audit that examined "
              "nothing must not read as a pass", file=sys.stderr)
        return 2
    return 1 if rep["tainted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
