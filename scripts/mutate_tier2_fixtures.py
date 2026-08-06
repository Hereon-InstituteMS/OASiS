#!/usr/bin/env python3
"""Prove each fixture goes RED when the pathology it tests is removed.

A tier-2 fixture that passes whether or not the pitfall is present is
decoration. The only way to know is to remove the pathology and watch the
fixture fail — and the only way for a reader to know is for that removal to be
recorded next to the fixture and re-runnable.

So a fixture declares its own antidote::

    "_mutation": {
      "note": "write the CORRECT element category, so the diagnostic vanishes",
      "file": "cmd.sh",
      "from": "SOLID QUAD4 1 2 3 4 MAT 1",
      "to":   "ALE2 QUAD4 1 2 3 4 MAT 1"
    }

This script applies that substitution in a scratch copy of the fixture, runs it
through the ordinary evaluator, and asserts the result is NOT ``passed``. The
fixture itself is never modified.

A list of mutations is allowed; each is applied in turn and each must kill the
fixture on its own.

Usage:
    python scripts/mutate_tier2_fixtures.py [backend] [--fixture NAME] [--json OUT]

Exit status is 0 only when every fixture that declares a mutation is killed by
it, and no fixture is missing its declaration.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "scripts" / "tier2_fixtures"
sys.path.insert(0, str(REPO / "scripts"))

from run_tier2_fixtures import _eval_fixture  # noqa: E402


def _mutations(meta: dict) -> list[dict]:
    m = meta.get("_mutation")
    if m is None:
        return []
    return m if isinstance(m, list) else [m]


def check_one(fixture_dir: Path) -> dict:
    meta = json.loads((fixture_dir / "fixture.json").read_text())
    muts = _mutations(meta)
    row = {"fixture": f"{fixture_dir.parent.name}/{fixture_dir.name}",
           "declared": len(muts), "killed": 0, "results": []}
    if not muts:
        row["verdict"] = "NO_MUTATION_DECLARED"
        return row

    for i, mut in enumerate(muts):
        target = mut.get("file") or ("cmd.sh" if (fixture_dir / "cmd.sh").is_file()
                                     else "source.py")
        frm, to = mut.get("from", ""), mut.get("to", "")
        with tempfile.TemporaryDirectory() as td:
            work = Path(td) / fixture_dir.name
            shutil.copytree(fixture_dir, work)
            p = work / target
            if not p.is_file():
                row["results"].append({"i": i, "status": "TARGET_MISSING",
                                       "target": target})
                continue
            text = p.read_text()
            if frm not in text:
                row["results"].append({"i": i, "status": "FROM_NOT_FOUND",
                                       "from": frm[:80]})
                continue
            n = text.count(frm)
            p.write_text(text.replace(frm, to))
            r = _eval_fixture(work, meta)
            killed = r.status != "passed"
            row["killed"] += int(killed)
            row["results"].append({
                "i": i, "occurrences_replaced": n,
                "mutant_status": r.status,
                "killed": killed,
                "note": mut.get("note", ""),
                "missing": [nt for nt in r.notes if "missing in output" in nt][:1],
            })
    row["verdict"] = ("KILLED" if row["killed"] == len(muts) and muts
                      else "SURVIVED")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", nargs="?", default=None)
    ap.add_argument("--fixture", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dirs = sorted(p.parent for p in FIXTURES.glob("*/*/fixture.json"))
    if args.backend:
        dirs = [d for d in dirs if d.parent.name == args.backend]
    if args.fixture:
        dirs = [d for d in dirs if d.name == args.fixture]

    rows = []
    bad = 0
    for d in dirs:
        row = check_one(d)
        rows.append(row)
        glyph = {"KILLED": "✓", "SURVIVED": "✗",
                 "NO_MUTATION_DECLARED": "—"}[row["verdict"]]
        print(f"  {glyph} {row['fixture']}: {row['verdict']}", flush=True)
        for r in row["results"]:
            if not r.get("killed", False):
                print(f"      arm {r['i']}: {r.get('status', r.get('mutant_status'))}"
                      f" {r.get('from', '')}")
        if row["verdict"] != "KILLED":
            bad += 1

    n = len(rows)
    killed = sum(1 for r in rows if r["verdict"] == "KILLED")
    print(f"\nmutation-killed {killed}/{n}; "
          f"{sum(1 for r in rows if r['verdict'] == 'NO_MUTATION_DECLARED')} "
          f"declare no mutation")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
