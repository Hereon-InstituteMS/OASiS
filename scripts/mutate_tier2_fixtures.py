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


def _stage(fixture_dir: Path, td: str) -> Path:
    """Copy the fixture into a scratch tree that still resolves its siblings.

    The first version copied ONLY the fixture directory. 174 of the 4C fixtures
    begin with

        . "$(dirname "$0")/../_lib/preamble.sh"

    and that path is one level ABOVE the fixture, so in the scratch tree it did
    not exist: the preamble never loaded, `run4c` and `upstream` were undefined
    and the copy could not run at all. Both the unmutated and the mutated copy
    then failed, and this script recorded KILLED — the same verdict a real
    detection produces. It would have reported KILLED for a no-op mutation just
    as readily, which makes the whole verdict worthless exactly where it is
    supposed to carry the weight.

    Worse than a bug: the preamble exists to STOP vacuous passes (it aborts with
    FIXTURE_ABORT=no_binary so a missing solver turns a fixture red). A scoping
    slip disabled the anti-vacuity machinery inside the tool whose only job is to
    prove fixtures are not vacuous.

    So the parent directory layout is preserved and every shared sibling — any
    directory whose name starts with `_` — is copied alongside.
    """
    parent = Path(td) / fixture_dir.parent.name
    parent.mkdir(parents=True, exist_ok=True)
    work = parent / fixture_dir.name
    shutil.copytree(fixture_dir, work)
    for sib in fixture_dir.parent.iterdir():
        if sib.is_dir() and sib.name.startswith("_"):
            shutil.copytree(sib, parent / sib.name, dirs_exist_ok=True)
    return work


def check_one(fixture_dir: Path) -> dict:
    meta = json.loads((fixture_dir / "fixture.json").read_text())
    muts = _mutations(meta)
    row = {"fixture": f"{fixture_dir.parent.name}/{fixture_dir.name}",
           "declared": len(muts), "killed": 0, "results": []}
    if not muts:
        row["verdict"] = "NO_MUTATION_DECLARED"
        return row

    # ── vacuity precheck ────────────────────────────────────────────────
    # Run the UNMUTATED copy in the scratch tree first. If it does not pass
    # there, every mutant will "fail" for reasons that have nothing to do with
    # the mutation, and a KILLED verdict would mean only that nothing ran. A
    # harness that cannot tell detection from silence is not evidence.
    with tempfile.TemporaryDirectory() as td:
        base = _stage(fixture_dir, td)
        r0 = _eval_fixture(base, meta)
        if r0.status != "passed":
            row["verdict"] = "VACUOUS_BASELINE"
            row["baseline_status"] = r0.status
            row["baseline_notes"] = [n[:200] for n in r0.notes[:2]]
            return row

    for i, mut in enumerate(muts):
        target = mut.get("file") or ("cmd.sh" if (fixture_dir / "cmd.sh").is_file()
                                     else "source.py")
        frm, to = mut.get("from", ""), mut.get("to", "")
        with tempfile.TemporaryDirectory() as td:
            work = _stage(fixture_dir, td)
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
                 "VACUOUS_BASELINE": "∅",
                 "NO_MUTATION_DECLARED": "—"}[row["verdict"]]
        print(f"  {glyph} {row['fixture']}: {row['verdict']}", flush=True)
        if row["verdict"] == "VACUOUS_BASELINE":
            print(f"      unmutated copy did not pass in the scratch tree "
                  f"({row.get('baseline_status')}); the mutation verdict would "
                  f"mean nothing")
            for n_ in row.get("baseline_notes", []):
                print(f"      {n_}")
        for r in row["results"]:
            if not r.get("killed", False):
                print(f"      arm {r['i']}: {r.get('status', r.get('mutant_status'))}"
                      f" {r.get('from', '')}")
        if row["verdict"] != "KILLED":
            bad += 1

    n = len(rows)
    killed = sum(1 for r in rows if r["verdict"] == "KILLED")
    surv = sum(1 for r in rows if r["verdict"] == "SURVIVED")
    vac = sum(1 for r in rows if r["verdict"] == "VACUOUS_BASELINE")
    nod = sum(1 for r in rows if r["verdict"] == "NO_MUTATION_DECLARED")
    print(f"\nmutation-killed {killed}/{n};  survived {surv};  "
          f"vacuous baseline {vac};  no mutation declared {nod}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
