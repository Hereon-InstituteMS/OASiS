#!/usr/bin/env python3
"""Put the execution evidence where the gate reads it.

THE GAP THIS CLOSES
-------------------
`scripts/scan_results/tier2_results.json` is what `test_results_file_is_not_stale`
and the signal-verification gate read as THE pass record. It holds 178 rows
against 1308 fixtures on disk — 14% of the tree — and its own header calls
itself a "PARTIAL RECORD ... never all produced by one run on one host".

Meanwhile `data/execution_ledger_*.json` holds 1033 rows of real execution,
each stamped with the commit its fixtures came from. The evidence exists. Nothing
carried it across, so a reviewer asking "show me this fixture detects what it
claims" got prose.

WHAT IT REFUSES TO DO
---------------------
**It never invents a pass.** A fixture with no ledger row is written as
`not_run`, with the reason, and stays visible in the count. Making the gate green
by asserting things that were never measured is the defect this whole effort
exists to remove; doing it in the importer would be the same error one level up.

**It will not launder a stale ledger.** Rows produced before the mutation arm was
fixed carry no `control` field — they ran `T2_MUTATE=1` against fixtures whose
control is a declared recipe, so the mutated run was byte-identical to the
unmutated one and `discriminates` is meaningless. Those rows are imported with
their pass verdict and an explicit `mutation_evidence: "none — pre-fix ledger"`,
never with a discrimination claim. The coupling ledger is exactly this case: 26
rows, 26 passing, 0 discriminating, because its controls were never applied.

**It distinguishes the two verdicts.** Passing and discriminating are different
evidence and are recorded separately. A fixture that passes proves its own output
matches; only a fixture that also goes red under mutation proves it is looking at
the pathology at all.

PROVENANCE
----------
Every row carries the commit its fixtures were at, the ledger's timestamp, and
the interpreter used. The last matters most for Kratos: three Pythons exist here
and only `/mnt/kratos-tier2/kv/bin/python` both imports Kratos and imports this
repo, so a row without that context cannot be interpreted.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEDGER_DIR = REPO / "data"
RESULTS = REPO / "scripts" / "scan_results" / "tier2_results.json"
FIXTURES = REPO / "scripts" / "tier2_fixtures"


def _fingerprint() -> str | None:
    """The live fixture-inventory fingerprint, or None if it cannot be taken.

    Deliberately not fabricated on failure: a wrong or invented fingerprint
    would make a stale file look current, which is the one thing the gate that
    reads it exists to prevent. None leaves the gate red.
    """
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        from run_tier2_fixtures import fixture_inventory_fingerprint
        return fixture_inventory_fingerprint()
    except Exception:
        return None


def _key(backend: str, spec: dict) -> str:
    """The results file keys `backend::physics::index`."""
    physics, idx = spec.get("physics"), spec.get("pitfall_index")
    if physics is None or idx is None:
        return ""
    return f"{backend}::{physics}::{idx}"


def collect() -> tuple[dict, dict]:
    rows, meta = {}, {}
    for lf in sorted(LEDGER_DIR.glob("execution_ledger*.json")):
        try:
            doc = json.loads(lf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        commit = doc.get("commit", "")
        for r in doc.get("rows", []):
            backend, fixture = r.get("backend", ""), r.get("fixture", "")
            d = FIXTURES / backend / fixture / "fixture.json"
            if not d.is_file():
                continue
            try:
                spec = json.loads(d.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            k = _key(backend, spec)
            if not k:
                continue
            un = (r.get("unmutated") or {}).get("status", "")
            # A ledger row with no `control` key predates the mutation fix: it
            # set T2_MUTATE=1 against recipe-declared controls, so the mutated
            # run was byte-identical and its verdict says nothing.
            pre_fix = "control" not in r
            rows[k] = {
                "status": "passed" if un == "PASS" else un.lower(),
                # BACKEND-QUALIFIED, the same form run_tier2_fixtures.py writes.
                # The bare directory name is not unique across the tree —
                # elasticity_mms_convergence, poisson_mms_convergence and
                # stokes_mms_convergence each exist under several backends — and
                # the results record is read row-wise by fixture_id, so a bare
                # name makes each backend's row overwrite the previous one's.
                # The backend qualifier lives in the dict KEY here, which is
                # enough for this script but not for a row-wise reader, and
                # test_results_keys_distinguish_different_fixtures reads rows.
                "fixture_id": f"{backend}/{fixture}",
                "backend": backend,
                "commit": commit,
                "source_ledger": lf.name,
                "mutation_evidence": (
                    "none — pre-fix ledger, mutation arm never applied"
                    if pre_fix else
                    "discriminates" if r.get("discriminates") is True else
                    "passes both ways — proves nothing"
                    if r.get("discriminates") is False else
                    f"no verdict — {(r.get('mutated') or {}).get('status', '?')}"
                ),
            }
            meta.setdefault(lf.name, {"commit": commit, "rows": 0,
                                      "pre_fix": pre_fix})["rows"] += 1
    return rows, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="write the results file (default: report only)")
    args = ap.parse_args()

    rows, meta = collect()
    on_disk = sorted(d for d in FIXTURES.glob("*/*/fixture.json"))
    missing = []
    for d in on_disk:
        backend = d.parent.parent.name
        try:
            spec = json.loads(d.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        k = _key(backend, spec)
        if k and k not in rows:
            missing.append((k, backend, d.parent.name))

    print(f"  ledgers imported      : {len(meta)}")
    for name, m in sorted(meta.items()):
        flag = "  [PRE-FIX: no mutation evidence]" if m["pre_fix"] else ""
        print(f"      {name:<36} {m['rows']:>4} rows @ {m['commit'][:12]}{flag}")
    print(f"  rows with evidence    : {len(rows)}")
    print(f"  fixtures on disk      : {len(on_disk)}")
    print(f"  NOT RUN (recorded as such, never as a pass): {len(missing)}")
    for k, be, fx in missing[:6]:
        print(f"      {be}/{fx}")

    disc = sum(1 for v in rows.values()
               if v["mutation_evidence"] == "discriminates")
    print(f"  of those, discriminating: {disc}")

    if not args.write:
        print("\n  (report only — pass --write to update the results file)")
        return 0

    for k, be, fx in missing:
        rows[k] = {"status": "not_run", "fixture_id": f"{be}/{fx}", "backend": be,
                   "commit": "", "source_ledger": "",
                   "mutation_evidence": "not run on this host"}
    doc = {
        "_what": "Execution record, imported from data/execution_ledger_*.json.",
        "_honesty": (
            "status=passed means the fixture ran and its expectations matched. "
            "It does NOT mean the fixture detects the pathology it names — that "
            "is mutation_evidence, recorded separately. Rows marked "
            "'pre-fix ledger' come from runs where the mutation arm was never "
            "applied and carry no discrimination claim. Rows marked 'not_run' "
            "were never executed here and are counted as such."),
        "_generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "_ledgers": meta,
        # CURRENCY AND COUNT, because this script is now the writer of the file
        # the floor gate reads. `run_tier2_fixtures.py --write-results` used to
        # be the only writer and stamped both; replacing the file without them
        # left test_tier2_runner_passed_count_meets_floor unable to tell whether
        # the counts describe the fixtures in the tree, which it correctly
        # refuses to read as green. The fingerprint comes from the SAME function
        # the gate compares against, so it goes stale the moment a fixture is
        # added, deleted or edited — which is the point.
        "fixture_fingerprint": _fingerprint(),
        "summary": {
            "passed": sum(1 for v in rows.values() if v["status"] == "passed"),
            "_meaning": (
                "rows whose recorded run matched the fixture's expectations. "
                "Not a discrimination claim — see mutation_evidence per row."),
        },
        "results": rows,
    }
    RESULTS.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\n  written {len(rows)} rows to {RESULTS.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
