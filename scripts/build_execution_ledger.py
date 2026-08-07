#!/usr/bin/env python3
"""Record which fixtures actually ran, passed, and were killed by their mutation.

THE NUMBER THIS EXISTS TO PRODUCE
---------------------------------
OASiS can say "1245 of 1377 knowledge warnings have a proof fixture" — 90%. Two
independent audits established what that sentence means: **a fixture exists**.
Not that it ran. Not that it passed.

The evidence of execution is `scripts/scan_results/tier2_results.json`, and it is
stale nearly everywhere:

    ofa-know-4c        116 recorded / 472 fixtures  (3 of them FAILED)
    ofa-verify-fenics  108 / 287
    ofa-verify-ngs     108 / 330
    ofa-know-febio     183 / 184     <- the only one doing this right

Three of four still hold the pre-campaign 108-fixture baseline. The cause is in
`docs/CONSOLIDATION.md`: the runner's `--write-results` flag was parsed and never
read during the campaigns, so every campaign's greenness lives in a session
transcript nobody can re-run.

So the two numbers a reviewer asks for first do not exist:

    N of <total> fixtures pass on commit X
    M of <controls> mutations go red

This produces both, per backend, with the commit sha and a timestamp.

WHY THE MUTATION HALF MATTERS AS MUCH AS THE PASS HALF
-----------------------------------------------------
A fixture greps stdout for strings it expects. Nothing in that mechanism can
tell a fixture that prints the right strings for the right reason from one that
prints them for the wrong reason — a script hardcoding its own expected output
passes identically. The mutation control is the whole difference: remove the
pathology, and the fixture must go red.

So a fixture that passes BOTH mutated and unmutated proves nothing, and finding
those is the most valuable output here. They are reported, never repaired — a
red row with a diagnosis is worth more than a green one somebody engineered.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not judge whether a fixture asserts the RIGHT thing. It cannot. Recent
passes found a DUNE fixture printing `zero_pressure_entry_claim_not_reproduced`
when the claim IS reproduced (its mask sliced the wrong leg of an interpolation),
and a skfem fixture printing `free_boundary_leaves_zero_immediately=True` while
its own next line printed `max_over_run=0.0` — because `np.argmax` returns 0 on
an all-False array. Both pass. Both would be killed by their mutation, because
the mutation flips the assertion either way. Only a person re-reading the probe
catches that class, and pretending otherwise would be the same error one level
up from the one this project exists to fix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "scripts" / "tier2_fixtures"

# The interpreter each backend's fixtures need. Getting this wrong does not
# produce an error — it produces a SKIP that reads as green, which is exactly
# the failure this ledger exists to stop. DUNE is the sharp case: with
# DUNE_PYTHON unset, every DUNE fixture skips silently.
INTERPRETERS = {
    "skfem": os.environ.get("SKFEM_PYTHON",
                            "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python"),
    "ngsolve": os.environ.get("NGSOLVE_PYTHON",
                              "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python"),
    "kratos": os.environ.get("KRATOS_PYTHON", "/usr/bin/python3"),
    "dune": os.environ.get("DUNE_PYTHON", ""),
    "fenics": os.environ.get("FENICS_PYTHON", ""),
    "sparta": os.environ.get("SPARTA_PYTHON",
                             "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python"),
    "febio": os.environ.get("FEBIO_PYTHON",
                            "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python"),
    "coupling": os.environ.get("COUPLING_PYTHON",
                               "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python"),
}


def claim_key(spec: dict) -> str:
    """The claim a fixture defends. Keyed by claim, never by fixture name.

    Fixture names drift and collide: `elasticity_mms_convergence` exists under
    four backends, and the current results file keys by name, so eleven distinct
    runs collapsed into three rows and whichever backend ran last is the one the
    record describes.
    """
    covers = spec.get("covers")
    if isinstance(covers, list) and covers:
        return ",".join(sorted(str(c) for c in covers))
    physics, idx = spec.get("physics"), spec.get("pitfall_index")
    if physics is not None and idx is not None:
        return f"{physics}:{idx}"
    return "(unkeyed)"


def claim_text_hash(backend: str, spec: dict) -> str:
    """Hash of the warning text, so a row is invalidated when it is reworded.

    Without this the ledger silently certifies a claim whose wording — and
    therefore whose meaning — has changed since the run. Returns "" when the
    warning cannot be resolved, which is itself worth recording.
    """
    physics, idx = spec.get("physics"), spec.get("pitfall_index")
    if physics is None or idx is None:
        return ""
    try:
        sys.path.insert(0, str(REPO / "src"))
        from core.registry import get_backend, load_all_backends
        load_all_backends()
        be = get_backend(backend)
        k = be.get_knowledge(str(physics)) if be else None
        pit = (k or {}).get("pitfalls") or []
        if 0 <= int(idx) < len(pit):
            return hashlib.sha256(pit[int(idx)].encode()).hexdigest()[:16]
    except Exception:
        pass
    return ""


def evaluate(out: str, spec: dict) -> tuple[bool, list[str]]:
    """The runner's own rule: every expect present, no forbid present."""
    missing = [e for e in (spec.get("expect_in_output") or [])
               if str(e) not in out]
    hit = [f for f in (spec.get("forbid_in_output") or []) if str(f) in out]
    return (not missing and not hit), missing + [f"FORBIDDEN:{f}" for f in hit]


def run_one(d: Path, backend: str, mutate: bool, timeout: int) -> dict:
    spec = json.loads((d / "fixture.json").read_text())
    interp = INTERPRETERS.get(backend, "")
    src = d / "source.py"
    cmd_sh = d / "cmd.sh"

    if cmd_sh.is_file():
        cmd = ["bash", str(cmd_sh)]
    elif src.is_file():
        if not interp:
            return {"status": "SKIPPED",
                    "reason": f"no interpreter configured for {backend} — set "
                              f"{backend.upper()}_PYTHON. A missing interpreter "
                              f"is NOT a pass."}
        cmd = [interp, str(src)]
    else:
        return {"status": "SKIPPED", "reason": "no runnable artefact"}

    env = dict(os.environ)
    if mutate:
        env["T2_MUTATE"] = "1"
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(d),
                           timeout=timeout, env=env, stdin=subprocess.DEVNULL)
        out = (p.stdout or "") + (p.stderr or "")
        ok, why = evaluate(out, spec)
        return {"status": "PASS" if ok else "FAIL", "rc": p.returncode,
                "seconds": round(time.time() - t0, 1),
                "missing": why[:4]}
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "seconds": timeout}
    except OSError as exc:
        return {"status": "ERROR", "reason": str(exc)[:120]}


def has_control(d: Path, spec: dict) -> bool:
    if spec.get("_mutation") or spec.get("mutations"):
        return True
    for f in d.iterdir():
        if f.is_file() and f.suffix in (".py", ".sh", ".cpp", ".cc"):
            try:
                if "T2_MUTATE" in f.read_text(errors="ignore"):
                    return True
            except OSError:
                pass
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("backends", nargs="*", help="default: all present")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--no-mutation", action="store_true",
                    help="skip the mutation half (faster, halves the evidence)")
    ap.add_argument("--out", default=str(REPO / "data" / "execution_ledger.json"))
    args = ap.parse_args()

    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                         capture_output=True, text=True).stdout.strip()[:12]
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=str(REPO),
                                capture_output=True, text=True).stdout.strip())

    names = args.backends or sorted(p.name for p in FIXTURES.iterdir()
                                    if p.is_dir())
    rows = []
    for backend in names:
        be = FIXTURES / backend
        if not be.is_dir():
            continue
        for d in sorted(be.iterdir()):
            if not (d / "fixture.json").is_file():
                continue
            spec = json.loads((d / "fixture.json").read_text())
            row = {"backend": backend, "fixture": d.name,
                   "claim": claim_key(spec),
                   "claim_text_sha": claim_text_hash(backend, spec),
                   "commit": sha, "tree_dirty": dirty,
                   "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            row["unmutated"] = run_one(d, backend, False, args.timeout)
            if not args.no_mutation and has_control(d, spec):
                row["mutated"] = run_one(d, backend, True, args.timeout)
                # The verdict that matters: green unmutated AND red mutated.
                row["discriminates"] = (row["unmutated"]["status"] == "PASS"
                                        and row["mutated"]["status"] != "PASS")
            else:
                row["mutated"] = {"status": "NO_CONTROL"}
                row["discriminates"] = None
            rows.append(row)
            print(f"  {backend}/{d.name[:44]:<46} "
                  f"{row['unmutated']['status']:<8} "
                  f"mut={row['mutated']['status']}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"commit": sha, "tree_dirty": dirty,
                               "rows": rows}, indent=2))

    passed = sum(1 for r in rows if r["unmutated"]["status"] == "PASS")
    disc = sum(1 for r in rows if r["discriminates"] is True)
    both = [r for r in rows if r["discriminates"] is False]
    print(f"\n=== LEDGER  commit {sha}{' (DIRTY TREE)' if dirty else ''} ===")
    print(f"  {passed} of {len(rows)} fixtures pass")
    print(f"  {disc} of {len(rows)} discriminate (green unmutated, red mutated)")
    if both:
        print(f"\n  {len(both)} PASS BOTH WAYS — these prove nothing:")
        for r in both[:12]:
            print(f"     {r['backend']}/{r['fixture']}")
    print(f"\n  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
