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


def _load_shared_matcher():
    """Import the runner's matcher so there is exactly ONE definition of 'present'.

    There used to be two, and they disagreed. The runner lowercases both sides;
    this file compared raw strings. So a fixture whose probe printed
    `space_has_Nedelec=False` against an expectation of `space_has_nedelec=False`
    passed the runner and failed the ledger — same fixture, same output, two
    verdicts. Whichever number got quoted was the one that happened to be run.

    The runner is the authority: it is what the merge gate calls, and its rule
    (case-insensitive, with a value boundary on `key=value` expectations) is the
    stricter of the two on the axis that actually spoofs — value prefixes —
    while being the more forgiving one on capitalisation, which spoofs nothing.

    Degrades loudly rather than crashing when the runner is older than this
    file: agents copy this ledger between worktrees, and a worktree whose branch
    predates the matcher fix has no `_needle_present`. An ImportError there would
    kill a multi-hour sweep at minute zero, so the fallback is the old plain
    substring rule with a banner saying which rule produced the numbers.
    """
    import importlib.util
    p = REPO / "scripts" / "run_tier2_fixtures.py"
    try:
        spec = importlib.util.spec_from_file_location("run_tier2_fixtures", p)
        mod = importlib.util.module_from_spec(spec)
        # Registered under its REAL name for two reasons: dataclasses resolve
        # annotations through sys.modules, and mutate_tier2_fixtures.py does
        # `from run_tier2_fixtures import _eval_fixture` — seeding the name here
        # means it gets THIS hardened runner instead of loading a second copy.
        sys.modules["run_tier2_fixtures"] = mod
        spec.loader.exec_module(mod)
        return mod._needle_present, mod
    except (AttributeError, OSError, ImportError) as exc:
        print(f"!! {p} has no hardened matcher ({type(exc).__name__}); falling "
              f"back to PLAIN SUBSTRING matching. Value-prefix spoofs "
              f"(key=1 matching key=10) are NOT caught in this run.",
              file=sys.stderr)
        return (lambda needle, low: needle.lower() in low), None


_needle_present, _RUNNER = _load_shared_matcher()


def _load_mutation_checker():
    """The recipe-driven mutation checker, or None.

    Two different things are both called "the mutation control", and conflating
    them is why 354 of 395 4C fixtures looked like they proved nothing:

      * an **env branch** — the fixture's own source reads `T2_MUTATE` and
        removes the pathology itself. Running it again with the variable set is
        the whole control.
      * a **declared recipe** — a `_mutation` block naming a file and a from/to
        substitution. `T2_MUTATE=1` does nothing for these: the recipe has to be
        APPLIED, in a staged copy, by mutate_tier2_fixtures.py.

    This ledger previously ran only the first kind and treated the second as if
    it were the first, so 354 fixtures ran byte-identically twice and were
    recorded PASS/PASS. They were never vacuous — they were never mutated.
    """
    import importlib.util
    for cand in (REPO / "scripts" / "mutate_tier2_fixtures.py",
                 FIXTURES.parent / "mutate_tier2_fixtures.py"):
        if not cand.is_file():
            continue
        try:
            spec = importlib.util.spec_from_file_location("_t2mut", cand)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_t2mut"] = mod
            spec.loader.exec_module(mod)
            return mod.check_one
        except Exception as exc:
            print(f"!! could not load {cand}: {exc}", file=sys.stderr)
    return None


def evaluate(out: str, spec: dict) -> tuple[bool, list[str]]:
    """The runner's own rule: every expect present, no forbid present."""
    low = out.lower()
    missing = [e for e in (spec.get("expect_in_output") or [])
               if not _needle_present(str(e), low)]
    # Forbidden strings stay plain substring, matching the runner: a tripwire
    # should fire on the widest reading.
    hit = [f for f in (spec.get("forbid_in_output") or [])
           if str(f).lower() in low]
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


def control_kind(d: Path, spec: dict) -> str | None:
    """Which KIND of mutation control this fixture carries: "env", "recipe", None.

    The env branch is checked first. A fixture may carry both, and when it does
    the in-source branch is the one its author wired to run.
    """
    for f in d.iterdir():
        if f.is_file() and f.suffix in (".py", ".sh", ".cpp", ".cc"):
            try:
                if "T2_MUTATE" in f.read_text(errors="ignore"):
                    return "env"
            except OSError:
                pass
    if spec.get("_mutation") or spec.get("mutations"):
        return "recipe"
    return None


def run_recipe_mutation(d: Path, check_one) -> dict:
    """Apply the declared from/to recipe and report whether it killed the fixture.

    Translated into the same vocabulary as the env-branch arm so both kinds land
    in one column. The distinctions that must NOT collapse into "pass":

      * FROM_NOT_FOUND — the recipe's anchor text is no longer in the file. The
        substitution silently does nothing, so the fixture "survives" a mutation
        that never happened. This is rot, and it reads exactly like a fixture
        that fails to discriminate unless it is named separately.
      * VACUOUS_BASELINE — the fixture does not even pass unmutated in the
        staged copy, so no verdict about the mutation means anything.
    """
    if check_one is None:
        return {"status": "NO_MUTATION_TOOL",
                "reason": "mutate_tier2_fixtures.py not found; declared "
                          "recipes could not be applied — NOT a pass"}
    try:
        row = check_one(d)
    except Exception as exc:
        return {"status": "ERROR", "reason": f"{type(exc).__name__}: {exc}"[:160]}

    verdict = row.get("verdict")
    if verdict == "KILLED":
        # Killed by its own antidote == the mutated run failed, which is what
        # "discriminates" means for the env arm too.
        return {"status": "FAIL", "detail": "recipe killed the fixture"}
    if verdict == "VACUOUS_BASELINE":
        return {"status": "VACUOUS_BASELINE",
                "reason": row.get("baseline_status", ""),
                "notes": row.get("baseline_notes", [])}
    if verdict == "NOT_MUTABLE_DECLARED":
        # A `_mutation` block with a reason and no from/to: the author declared
        # that no faithful mutation exists. It must NOT be credited as
        # discriminating, and it must not be reported as having "survived its
        # own antidote" either — it never had one, and the two are different
        # facts about the fixture.
        return {"status": "NOT_MUTABLE_DECLARED",
                "reason": row.get("note", "")[:300]}
    stale = [r for r in row.get("results", [])
             if r.get("status") in ("FROM_NOT_FOUND", "TARGET_MISSING")]
    if stale:
        return {"status": "MUTATION_STALE",
                "reason": f"{len(stale)} recipe step(s) no longer match the "
                          f"fixture: {stale[0].get('status')}",
                "detail": stale[0]}
    return {"status": "PASS", "detail": "survived its own antidote"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("backends", nargs="*", help="default: all present")
    ap.add_argument("--fixtures", default=None, metavar="DIR",
                    help="fixture root to run instead of this worktree's. The "
                         "campaign fixtures live on the branch that built them, "
                         "so the ledger has to reach across worktrees; copying "
                         "the scripts in instead leaves untracked files behind "
                         "for the consolidation merge to trip over.")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--no-mutation", action="store_true",
                    help="skip the mutation half (faster, halves the evidence)")
    ap.add_argument("--out", default=str(REPO / "data" / "execution_ledger.json"))
    args = ap.parse_args()

    global FIXTURES
    if args.fixtures:
        FIXTURES = Path(args.fixtures).resolve()
        if not FIXTURES.is_dir():
            print(f"no such fixture root: {FIXTURES}")
            return 2

    # Stamp the commit of the tree the FIXTURES came from, not the tree this
    # script lives in. With --fixtures those differ, and recording the script's
    # sha would attribute every result to the wrong branch.
    prov = FIXTURES.parent.parent
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(prov),
                         capture_output=True, text=True).stdout.strip()[:12]
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=str(prov),
                                capture_output=True, text=True).stdout.strip())

    CHECK_ONE = _load_mutation_checker()
    if CHECK_ONE is None:
        print("!! mutate_tier2_fixtures.py not found — fixtures whose control "
              "is a DECLARED RECIPE cannot be exercised. They are recorded "
              "NO_MUTATION_TOOL, never PASS.", file=sys.stderr)

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
            kind = control_kind(d, spec)
            row["control"] = kind or "none"
            if args.no_mutation or kind is None:
                row["mutated"] = {"status": "NO_CONTROL"}
                row["discriminates"] = None
            elif kind == "env":
                row["mutated"] = run_one(d, backend, True, args.timeout)
                mst = row["mutated"]["status"]
                # The verdict that matters: green unmutated AND red mutated.
                #
                # "Red" means FAIL — the fixture ran and its assertions did not
                # hold. It does NOT mean TIMEOUT or ERROR. Crediting those was
                # an overcount: a sweep found five rows whose entire claim to
                # discriminating rested on the mutated run being killed by the
                # clock, which says nothing about whether the mutation was
                # detected. A fixture slow enough to time out under mutation
                # would earn the same credit if its assertion were absent.
                row["discriminates"] = (
                    True if (row["unmutated"]["status"] == "PASS"
                             and mst == "FAIL")
                    else False if mst == "PASS"
                    else None)
            else:
                row["mutated"] = run_recipe_mutation(d, CHECK_ONE)
                st = row["mutated"]["status"]
                # Only FAIL earns credit. A stale recipe, a vacuous baseline or
                # a missing tool are all "we do not know", recorded as None so
                # they can never be counted as either evidence or a defect.
                row["discriminates"] = (
                    True if (row["unmutated"]["status"] == "PASS"
                             and st == "FAIL")
                    else False if st == "PASS"
                    else None)
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
    import collections as _c
    unknown = _c.Counter(r["mutated"]["status"] for r in rows
                         if r["discriminates"] is None)
    print(f"\n=== LEDGER  commit {sha}{' (DIRTY TREE)' if dirty else ''} ===")
    print(f"  {passed} of {len(rows)} fixtures pass")
    print(f"  {disc} of {len(rows)} discriminate (green unmutated, red mutated)")
    if both:
        print(f"\n  {len(both)} PASS BOTH WAYS — these prove nothing:")
        for r in both[:12]:
            print(f"     {r['backend']}/{r['fixture']}")
    if unknown:
        print(f"\n  {sum(unknown.values())} give NO VERDICT (not evidence, "
              f"not a defect — they did not run a real mutation):")
        for st, n in unknown.most_common():
            print(f"     {n:>5}  {st}")
    print(f"\n  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
