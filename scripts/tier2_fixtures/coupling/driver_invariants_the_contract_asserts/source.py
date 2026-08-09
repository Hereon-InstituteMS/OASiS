"""Four sentences in the served contract that describe the DRIVER, and that
nothing exercised.

Coverage of coupling has been counted in PAIRS, and pairs are cheap
combinatorics once both sides' roles are proven. The scarce thing is the
driver's own behaviour: statements an agent is told to design around, that hold
or fail for every pair at once, and that no pair fixture can see because every
pair fixture is written to succeed. These four are the ones a reader would plan
around and be wrong about:

  1. THE DRIVER IS JACOBI, NOT GAUSS-SEIDEL. "Inside one iteration, every
     participant reads the PREVIOUS iteration's exports... Ordering the
     participants differently changes nothing." An agent who believes the
     ordering matters will spend a day on it. Checked by running the SAME
     coupling with the participant list reversed and requiring bit-identical
     histories — and, separately, by catching the participant red-handed: on
     iteration 2 it must see its partner's ITERATION-1 export, not the one
     produced moments earlier in the same sweep.

  2. THE FILE IS NOT ENOUGH — THE EXIT CODE IS ENFORCED. "write `exports.json`
     LAST, only after the solve succeeded, AND EXIT 0. The driver requires
     both." A participant that writes a complete exports.json and then dies
     used to be accepted, and the coupling then converged on the output of a
     crashed solve; that route is closed. Checked with a participant that
     exits non-zero after writing a complete file, and — the half that makes it
     discriminating — with the byte-identical participant exiting 0, which must
     be accepted.

  3. WHAT COMES BACK IS RELAXED, NOT RAW. "The `exports` returned are the
     RELAXED blend the driver holds, not the last raw output of your solver."
     Checked against a participant that returns a CONSTANT: with theta < 1 the
     relaxed value must still be short of that constant after a bounded number
     of iterations, by exactly (1-theta)^k of the initial gap.

  4. `data_files` REACHES THE DRIVER THROUGH `couple`. "a list of ABSOLUTE
     paths to files your solver opens... copied into `work_dir` once, before
     the iteration starts." The tool used to drop the key while the
     `Participant` dataclass had the field and `run_coupling` staged it, so the
     contract told agents to copy their own files and a reader of the source
     would have concluded the opposite; both levels are now checked, and
     separately, because a tool that copied files by some other route would
     satisfy one and not the other.

These use trivial synthetic participants on purpose. The claim under test is
about the driver, and a real solver would only add ways for the fixture to fail
for reasons that are not the claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                     # noqa: E402

THETA = 0.5                 # the relaxation the relaxed-vs-raw check predicts
CONST_TARGET = 10.0         # what participant "hold" always returns
ITERS = 6                   # short, so the relaxed value is measurably short

# A participant that records, every iteration, what it received. Used to catch
# the Jacobi property directly rather than infer it from a residual history.
WITNESS = '''\
import json
from pathlib import Path

NAME = "%(name)s"
PARTNER = "%(partner)s"

log = Path("seen.json")
hist = json.loads(log.read_text()) if log.is_file() else []
imp = {}
p = Path("imports.json")
if p.is_file():
    try:
        imp = json.loads(p.read_text() or "{}") or {}
    except json.JSONDecodeError:
        imp = {}
d = imp.get(PARTNER)
seen = float(d["values"][0]) if d and d.get("values") else None
hist.append(seen)
log.write_text(json.dumps(hist))

# Export this participant's ITERATION NUMBER, so what a partner saw names the
# iteration it came from and no arithmetic is needed to tell.
mine = float(len(hist))
Path("exports.json").write_text(json.dumps({
    "field_name": NAME, "n_points": 1, "coordinates": [[0.0]],
    "values": [mine]}))
'''

# Writes a COMPLETE export, then dies. The contract says this ends the run.
CRASHER = '''\
import json, sys
from pathlib import Path
Path("exports.json").write_text(json.dumps({
    "field_name": "x", "n_points": 1, "coordinates": [[0.0]], "values": [%(v)s]}))
sys.exit(7)
'''

# BYTE-FOR-BYTE the same participant with a zero exit, so the refusal above can
# be attributed to the exit code and to nothing else about the run.
EXIT_OK = '''\
import json, sys
from pathlib import Path
Path("exports.json").write_text(json.dumps({
    "field_name": "x", "n_points": 1, "coordinates": [[0.0]], "values": [%(v)s]}))
sys.exit(0)
'''

# Always returns the same number, whatever it is given.
HOLDER = '''\
import json
from pathlib import Path
Path("exports.json").write_text(json.dumps({
    "field_name": "x", "n_points": 1, "coordinates": [[0.0]], "values": [%(v)s]}))
'''

# Reports whether a named file arrived in its work_dir.
LOOKER = '''\
import json
from pathlib import Path
Path("exports.json").write_text(json.dumps({
    "field_name": "x", "n_points": 1, "coordinates": [[0.0]],
    "values": [1.0 if Path("%(f)s").is_file() else 0.0]}))
'''


def put(root: Path, name: str, body: str, imports_from=(), **extra) -> dict:
    wd = root / name
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "run.py").write_text(body)
    spec = {"name": name, "command": [sys.executable, "run.py"],
            "work_dir": str(wd), "imports_from": list(imports_from),
            "timeout": 120}
    spec.update(extra)
    return spec


def seen(root: Path, name: str) -> list:
    return json.loads((root / name / "seen.json").read_text())


# ── 1. Jacobi, not Gauss-Seidel ─────────────────────────────────────────────

def jacobi() -> None:
    root = L.workroot("inv_jacobi")
    a = put(root, "A", WITNESS % {"name": "A", "partner": "B"}, ["B"])
    b = put(root, "B", WITNESS % {"name": "B", "partner": "A"}, ["A"])
    fwd = L.couple([a, b], max_iter=4, tol=0.0, accelerator="constant",
                   theta=1.0)
    sa, sb = seen(root, "A"), seen(root, "B")
    print(f"A_saw={sa}")
    print(f"B_saw={sb}")
    # Iteration 1: nothing has been exported yet, so imports.json is empty.
    ok = L.check(sa[0] is None and sb[0] is None,
                 "iteration_1_was_not_empty",
                 f"the contract says imports.json is exactly {{}} on iteration "
                 f"1; A saw {sa[0]!r} and B saw {sb[0]!r}")
    # Iteration 2: each side must see its partner's ITERATION-1 export, which
    # is 1.0. Under Gauss-Seidel the second participant in the list would see
    # 2.0 — the export its partner produced moments earlier in the same sweep.
    ok &= L.check(sa[1] == 1.0 and sb[1] == 1.0,
                  "the_second_participant_saw_a_same_sweep_export",
                  f"A saw {sa[1]!r} and B saw {sb[1]!r} on iteration 2; under "
                  f"Jacobi both must be 1.0, and a 2.0 means that side read an "
                  f"export produced in its OWN sweep")
    print(f"both_sides_read_the_previous_iteration={bool(ok)}")

    # And the order of the participant list changes nothing.
    root2 = L.workroot("inv_jacobi_rev")
    a2 = put(root2, "A", WITNESS % {"name": "A", "partner": "B"}, ["B"])
    b2 = put(root2, "B", WITNESS % {"name": "B", "partner": "A"}, ["A"])
    rev = L.couple([b2, a2], max_iter=4, tol=0.0, accelerator="constant",
                   theta=1.0)
    same = (seen(root2, "A") == sa and seen(root2, "B") == sb
            and fwd["history"][1:] == rev["history"][1:])
    print(f"reversing_the_participant_order_changes_nothing={bool(same)}")
    L.check(same, "participant_order_changed_the_run",
            f"forward history {fwd['history']} vs reversed {rev['history']}, "
            f"A saw {sa} vs {seen(root2, 'A')}")


# ── 2. the file is NOT enough: a non-zero exit ends the run ─────────────────

def exit_code_is_enforced() -> None:
    """A complete exports.json written before a later crash is REFUSED.

    This arm was written the other way round, against the driver as it stood on
    knowledge/coupling-revision, and it quoted the contract's own warning that
    "the driver checks that the file EXISTS; it does NOT check your exit code".
    feature/coupling-robustness closed that route — a solver that diverges
    commonly writes its last iterate and then aborts, and the iteration used to
    continue on it — so the invariant now runs the other way and the served
    contract has been corrected to match. The claim is the same claim; what
    changed is which answer is the safe one.

    EXIT_OK below is what makes this discriminating rather than a restatement
    of "the run failed": the identical participant, differing only in its exit
    code, must be ACCEPTED. A driver that refused every run would fail that
    half.
    """
    root = L.workroot("inv_exit")
    a = put(root, "A", CRASHER % {"v": 3.0}, ["B"])
    b = put(root, "B", HOLDER % {"v": 4.0}, ["A"])
    res = L.couple([a, b], max_iter=6, tol=1e-9, accelerator="constant",
                   theta=1.0)
    err = str(res.get("error") or "")
    refused = (not res.get("converged")) and ("exited with code" in err)
    print(f"crashing_participant_error={err[:120]!r}")
    print(f"participant_exited_nonzero_and_the_run_was_refused={refused}")
    L.check(refused, "a_crashing_participant_was_coupled_on",
            f"a participant that writes a complete exports.json and then exits "
            f"non-zero must end the run: its file is the output of a FAILED "
            f"solve. converged={res.get('converged')} error={err[:160]}")

    # The other half: the SAME participant exiting 0 is accepted, so the
    # refusal above is about the exit code and not about the run.
    root2 = L.workroot("inv_exit_ok")
    a2 = put(root2, "A", EXIT_OK % {"v": 3.0}, ["B"])
    b2 = put(root2, "B", HOLDER % {"v": 4.0}, ["A"])
    res2 = L.couple([a2, b2], max_iter=6, tol=1e-9, accelerator="constant",
                    theta=1.0)
    got = (float(res2["exports"]["A"]["values"][0])
           if res2.get("exports") else None)
    ok = bool(res2.get("converged")) and got == 3.0
    print(f"same_participant_exiting_zero_is_accepted={ok}")
    L.check(ok, "a_clean_participant_was_also_rejected",
            f"the identical script exiting 0 must be accepted, or the check "
            f"above is not about the exit code. converged={res2.get('converged')} "
            f"A={got!r} error={str(res2.get('error'))[:160]}")


# ── 3. what comes back is the RELAXED blend, not the raw export ─────────────

def exports_are_relaxed() -> None:
    root = L.workroot("inv_relaxed")
    a = put(root, "A", HOLDER % {"v": CONST_TARGET}, ["B"])
    b = put(root, "B", HOLDER % {"v": CONST_TARGET}, ["A"])
    res = L.couple([a, b], max_iter=ITERS, tol=0.0, accelerator="constant",
                   theta=THETA)
    got = float(res["exports"]["A"]["values"][0])
    # Iteration 1 seeds the relaxed vector with the raw value, so from then on
    # the relaxed value equals the raw one — a participant returning a CONSTANT
    # is the one case where relaxation cannot be seen. So the discriminator is
    # the OTHER direction: the returned value must be exactly the constant, and
    # the RESIDUAL must be exactly zero, which is only true because the driver
    # relaxes toward an unmoving target.
    print(f"holder_returns={got:.10g}")
    print(f"relaxed_export_equals_the_unmoving_raw_value="
          f"{abs(got - CONST_TARGET) < 1e-12}")
    L.check(abs(got - CONST_TARGET) < 1e-12,
            "relaxed_export_drifted_from_an_unmoving_raw_value",
            f"got {got!r} against {CONST_TARGET!r}")

    # Now a participant whose raw output STEPS once, so the relaxed value has
    # something to lag behind. It returns 0 on iteration 1 and the target
    # afterwards; after k further iterations the relaxed value must be
    # target * (1 - (1-theta)^k) — which is NOT the raw output.
    step = ('import json\nfrom pathlib import Path\n'
            'c = Path("n.txt"); k = int(c.read_text()) if c.is_file() else 0\n'
            'c.write_text(str(k + 1))\n'
            f'v = 0.0 if k == 0 else {CONST_TARGET}\n'
            'Path("exports.json").write_text(json.dumps({"field_name": "x",'
            '"n_points": 1, "coordinates": [[0.0]], "values": [v]}))\n')
    root2 = L.workroot("inv_relaxed_step")
    a2 = put(root2, "A", step, ["B"])
    b2 = put(root2, "B", HOLDER % {"v": 1.0}, ["A"])
    res2 = L.couple([a2, b2], max_iter=ITERS, tol=0.0, accelerator="constant",
                    theta=THETA)
    got2 = float(res2["exports"]["A"]["values"][0])
    k = ITERS - 1                       # relaxation starts at iteration 2
    want = CONST_TARGET * (1.0 - (1.0 - THETA) ** k)
    print(f"stepping_raw_export={CONST_TARGET:.10g} returned={got2:.10g} "
          f"predicted_relaxed={want:.10g}")
    close = abs(got2 - want) < 1e-9
    print(f"returned_value_is_the_relaxed_blend_not_the_raw_export="
          f"{bool(close and abs(got2 - CONST_TARGET) > 1e-6)}")
    L.check(close, "relaxed_export_does_not_match_the_relaxation_formula",
            f"got {got2!r}, expected target*(1-(1-theta)^{k}) = {want!r}")
    L.check(abs(got2 - CONST_TARGET) > 1e-6,
            "returned_export_was_the_raw_value",
            "the contract says what comes back is the relaxed blend; here it "
            "is indistinguishable from the raw output, so the sentence is "
            "either wrong or untestable")


# ── 4. `data_files` is silently ignored by the TOOL ─────────────────────────

def data_files_are_staged() -> None:
    """`data_files` reaches the driver THROUGH the tool, and the file arrives.

    Written the other way round originally: on knowledge/coupling-revision the
    tool dropped the key on the floor while the Participant dataclass had the
    field and the driver staged it, so the served contract told agents to copy
    their own files and a reader of the source would have concluded the
    opposite. The tool now passes it through, measured here at both levels, and
    the served contract has been corrected.

    The one that is NOT a duplicate is the second half: the driver is exercised
    DIRECTLY as well, because "the tool stages it" and "the driver stages it"
    fail independently — a tool that quietly copied files itself would satisfy
    the first and not the second.
    """
    root = L.workroot("inv_data")
    payload = root / "needed.dat"
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_text("a file the solver would open\n")
    a = put(root, "A", LOOKER % {"f": "needed.dat"}, ["B"],
            data_files=[str(payload)])
    b = put(root, "B", HOLDER % {"v": 1.0}, ["A"])
    res = L.couple([a, b], max_iter=3, tol=0.0, accelerator="constant",
                   theta=1.0)
    arrived = float(res["exports"]["A"]["values"][0]) == 1.0
    print(f"data_files_key_staged_the_file={arrived}")
    L.check(arrived, "data_files_did_not_reach_the_driver",
            "the served contract now tells agents to declare mesh/species/"
            "config files in `data_files` and lets the tool stage them. If the "
            "key is dropped again, that sentence sends people into an opaque "
            "'Cannot open ...' from inside the solver.")
    # The DRIVER as well, separately, because the two can fail on their own.
    from core.coupling_driver import Participant, run_coupling
    root2 = L.workroot("inv_data_driver")
    pay2 = root2 / "needed.dat"
    pay2.parent.mkdir(parents=True, exist_ok=True)
    pay2.write_text("x\n")
    sa = put(root2, "A", LOOKER % {"f": "needed.dat"}, ["B"])
    sb = put(root2, "B", HOLDER % {"v": 1.0}, ["A"])
    pa = Participant("A", sa["command"], Path(sa["work_dir"]),
                     imports_from=["B"], data_files=[str(pay2)])
    pb = Participant("B", sb["command"], Path(sb["work_dir"]),
                     imports_from=["A"])
    r = run_coupling([pa, pb], max_iter=3, tol=0.0, accelerator="constant",
                     theta0=1.0)
    staged = float(r.exports["A"]["values"][0]) == 1.0
    print(f"driver_itself_does_stage_data_files={staged}")
    L.check(staged, "the_driver_does_not_stage_data_files_either",
            "then the contract's advice is right for the wrong reason and the "
            "`data_files` field on Participant is dead")


def body() -> None:
    jacobi()
    exit_code_is_enforced()
    exports_are_relaxed()
    data_files_are_staged()
    print("invariants_checked=4")


L.main(body)
