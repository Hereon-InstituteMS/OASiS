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

  2. THE FILE IS THE RESULT, NOT THE EXIT CODE. "The driver checks that the
     file EXISTS; it does NOT check your exit code. A complete file written
     before a later crash is accepted as a result." That is the contract's own
     warning and it is a silent-wrong route: a participant that writes
     exports.json and then dies leaves a coupling that converges on stale
     physics. Checked with a participant that exits non-zero every iteration
     after writing a complete file.

  3. WHAT COMES BACK IS RELAXED, NOT RAW. "The `exports` returned are the
     RELAXED blend the driver holds, not the last raw output of your solver."
     Checked against a participant that returns a CONSTANT: with theta < 1 the
     relaxed value must still be short of that constant after a bounded number
     of iterations, by exactly (1-theta)^k of the initial gap.

  4. `data_files` IS SILENTLY IGNORED BY `couple`. "an extra key in the spec is
     silently ignored. Copy every mesh/species/config file your solver opens
     into `work_dir` yourself." The `Participant` dataclass HAS the field and
     `run_coupling` stages it; the TOOL does not pass it through. A reader who
     saw the dataclass would conclude the opposite. Checked by handing `couple`
     a `data_files` entry and asserting the file does not arrive.

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

# Writes a COMPLETE export, then dies. The contract says this is accepted.
CRASHER = '''\
import json, sys
from pathlib import Path
Path("exports.json").write_text(json.dumps({
    "field_name": "x", "n_points": 1, "coordinates": [[0.0]], "values": [%(v)s]}))
sys.exit(7)
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


# ── 2. the file is the result, the exit code is not read ────────────────────

def exit_code_is_not_read() -> None:
    root = L.workroot("inv_exit")
    a = put(root, "A", CRASHER % {"v": 3.0}, ["B"])
    b = put(root, "B", HOLDER % {"v": 4.0}, ["A"])
    res = L.couple([a, b], max_iter=6, tol=1e-9, accelerator="constant",
                   theta=1.0)
    got = float(res["exports"]["A"]["values"][0]) if res.get("exports") else None
    accepted = bool(res.get("converged")) and got == 3.0
    print(f"participant_exited_nonzero_and_was_still_accepted={accepted}")
    L.check(accepted, "a_crashing_participant_was_rejected",
            f"the contract warns that a complete exports.json written before a "
            f"later crash IS accepted — that is the silent-wrong route it "
            f"names. converged={res.get('converged')} A={got!r} "
            f"error={str(res.get('error'))[:160]}")


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

def data_files_ignored() -> None:
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
    L.check(not arrived, "data_files_is_no_longer_ignored",
            "the served contract says `data_files` is NOT supported by this "
            "tool and the extra key is silently ignored, so an agent is told "
            "to copy its own files. If the tool now stages them, that sentence "
            "sends people to do unnecessary work and must be corrected.")
    # And the underlying driver DOES support it — which is exactly why the
    # tool-level silence is worth a fixture: the dataclass field exists and a
    # reader of the source would conclude the opposite of the truth.
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
    exit_code_is_not_read()
    exports_are_relaxed()
    data_files_ignored()
    print("invariants_checked=4")


L.main(body)
