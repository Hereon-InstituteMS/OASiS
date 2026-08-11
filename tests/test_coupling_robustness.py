"""What a partitioned coupling gets WRONG quietly, and that OASiS now says so.

Every case here was first run against the previous machinery and reported as a
clean success, or reported nothing at all. They are the adversarial suite for
`couple`: sign convention, unit mismatch, non-matching interfaces, a participant
that exits 0 having done nothing, a crash mid-iteration, NaN in the exchanged
data, non-convergence, one-way/two-way confusion, stale data, and relaxation.

TWO RULES THESE TESTS ENFORCE ABOUT THEMSELVES.

  * A test that passes whether or not the check exists is decoration. The
    `*_discriminates` tests therefore stub the check out and assert the verdict
    flips back to trustworthy — the check is what makes the difference, in this
    suite, mechanically.
  * A check that cannot look at anything must never report a pass. Findings
    (`validation`) and coverage (`checks_not_run`) are separate channels, and
    the second is asserted to reach the verdict text.

The participants are deliberately small, opaque maps rather than FEM solves: the
driver is physics-agnostic and these tests are about the driver. Cross-code runs
with dolfinx / NGSolve / scikit-fem / 4C are how the physics side is exercised.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.coupling_driver import Participant, run_coupling          # noqa: E402
from core.quality_checks import (                                    # noqa: E402
    check_coupling_directionality, check_interface_balance,
    check_interface_meshes, check_monolithic_consistency,
    check_participant_responsiveness, check_residual_blocks,
    check_returncodes,
)


# ── participants ─────────────────────────────────────────────────────────────
# A: x <- 0.5*y + 1 ;  B: y <- 0.5*x + 2  =>  fixed point x = 8/3, y = 10/3.
# Each also exports a normal flux with respect to its own outward normal, so the
# two cancel at the fixed point exactly as a conservative interface must.
_A = """\
import json
from pathlib import Path
imp = json.loads(Path("imports.json").read_text() or "{}")
y = imp["B"]["values"][0] if "B" in imp else 0.0
x = 0.5 * y + 1.0
json.dump({"field_name": "x", "n_points": 1, "coordinates": [[0.0, 0.0]],
           "values": [x], "normal_fluxes": [x]}, open("exports.json", "w"))
"""
_B = """\
import json
from pathlib import Path
imp = json.loads(Path("imports.json").read_text() or "{}")
x = imp["A"]["values"][0] if "A" in imp else 0.0
json.dump({"field_name": "y", "n_points": 1, "coordinates": [[0.0, 0.0]],
           "values": [OFFSET + SCALE * x + 2.0], "normal_fluxes": [-x]},
          open("exports.json", "w"))
"""
# The un-split solve of the SAME system, in one place: substitute and solve.
_MONO = """\
import json
x = (1.0 + 0.5 * 2.0) / (1.0 - 0.25)
json.dump({"field_name": "x", "n_points": 1, "coordinates": [[0.0, 0.0]],
           "values": [x]}, open("monolithic.json", "w"))
"""


def _b(offset: float = 0.0, scale: float = 0.5) -> str:
    return _B.replace("OFFSET", repr(offset)).replace("SCALE", repr(scale))


def _mk(tmp_path: Path, name: str, body: str) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(body)
    return d


def _parts(tmp_path, a_body=_A, b_body=None, a_from=("B",), b_from=("A",)):
    b_body = _b() if b_body is None else b_body
    return [
        Participant("A", [sys.executable, "run.py"], _mk(tmp_path, "A", a_body),
                    imports_from=list(a_from)),
        Participant("B", [sys.executable, "run.py"], _mk(tmp_path, "B", b_body),
                    imports_from=list(b_from)),
    ]


# ── the tool, invoked for real ───────────────────────────────────────────────
def _tool(name: str):
    from mcp.server.fastmcp import FastMCP
    from tools.consolidated import register_consolidated_tools
    mcp = FastMCP("t")
    register_consolidated_tools(mcp)
    return mcp._tool_manager._tools[name].fn


@pytest.fixture(autouse=True)
def _fresh_critic(monkeypatch):
    import tools.consolidated as C
    from core.critic_gate import CriticRegistry
    monkeypatch.setattr(C, "_CRITIC_REGISTRY", CriticRegistry())


def _spec(parts):
    return json.dumps([{"name": p.name, "command": p.command,
                        "work_dir": str(p.work_dir),
                        "imports_from": p.imports_from} for p in parts])


def _couple(parts, *, reviewed=True, **kw):
    """Call the real `couple` tool, with a critic review on record by default.

    Without the review every verdict is NOT VERIFIED for that reason alone, and
    a check could be removed without any test noticing. The interesting question
    is whether the NUMERICAL checks flip the verdict, so the critic is satisfied
    and then held constant.
    """
    from tools.consolidated import _coupling_setup_text, _CRITIC_REGISTRY
    from core.critic_gate import setup_digest
    kw.setdefault("max_iter", 60)
    kw.setdefault("tol", 1e-8)
    args = dict(participants=_spec(parts), **kw)
    if reviewed:
        setup = _coupling_setup_text(
            participants=args["participants"], max_iter=args["max_iter"],
            tol=args["tol"], accelerator=args.get("accelerator", "aitken"),
            theta=args.get("theta", 0.5), monolithic=args.get("monolithic", ""),
            probe=args.get("probe", True))
        _CRITIC_REGISTRY.submit_review(
            solver="couple", digest=setup_digest("couple", setup),
            findings="Reviewed the participant maps, the units on both sides "
                 "of the interface, the boundary conditions and the "
                 "interface discretisation; no issue found that would "
                 "invalidate the coupled run.")
    loop = asyncio.new_event_loop()
    try:
        return json.loads(loop.run_until_complete(_tool("couple")(
            critic_approved=True, **args)))
    finally:
        loop.close()


def _hits(d, needle):
    return [w for w in (d.get("validation") or []) if needle in w]


# ═══════════════════════════════════════════════════════════════════════════
# The control: a CORRECT coupling must come back clean. Every check below is
# worthless if it also fires on this.
# ═══════════════════════════════════════════════════════════════════════════
def test_a_correct_coupling_is_verified_and_produces_no_findings(tmp_path):
    d = _couple(_parts(tmp_path))
    assert d["converged"] is True
    assert abs(d["exports"]["A"]["values"][0] - 8 / 3) < 1e-6
    assert d["validation"] == [], d["validation"]
    assert d["trustworthy_result"] is True
    assert d["responsiveness"] == {"A": "responsive", "B": "responsive"}
    assert d["returncodes"] == {"A": 0, "B": 0}


# ═══════════════════════════════════════════════════════════════════════════
# 4 + 9. A participant that exits 0 having done nothing / re-serves a cached
# answer. The coupling "converges" at iteration 2 with residual exactly 0.
# ═══════════════════════════════════════════════════════════════════════════
_NOOP = """\
import json
json.dump({"field_name": "y", "n_points": 1, "coordinates": [[0.0, 0.0]],
           "values": [0.0], "normal_fluxes": [0.0]}, open("exports.json", "w"))
"""


def test_do_nothing_participant_converges_but_is_not_a_result(tmp_path):
    d = _couple(_parts(tmp_path, b_body=_NOOP))
    # It really does converge, and the residual really is zero. That is the
    # problem: nothing about the iteration itself is wrong.
    assert d["converged"] is True and d["residual"] == 0.0
    assert d["responsiveness"]["B"] == "unresponsive"
    assert _hits(d, "byte-identical")
    assert d["trustworthy_result"] is False


# Values-only exchange: no fluxes, so the interface-balance check has nothing to
# look at and the responsiveness check is the ONLY thing standing between a
# do-nothing participant and a VERIFIED verdict. That is the case the
# discrimination test has to use.
_A_NOFLUX = _A.replace(', "normal_fluxes": [x]', "")
_NOOP_NOFLUX = """\
import json
json.dump({"field_name": "y", "n_points": 1, "coordinates": [[0.0, 0.0]],
           "values": [0.0]}, open("exports.json", "w"))
"""


def test_do_nothing_detection_discriminates(tmp_path, monkeypatch):
    """Two independent checks catch this, and BOTH are load-bearing.

    Disabling either one alone still catches it — that is the point of having
    two. Disabling both produces a VERIFIED coupling in which one participant
    did nothing, which is what they are for.
    """
    import core.quality_checks as Q
    dead = lambda *_a, **_k: ([], [])                       # noqa: E731

    both = _couple(_parts(tmp_path / "b", a_body=_A_NOFLUX, b_body=_NOOP_NOFLUX))
    assert both["converged"] is True and both["trustworthy_result"] is False

    # only the interface-sensitivity probe left
    monkeypatch.setattr(Q, "check_participant_responsiveness", dead)
    probe_only = _couple(_parts(tmp_path / "p", a_body=_A_NOFLUX,
                                b_body=_NOOP_NOFLUX))
    assert probe_only["trustworthy_result"] is False
    assert any("NOT COUPLED" in w for w in probe_only["validation"])
    monkeypatch.undo()

    # only the byte-identity responsiveness check left
    resp_only = _couple(_parts(tmp_path / "r", a_body=_A_NOFLUX,
                               b_body=_NOOP_NOFLUX), probe=False)
    assert resp_only["trustworthy_result"] is False
    assert any("byte-identical" in w for w in resp_only["validation"])

    # neither
    monkeypatch.setattr(Q, "check_participant_responsiveness", dead)
    neither = _couple(_parts(tmp_path / "n", a_body=_A_NOFLUX,
                             b_body=_NOOP_NOFLUX), probe=False)
    assert neither["converged"] is True
    assert neither["trustworthy_result"] is True, (
        "with both stubbed out, a participant that did nothing produces a "
        "VERIFIED coupling — which is what they are for")


# ── the checks the critic pass added, each with its discrimination ───────────
_STATEFUL_B = """\
import json
from pathlib import Path
n = int(Path("n.txt").read_text()) + 1 if Path("n.txt").exists() else 1
Path("n.txt").write_text(str(n))
y = 10.0 * (1.0 - 0.5 ** n)          # converges on its own, reads nothing
json.dump({"field_name": "y", "n_points": 1, "coordinates": [[0.0, 0.0]],
           "values": [y]}, open("exports.json", "w"))
"""


def test_a_participant_with_hidden_state_is_caught_by_the_probe(tmp_path):
    """It never opens imports.json, but its export moves every iteration, so it
    reads as responsive throughout and the coupling converges."""
    d = _couple(_parts(tmp_path, a_body=_A_NOFLUX, b_body=_STATEFUL_B))
    assert d["converged"] is True
    assert d["responsiveness"]["B"] == "responsive"      # the cheap check is fooled
    assert any("NOT A FUNCTION" in w for w in d["validation"])
    assert d["trustworthy_result"] is False


def test_hidden_state_detection_discriminates(tmp_path):
    d = _couple(_parts(tmp_path, a_body=_A_NOFLUX, b_body=_STATEFUL_B),
                probe=False)
    assert d["converged"] is True
    assert d["trustworthy_result"] is True, (
        "without the probe, a participant carrying hidden state between calls "
        "converges and is stamped VERIFIED")
    assert any("NOT probed" in c for c in d["checks_not_run"])


# B pins its physics to a constant but echoes the imported value back in its
# flux, so the STACKED export responds fully and the frozen half is invisible.
_FROZEN_BLOCK_B = """\
import json
from pathlib import Path
imp = json.loads(Path("imports.json").read_text() or "{}")
x = imp["A"]["values"][0] if "A" in imp else 0.0
json.dump({"field_name": "y", "n_points": 1, "coordinates": [[0.0, 0.0]],
           "values": [10.0], "normal_fluxes": [-x]}, open("exports.json", "w"))
"""


def test_a_frozen_block_inside_a_responsive_export_is_caught(tmp_path):
    d = _couple(_parts(tmp_path, b_body=_FROZEN_BLOCK_B))
    assert d["converged"] is True
    assert d["responsiveness"]["B"] == "responsive"
    sens = d["interface_sensitivity"]["B"]
    assert sens["S"] > 1e-9, "the export as a whole DOES respond"
    assert any("do NOT respond" in w for w in d["validation"])
    assert d["trustworthy_result"] is False


def test_frozen_block_detection_discriminates(tmp_path):
    d = _couple(_parts(tmp_path, b_body=_FROZEN_BLOCK_B), probe=False)
    assert d["converged"] is True and d["trustworthy_result"] is True


def test_the_probe_costs_nothing_in_correctness_on_a_good_coupling(tmp_path):
    """It must not fire on a real coupling — the control for all of the above."""
    d = _couple(_parts(tmp_path))
    assert d["validation"] == [], d["validation"]
    assert d["trustworthy_result"] is True
    for name, rec in d["interface_sensitivity"].items():
        assert rec["S"] > 1e-6, (name, rec)
        assert rec["noise"] == 0.0, "a deterministic solver must repeat exactly"


def test_a_genuinely_converged_participant_is_not_called_unresponsive(tmp_path):
    """The check must key on 'output frozen while input moved', not on 'output
    stopped moving' — otherwise every converged run trips it."""
    d = _couple(_parts(tmp_path), tol=1e-14, max_iter=90)
    assert d["responsiveness"] == {"A": "responsive", "B": "responsive"}
    assert not _hits(d, "byte-identical")


# ═══════════════════════════════════════════════════════════════════════════
# 8. One-way vs two-way confusion.
# ═══════════════════════════════════════════════════════════════════════════
def test_partner_name_typo_is_refused_not_silently_dropped(tmp_path):
    parts = _parts(tmp_path, b_from=("Bee",))
    r = run_coupling(parts, max_iter=5, tol=1e-8)
    assert r.converged is False
    assert "no such participant" in (r.error or "")
    assert r.iterations == 0, "nothing must run before the graph is checked"


def test_partner_name_typo_would_otherwise_converge_cleanly(tmp_path):
    """Discrimination: with the edge dropped instead of refused — which is what
    the driver used to do — the run converges and looks excellent."""
    parts = _parts(tmp_path, b_from=())          # the dropped-edge outcome
    r = run_coupling(parts, max_iter=20, tol=1e-8)
    assert r.converged is True and r.residual < 1e-8
    assert abs(r.exports["A"]["values"][0] - 8 / 3) > 0.5, (
        "and the answer it converges to is not the coupled one")


def test_one_way_graph_is_flagged_when_iterated(tmp_path):
    d = _couple(_parts(tmp_path, b_from=()))
    assert d["converged"] is True                 # it converges beautifully
    assert _hits(d, "ONE-WAY")
    assert d["trustworthy_result"] is False


def test_one_way_declared_as_a_single_pass_is_not_flagged():
    graph = {"participants": ["A", "B"], "declared_edges": {"A": ["B"], "B": []}}
    iterated, _ = check_coupling_directionality(graph, max_iter=20)
    single, not_run = check_coupling_directionality(graph, max_iter=1)
    assert iterated and not single
    assert not_run, "a declared single pass must still say no fixed point was found"


# ═══════════════════════════════════════════════════════════════════════════
# 5. A participant that crashes mid-iteration but leaves exports.json behind.
# ═══════════════════════════════════════════════════════════════════════════
_CRASH_AFTER_WRITE = _b() + """
import sys
from pathlib import Path
n = int(Path("n.txt").read_text()) + 1 if Path("n.txt").exists() else 1
Path("n.txt").write_text(str(n))
if n >= 3:
    sys.stderr.write("solver diverged\\n")
    sys.exit(1)
"""


def test_nonzero_exit_is_refused_even_with_a_well_formed_export(tmp_path):
    parts = _parts(tmp_path, b_body=_CRASH_AFTER_WRITE)
    r = run_coupling(parts, max_iter=20, tol=1e-14)
    assert r.converged is False
    assert "exited with code 1" in (r.error or "")
    assert r.iterations == 3
    assert check_returncodes(r.returncodes)[0], "and the validator names it"


def test_returncode_check_discriminates():
    assert check_returncodes({"A": 0, "B": 1})[0]
    assert not check_returncodes({"A": 0, "B": 0})[0]
    # "no exit codes recorded" is NOT a pass.
    findings, not_run = check_returncodes({})
    assert not findings and not_run


def test_a_participant_that_hangs_is_killed_and_reported(tmp_path):
    hang = "import time\ntime.sleep(600)\n"
    parts = _parts(tmp_path, b_body=hang)
    parts[1].timeout = 2
    r = run_coupling(parts, max_iter=3, tol=1e-8)
    assert r.converged is False and "timed out" in (r.error or "")


# ═══════════════════════════════════════════════════════════════════════════
# 6. NaN / Inf in the exchanged data.
# ═══════════════════════════════════════════════════════════════════════════
def test_nonfinite_flux_is_not_silently_passed_by_the_balance_check():
    """`nan > rtol` is False, so an unguarded comparison reports NOTHING on the
    most broken data the check can be handed."""
    a = {"normal_fluxes": [1.0, 1.0]}
    bad = {"normal_fluxes": [float("nan"), -1.0]}
    good = {"normal_fluxes": [-1.0, -1.0]}
    assert check_interface_balance(a, bad)
    assert "could NOT be evaluated" in check_interface_balance(a, bad)[0]
    assert not check_interface_balance(a, good)


def test_a_component_that_is_zero_on_both_sides_is_not_an_imbalance():
    """A validator that condemns a CORRECT coupling is worse than one that misses.

    The per-component branch judges each component on its own scale. A component
    that is zero on both sides — the tangential traction of a frictionless
    interface, which is the ordinary case and not a corner one — then has a
    denominator made of roundoff, and two entries that should cancel report tens
    of percent. Measured before the floor: a normal traction of 1e5 cancelling
    exactly, with 1e-17 tangential components, came back "NOT balanced ... 91.6%".

    Remove the `floor` argument from the recursive call in
    check_interface_balance and this test fails.
    """
    n = 11
    rng = np.random.default_rng(0)
    fa = np.stack([np.full(n, 1.0e5), rng.normal(0, 3e-17, n)], axis=1)
    fb = np.stack([np.full(n, -1.0e5), rng.normal(0, 3e-17, n)], axis=1)
    co = [[float(x), 0.0] for x in np.linspace(0, 1, n)]
    assert check_interface_balance(
        {"coordinates": co, "normal_fluxes": fa.tolist()},
        {"coordinates": co, "normal_fluxes": fb.tolist()}) == []
    # and the floor must not blind the check to a REAL imbalance in the same
    # small component: comp1 off by a factor of two is still caught.
    fb2 = np.stack([np.full(n, -1.0e5), np.full(n, -0.25)], axis=1)
    fa2 = np.stack([np.full(n, 1.0e5), np.full(n, 0.5)], axis=1)
    got = check_interface_balance(
        {"coordinates": co, "normal_fluxes": fa2.tolist()},
        {"coordinates": co, "normal_fluxes": fb2.tolist()})
    assert got and "[1]" in got[0], got


def test_nonfinite_export_is_reported_by_the_tool(tmp_path):
    nan_b = _b().replace('"normal_fluxes": [-x]', '"normal_fluxes": [float("nan")]')
    d = _couple(_parts(tmp_path, b_body=nan_b), max_iter=5)
    assert any("non-finite" in w for w in d["validation"])
    assert d["trustworthy_result"] is False


def test_nonfinite_warnings_do_not_bury_every_other_finding(tmp_path):
    """One NaN used to produce a warning per participant per iteration — about
    eighty near-identical lines, with every specific finding lost among them."""
    nan_b = _b().replace('"normal_fluxes": [-x]', '"normal_fluxes": [float("nan")]')
    parts = _parts(tmp_path, b_body=nan_b)
    r = run_coupling(parts, max_iter=40, tol=1e-8)
    assert len(r.warnings) <= 6, r.warnings
    assert any("suppressed" in w for w in r.warnings)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Non-convergence, and the residual that is not representative.
# ═══════════════════════════════════════════════════════════════════════════
def test_nonconvergence_is_reported_as_failure_not_as_a_result(tmp_path):
    d = _couple(_parts(tmp_path, b_body=_b(scale=3.0)),
                max_iter=8, tol=1e-12, accelerator="constant", theta=1.0)
    assert d["converged"] is False
    assert _hits(d, "NOT CONVERGED")
    assert d["trustworthy_result"] is False


def test_a_large_settled_block_cannot_hide_a_small_moving_one():
    """The FSI/TSI conditioning trap: one global relative norm is set by the
    largest-magnitude block, so a small quantity can be 100% wrong underneath a
    converged residual."""
    blocks = {"A.values[0]": 1e-12, "A.values[1]": 0.4, "B.normal_fluxes": 1e-12}
    findings, not_run = check_residual_blocks(blocks, tol=1e-8)
    assert findings and "A.values[1]" in findings[0]
    assert not not_run
    clean, _ = check_residual_blocks({k: 1e-12 for k in blocks}, tol=1e-8)
    assert not clean
    # and an empty record is NOT a pass
    assert check_residual_blocks({}, tol=1e-8)[1]


def test_block_residuals_are_recorded_per_component(tmp_path):
    """A TSI interface carries temperature and displacement inside ONE `values`
    array on wildly different scales; per-array residuals would still lump them."""
    two = _A.replace('"values": [x], "normal_fluxes": [x]',
                     '"values": [[x, 1e-9 * x]], "normal_fluxes": [x]')
    parts = _parts(tmp_path, a_body=two)
    parts[1].work_dir  # noqa: B018 - keep B as-is; it reads values[0] via [0]
    b2 = _b().replace('imp["A"]["values"][0]', 'imp["A"]["values"][0][0]')
    parts = _parts(tmp_path, a_body=two, b_body=b2)
    r = run_coupling(parts, max_iter=30, tol=1e-10)
    assert "A.values[0]" in r.block_residuals
    assert "A.values[1]" in r.block_residuals


# ═══════════════════════════════════════════════════════════════════════════
# 1 + 2. Sign convention and unit mismatch: same imbalance number, opposite
# diagnoses, and the agent is sent looking in a different place by each.
# ═══════════════════════════════════════════════════════════════════════════
def test_matching_magnitudes_with_agreeing_signs_reads_as_a_sign_convention_error():
    w = check_interface_balance({"normal_fluxes": [3.0]}, {"normal_fluxes": [3.0]})
    assert w and "SIGN-CONVENTION" in w[0]
    assert "same number on both sides" in w[0]


def test_a_clean_power_of_ten_reads_as_a_unit_mismatch():
    w = check_interface_balance({"normal_fluxes": [-100.0]},
                                {"normal_fluxes": [0.1]})
    assert w and "UNIT MISMATCH" in w[0]
    assert "SIGN-CONVENTION" not in w[0]


def test_an_ordinary_imbalance_names_neither():
    w = check_interface_balance({"normal_fluxes": [-100.0]},
                                {"normal_fluxes": [83.0]})
    assert w and "UNIT MISMATCH" not in w[0] and "SIGN-CONVENTION" not in w[0]


def test_balance_check_says_nothing_when_it_cannot_look(tmp_path):
    """Values-only exchange is common and legitimate. What must not happen is an
    empty finding list reading as 'conservation was checked and is fine'."""
    a = _A.replace(', "normal_fluxes": [x]', "")
    b = _b().replace(', "normal_fluxes": [-x]', "")
    d = _couple(_parts(tmp_path, a_body=a, b_body=b))
    assert d["converged"] is True
    assert not _hits(d, "balance")
    assert any("flux balance" in c and "NOT checked" in c
               for c in d["checks_not_run"])
    assert "COVERAGE" in d["verification"]


# ═══════════════════════════════════════════════════════════════════════════
# 3. Non-matching interface discretisations.
# ═══════════════════════════════════════════════════════════════════════════
def test_non_matching_interfaces_are_reported_with_what_that_costs():
    a = {"coordinates": [[0.5, y / 4] for y in range(5)],
         "normal_fluxes": [1.0] * 5}
    b = {"coordinates": [[0.5, y / 2] for y in range(3)],
         "normal_fluxes": [-1.0] * 3}
    findings, not_run = check_interface_meshes(a, b, "A", "B")
    assert not findings, "a non-matching interface is legitimate, not an error"
    assert not_run and "NON-MATCHING" in not_run[0]
    assert "flux balance" in not_run[0]


def test_non_matching_without_any_flux_says_conservation_was_not_checked():
    a = {"coordinates": [[0.5, y / 4] for y in range(5)]}
    b = {"coordinates": [[0.5, y / 2] for y in range(3)]}
    _, not_run = check_interface_meshes(a, b, "A", "B")
    assert not_run and "NOTHING here checked" in not_run[0]


def test_matching_interfaces_produce_no_note():
    co = [[0.5, y / 4] for y in range(5)]
    findings, not_run = check_interface_meshes({"coordinates": co},
                                               {"coordinates": co})
    assert not findings and not not_run


def test_missing_coordinates_is_not_checked_rather_than_matching():
    findings, not_run = check_interface_meshes({}, {})
    assert not findings and not_run


# ═══════════════════════════════════════════════════════════════════════════
# An export whose layout changes between iterations.
# ═══════════════════════════════════════════════════════════════════════════
_GROWING = """\
import json
from pathlib import Path
n = int(Path("n.txt").read_text()) + 1 if Path("n.txt").exists() else 1
Path("n.txt").write_text(str(n))
k = 1 if n < 2 else 3
json.dump({"field_name": "y", "n_points": k,
           "coordinates": [[0.0, 0.0]] * k, "values": [2.0] * k},
          open("exports.json", "w"))
"""


def test_an_export_that_changes_length_is_refused(tmp_path):
    parts = _parts(tmp_path, b_body=_GROWING)
    r = run_coupling(parts, max_iter=5, tol=1e-8)
    assert r.converged is False
    assert "changed its export length" in (r.error or "")


# ═══════════════════════════════════════════════════════════════════════════
# 10. Relaxation. theta had no way in through the tool at all, and the
# accelerator was not Aitken.
# ═══════════════════════════════════════════════════════════════════════════
def test_theta_reaches_the_driver_and_the_applied_value_is_reported(tmp_path):
    d = _couple(_parts(tmp_path), accelerator="constant", theta=0.25)
    assert d["relaxation"]["mode"] == "constant"
    assert d["relaxation"]["theta0"] == 0.25
    assert d["relaxation"]["applied"] == 0.25


def test_constant_relaxation_actually_uses_the_given_theta(tmp_path):
    """Discrimination for the parameter itself: a theta that is ignored gives
    the same iteration count for every value."""
    n = {}
    for th in (0.25, 0.5, 1.0):
        parts = _parts(tmp_path / f"t{th}")
        r = run_coupling(parts, max_iter=200, tol=1e-10,
                         accelerator="constant", theta0=th)
        assert r.converged
        n[th] = r.iterations
    assert len(set(n.values())) == 3, n


def test_aitken_recovers_from_a_theta0_that_does_not_converge(tmp_path):
    """The point of the accelerator, and the regression that mattered: Aitken
    was handed the previous raw export where the formula wants the previous
    residual, and relaxed each participant by a different theta.

    scale=-2.0, not -1.0. The composite Jacobi map over the stacked state
    (A.values, A.fluxes, B.values, B.fluxes) is J = [[0,0,.5,0], [0,0,.5,0],
    [s,0,0,0], [s,0,0,0]], whose non-zero eigenvalues satisfy lambda^2 = 0.5*s.
    At s=-1 that is |lambda| = 0.707, so theta=1 CONVERGES — in 68 iterations,
    and `assert not converged` at max_iter=60 was passing on the budget being 8
    short rather than on the premise in the comment. At s=-2, |lambda| = 1
    exactly, the relaxed spectral radius is |1-theta+i*theta| >= 1 at theta=1,
    and theta=1 provably cannot converge however long it runs (measured: still
    2.65e+00 after 400 iterations). Same map, real premise.
    """
    body = _b(scale=-2.0)          # |lambda(J)| = 1 exactly: theta=1 cannot converge
    fixed = run_coupling(_parts(tmp_path / "c", b_body=body), max_iter=120,
                         tol=1e-10, accelerator="constant", theta0=1.0)
    assert not fixed.converged, "theta0=1 must be the bad choice here"
    # and not merely short of budget: the residual is still O(1), not creeping down
    assert fixed.residual > 1e-3, fixed.residual
    ait = run_coupling(_parts(tmp_path / "a", b_body=body), max_iter=120,
                       tol=1e-10, accelerator="aitken", theta0=1.0)
    assert ait.converged, "Aitken must find a theta that works from a bad start"
    assert 0.05 <= ait.theta["applied"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# THE STRONGEST CHECK: compare against the same problem solved un-split.
# ═══════════════════════════════════════════════════════════════════════════
def _mono_spec(tmp_path, body=_MONO, timeout=60):
    d = tmp_path / "MONO"
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(body)
    return json.dumps({"command": [sys.executable, "run.py"],
                       "work_dir": str(d), "timeout": timeout})


def test_monolithic_check_catches_a_coupling_that_passes_everything_else(tmp_path):
    """A consistent offset in what one side reads — the K vs degC mistake.

    The iteration converges, the interface flux balances exactly, every export
    is finite, both participants respond to their inputs, the graph is two-way
    and the interfaces match. The answer is wrong by a factor of four, and the
    un-split re-solve is the only thing here that can see it.
    """
    parts = _parts(tmp_path, b_body=_b(offset=-5.0))
    clean = _couple(parts)
    assert clean["converged"] is True
    assert clean["validation"] == [], clean["validation"]
    assert clean["trustworthy_result"] is True          # nothing else objects
    assert abs(clean["exports"]["A"]["values"][0] - 8 / 3) > 1.0

    checked = _couple(_parts(tmp_path / "m", b_body=_b(offset=-5.0)),
                      monolithic=_mono_spec(tmp_path))
    assert checked["monolithic_check"]["status"] == "checked"
    assert _hits(checked, "likely WRONG")
    assert checked["trustworthy_result"] is False


def test_monolithic_check_agrees_with_a_correct_coupling(tmp_path):
    d = _couple(_parts(tmp_path), monolithic=_mono_spec(tmp_path))
    assert d["monolithic_check"]["status"] == "checked"
    assert d["monolithic_check"]["A"]["relative_l2"] < 1e-6
    assert d["validation"] == [], d["validation"]
    assert d["trustworthy_result"] is True


def test_monolithic_check_refuses_to_compare_a_different_quantity(tmp_path):
    """B exports y, the reference solves for x. Comparing them would report a
    large disagreement that means nothing at all."""
    d = _couple(_parts(tmp_path), monolithic=_mono_spec(tmp_path))
    assert "A" in d["monolithic_check"] and "B" not in d["monolithic_check"]
    assert any("nothing to compare" in c for c in d["checks_not_run"])


def test_absence_of_a_monolithic_reference_is_stated_in_the_verdict(tmp_path):
    d = _couple(_parts(tmp_path))
    assert d["monolithic_check"] == {"status": "not supplied"}
    assert any("monolithic consistency: NOT CHECKED" in c
               for c in d["checks_not_run"])
    assert "COVERAGE" in d["verification"]
    assert d["trustworthy_result"] is True, (
        "not supplying the reference is not evidence of a wrong answer — it is "
        "an unchecked one, and the verdict must say which")


def test_a_failed_reference_solve_is_not_read_as_agreement(tmp_path):
    """The coupling is not innocent because its reference could not be produced."""
    d = _couple(_parts(tmp_path),
                monolithic=_mono_spec(tmp_path, body="import sys; sys.exit(2)"))
    assert d["monolithic_check"]["status"] == "reference solve failed"
    assert not _hits(d, "likely WRONG")
    assert any("NOT CHECKED" in c for c in d["checks_not_run"])
    assert "COVERAGE" in d["verification"]


def test_monolithic_consistency_unit():
    assert check_monolithic_consistency(96.9, 50.0)
    assert not check_monolithic_consistency(50.0, 50.5)
    assert check_monolithic_consistency(float("nan"), 50.0)
    assert "not corroborated" in check_monolithic_consistency(float("nan"), 50.0)[0]
    assert not check_monolithic_consistency(None, 50.0)


# ═══════════════════════════════════════════════════════════════════════════
# Coverage is a separate channel from findings, in the verdict itself.
# ═══════════════════════════════════════════════════════════════════════════
def test_checks_that_could_not_run_never_flip_the_verdict_but_always_appear(tmp_path):
    a = _A.replace(', "normal_fluxes": [x]', "")
    b = _b().replace(', "normal_fluxes": [-x]', "")
    d = _couple(_parts(tmp_path, a_body=a, b_body=b))
    assert d["trustworthy_result"] is True
    assert d["verification"].startswith("VERIFIED")
    assert "COVERAGE" in d["verification"]
    for note in d["checks_not_run"]:
        assert note in d["verification"]


# ═══════════════════════════════════════════════════════════════════════════
# The three limits that were true of the machinery but were stated only in
# docstrings, which no agent reads. A limit that is not SERVED is a limit the
# reader does not have, and VERIFIED then implies more than was established.
# ═══════════════════════════════════════════════════════════════════════════
def test_the_digest_says_which_files_it_actually_covers(tmp_path):
    """The fingerprint reaches the paths the SPEC NAMES. A helper module the
    script imports at runtime is outside it: rewriting `model.py` moved a
    reviewed x=2.666667 to x=334.666665 with the verdict still VERIFIED. That
    is a real limit, so it is stated where the verdict is read.
    """
    d = _couple(_parts(tmp_path))
    assert d["trustworthy_result"] is True
    hit = [c for c in d["checks_not_run"] if "review-to-run binding SCOPE" in c]
    assert hit, d["checks_not_run"]
    assert "data_files" in hit[0]        # says how to bring such a file inside
    assert hit[0] in d["verification"]


def test_the_sensitivity_frontier_is_stated_where_the_verdict_is_read(tmp_path):
    """S above the floor proves the export MOVES, not that it moves correctly.
    Measured: values = 6.0 + 1e-6*import is 50% wrong and produces no finding.
    """
    d = _couple(_parts(tmp_path), probe=True)
    hit = [c for c in d["checks_not_run"] if "sensitivity FRONTIER" in c]
    assert hit, d["checks_not_run"]
    assert "monolithic" in hit[0]
    assert hit[0] in d["verification"]


def test_the_wrong_surface_limit_is_stated_where_the_verdict_is_read(tmp_path):
    """Overlapping coordinates are the coordinates the participants REPORTED."""
    d = _couple(_parts(tmp_path))
    hit = [c for c in d["checks_not_run"] if "interface identity" in c]
    assert hit, d["checks_not_run"]
    assert hit[0] in d["verification"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ═══════════════════════════════════════════════════════════════════════════
# The degenerate case every value-based check is blind to: nothing exchanged.
# ═══════════════════════════════════════════════════════════════════════════
_EMPTY = """\
import json
json.dump({"field_name": "y", "n_points": 0, "coordinates": [], "values": []},
          open("exports.json", "w"))
"""


def test_an_empty_interface_is_not_a_converged_coupling(tmp_path):
    """With nothing in the stacked vector the residual is 0 at iteration 2, and
    every value-based check has nothing to look at and therefore says nothing."""
    parts = _parts(tmp_path, b_body=_EMPTY)
    r = run_coupling(parts, max_iter=10, tol=1e-8)
    assert r.converged is False
    assert "EMPTY interface" in (r.error or "")


def test_empty_interface_would_otherwise_report_a_zero_residual(tmp_path):
    """Discrimination: the same participants with ONE value converge honestly,
    so it is the emptiness the guard reacts to, not the participant."""
    one = _EMPTY.replace('"n_points": 0, "coordinates": [], "values": []',
                         '"n_points": 1, "coordinates": [[0.0, 0.0]], "values": [2.0]')
    r = run_coupling(_parts(tmp_path / "b", b_body=one), max_iter=10, tol=1e-8)
    assert r.converged is True and r.residual == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Conservation is not one number, and an interface is not just a node count.
# ═══════════════════════════════════════════════════════════════════════════
from core.quality_checks import (                                    # noqa: E402
    check_interface_flux_profile, check_interfaces_are_the_same_surface,
    check_interface_sensitivity,
)

_CO4 = [[0.5, y / 3] for y in range(4)]


def test_a_wrong_flux_distribution_that_sums_to_zero_is_caught():
    """The net balance is a single number, and a redistribution along the
    interface cancels in the sum. A exports a uniform flux; B piles it all onto
    one node and takes it off the others. The totals cancel EXACTLY."""
    a = {"coordinates": _CO4, "normal_fluxes": [1.0, 1.0, 1.0, 1.0]}
    b = {"coordinates": _CO4, "normal_fluxes": [-40.0, 12.0, 12.0, 12.0]}
    assert abs(sum(a["normal_fluxes"]) + sum(b["normal_fluxes"])) < 1e-12
    assert not check_interface_balance(a, b), "the total balance is satisfied"
    findings, not_run = check_interface_flux_profile(a, b)
    assert findings and "POINT BY POINT" in findings[0]
    assert not not_run


def test_flux_profile_check_passes_a_correct_distribution():
    a = {"coordinates": _CO4, "normal_fluxes": [1.0, 2.0, 3.0, 4.0]}
    b = {"coordinates": _CO4, "normal_fluxes": [-1.0, -2.0, -3.0, -4.0]}
    findings, not_run = check_interface_flux_profile(a, b)
    assert not findings and not not_run


def test_flux_profile_says_it_could_not_look_rather_than_passing():
    """Different node counts, or coordinates that do not line up: the profile
    cannot be compared, and that must not read as conservation being fine."""
    a = {"coordinates": _CO4, "normal_fluxes": [1.0] * 4}
    b = {"coordinates": _CO4[:2], "normal_fluxes": [-2.0] * 2}
    findings, not_run = check_interface_flux_profile(a, b)
    assert not findings and not_run and "ONLY conservation evidence" in not_run[0]
    findings, not_run = check_interface_flux_profile({"coordinates": _CO4}, b)
    assert not findings and not_run


def test_two_interfaces_that_do_not_overlap_are_not_one_interface():
    a = {"coordinates": [[0.5, 0.0], [0.5, 1.0]]}
    b = {"coordinates": [[0.5, 5.0], [0.5, 6.0]]}
    findings, _ = check_interfaces_are_the_same_surface(a, b)
    assert findings and "do NOT overlap" in findings[0]


def test_overlapping_interfaces_of_different_resolution_are_fine():
    a = {"coordinates": [[0.5, y / 10] for y in range(11)]}
    b = {"coordinates": [[0.5, y / 2] for y in range(3)]}
    findings, not_run = check_interfaces_are_the_same_surface(a, b)
    # No FINDING: differing resolution on the same surface is legitimate.
    assert not findings
    # Coverage is a different channel, and overlapping-as-reported is exactly
    # where the reader needs telling that "as reported" is all that was checked.
    assert [n for n in not_run if "interface identity" in n], not_run


def test_missing_coordinates_is_reported_not_passed():
    findings, not_run = check_interfaces_are_the_same_surface({}, {})
    assert not findings and not_run


# ── intra-block scale masking: a huge and a small entry in ONE flat array ────
_BIG_A = """\
import json
from pathlib import Path
imp = json.loads(Path("imports.json").read_text() or "{}")
s = imp["B"]["values"][1] if "B" in imp else 0.0
json.dump({"field_name": "v", "n_points": 2, "coordinates": [[0.0, 0.0], [0.0, 1.0]],
           "values": [1e12, -1.0 * s + 1.0]}, open("exports.json", "w"))
"""
_BIG_B = """\
import json
from pathlib import Path
imp = json.loads(Path("imports.json").read_text() or "{}")
s = imp["A"]["values"][1] if "A" in imp else 0.0
json.dump({"field_name": "v", "n_points": 2, "coordinates": [[0.0, 0.0], [0.0, 1.0]],
           "values": [1e12, -1.0 * s + 1.0]}, open("exports.json", "w"))
"""


def test_a_huge_entry_cannot_mask_a_small_one_in_the_same_array(tmp_path):
    """`values` is often a flat list of mixed quantities. A block NORM is set by
    the largest entry, so a small entry oscillating by 100% of itself leaves the
    block residual at 1e-12 and the coupling reports convergence at iteration 2.
    """
    d = _couple(_parts(tmp_path, a_body=_BIG_A, b_body=_BIG_B),
                max_iter=40, tol=1e-8)
    assert d["converged"] is True and d["residual"] < 1e-8
    assert d["block_residuals"]["A.values"] > 0.1, (
        "the entry-wise measure must see the oscillation the norm hides")
    assert any("NOT representative" in w for w in d["validation"])
    assert d["trustworthy_result"] is False


def test_intra_block_masking_discriminates(tmp_path, monkeypatch):
    """Restore the block-NORM measure and the oscillation disappears.

    The probe is disabled here on purpose: it measures its per-block
    sensitivity with the same function, so it independently catches this case
    too. Isolating one check at a time is the only way to show that each is
    doing the work the test claims.
    """
    import core.coupling_driver as D
    import numpy as _np

    def _norm_ratio(new, prev):          # the block-NORM measure it replaced
        if new.shape != prev.shape or new.size == 0:
            return float("nan")
        ref = float(_np.linalg.norm(prev)) + float(_np.linalg.norm(new))
        return 0.0 if ref <= 0 else float(_np.linalg.norm(new - prev)) * 2.0 / ref

    monkeypatch.setattr(D, "_rel_change", _norm_ratio)
    d = _couple(_parts(tmp_path, a_body=_BIG_A, b_body=_BIG_B),
                max_iter=40, tol=1e-8, probe=False)
    assert d["converged"] is True
    assert d["block_residuals"]["A.values"] < 1e-8, (
        "with the norm measure the oscillation is invisible")
    assert d["trustworthy_result"] is True, (
        "and the coupling is stamped VERIFIED with one exchanged quantity "
        "oscillating by 100% of itself")


# ═══════════════════════════════════════════════════════════════════════════
# The critic gate binds a review to a SETUP. Anything that changes what is
# solved and is not in the digest is a review of one thing approving another.
# ═══════════════════════════════════════════════════════════════════════════
def test_rewriting_a_participant_script_after_review_is_refused(tmp_path):
    """The spec names `["python", "run.py"]`; the physics is inside run.py.

    Without the file contents in the digest, reviewing a correct coupling
    approved any other coupling that reused the same file names.
    """
    parts = _parts(tmp_path)
    ok = _couple(parts)
    assert ok["trustworthy_result"] is True
    x_reviewed = ok["exports"]["A"]["values"][0]

    # same file name, same command, same work_dir — different physics
    (parts[1].work_dir / "run.py").write_text(_b(offset=900.0))
    swapped = _couple(parts, reviewed=False)
    assert swapped["converged"] is True
    assert abs(swapped["exports"]["A"]["values"][0] - x_reviewed) > 100.0
    assert swapped["trustworthy_result"] is False
    assert "changed after it was reviewed" in swapped["critic_review"]


def test_script_fingerprint_discriminates(tmp_path, monkeypatch):
    """Strip the file contents back out of the digest and the swap goes through."""
    import tools.consolidated as C
    monkeypatch.setattr(C, "_participant_fingerprints", lambda *_a, **_k: {})
    parts = _parts(tmp_path)
    assert _couple(parts)["trustworthy_result"] is True
    (parts[1].work_dir / "run.py").write_text(_b(offset=900.0))
    swapped = _couple(parts, reviewed=False)
    assert swapped["converged"] is True
    assert swapped["trustworthy_result"] is True, (
        "without the script fingerprint, a review of one coupling stamps a "
        "completely different one as VERIFIED")


def test_a_participant_script_that_does_not_exist_yet_cannot_be_reviewed(tmp_path):
    """Fail closed: an absent file is recorded as absent, so writing it later
    changes the digest."""
    from tools.consolidated import _participant_fingerprints
    spec = json.dumps([{"name": "A", "command": [sys.executable, "later.py"],
                        "work_dir": str(tmp_path)}])
    before = _participant_fingerprints(spec)
    (tmp_path / "later.py").write_text("pass\n")
    after = _participant_fingerprints(spec)
    assert before != after
    assert before["A"]["later.py"] == "absent"
    assert after["A"]["later.py"].startswith("sha256:")


def test_data_files_are_part_of_the_setup(tmp_path):
    """A compiled solver's physics is in its deck, which arrives via data_files."""
    from tools.consolidated import _participant_fingerprints
    deck = tmp_path / "deck.yaml"
    deck.write_text("k: 1.0\n")
    spec = json.dumps([{"name": "A", "command": ["/bin/true"],
                        "work_dir": str(tmp_path), "data_files": [str(deck)]}])
    before = _participant_fingerprints(spec)
    deck.write_text("k: 1000.0\n")
    assert _participant_fingerprints(spec) != before


def test_submit_critic_review_uses_the_same_definition_as_the_run(tmp_path):
    """The review path used to build its own setup text, so the two could — and
    did — cover different sets of arguments."""
    import tools.consolidated as C
    parts = _parts(tmp_path)
    args = {"participants": _spec(parts), "max_iter": 60, "tol": 1e-8,
            "accelerator": "aitken", "theta": 0.5, "monolithic": "",
            "probe": True}
    loop = asyncio.new_event_loop()
    try:
        out = json.loads(loop.run_until_complete(_tool("submit_critic_review")(
            solver="couple",
            findings="Checked the participant maps, the interface units, the "
                     "boundary conditions and the discretisation; nothing found "
                     "that would invalidate the coupled run.",
            coupling_args=json.dumps(args))))
        assert out["accepted"] is True
        d = json.loads(loop.run_until_complete(_tool("couple")(
            critic_approved=True, **args)))
    finally:
        loop.close()
    assert d["trustworthy_result"] is True, d["critic_review"]


# ═══════════════════════════════════════════════════════════════════════════
# Mutation testing found these SIX checks untested. A check whose call site can
# be deleted without a test noticing is not wired, whatever its unit test says.
# ═══════════════════════════════════════════════════════════════════════════
def test_vector_flux_components_are_balanced_one_by_one():
    """Summing a vector traction across components lets a +x imbalance cancel a
    -y one and report perfect conservation across an interface that conserves
    nothing."""
    a = {"normal_fluxes": [[10.0, 3.0]]}
    b = {"normal_fluxes": [[-3.0, -10.0]]}          # totals: 13 and -13
    assert abs(sum(sum(v) for v in a["normal_fluxes"])
               + sum(sum(v) for v in b["normal_fluxes"])) < 1e-12
    w = check_interface_balance(a, b)
    assert w, "component-wise, neither x nor y balances"
    assert any("[0]" in m for m in w) and any("[1]" in m for m in w)
    ok = check_interface_balance({"normal_fluxes": [[10.0, 3.0]]},
                                 {"normal_fluxes": [[-10.0, -3.0]]})
    assert not ok


def test_mismatched_flux_component_counts_are_reported_not_skipped():
    w = check_interface_balance({"normal_fluxes": [[1.0, 2.0]]},
                                {"normal_fluxes": [[-1.0, -2.0, 0.0]]})
    assert w and "could NOT be evaluated" in w[0]


# The three tool-level wirings. Each of these was reachable only through a unit
# test, so deleting the call in `couple` changed nothing that any test saw.
_CO11 = [[0.5, y / 10] for y in range(11)]


def _iface_participant(coords, fluxes, value="x") -> str:
    return (
        "import json\nfrom pathlib import Path\n"
        'imp = json.loads(Path("imports.json").read_text() or "{}")\n'
        f'v = imp["{"B" if value == "x" else "A"}"]["values"][0] if imp else 0.0\n'
        f"co = {coords!r}\nfl = {fluxes!r}\n"
        'json.dump({"field_name": "t", "n_points": len(co), "coordinates": co,\n'
        '           "values": [0.5 * v + 1.0] * len(co), "normal_fluxes": fl},\n'
        '          open("exports.json", "w"))\n')


def test_flux_profile_check_is_wired_into_couple(tmp_path):
    """Totals cancel exactly; the distribution is nonsense."""
    a = _iface_participant(_CO11, [1.0] * 11, "x")
    b = _iface_participant(_CO11, [-11.0] + [0.0] * 10, "y")
    d = _couple(_parts(tmp_path, a_body=a, b_body=b), max_iter=30, tol=1e-8)
    assert d["converged"] is True
    assert not any("NOT balanced" in w for w in d["validation"]), \
        "the NET balance is satisfied — only the profile is wrong"
    assert any("POINT BY POINT" in w for w in d["validation"])
    assert d["trustworthy_result"] is False


def test_same_surface_check_is_wired_into_couple(tmp_path):
    a = _iface_participant([[0.5, 0.0], [0.5, 1.0]], [1.0, 1.0], "x")
    b = _iface_participant([[0.5, 90.0], [0.5, 91.0]], [-1.0, -1.0], "y")
    d = _couple(_parts(tmp_path, a_body=a, b_body=b), max_iter=30, tol=1e-8)
    assert d["converged"] is True
    assert any("do NOT overlap" in w for w in d["validation"])
    assert d["trustworthy_result"] is False


def test_mesh_conformity_note_is_wired_into_couple(tmp_path):
    a = _iface_participant(_CO11, [1.0] * 11, "x")
    b = _iface_participant([[0.5, y / 2] for y in range(3)], [-11.0 / 3] * 3, "y")
    d = _couple(_parts(tmp_path, a_body=a, b_body=b), max_iter=30, tol=1e-8)
    assert any("NON-MATCHING" in c for c in d["checks_not_run"])
    assert "NON-MATCHING" in d["verification"]


def test_aitken_beats_the_theta_it_started_from(tmp_path):
    """Discrimination for the Aitken fix itself.

    Handing the formula the previous RAW EXPORT where it wants the previous
    RESIDUAL leaves theta wandering inside its clamp, which costs convergence
    RATE rather than correctness — so a test that only asks "did it converge"
    passes with the bug in place. This asks for the rate: on a map whose
    un-relaxed iteration does not converge at all, correct Aitken has to find
    the theta that does, and get there in a comparable number of iterations to
    the best constant choice.
    """
    body = _b(scale=-1.0)                       # oscillates at theta = 1
    best = run_coupling(_parts(tmp_path / "k", b_body=body), max_iter=400,
                        tol=1e-10, accelerator="constant", theta0=0.5)
    assert best.converged
    ait = run_coupling(_parts(tmp_path / "a", b_body=body), max_iter=400,
                       tol=1e-10, accelerator="aitken", theta0=1.0)
    assert ait.converged
    assert ait.iterations <= best.iterations + 4, (
        f"Aitken took {ait.iterations} where the best constant theta took "
        f"{best.iterations} — it is not tracking the residual")
    # and it must have MOVED off the starting guess it was given, which is the
    # whole claim. The optimum for this map is not derived here, so its value is
    # not asserted — the rate above is the substantive check.
    assert ait.theta["applied"] < 0.95, ait.theta


def test_tool_level_returncode_check_is_wired(monkeypatch, tmp_path):
    """Defence in depth, made testable.

    The driver refuses a non-zero exit before it can reach the tool, so deleting
    the tool's own check changes nothing any other test sees. That makes the call
    look like decoration. It is not — it is the guard for a result that arrives
    carrying a failed exit code — so exercise it directly by handing `couple` a
    driver result the driver itself would never produce.
    """
    import tools.consolidated as C
    from core.coupling_driver import CouplingResult

    ex = {"field_name": "x", "n_points": 1, "coordinates": [[0.0, 0.0]],
          "values": [1.0], "normal_fluxes": [1.0]}
    exb = dict(ex, field_name="y", normal_fluxes=[-1.0])
    fake = CouplingResult(
        converged=True, iterations=3, residual=1e-12,
        exports={"A": ex, "B": exb}, history=[float("nan"), 1e-12],
        returncodes={"A": 0, "B": 137},          # killed, but "converged"
        block_residuals={"A.values": 0.0, "B.values": 0.0},
        responsiveness={"A": "responsive", "B": "responsive"},
        graph={"participants": ["A", "B"],
               "declared_edges": {"A": ["B"], "B": ["A"]}},
        theta={"mode": "aitken", "theta0": 0.5, "applied": 0.5},
        sensitivity={"A": {"noise": 0.0, "signal": 1e-3, "S": 1.0, "blocks": {}},
                     "B": {"noise": 0.0, "signal": 1e-3, "S": 1.0, "blocks": {}}})
    monkeypatch.setattr(C, "run_coupling", lambda *a, **k: fake, raising=False)
    import core.coupling_driver as D
    monkeypatch.setattr(D, "run_coupling", lambda *a, **k: fake)
    d = _couple(_parts(tmp_path))
    assert d["converged"] is True
    assert any("exited non-zero" in w for w in d["validation"]), d["validation"]
    assert d["trustworthy_result"] is False


def test_aitken_formula_matches_the_hand_computation():
    """Pin the formula itself: theta = -theta_prev (r_prev . dr) / (dr . dr).

    With prev_relaxed = 0, new_raw = 2 the residual is r = 2; against r_prev = -1
    that gives dr = 3 and theta = -0.5 * (-1 * 3) / 9 = 1/6. The function must
    also RETURN r, since the caller has to store it for the next step.
    """
    from core.coupling_driver import _aitken
    th, r_k = _aitken(np.array([0.0]), np.array([2.0]), np.array([-1.0]), 0.5)
    assert abs(th - 1.0 / 6.0) < 1e-12, th
    assert r_k.tolist() == [2.0]
    # no previous residual yet -> hold theta, and still return the residual
    th0, r0 = _aitken(np.array([0.0]), np.array([2.0]), None, 0.4)
    assert th0 == 0.4 and r0.tolist() == [2.0]
    # a zero denominator must not divide by zero
    th1, _ = _aitken(np.array([0.0]), np.array([2.0]), np.array([2.0]), 0.3)
    assert th1 == 0.3


def test_aitken_is_given_the_residual_and_not_the_raw_export(tmp_path):
    """Direct discrimination for the Aitken bookkeeping.

    The formula needs the previous RESIDUAL r = G(x) - x; it used to be handed
    the previous raw export G(x). Both are vectors of the right shape, so nothing
    complains and the only symptom is a theta that wanders inside its clamp — a
    convergence-RATE loss that a "did it converge" test does not see. What DOES
    separate them is size: at convergence the residual is by definition tiny,
    while the raw export is the size of the solution.

    LIMIT, stated: this pins WHAT the driver stores. Mutation testing shows that
    corrupting only what is PASSED BACK IN — the input to the formula — is caught
    indirectly, by three tests that are sensitive to iteration count, and not by
    any assertion written for it. Deriving the analytic optimum theta for this
    map would close that, and is not done here.
    """
    r = run_coupling(_parts(tmp_path), max_iter=90, tol=1e-9,
                     accelerator="aitken", theta0=0.5)
    assert r.converged
    solution_scale = abs(r.exports["A"]["values"][0])
    assert solution_scale > 1.0
    assert r.theta["residual_norm"] is not None
    assert r.theta["residual_norm"] < 1e-6, (
        f"Aitken is holding a vector of norm {r.theta['residual_norm']:.3g} "
        f"where the converged residual must be tiny (the solution is of order "
        f"{solution_scale:.3g}) — it is being handed the raw export, not the "
        "residual")
