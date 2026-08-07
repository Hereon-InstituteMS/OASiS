"""The coupled redesign, and the harness holes it exposed.

Every test here corresponds to something that was WRONG and is now checked, so
that a future edit which reintroduces it fails rather than ships.  The three
groups are: the tautology (a check that cannot fail), the insensitivity (a
graded quantity that cannot move), and the custody/evidence gates that a
previous amendment declared fixed while the code still carried the defect.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import sympy as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from blind_eval import coupled as C          # noqa: E402
from blind_eval import evidence as EV        # noqa: E402
from blind_eval import interface as IF       # noqa: E402
from blind_eval import keyvault              # noqa: E402

x, y = C.X, C.Y
LX, XI = sp.Rational(3, 2), sp.Rational(3, 5)


def _two_material(kA=1, kB=4):
    pm = C.ProductMaterial([XI], [sp.Integer(kA), sp.Integer(kB)], [], [1])
    H = pm.eta_at(LX)

    def U(s, w):
        return s * (H - s) * w * (1 - w) * (sp.Rational(2, 3)
                                            + sp.Rational(1, 5) * s
                                            + sp.Rational(3, 7) * w)
    cells = pm.field(lambda s, w: U(s, w))
    return cells[(0, 0)], cells[(1, 0)]


# ── group 1: a check that cannot fail must say so ─────────────────────
def test_same_field_same_material_is_reported_vacuous_not_pass():
    """D1/D2/D4's shape. jump_u = 0 holds for ANY input; that is not a pass."""
    u, _ = _two_material()
    rep = C.check_scalar_transmission(u, u, sp.eye(2), sp.eye(2), (x, y), x, XI)
    assert not rep.ok
    assert "non_tautological" in rep.vacuous
    assert "VACUOUS" in rep.summary()
    # same field, DIFFERENT material: still vacuous in the field condition
    rep2 = C.check_scalar_transmission(u, u, sp.eye(2), 4 * sp.eye(2),
                                       (x, y), x, XI)
    assert "fields_differ" in rep2.vacuous
    # different field, SAME material: the flux condition carries no information
    uA, uB = _two_material()
    rep3 = C.check_scalar_transmission(uA, uB, sp.eye(2), sp.eye(2),
                                       (x, y), x, XI)
    assert "materials_differ" in rep3.vacuous


def test_two_material_construction_is_binding_and_passes():
    uA, uB = _two_material()
    rep = C.check_scalar_transmission(uA, uB, sp.eye(2), 4 * sp.eye(2),
                                      (x, y), x, XI)
    assert rep.ok, rep.summary()
    assert not rep.vacuous
    # the constraint is real: the two fields are genuinely different functions
    assert sp.simplify(uA - uB) != 0


def test_transmission_check_is_falsifiable():
    """A verification never seen to fail is not known to be able to."""
    uA, uB = _two_material()
    c = C.check_transmission_is_falsifiable(uA, uB, sp.eye(2), 4 * sp.eye(2),
                                            (x, y), x, XI)
    assert c.ok, c.detail


def test_interface_value_and_flux_must_be_nonzero():
    """Continuity satisfied by both sides being zero proves nothing."""
    zero = sp.Integer(0) * x
    rep = C.check_scalar_transmission(zero, zero, sp.eye(2), 4 * sp.eye(2),
                                      (x, y), x, XI)
    assert "interface_value_nonzero" in rep.vacuous


def test_anisotropic_offdiagonal_must_be_shared():
    """The family has a real constraint; it is checked, not assumed."""
    KA = sp.Matrix([[1, sp.Rational(1, 2)], [sp.Rational(1, 2), 2]])
    good = sp.Matrix([[3, sp.Rational(1, 2)], [sp.Rational(1, 2), 5]])
    bad = sp.Matrix([[3, sp.Rational(1, 3)], [sp.Rational(1, 3), 5]])
    pm = C.ProductMaterial([XI], [KA[0, 0], good[0, 0]], [], [1])
    H = pm.eta_at(LX)

    def U(s):
        return s * (H - s) * y * (1 - y) * (sp.Rational(2, 3)
                                            + sp.Rational(1, 5) * s)
    uA, uB = U(pm.eta[0][2]), U(pm.eta[1][2])
    assert C.check_scalar_transmission(uA, uB, KA, good, (x, y), x, XI).ok
    r = C.check_scalar_transmission(uA, uB, KA, bad, (x, y), x, XI)
    assert "flux_continuity" in r.failed


def test_vector_transmission_requires_nonzero_shear():
    """Otherwise the vector instance is two scalar problems wearing a coat."""
    lam, muA, muB = sp.Integer(577), sp.Integer(385), sp.Integer(1155)
    g = y ** 2 * (1 - y) ** 2 * (2 + y / 3)
    P_a, Q_a = x * (1 + x / 2), x * (2 - x / 3)
    P_b, Q_b = C.solve_vector_interface(P_a, Q_a, g, (x, y), lam, muA, muB,
                                        XI, LX)
    uA = (P_a * g, Q_a * sp.diff(g, y))
    uB = (P_b * g, Q_b * sp.diff(g, y))
    rep = C.check_vector_transmission(uA, uB, (x, y), (lam, muA), (lam, muB),
                                      x, XI)
    assert rep.ok, rep.summary()
    assert "shear_traction_nonzero" not in rep.vacuous
    # and the same-material version is vacuous, as D4 was
    same = C.check_vector_transmission(uA, uA, (x, y), (lam, muA), (lam, muA),
                                       x, XI)
    assert "non_tautological" in same.vacuous


# ── group 2: the graded quantity must be able to move ─────────────────
def test_interface_temperature_is_invariant_under_ratio_preserving_error():
    """The exact statement of defect 2, in the setting the paper bands."""
    xl, xi, xr, tl, tr = 0.0, 0.6, 1.1, 320.0, 300.0

    def cf(kl, kr):
        cl, cr = kl / (xi - xl), kr / (xr - xi)
        t = (cl * tl + cr * tr) / (cl + cr)
        return t, cl * (tl - t)

    t0, q0 = cf(0.8, 1.5)
    for s in (2.0, 5.0, 0.25):
        t, q = cf(0.8 * s, 1.5 * s)
        assert abs(t - t0) < 1e-9, "interface temperature must be invariant"
        assert abs(q - q0) / q0 > 0.5, "the flux must move a lot"


def test_flux_jump_detects_a_wrongly_transmitted_quantity():
    """q_A + q_B is O(1) when the coupling transmits du/dn instead of k du/dn."""
    kA, kB = 1.0, 4.0
    qa = [(1.0,), (2.0,), (3.0,)]
    pts = [(0.6, 0.1), (0.6, 0.2), (0.6, 0.3)]
    ua = [(0.5,), (0.6,), (0.7,)]
    correct = ([p for p in pts], ua, [(-v[0],) for v in qa])
    broken = ([p for p in pts], ua, [(-v[0] * kB / kA,) for v in qa])
    good, _ = IF.two_sided_jumps((pts, ua, qa), correct)
    bad, _ = IF.two_sided_jumps((pts, ua, qa), broken)
    assert good["jump_q_rel"] < 1e-12
    assert bad["jump_q_rel"] > 1.0


def test_scalar_interface_summary_is_blind_to_a_reversed_mapping():
    """Why the PROFILE is graded and not one number."""
    prof = [1.0, 2.0, 5.0, 9.0]
    assert sum(prof) == sum(reversed(prof))                 # net flux: invariant
    assert sum(prof) / len(prof) == sum(reversed(prof)) / len(prof)
    pts = [(0.6, 0.1), (0.6, 0.2), (0.6, 0.3), (0.6, 0.4)]
    a = (pts, [(v,) for v in prof], [(v,) for v in prof])
    b = (pts, [(v,) for v in reversed(prof)],
         [(-v,) for v in reversed(prof)])
    d, _ = IF.two_sided_jumps(a, b)
    assert d["jump_q_rel"] > 0.1, "the profile must see what the sum cannot"


def test_flux_consistency_catches_a_fabricated_flux():
    field_pts = [(0.5, 0.25), (0.55, 0.25), (0.58, 0.25)]
    field_vals = [(0.5,), (0.55,), (0.58,)]        # u = x, so du/dx = 1
    got = IF.recover_flux_from_field(field_pts, field_vals, [(0.6, 0.25)],
                                     k_normal=2.0, normal_axis=0,
                                     iface_coord=0.6, outward_sign=1.0)
    assert got[0] is not None
    assert abs(got[0][0] - (-2.0)) < 1e-6
    honest = IF.flux_consistency([(-2.0,)], got)
    assert honest["verdict"] == "CONSISTENT"
    faked = IF.flux_consistency([(-8.0,)], got)
    assert faked["verdict"] == "INCONSISTENT"


def test_order_from_halving_cannot_see_a_wrong_limit(tmp_path):
    """The recorded reason the interface quantities exist at all."""
    data = json.loads((REPO / "data" /
                       "coupled_grading_sensitivity.json").read_text())
    new = [i for i in data["instances"] if i["kA"] != i["kB"]][0]
    correct = new["variants"]["correct"]
    broken = new["variants"]["MUT_KRATIO"]
    assert abs(broken["u_order_halving"] - correct["u_order_halving"]) < 0.2, \
        "the mesh-halving order must be shown NOT to separate them"
    assert broken["q_jump_two_sided"] > 1.0
    assert correct["q_jump_two_sided"] < 1e-6
    old = [i for i in data["instances"] if i["kA"] == i["kB"]][0]
    assert old["variants"]["MUT_KRATIO"]["q_jump_two_sided"] < 1e-6, \
        "on the OLD equal-material shape the mutation is invisible to everything"
    assert old["transmission_vacuous"], "the old shape's check is vacuous"


# ── group 3: the gates that were declared fixed but were not ──────────
def test_missing_or_empty_keys_directory_is_not_sealed(tmp_path):
    assert keyvault.is_sealed(tmp_path / "nope") is False
    (tmp_path / "empty").mkdir()
    assert keyvault.is_sealed(tmp_path / "empty") is False


def test_runner_imports_the_real_seal_check():
    """run_blind.py carried its own broken copy and never imported the fix."""
    src = (REPO / "campaign3_blind" / "run_blind.py").read_text()
    assert "keyvault" in src and "_kv.is_sealed" in src
    assert "if not keys.exists():\n        return True" not in src
    assert "preflight_or_die" in src
    assert "per_cell_exposure_check" in src


def test_evidence_gate_rejects_a_name_in_a_text_file(tmp_path):
    """The whole of the previous gate: write 'deal.II' anywhere and pass."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "my_notes.txt").write_text(
        "I will use FEniCSx on A and deal.II on B, as the task says.")
    (work / "run.log").write_text("dolfinx solve finished")
    for code in ("fenics", "dealii"):
        assert EV.code_evidence(work, code).verdict == "NOT_PROVEN"


def test_evidence_gate_accepts_structured_solver_output(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "a.log").write_text("dolfinx: num_dofs 4225\nKSP converged")
    (work / "b.log").write_text("Number of active cells: 2048\n"
                                "Number of degrees of freedom: 1089\n"
                                "12 CG iterations needed")
    assert EV.code_evidence(work, "fenics").verdict == "PROVEN"
    assert EV.code_evidence(work, "dealii").verdict == "PROVEN"


def test_coupled_run_without_a_residual_history_is_not_proven(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "a.log").write_text("dolfinx num_dofs 4225")
    (work / "b.log").write_text("Number of active cells: 2048")
    rep = EV.assess(work, ["fenics", "dealii"], coupled=True)
    assert rep.verdict == "NOT_PROVEN"
    assert "residual history" in rep.coupling["detail"]


def test_residual_history_must_actually_converge(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "a.log").write_text("dolfinx num_dofs 4225")
    (work / "b.log").write_text("Number of active cells: 2048")
    (work / "residual_level1.csv").write_text(
        "iteration,residual\n1,1e-7\n2,1e-7\n3,1e-7\n")
    rep = EV.assess(work, ["fenics", "dealii"], coupled=True)
    assert rep.coupling["verdict"] == "CONTRADICTED"

    (work / "residual_level1.csv").write_text(
        "iteration,residual\n1,3.0e-1\n2,1.1e-3\n3,4.0e-6\n4,7.0e-8\n")
    rep = EV.assess(work, ["fenics", "dealii"], coupled=True,
                    claimed_iterations=4)
    assert rep.verdict == "PROVEN", rep.notes

    # and the claimed iteration count must match the history
    rep = EV.assess(work, ["fenics", "dealii"], coupled=True,
                    claimed_iterations=99)
    assert rep.coupling["verdict"] == "CONTRADICTED"


def test_leak_invalidate_survives_a_ledger_with_no_outcome(tmp_path):
    """Campaign-3 ledgers have no 'outcome' key; the old code raised KeyError."""
    sys.path.insert(0, str(REPO / "scripts"))
    import leak_invalidate as LI
    run = tmp_path / "runs" / "D3_27b_MCP_seed0"
    run.mkdir(parents=True)
    (run / "ledger.json").write_text(json.dumps(
        {"problem": "D3", "graded": False}))
    done, skipped = LI.invalidate(
        {"tainted": [{"run": str(run), "ledger": str(run / "ledger.json"),
                      "findings": ["sealed solution in trajectory"]}]},
        queue=tmp_path / "q.txt")
    assert len(done) == 1
    d = json.loads((run / "ledger.json").read_text())
    assert d["outcome"] == "INVALID_INFRA"
    assert "outcome_pre_leak_audit" not in d


def test_audit_leaks_reaches_campaign3_runs(tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    import audit_leaks as AL
    t = tmp_path / "campaign3_blind" / "runs" / "D3_27b_BARE_seed0" / "work"
    t.mkdir(parents=True)
    (t / "trajectory.txt").write_text(
        "let me look at open-fem-agent/src/backends/fenics/backend.py")
    rep = AL.audit(tmp_path)
    assert rep["audited"] == 1, "the blind campaign was the one never audited"
    assert rep["tainted"] and "OASiS-source access" in rep["tainted"][0]["findings"]


def test_audit_leaks_bad_both_regex_is_live():
    """It used to be `if False else []` — compiled, never evaluated."""
    sys.path.insert(0, str(REPO / "scripts"))
    import audit_leaks as AL
    assert AL.BAD_BOTH.search("cat ../../build_extra.py")
    assert AL.BAD_BOTH.search("campaign3_blind/keys/D3/key.json")
    assert AL.BAD_BOTH.search("paper_experiments/runs/x")
    import ast
    src = (REPO / "scripts" / "audit_leaks.py").read_text()
    body = ast.parse(src)
    code_only = "\n".join(
        ast.unparse(n) for n in body.body
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)))
    assert "if False else []" not in code_only


def test_probe_grid_incommensurability_is_enforced():
    sys.path.insert(0, str(REPO / "campaign3_blind"))
    import grade_blind as GB
    GB.assert_probe_grid_incommensurate(2, [8, 16, 32])       # M = 44, fine
    with pytest.raises(ValueError):
        old = GB.PROBE_M[2]
        try:
            GB.PROBE_M[2] = 32
            GB.assert_probe_grid_incommensurate(2, [8, 16, 32])
        finally:
            GB.PROBE_M[2] = old


def test_grade_blind_no_longer_uses_a_substring_scan():
    src = (REPO / "campaign3_blind" / "grade_blind.py").read_text()
    assert "from blind_eval.evidence import code_evidence" in src
    assert "PROBE_M = {2: 44, 3: 21}" in src
