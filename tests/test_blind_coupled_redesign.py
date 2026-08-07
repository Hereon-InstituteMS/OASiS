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

# A DEMONSTRATION seed. NEVER the campaign's: the hidden fields are drawn from a
# CSPRNG at build time and the seed is written only into the sealed key, so a
# seed in a tracked file would let anyone with the repository re-derive every
# answer. These tests check the FAMILY's structure, for which any seed serves.
DEMO_SEED = 11111111


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


# ── group 4: the verification of the constructions is itself falsifiable ──
def _d3():
    sys.path.insert(0, str(REPO / "campaign3_blind"))
    import build_coupled_v2 as BV
    d = BV.Draw(DEMO_SEED)
    spec, fields, sources, coords, mats = BV.instance_D3(d)
    return BV, spec, fields, sources, coords, mats


def test_finite_difference_substitution_catches_a_perturbed_source():
    """The construction check must be able to fail, or it verifies nothing.

    The first two versions of this check could not: forming the residual
    symbolically let sympy collapse it to exactly zero, and lambdifying the two
    sides separately still hit sympy's identical canonical form. It now applies
    the operator by finite differences, which shares no machinery with the
    symbolic derivation.
    """
    BV, spec, fields, sources, coords, mats = _d3()
    box = [(0.05, 0.55), (0.05, 0.95)]
    good = BV.numeric_residual(fields["A"], sources["A"], coords, mats[0],
                               box=box)
    assert good < 1e-6, f"the correct construction must pass, got {good:.2e}"
    perturbed = BV.numeric_residual(
        fields["A"], sources["A"] * sp.Rational(1001, 1000), coords, mats[0],
        box=box)
    assert perturbed > 1e-4, "a 0.1% source error must be caught"
    flipped = BV.numeric_residual(fields["A"], -sources["A"], coords, mats[0],
                                  box=box)
    assert flipped > 1.0, "a sign error must be caught"


def test_finite_difference_interface_check_catches_a_broken_transmission():
    BV, spec, fields, sources, coords, mats = _d3()
    ju, jq = BV.numeric_interface_jump(fields["A"], fields["B"], mats[0],
                                       mats[1], coords, BV.x, BV.XI)
    assert max(ju, jq) < 1e-6
    ju2, jq2 = BV.numeric_interface_jump(
        fields["A"], fields["B"] * sp.Rational(101, 100), mats[0], mats[1],
        coords, BV.x, BV.XI)
    assert max(ju2, jq2) > 1e-3, "a 1% error on one side must break continuity"


def test_interface_lies_on_a_mesh_line_at_every_level():
    """It did not: x = 3/5 against h = 1/8, 1/16, 1/32 is on no mesh line.

    A correct monolithic vector solve then graded 1.475 against a theoretical
    2.0, inside a tolerance of 0.4 — CONFIDENTLY_WRONG for being right.
    """
    sys.path.insert(0, str(REPO / "campaign3_blind"))
    import build_coupled_v2 as BV
    for n in BV.MESH_N:
        for edge in (BV.XI, BV.LX):
            cells = edge * n
            assert cells == int(cells), (
                f"{edge} is not a multiple of h = 1/{n}: the subdomain cannot "
                f"be meshed with the h the task prescribes")


def test_probe_grid_never_lands_on_a_material_line():
    """A probe ON the interface has no well-defined value: k jumps there.

    The grids are cell-centred within each subdomain's own extent, and the
    extents are bounded BY the material lines, so this holds structurally. It is
    asserted because the property is easy to lose: over the unit interval,
    M = 45 puts a probe exactly on 0.5 (22.5/45) and M = 44 does not.
    """
    sys.path.insert(0, str(REPO / "campaign3_blind"))
    import build_coupled_v2 as BV
    M = BV.PROBE_M[2]
    for n in BV.MESH_N:
        assert M % n != 0 and n % M != 0, "probe count aliases with the mesh"
    for fn in BV.BUILDERS:
        spec = fn(BV.Draw(DEMO_SEED))[0]
        if spec["dim"] != 2:
            continue
        for extent in (spec["extent_a"], spec["extent_b"]):
            for lo, hi in extent:
                pts = [lo + (i + 0.5) * (hi - lo) / M for i in range(M)]
                assert all(min(abs(p - lo), abs(p - hi)) > 1e-12 for p in pts)
    # and over the unit interval, which is what D5's per-cell grids use
    assert all(abs((i + 0.5) / 45 - 0.5) > 1e-12 for i in range(45)) is False
    assert all(abs((i + 0.5) / M - 0.5) > 1e-12 for i in range(M))


def test_the_family_is_varied_along_every_axis_claimed():
    """A benchmark is a family; one problem plumbed nine ways is an anecdote."""
    sys.path.insert(0, str(REPO / "campaign3_blind"))
    import build_coupled_v2 as BV
    specs = []
    for fn in BV.BUILDERS:
        d = BV.Draw(DEMO_SEED)
        specs.append(fn(d)[0])
    assert len(specs) >= 8
    assert all(s["evidence_grade"] == 1 for s in specs)
    assert {2, 3} <= {s["dim"] for s in specs}, "must include a 3D instance"
    assert any(s.get("components") == ["ux", "uy"] for s in specs), \
        "must include a VECTOR interface"
    assert any("notched" in s.get("arrangement", "") for s in specs), \
        "must include an arrangement that is not the 2D rectangle"
    assert any("1000" in s["material_contrast"] for s in specs), \
        "must include a severe material contrast"
    assert any("reaction" in s["physics"] for s in specs), \
        "must include different operators either side"
    assert any("TRANSIENT" in s["physics"].upper() for s in specs)
    # every instance has a real material contrast: that is what makes the
    # interface condition carry information at all
    assert all(s["material_contrast"] for s in specs)


# ── group 5: the grader, end to end, on a correct and a broken submission ──
def _synthetic_submission(tmp_path, spec, flux_factor_b=1.0):
    import csv
    import math
    run = tmp_path / "D3_27b_MCP_seed0"
    w = run / "work"
    w.mkdir(parents=True)
    M = spec["probe_M"]

    def grid(bx, by):
        return [(bx[0] + (i + 0.5) * (bx[1] - bx[0]) / M,
                 by[0] + (j + 0.5) * (by[1] - by[0]) / M)
                for i in range(M) for j in range(M)]

    xi = float(spec["extent_b"][0][0])
    for lvl, h in enumerate([1 / 8, 1 / 16, 1 / 32], 1):
        for side, ext in (("A", spec["extent_a"]), ("B", spec["extent_b"])):
            with open(w / f"solution_level{lvl}_{side}.csv", "w",
                      newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["x", "y", "u"])
                for (X, Y) in grid(tuple(ext[0]), tuple(ext[1])):
                    wr.writerow([X, Y, math.sin(3 * X) * Y * (1 - Y)
                                 + 0.7 * h * h * math.cos(5 * X + Y)])
            with open(w / f"interface_level{lvl}_{side}.csv", "w",
                      newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["x", "y", "u", "qn"])
                for i in range(M):
                    Y = (i + 0.5) / M
                    u = math.sin(3 * xi) * Y * (1 - Y)
                    q = math.cos(3 * xi) * Y * (1 - Y)
                    wr.writerow([xi, Y, u,
                                 q if side == "A" else -q * flux_factor_b])
        with open(w / f"residual_level{lvl}.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["iteration", "interface_residual"])
            for k, v in enumerate([3e-1, 1.1e-3, 4e-6, 7e-8], 1):
                wr.writerow([k, v])
    (w / "RESULT.txt").write_text(
        "LEVELS = 3\nCOUPLING_ITERATIONS = 4\n"
        "MESH_INDEPENDENCE = CONVERGED\nMAX_REL_CHANGE = 0.001\n")
    (w / "fenics_run.log").write_text("dolfinx num_dofs 4225\nKSP converged")
    (w / "kratos_run.log").write_text(
        "KRATOS Multiphysics 9.5.1\nSolving time: 0.412")
    return run


def test_grader_order_is_blind_to_a_broken_coupling_and_the_interface_is_not(
        tmp_path):
    """The whole redesign, demonstrated through the grader that will grade.

    A submission that converges cleanly to the WRONG fixed point is graded
    SELF_CONVERGED at order 2.000 by the primary, key-free grade — identically
    to a correct one. The interface flux jump separates them, with the keys
    still sealed, which is the capability mesh halving does not have.
    """
    spec_path = REPO / "campaign3_blind" / "problems" / "D3" / "spec_public.json"
    if not spec_path.is_file():
        pytest.skip("D3 has not been built on this machine")
    spec = json.loads(spec_path.read_text())
    sys.path.insert(0, str(REPO / "scripts"))
    import blind_grade as BG

    ok = BG.grade(_synthetic_submission(tmp_path / "ok", spec), "D3")
    assert ok["phase1_key_free"]["verdict"] == "SELF_CONVERGED"
    assert abs(ok["phase1_key_free"]["observed_order_mesh_halving"] - 2.0) < 0.05
    assert ok["execution_evidence"]["verdict"] == "PROVEN"
    assert ok["phase1_interface"]["verdict"] == "INTERFACE_SATISFIED"
    assert ok["evidence_grade"]["grade"] == 1

    bad = BG.grade(_synthetic_submission(tmp_path / "bad", spec,
                                         flux_factor_b=4.0), "D3")
    # the primary key-free grade cannot tell them apart
    assert bad["phase1_key_free"]["verdict"] == "SELF_CONVERGED"
    assert (abs(bad["phase1_key_free"]["observed_order_mesh_halving"]
                - ok["phase1_key_free"]["observed_order_mesh_halving"]) < 1e-9)
    # the interface quantity does
    assert bad["phase1_interface"]["verdict"] == "INTERFACE_NOT_SATISFIED"
    assert min(bad["phase1_interface"]["jump_q_rel"]) > 1.0
    assert bad["phase1_interface"]["flux_jump_falls_under_refinement"] in (
        True, False)


# ── group 6: the interface ENDS are not gradeable ─────────────────────
def test_interface_probes_stay_clear_of_the_interface_ends():
    """Where the interface meets a constrained boundary the split problem has a
    Dirichlet-Neumann corner, and the recovered flux there gets WORSE under
    refinement (measured 2.11x -> 2.51x the true value over a 4x refinement).
    Grading those points fails a correct submission."""
    sys.path.insert(0, str(REPO / "campaign3_blind"))
    import build_coupled_v2 as BV
    for fn in BV.BUILDERS:
        spec = fn(BV.Draw(DEMO_SEED))[0]
        band = spec.get("iface_graded_band")
        assert band, f"{spec['id']} has no graded interface band"
        lo, hi = band
        assert lo > 0.0 and hi < 1.0, f"{spec['id']} grades an interface end"
        assert hi - lo >= 0.2, f"{spec['id']}'s band is too narrow to grade"


def test_task_states_that_interface_corners_belong_to_the_outer_boundary():
    """Getting this wrong converges, balances to 1e-10, and is 4.7% wrong in
    displacement and 28% wrong in traction — a silent wrong answer."""
    probs = REPO / "campaign3_blind" / "problems"
    seen = 0
    for d in sorted(probs.glob("D*")):
        t = d / "task.txt"
        if not t.is_file():
            continue
        seen += 1
        txt = t.read_text()
        assert "INTERFACE CORNERS" in txt, f"{d.name} does not state corner ownership"
        assert "BOTH" in txt.split("INTERFACE CORNERS")[1][:400]
    assert seen >= 8


def test_vector_relaxation_uses_the_worst_component():
    """theta = 1/(1 + max_c rho_c). A single scalar rho can diverge in the
    stiff component while converging in the other — measured spread 10.7x."""
    from blind_eval import femdd as F
    lam, muA, muB, wA, wB = 577.0, 385.0, 1155.0, 0.625, 0.875
    rho_n = ((lam + 2 * muA) / wA) / ((lam + 2 * muB) / wB)
    rho_s = (muA / wA) / (muB / wB)
    got = F.vector_theta(lam, muA, lam, muB, wA, wB)
    assert abs(got - 1.0 / (1.0 + max(rho_n, rho_s))) < 1e-12
    assert got <= 1.0 / (1.0 + min(rho_n, rho_s)), \
        "must be the WORST component, not the friendlier one"


def test_grader_refuses_interface_points_outside_the_graded_band(tmp_path):
    """Excluding the ends must not become a way to choose where you are graded."""
    spec_path = REPO / "campaign3_blind" / "problems" / "D3" / "spec_public.json"
    if not spec_path.is_file():
        pytest.skip("D3 has not been built on this machine")
    spec = json.loads(spec_path.read_text())
    if not spec.get("iface_graded_band"):
        pytest.skip("instance predates the graded band")
    sys.path.insert(0, str(REPO / "scripts"))
    import blind_grade as BG
    import csv as _csv
    run = _synthetic_submission(tmp_path / "ok", spec)
    good = BG.grade(run, "D3")
    assert good["phase1_interface"]["verdict"] in (
        "INTERFACE_SATISFIED", "INTERFACE_NOT_SATISFIED")
    # now move the interface probes out to the ends
    for lvl in (1, 2, 3):
        for side in ("A", "B"):
            p = run / "work" / f"interface_level{lvl}_{side}.csv"
            rows = list(_csv.reader(open(p)))
            with open(p, "w", newline="") as f:
                wr = _csv.writer(f)
                wr.writerow(rows[0])
                for i, r in enumerate(rows[1:]):
                    wr.writerow([r[0], 0.001 + i * 1e-6, r[2], r[3]])
    bad = BG.grade(run, "D3")
    assert bad["phase1_interface"]["verdict"] == "NOT_ASSESSED" or \
        any("graded band" in n for n in bad["phase1_interface"]["notes"])
