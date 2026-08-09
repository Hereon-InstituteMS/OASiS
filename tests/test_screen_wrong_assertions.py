"""Ratchet for `scripts/screen_wrong_assertions.py`.

WHY THIS EXISTS
---------------
`test_expectations_assert_values.py` catches a fixture whose EVERY expectation is
a bare `key=` prefix. This catches seven more shapes of the same disease: an
assertion that passes for a reason unrelated to the name it carries. A mutation
control cannot see any of them, because they are true with the pathology and
without it.

The screen's own docstring documents each shape. What is enforced here:

  ZERO CATEGORIES -- SELFSAME, EMPTYEXC, SUCCESS, ARGMAX. Every hit these
  produced was read and every one was real, so any new hit is a defect and the
  gate has no baseline for them. They were 1, 2, 4 and 1 at f1d4fbf5 and are 0
  now.

  BASELINED CATEGORIES -- BAKED_QUOTE, BAKED_ALL, SYNTHETIC, CONSTCMP. These
  carry known false positives, each read and each explained below, so a new hit
  fails the gate while the read ones do not. The baselines shrink only.

WHAT A NEW HIT MEANS
--------------------
Not "this fixture is broken" -- "read this line and decide". The screen is
static; it cannot see a fixture whose discrimination lives in
`forbid_in_output`, and it cannot see an assertion that is a substring grep over
generated source text. Both blind spots were found by reading a random sample of
what the screen did NOT flag, and both are recorded in the screen's docstring.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "screen_wrong_assertions", REPO / "scripts" / "screen_wrong_assertions.py")
screen_mod = importlib.util.module_from_spec(_spec)
sys.modules["screen_wrong_assertions"] = screen_mod
_spec.loader.exec_module(screen_mod)

# No fixture may land in these again. Every hit ever produced was confirmed.
ZERO_CATEGORIES = ("SELFSAME", "EMPTYEXC", "SUCCESS", "ARGMAX")

BASELINE: dict[str, set[str]] = {
    # Read 2026-08-08. All nine are false positives of the "quoted diagnostic"
    # rule, kept as a baseline rather than special-cased in the screen so that a
    # NEW baked expectation still fails.
    "BAKED_QUOTE": {
        # `ok ? "FIXTURE_OK" : "FIXTURE_MISMATCH"`, with ok computed from the
        # triangulation and MISMATCH forbidden.
        "dealii/grid_in_unsupported_cell_type",
        # "SolverControl" is written only in the catch arm; the converged arm
        # prints "Unexpected: SolverCG converged".
        "dealii/iterative_method_failed_to_converge",
        # "is not registered" is str(exc) from Kratos, not a fixture literal.
        "kratos/contact_mortar_entities_are_conditions",
        "kratos/poromechanics_upl_not_upw",
        "kratos/shallowwater_element_is_boussinesq_stem",
        # The baked prose was dropped from all three on 2026-08-08; what the
        # screen still sees is the surviving gated needle ("do not match",
        # "CFL violated", "singular"), each printed only past a measured guard.
        "skfem/element_vector_dof_interleaving",
        "skfem/forward_euler_instability",
        "skfem/missing_condense_singular",
        # "global_differs" is inside the measured Nbfun==3 && Nbfun!=N check.
        "skfem/nbfun_vs_global_dof",
    },
    # Read 2026-08-08. Every one of these routes its discrimination through
    # `forbid_in_output` -- a success token gated on measured quantities, plus a
    # forbidden `FAIL:` -- which the screen cannot see from the expect list.
    # The MMS group additionally leaves its numbers unpinned on purpose, so the
    # measured values do not leak into the corpus as answers.
    "BAKED_ALL": {
        "febio/degenerate_hex8_clean_wrong_result",
        "febio/hex8_patch_test_exact_stress",
        "febio/missing_control_silent_zero_result",
        "febio/null_test_linear_solver_reports_success",
        "fenics/stokes_mms_convergence",
        "fourc/structural_2d_solid_quad4_not_wall",
        "kratos/elasticity_mms_convergence",
        "kratos/poisson_mms_convergence",
        "kratos/poromechanics_upl_not_upw",
        "kratos/shallowwater_element_is_boussinesq_stem",
        "ngsolve/stokes_mms_convergence",
        "skfem/elasticity_mms_convergence",
        "skfem/element_vector_dof_interleaving",
        "skfem/forward_euler_instability",
        "skfem/missing_condense_singular",
        "skfem/stokes_mms_convergence",
    },
    # Read 2026-08-08. In all but one the arange/eye is incidental -- a time
    # axis, a free-DOF index set, a reference identity compared against a
    # computed matrix -- and the assertion does measure a real outcome.
    "SYNTHETIC": {
        "fenics/dg_mass_matrix_is_block_diagonal",
        "fenics/entity_maps_is_a_sequence_of_entitymap",
        # The one real case, recorded as NOT repairable in its own _comment:
        # the element mass matrix is typed by the fixture, so both flagged lines
        # are properties of that formula. Repairing means taking the matrix from
        # a Kratos element, i.e. redesigning what the fixture exercises.
        "kratos/heat_transient_consistent_mass_beats_lumped",
        "skfem/adapt_f2t_boundary_marker",
        "skfem/hyd_bmat_zero_block",
        "skfem/td_explicit_euler_mass_not_diagonal",
        "skfem/td_implicit_dt_accuracy_not_stability",
        "skfem/td_recondense_time_varying_bc",
    },
    # Read 2026-08-08. `AITKEN != CONSTANT and a["accelerator"] !=
    # c["accelerator"]`: the first conjunct is two constants, the second is read
    # off the two runs' results, so the line does measure that the arms differed.
    "CONSTCMP": {
        "coupling/aitken_survives_where_constant_theta_diverges",
    },
}


def _by_category() -> dict[str, set[str]]:
    found = screen_mod.screen()
    return {cat: {h["fixture"] for h in found.get(cat, [])}
            for cat in screen_mod.CATEGORIES}


def test_no_fixture_in_a_zero_category() -> None:
    got = _by_category()
    offenders = {c: sorted(got[c]) for c in ZERO_CATEGORIES if got[c]}
    assert not offenders, (
        "these shapes are always real -- every hit they have ever produced was "
        "confirmed by reading the probe -- so there is no baseline for them:\n"
        + "\n".join(f"  {c}: {v}" for c, v in offenders.items())
        + "\n\nSee scripts/screen_wrong_assertions.py for what each category "
          "means and what the repaired form looks like.")


def test_no_new_hit_in_a_baselined_category() -> None:
    got = _by_category()
    new = {c: sorted(got[c] - BASELINE[c]) for c in BASELINE if got[c] - BASELINE[c]}
    assert not new, (
        "new fixture(s) flagged by the wrong-assertion screen. Read the probe: "
        "either repair the assertion so it measures what its name says, or add "
        "it to BASELINE with the sentence that says why the hit is false.\n"
        + "\n".join(f"  {c}: {v}" for c, v in new.items()))


def test_baselines_only_shrink() -> None:
    """A fixture that has been repaired must leave the baseline.

    Otherwise the list becomes a graveyard and a fixture that regresses is
    silently re-permitted by its own stale entry.
    """
    got = _by_category()
    stale = {c: sorted(BASELINE[c] - got[c]) for c in BASELINE
             if BASELINE[c] - got[c]}
    assert not stale, (
        "these no longer trip the screen and must leave BASELINE:\n"
        + "\n".join(f"  {c}: {v}" for c, v in stale.items()))
