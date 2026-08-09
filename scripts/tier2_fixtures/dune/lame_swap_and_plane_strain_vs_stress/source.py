"""Tier-2: three ways an elasticity form is wrong while still
converging.

  linear_elasticity#3   swapping the mu and lam formulae still
                        assembles, still converges and still prints a
                        plausible displacement; the detector is the SIGN
                        of the lateral contraction.
  linear_elasticity#4   writing the strain as a bare grad(u) instead of
                        sym(grad(u)) couples rotation into stress: a
                        rigid-body rotation then carries energy.
  linear_elasticity#6   plane STRAIN and plane STRESS differ only in
                        lam and DUNE-fem has no switch; judging a
                        plane-strain run against the plane-stress
                        formula gives a ratio of about 1 - nu^2, i.e.
                        the run looks ~9% too SOFT, and it is the
                        reverse pairing that gives 1/(1 - nu^2).

Cost control: mu and lam are dune.ufl.Constants, so the swap and the
plane-stress substitution are VALUE changes and cost no rebuild. Only
two modules are needed — the correct sym(grad(u)) form and the wrong
bare-grad(u) one.

Verified by execution against dune-fem 2.12.0.2.

MUTATION CONTROL. T2_MUTATE=1 substitutes the PLANE-STRESS Lame
parameter lam_ps = 2 mu lam / (lam + 2 mu) before the first run — the
pathology (a plane-strain run judged against the plane-stress formula)
removed. The ratio then lands on 1.0 instead of 1 - nu^2, so
'plane_strain_over_plane_stress_ratio=0.9114' and
'strain_run_judged_by_stress_formula_looks_soft=True' are no longer
printed and a FAIL: line appears. lam is a dune.ufl.Constant, so the
mutation costs no rebuild.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dune.grid import structuredGrid                            # noqa: E402
from dune.fem.space import lagrange                             # noqa: E402
from dune.fem.scheme import galerkin                            # noqa: E402
from dune.ufl import Constant, DirichletBC                      # noqa: E402
import dune.fem as dfem                                         # noqa: E402
from ufl import (TrialFunction, TestFunction,                    # noqa: E402
                 SpatialCoordinate, Identity, as_vector,
                 grad, inner, sym, tr, dx, ds, conditional, lt)

E, NU, TRACTION = 210e9, 0.3, 1.0e6
MU_OK = E / (2 * (1 + NU))
LAM_OK = E * NU / ((1 + NU) * (1 - 2 * NU))
TOL = 1e-8
PARAMS = {"linear.tolerance": 1e-14, "linear.maxiterations": 100000}


def main() -> int:
    fail: list[str] = []
    # 32x32: the plane-strain/plane-stress ratio is a DISCRETISATION-
    # sensitive number (measured 0.9330 at 8x8, against 1-nu^2 =
    # 0.9100), and the grid size is not part of the generated code,
    # so refining it is free.
    gridView = structuredGrid([0, 0], [1, 1], [32, 32])
    space = lagrange(gridView, dimRange=2, order=1)
    u, v = TrialFunction(space), TestFunction(space)
    x = SpatialCoordinate(space)
    I = Identity(2)
    mu = Constant(MU_OK, name="mu")
    lam = Constant(LAM_OK, name="lam")

    def sigma(w, strain):
        return lam * tr(strain(w)) * I + 2 * mu * strain(w)

    def sym_grad(w):
        return sym(grad(w))

    def bare_grad(w):
        return grad(w)

    a_ok = inner(sigma(u, sym_grad), sym_grad(v)) * dx
    L = conditional(lt(1.0 - x[0], TOL), TRACTION * v[0], 0.0) * ds
    bcs = [DirichletBC(space, [0, None], conditional(lt(x[0], TOL), 1, 0)),
           DirichletBC(space, [None, 0], conditional(lt(x[1], TOL), 1, 0))]
    scheme = galerkin([a_ok == L] + bcs, solver="cg", parameters=PARAMS)

    def run(name):
        uh = space.interpolate([0, 0], name=name)
        info = scheme.solve(target=uh)
        vals = np.array(uh.as_numpy).reshape(-1, 2)
        return info, vals

    # ── correct parameters: tension stretches, sides pull IN ─────────
    if MUTATE:
        print("mutation=the_first_run_uses_the_plane_stress_lame_"
              "parameter")
        lam.value = 2 * MU_OK * LAM_OK / (LAM_OK + 2 * MU_OK)
    info_ok, vals_ok = run("ok")
    ux_max, uy_min = float(vals_ok[:, 0].max()), float(vals_ok[:, 1].min())
    print(f"correct_converged={bool(info_ok['converged'])}")
    print(f"correct_ux_max={ux_max:.6e}")
    print(f"correct_uy_min={uy_min:.6e}")
    print(f"correct_lateral_contraction_is_negative={uy_min < 0.0}")
    if not info_ok["converged"] or uy_min >= 0.0:
        fail.append(f"the correctly parameterised run gave uy_min "
                    f"{uy_min:.6e}; under tension the lateral "
                    f"displacement must be negative, and without that "
                    f"control the swapped run proves nothing")

    # ── linear_elasticity#6: plane strain vs plane stress ───────────
    # the plane-STRESS closed form for this uniaxial pull is u_x(1)=t/E
    ux_analytic_ps = TRACTION / E
    ratio = ux_max / ux_analytic_ps
    print(f"plane_stress_analytic_ux={ux_analytic_ps:.6e}")
    print(f"plane_strain_over_plane_stress_ratio={ratio:.4f}")
    print(f"one_minus_nu_squared={1 - NU ** 2:.4f}")
    print(f"strain_run_judged_by_stress_formula_looks_soft="
          f"{abs(ratio - (1 - NU ** 2)) < 0.02}")
    if abs(ratio - (1 - NU ** 2)) >= 0.02:
        fail.append(f"a plane-STRAIN run judged against the "
                    f"plane-STRESS value gave a ratio of {ratio:.4f}; "
                    f"the claim is about 1-nu^2 = {1 - NU ** 2:.4f}, "
                    f"i.e. the run looks ~9% too soft, NOT 1/(1-nu^2)")

    # and the reverse pairing: substituting lam_ps makes it match t/E
    lam.value = 2 * MU_OK * LAM_OK / (LAM_OK + 2 * MU_OK)
    info_ps, vals_ps = run("ps")
    ux_ps = float(vals_ps[:, 0].max())
    ratio_ps = ux_ps / ux_analytic_ps
    print(f"plane_stress_lam_substituted_ratio={ratio_ps:.4f}")
    print(f"plane_stress_substitution_recovers_the_formula="
          f"{abs(ratio_ps - 1.0) < 0.02}")
    if abs(ratio_ps - 1.0) >= 0.02:
        fail.append(f"substituting lam_ps = 2*mu*lam/(lam+2*mu) gave a "
                    f"ratio of {ratio_ps:.4f} against t/E; the claim is "
                    f"that this is the plane-stress switch DUNE does "
                    f"not provide")
    lam.value = LAM_OK

    # ── linear_elasticity#3: the swap ──────────────────────────────
    mu.value, lam.value = LAM_OK, MU_OK
    info_swap, vals_swap = run("swap")
    ux_swap = float(vals_swap[:, 0].max())
    uy_swap = float(vals_swap[:, 1].min())
    print(f"swapped_converged={bool(info_swap['converged'])}")
    print(f"swapped_ux_max={ux_swap:.6e}")
    print(f"swapped_uy_min={uy_swap:.6e}")
    print(f"swapped_still_converges={bool(info_swap['converged'])}")
    print(f"swapped_displacement_is_plausible="
          f"{0.0 < ux_swap < 10 * ux_max}")
    print(f"swapped_changes_the_answer="
          f"{abs(ux_swap - ux_max) > 1e-12}")
    if not info_swap["converged"]:
        fail.append("the swapped-parameter run did not converge; the "
                    "claim is that it converges and looks plausible")
    if abs(ux_swap - ux_max) <= 1e-12:
        fail.append("swapping mu and lam changed nothing at all, so "
                    "there is no silent-wrong to detect")
    mu.value, lam.value = MU_OK, LAM_OK

    # ── linear_elasticity#4: bare grad(u) carries rotation energy ───
    theta = 1e-3
    rot = space.interpolate(as_vector([-theta * x[1], theta * x[0]]),
                            name="rot")
    e_sym = float(dfem.integrate(
        inner(sym(grad(rot)), sym(grad(rot))),
        gridView=gridView, order=4))
    e_bare = float(dfem.integrate(
        inner(grad(rot), grad(rot)), gridView=gridView, order=4))
    print(f"rigid_rotation_sym_energy={e_sym:.6e}")
    print(f"rigid_rotation_bare_grad_energy={e_bare:.6e}")
    print(f"sym_grad_sees_no_strain_in_a_rotation={e_sym < 1e-20}")
    print(f"bare_grad_invents_strain_in_a_rotation="
          f"{e_bare > 1e-8 * theta ** 2}")
    if e_sym >= 1e-20:
        fail.append(f"sym(grad(u)) gave {e_sym:.6e} on a rigid-body "
                    f"rotation; it must be exactly zero")
    if e_bare <= 1e-8 * theta ** 2:
        fail.append(f"bare grad(u) gave {e_bare:.6e} on a rigid-body "
                    f"rotation; the claim is that it couples rotation "
                    f"into stress")

    # …and the wrong FORM assembles and solves anyway
    a_bad = inner(sigma(u, bare_grad), bare_grad(v)) * dx
    scheme_bad = galerkin([a_bad == L] + bcs, solver="bicgstab",
                          parameters=PARAMS)
    uh_bad = space.interpolate([0, 0], name="bad")
    info_bad = scheme_bad.solve(target=uh_bad)
    vals_bad = np.array(uh_bad.as_numpy).reshape(-1, 2)
    print(f"bare_grad_form_converged={bool(info_bad['converged'])}")
    print(f"bare_grad_form_ux_max={float(vals_bad[:, 0].max()):.6e}")
    print(f"bare_grad_form_assembles_and_solves="
          f"{bool(info_bad['converged'])}")
    if not info_bad["converged"]:
        fail.append("the bare-grad(u) form did not converge, so it "
                    "cannot be shown to be a silent-wrong")

    if not fail:
        print("dune_elasticity_constitutive_traps_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
