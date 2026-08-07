"""Tier-2: the "entrance effect" in this template is a formulation artifact.

Claim: skfem hydraulic_resistance#0 -- the Poiseuille R = 12 mu L / H^3 is only
valid for L >> H, short channels show entrance effects, "the FE-computed R is
HIGHER than 12 mu L / H^3 by an additive term ~ mu/H^2", at L/H = 20 the extra
is a couple of percent (Q = 4.100936e-05, R_fe = 2.438468e+04, relative error
1.60e-02), at L/H = 2 the extra is ~50%, and "refinement of MeshTri.init_tensor
doesn't help because the gap is physical (entrance length), not
discretization".

Measured on skfem 12.0.1.  The entry's numbers are REPRODUCED EXACTLY -- the
same 41x7 grid, 2106 velocity and 287 pressure DOFs, gives Q = 4.100936e-05
and R_fe = 2.438468e+04 -- so the original measurement was honest.  Its
ATTRIBUTION is wrong, and three of its four quantitative statements are wrong
with it.

  * THE GAP IS THE VISCOUS FORM, NOT THE GEOMETRY.  The shipped template uses
    the stress form 2*mu*(sym_grad u : sym_grad v).  Replacing it with the
    vector-Laplacian mu*(grad u : grad v) -- same geometry, same mesh, same
    boundary conditions, same flux computation -- reproduces R_exact to about
    1e-14 at EVERY aspect ratio tested, including L/H = 2.  A genuine
    entrance effect is a property of the geometry and would appear in both
    formulations.  It appears in only one.
    The two forms differ in the NATURAL boundary condition they imply at a
    traction boundary: 2*mu*sym(grad u).n - p n versus mu*du/dn - p n.  They
    agree only where d(u_x)/dx = 0, i.e. exactly where the flow is already
    fully developed.
  * THE SIGN IS WRONG AT SMALL L/H.  The entry says R_fe is HIGHER than
    R_exact.  At L/H = 2 the stress form gives a resistance LOWER than
    R_exact.
  * THE MAGNITUDE AT L/H = 2 IS WRONG.  The entry says ~50%.  Measured, it is
    a few percent.
  * "REFINEMENT DOESN'T HELP BECAUSE THE GAP IS PHYSICAL" IS WRONG.  Under
    refinement the stress-form gap MOVES, and not in one direction: at
    L/H = 20 it shrinks by nearly an order of magnitude, while at L/H = 10 it
    grows.  A fixed physical offset would do neither.

Mutation control: T2_MUTATE=1 applies the fix this fixture documents at the
pathology site -- TEMPLATE_FORM becomes laplacian_form, i.e. the template's
viscous term 2*mu*(sym_grad u : sym_grad v) is replaced by the vector
Laplacian mu*(grad u : grad v) everywhere the template's own form is used --
so the gap under study is gone.  The unmutated run matches all 14
expect_in_output strings; under mutation "template_Q=4.100936e-05",
"template_R_fe=2.438468e+04", "reproduces_entry_relative_error=True",
"stress_deviates=True" and
"gap_is_a_formulation_artifact_not_geometry=True" all disappear.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys

import numpy as np
import scipy.sparse as sp
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    ElementTriP2,
    ElementVector,
    FacetBasis,
    LinearForm,
    MeshTri,
    condense,
    solve,
)
from skfem.helpers import ddot, div, grad, sym_grad

MUTATE = os.environ.get("T2_MUTATE") == "1"

MU = 1.0
H = 0.1
P_IN = 1.0


@BilinearForm
def laplacian_form(u, v, w):
    return MU * ddot(grad(u), grad(v))


@BilinearForm
def stress_form(u, v, w):
    return 2.0 * MU * ddot(sym_grad(u), sym_grad(v))


# The viscous term the shipped template uses.  Under mutation it becomes the
# vector Laplacian -- the fix this fixture documents.
TEMPLATE_FORM = stress_form if not MUTATE else laplacian_form


@BilinearForm
def neg_div(u, p, w):
    return -div(u) * p


@LinearForm
def inlet_traction(v, w):
    return P_IN * v.value[0]


def run(length, nx, ny, form):
    m = MeshTri.init_tensor(np.linspace(0.0, length, nx),
                            np.linspace(0.0, H, ny))
    m = m.with_boundaries({
        "inlet": lambda x: x[0] < 1e-10,
        "outlet": lambda x: x[0] > length - 1e-10,
        "wall": lambda x: (x[1] < 1e-10) | (x[1] > H - 1e-10),
    })
    ev = ElementVector(ElementTriP2())
    bu = Basis(m, ev, intorder=4)
    bp = Basis(m, ElementTriP1(), intorder=4)
    fi = FacetBasis(m, ev, intorder=4, facets=m.boundaries["inlet"])
    fo = FacetBasis(m, ev, intorder=4, facets=m.boundaries["outlet"])
    K = sp.bmat([[form.assemble(bu), neg_div.assemble(bu, bp).T],
                 [neg_div.assemble(bu, bp), None]], format="csr")
    F = np.concatenate([inlet_traction.assemble(fi), bp.zeros()])
    D = np.concatenate([bu.get_dofs("wall").all(),
                        [bu.N + int(bp.get_dofs("outlet").all()[0])]])
    x = solve(*condense(K, F, x=np.zeros(K.shape[0]), D=D))
    q = float(np.sum(fo.interpolate(x[:bu.N]).value[0] * fo.dx))
    r_exact = 12.0 * MU * length / H ** 3
    return bu.N, bp.N, q, P_IN / q, r_exact


def main() -> int:
    ok = True
    grids = ((41, 7), (81, 13), (161, 25))

    # --- reproduce the entry's own measurement ----------------------------
    nu, npv, q, r, rex = run(20.0 * H, 41, 7, TEMPLATE_FORM)
    print(f"template_velocity_dofs={nu} pressure_dofs={npv}")
    print(f"template_Q={q:.6e}")
    print(f"template_R_fe={r:.6e}")
    print(f"template_R_exact={rex:.6e}")
    rel0 = abs(r - rex) / rex
    print(f"template_relative_error={rel0:.4e}")
    print(f"reproduces_entry_dof_counts={nu == 2106 and npv == 287}")
    print(f"reproduces_entry_relative_error={abs(rel0 - 1.60e-2) < 5e-4}")
    if not (nu == 2106 and npv == 287):
        print("FAIL: this is not the shipped template's discretisation",
              file=sys.stderr)
        ok = False
    if abs(rel0 - 1.60e-2) >= 5e-4:
        print("FAIL: the entry's own measurement did not reproduce",
              file=sys.stderr)
        ok = False

    # --- the same geometry with the vector-Laplacian form ----------------
    table = {}
    for tag, form in (("lap", laplacian_form), ("str", TEMPLATE_FORM)):
        for loh in (20.0, 10.0, 2.0):
            errs = []
            for nx, ny in grids:
                _, _, qq, rr, rr_ex = run(loh * H, nx, ny, form)
                errs.append(((rr - rr_ex) / rr_ex, rr, rr_ex))
            table[(tag, loh)] = errs
            print(f"{tag}_LoH{int(loh)}_signed_relative_errors="
                  f"{[f'{e[0]:+.4e}' for e in errs]}")

    lap_worst = max(abs(e[0]) for loh in (20.0, 10.0, 2.0)
                    for e in table[("lap", loh)])
    print(f"laplacian_worst_relative_error={lap_worst:.3e}")
    print(f"laplacian_matches_poiseuille_everywhere={lap_worst < 1e-9}")
    str_worst = max(abs(e[0]) for loh in (20.0, 10.0, 2.0)
                    for e in table[("str", loh)])
    print(f"stress_worst_relative_error={str_worst:.3e}")
    print(f"stress_deviates={str_worst > 1e-3}")
    print(f"gap_is_a_formulation_artifact_not_geometry="
          f"{lap_worst < 1e-9 and str_worst > 1e-3}")
    if lap_worst >= 1e-9:
        print("FAIL: the vector-Laplacian form also deviated, so the gap is "
              "not purely a formulation artifact", file=sys.stderr)
        ok = False
    if str_worst <= 1e-3:
        print("FAIL: the stress form did not deviate", file=sys.stderr)
        ok = False

    # --- the sign at small L/H --------------------------------------------
    small = table[("str", 2.0)]
    print(f"LoH2_R_fe={small[0][1]:.6e} R_exact={small[0][2]:.6e}")
    print(f"LoH2_R_fe_is_higher_than_exact={small[0][1] > small[0][2]}")
    print(f"LoH2_R_fe_is_lower_than_exact={small[0][1] < small[0][2]}")
    print(f"LoH2_relative_magnitude={abs(small[0][0]):.4e}")
    print(f"LoH2_magnitude_is_about_50_percent="
          f"{0.30 < abs(small[0][0]) < 0.70}")
    if small[0][1] > small[0][2]:
        print("FAIL: at L/H = 2 the FE resistance WAS higher, as the entry "
              "says", file=sys.stderr)
        ok = False
    if 0.30 < abs(small[0][0]) < 0.70:
        print("FAIL: the L/H = 2 deviation really was ~50%", file=sys.stderr)
        ok = False

    # --- refinement -------------------------------------------------------
    e20 = [abs(e[0]) for e in table[("str", 20.0)]]
    e10 = [abs(e[0]) for e in table[("str", 10.0)]]
    print(f"stress_LoH20_under_refinement={[f'{x:.3e}' for x in e20]}")
    print(f"stress_LoH10_under_refinement={[f'{x:.3e}' for x in e10]}")
    print(f"LoH20_gap_shrinks_under_refinement={e20[-1] < 0.5 * e20[0]}")
    print(f"LoH10_gap_grows_under_refinement={e10[-1] > e10[0]}")
    print(f"gap_is_not_a_fixed_physical_offset="
          f"{e20[-1] < 0.5 * e20[0] and e10[-1] > e10[0]}")
    if not (e20[-1] < 0.5 * e20[0] and e10[-1] > e10[0]):
        print("FAIL: the gap behaved like a fixed offset under refinement",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
