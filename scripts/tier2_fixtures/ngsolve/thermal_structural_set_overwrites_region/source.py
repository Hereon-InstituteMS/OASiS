"""Tier-2: GridFunction.Set() is a whole-vector overwrite, not an accumulate.

Claim: ngsolve thermal_structural#1 — every Set() call zeroes all DOFs outside
its `definedon` region first, so the two-call idiom

    gfT.Set(100.0, definedon=mesh.Boundaries('left'))
    gfT.Set(0.0,   definedon=mesh.Boundaries('right'))

silently discards the 100.0. Fix: ONE Set with a single CoefficientFunction
covering the whole Dirichlet region, e.g.
gfT.Set(IfPos(0.5-x, 100.0, 0.0), definedon=mesh.Boundaries('left|right')).
This is what makes the shipped thermal_structural_2d template print
'Temperature: [0.00, 0.00]' and a zero displacement while still exiting rc=0.

Wrong variant: the two-call idiom, run exactly as the shipped template has it,
then carried through the template's own heat solve and thermal-expansion solve.

Observed on NGSolve 6.2.2604 (2026-08-03), unit_square maxh=0.3,
H1(order=2, dirichlet='left|right'):
  * after the FIRST Set the vector spans [-2.92e-13, 100.0];
  * after the SECOND it is exactly [0.0, 0.0] -- the hot edge is gone;
  * swapping the call order leaves [-2.92e-13, 100.0], i.e. only the LAST call
    survives, which is the same defect seen from the other side;
  * the single IfPos Set gives [-2.92e-13, 100.0], the intended state;
  * downstream, the two-call template prints Temperature: [0.00, 0.00] and a
    displacement of 0.0 at (1, 0.5) with return code 0, while the one-call
    version transports heat and displaces the tip.
"""
from __future__ import annotations

import sys

import ngsolve as ngs
from netgen.geom2d import unit_square

# WRONG: one Set() call per boundary region
SET_CALLS_PER_REGION = "two"

T_HOT, T_COLD = 100.0, 0.0
E, NU, ALPHA = 200e3, 0.3, 12e-6
MU = E / (2.0 * (1.0 + NU))
LAM = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))


def set_dirichlet(gfT, mesh, style: str) -> None:
    if style == "two":
        gfT.Set(ngs.CoefficientFunction(T_HOT),
                definedon=mesh.Boundaries("left"))
        gfT.Set(ngs.CoefficientFunction(T_COLD),
                definedon=mesh.Boundaries("right"))
    elif style == "two_swapped":
        gfT.Set(ngs.CoefficientFunction(T_COLD),
                definedon=mesh.Boundaries("right"))
        gfT.Set(ngs.CoefficientFunction(T_HOT),
                definedon=mesh.Boundaries("left"))
    elif style == "one":
        gfT.Set(ngs.IfPos(0.5 - ngs.x, T_HOT, T_COLD),
                definedon=mesh.Boundaries("left|right"))
    else:
        raise ValueError(style)


def coupled_run(style: str) -> tuple[float, float, float, float]:
    """The shipped thermal_structural_2d recipe, with the BC style swapped in."""
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.3))
    V_T = ngs.H1(mesh, order=2, dirichlet="left|right")
    uT, vT = V_T.TnT()
    aT = ngs.BilinearForm(ngs.grad(uT) * ngs.grad(vT) * ngs.dx).Assemble()
    fT = ngs.LinearForm(V_T)
    fT.Assemble()
    gfT = ngs.GridFunction(V_T)
    set_dirichlet(gfT, mesh, style)
    t_after_bc = (float(min(gfT.vec)), float(max(gfT.vec)))
    fT.vec.data -= aT.mat * gfT.vec
    gfT.vec.data += aT.mat.Inverse(V_T.FreeDofs()) * fT.vec

    V_u = ngs.VectorH1(mesh, order=2, dirichlet="left")
    u, v = V_u.TnT()

    def strain(w):
        return 0.5 * (ngs.Grad(w) + ngs.Grad(w).trans)

    a_u = ngs.BilinearForm(ngs.InnerProduct(
        2 * MU * strain(u) + LAM * ngs.Trace(strain(u)) * ngs.Id(2),
        strain(v)) * ngs.dx).Assemble()
    f_u = ngs.LinearForm(ngs.InnerProduct(
        (3 * LAM + 2 * MU) * ALPHA * gfT * ngs.Id(2), strain(v)) * ngs.dx
    ).Assemble()
    gfu = ngs.GridFunction(V_u)
    gfu.vec.data = a_u.mat.Inverse(V_u.FreeDofs()) * f_u.vec
    ux = float(gfu.components[0](mesh(1.0, 0.5)))
    return t_after_bc[0], t_after_bc[1], float(max(gfT.vec)), ux


def main() -> int:
    ok = True
    print(f"ngsolve_version={ngs.__version__}")
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.3))
    V = ngs.H1(mesh, order=2, dirichlet="left|right")
    print(f"thermal_space_type={V.type} ndof={V.ndof}")

    # --- WRONG variant: watch the first Set() get erased ------------------
    gfT = ngs.GridFunction(V)
    gfT.Set(ngs.CoefficientFunction(T_HOT), definedon=mesh.Boundaries("left"))
    lo1, hi1 = float(min(gfT.vec)), float(max(gfT.vec))
    print(f"after_first_set=[{lo1:.6g}, {hi1:.6g}]")
    print(f"first_set_reached_t_hot={abs(hi1 - T_HOT) < 1e-9}")
    gfT.Set(ngs.CoefficientFunction(T_COLD), definedon=mesh.Boundaries("right"))
    lo2, hi2 = float(min(gfT.vec)), float(max(gfT.vec))
    print(f"after_second_set=[{lo2:.6g}, {hi2:.6g}]")
    print(f"second_set_wiped_t_hot={hi2 == 0.0 and lo2 == 0.0}")
    if abs(hi1 - T_HOT) >= 1e-9:
        print("FAIL: the first Set() did not reach T_hot at all",
              file=sys.stderr)
        ok = False
    if not (hi2 == 0.0 and lo2 == 0.0):
        print(f"FAIL: the second Set() left [{lo2:.6g}, {hi2:.6g}] -- Set() no "
              f"longer overwrites the whole vector", file=sys.stderr)
        ok = False

    # only the LAST call survives, whichever order it is in
    gf_sw = ngs.GridFunction(V)
    set_dirichlet(gf_sw, mesh, "two_swapped")
    hi_sw = float(max(gf_sw.vec))
    print(f"swapped_order_max={hi_sw:.6g}")
    print(f"swapped_order_keeps_only_last_call={abs(hi_sw - T_HOT) < 1e-9}")
    if abs(hi_sw - T_HOT) >= 1e-9:
        print("FAIL: swapping the call order did not move which value survives",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant: one Set over the whole Dirichlet region -----------
    gf_ok = ngs.GridFunction(V)
    set_dirichlet(gf_ok, mesh, "one")
    lo3, hi3 = float(min(gf_ok.vec)), float(max(gf_ok.vec))
    print(f"single_ifpos_set=[{lo3:.6g}, {hi3:.6g}]")
    print(f"single_set_keeps_t_hot={abs(hi3 - T_HOT) < 1e-9}")
    print(f"single_set_keeps_t_cold={abs(lo3) < 1e-9}")
    if abs(hi3 - T_HOT) >= 1e-9 or abs(lo3) >= 1e-9:
        print(f"FAIL: the documented one-call fix gave [{lo3:.6g}, {hi3:.6g}]",
              file=sys.stderr)
        ok = False

    # --- downstream consequence in the shipped template -------------------
    _, _, tmax_bad, ux_bad = coupled_run(SET_CALLS_PER_REGION)
    _, _, tmax_ok, ux_ok = coupled_run("one")
    print(f"two_call_solved_t_max={tmax_bad:.6g} tip_ux={ux_bad:.6e}")
    print(f"one_call_solved_t_max={tmax_ok:.6g} tip_ux={ux_ok:.6e}")
    print(f"two_call_template_temperature_is_all_zero={tmax_bad == 0.0}")
    print(f"two_call_template_displacement_is_zero={ux_bad == 0.0}")
    print(f"one_call_template_transports_heat={abs(tmax_ok - T_HOT) < 1e-9}")
    print(f"one_call_template_displaces_tip={abs(ux_ok) > 1e-9}")
    if tmax_bad != 0.0 or ux_bad != 0.0:
        print(f"FAIL: the two-call template no longer collapses to zero "
              f"(T_max={tmax_bad:.6g}, u_x={ux_bad:.6e})", file=sys.stderr)
        ok = False
    if abs(tmax_ok - T_HOT) >= 1e-9 or abs(ux_ok) <= 1e-9:
        print("FAIL: the one-call fix did not give a live thermal-structural "
              "solution", file=sys.stderr)
        ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
