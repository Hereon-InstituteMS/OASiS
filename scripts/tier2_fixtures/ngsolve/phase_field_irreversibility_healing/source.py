"""Tier-2: without the max(d, d_prev) projection the crack heals on
unloading - and the shipped fracture template never gets that far, because
it reports d_max = 0 forever.

Claim: ngsolve phase_field#3 - phase-field fracture needs irreversibility
d >= d_prev, enforced pointwise; otherwise d decreases between steps
(unphysical healing).

Wrong variant A (the shipped phase_field_fracture_2d template): solve the
  elastic step as `gfu.vec.data = a_u.mat.Inverse(Vu.FreeDofs()) * f_u.vec`
  with an all-zero load vector.  That overwrites the whole coefficient
  vector, Dirichlet entries included, so the prescribed displacement is
  destroyed, u == 0, psi_plus == 0 and d == 0 at every load step - silently.
Wrong variant B: fix the elastic step by homogenising, then run a
  load-unload schedule with no history projection - d follows the load back
  down and heals to exactly 0.
Right variant  : same, with gfd = max(gfd_unconstrained, gfd_prev).

Observed on NGSolve 6.2.2604, unit_square maxh=0.15 (ne=96, VectorH1 order 1
ndof=126 dirichlet='bottom|top', H1 ndof=63), E=1, nu=0.3, Gc=1e-3, l0=0.1,
schedule 0.01 -> 0.02 -> 0.03 -> 0.02 -> 0.01 -> 0.0 (2026-08-06):
  shipped elastic step : |u|_inf = 0 and d_max = 0 at all six steps
  homogenised, no proj : d_max 0.014295, 0.055392, 0.118524, then back down
                         0.055392, 0.014295, 0.0  - full healing
  homogenised + proj   : 0.014295, 0.055392, 0.118524, 0.118524, 0.118524,
                         0.118524 - monotone, as required

Second defect of the shipped template recorded here: GridFunction.Set with
definedon= zeroes every other DOF, so its two consecutive Set calls (top,
then bottom) leave ONLY the bottom displacement applied - the top boundary
integral of u_y drops from 0.03 to exactly 0.0.
"""
from __future__ import annotations

import sys

from netgen.geom2d import unit_square
from ngsolve import (BilinearForm, CoefficientFunction, GridFunction, Grad,
                     H1, Id, Integrate, InnerProduct, LinearForm, Mesh, Trace,
                     VectorH1, ds, dx, grad)

E_MOD, NU = 1.0, 0.3
MU = E_MOD / (2 * (1 + NU))
LAM = E_MOD * NU / ((1 + NU) * (1 - 2 * NU))
GC, L0, KRES = 1e-3, 0.1, 1e-10
SCHEDULE = (0.01, 0.02, 0.03, 0.02, 0.01, 0.0)

mesh = Mesh(unit_square.GenerateMesh(maxh=0.15))
Vu = VectorH1(mesh, order=1, dirichlet="bottom|top")
Vd = H1(mesh, order=1)


def strain(w):
    return 0.5 * (Grad(w) + Grad(w).trans)


def sig0(w):
    e = strain(w)
    return 2 * MU * e + LAM * Trace(e) * Id(2)


def psi(w):
    e = strain(w)
    return 0.5 * LAM * Trace(e) ** 2 + MU * InnerProduct(e, e)


def run(homogenize, project):
    u, v = Vu.TnT()
    d, phi = Vd.TnT()
    gfu, gfd, gprev = GridFunction(Vu), GridFunction(Vd), GridFunction(Vd)
    gfd.Set(0)
    gprev.Set(0)
    hist = []
    for disp in SCHEDULE:
        for _ in range(40):
            gfu.Set(CoefficientFunction((0, disp)),
                    definedon=mesh.Boundaries("top"))
            a = BilinearForm(Vu)
            a += ((1 - gfd) ** 2 + KRES) * InnerProduct(sig0(u),
                                                        strain(v)) * dx
            a.Assemble()
            f = LinearForm(Vu)
            f.Assemble()
            inv = a.mat.Inverse(Vu.FreeDofs())
            if homogenize:
                r = f.vec.CreateVector()
                r.data = f.vec - a.mat * gfu.vec
                gfu.vec.data += inv * r
            else:
                gfu.vec.data = inv * f.vec
            hfield = psi(gfu)
            ad = BilinearForm(Vd, symmetric=True)
            ad += ((GC / L0 + 2 * hfield) * d * phi
                   + GC * L0 * grad(d) * grad(phi)) * dx
            ad.Assemble()
            fd = LinearForm(2 * hfield * phi * dx)
            fd.Assemble()
            gnew = GridFunction(Vd)
            gnew.vec.data = ad.mat.Inverse(Vd.FreeDofs()) * fd.vec
            if project:
                for i in range(len(gnew.vec)):
                    gnew.vec[i] = max(float(gnew.vec[i]),
                                      float(gprev.vec[i]))
            delta = (gnew.vec - gfd.vec).Norm()
            gfd.vec.data = gnew.vec
            if delta < 1e-9:
                break
        gprev.vec.data = gfd.vec
        hist.append((max(gfd.vec), max(abs(float(q)) for q in gfu.vec)))
    return hist


def main() -> int:
    ok = True
    print(f"mesh_ne={mesh.ne} ndof_u={Vu.ndof} ndof_d={Vd.ndof}")
    print(f"vector_space_class={type(Vu).__name__}")

    # ---- WRONG A: the shipped elastic step ------------------------------
    shipped = run(False, False)
    print(f"shipped_dmax_sequence={[f'{d:.6g}' for d, _ in shipped]}")
    print(f"shipped_dmax_is_zero_at_every_step={all(d == 0 for d, _ in shipped)}")
    print("shipped_displacement_is_zero_at_every_step="
          f"{all(u == 0 for _, u in shipped)}")
    if not all(d == 0 for d, _ in shipped):
        print(f"FAIL: the shipped elastic step produced damage "
              f"{[d for d, _ in shipped]}; the silent no-op is gone",
              file=sys.stderr)
        ok = False

    # ---- WRONG B: homogenised, but no history projection ----------------
    healing = run(True, False)
    peak = max(d for d, _ in healing)
    final = healing[-1][0]
    print(f"healing_dmax_sequence={[f'{d:.6g}' for d, _ in healing]}")
    print("homogenized_dirichlet_value_applied_exactly="
          f"{all(abs(u - s) < 1e-12 for (_, u), s in zip(healing, SCHEDULE))}")
    print(f"homogenized_dmax_grows={peak > 0.01}")
    print("no_projection_damage_decreased_on_unload="
          f"{any(healing[i + 1][0] < healing[i][0] - 1e-12
                 for i in range(len(healing) - 1))}")
    print(f"no_projection_final_dmax_below_half_peak={final < 0.5 * peak}")
    if not peak > 0.01:
        print(f"FAIL: with a correctly applied load the damage still did not "
              f"grow (peak {peak:.3e})", file=sys.stderr)
        ok = False
    if not final < 0.5 * peak:
        print(f"FAIL: no healing observed - peak {peak:.6g}, final "
              f"{final:.6g}", file=sys.stderr)
        ok = False

    # ---- RIGHT: pointwise max(d, d_prev) --------------------------------
    monotone = run(True, True)
    seq = [d for d, _ in monotone]
    is_mono = all(seq[i + 1] >= seq[i] - 1e-12 for i in range(len(seq) - 1))
    print(f"projected_dmax_sequence={[f'{d:.6g}' for d in seq]}")
    print(f"with_projection_damage_monotone={is_mono}")
    print("with_projection_final_equals_peak="
          f"{abs(seq[-1] - max(seq)) < 1e-12}")
    print("with_projection_matches_loading_branch="
          f"{abs(seq[2] - healing[2][0]) < 1e-12}")
    if not (is_mono and abs(seq[-1] - max(seq)) < 1e-12):
        print(f"FAIL: the max(d, d_prev) projection did not make the damage "
              f"monotone: {seq}", file=sys.stderr)
        ok = False

    # ---- the shipped template's second defect ---------------------------
    g = GridFunction(Vu)
    g.Set(CoefficientFunction((0, 0.03)), definedon=mesh.Boundaries("top"))
    top_before = Integrate(g[1] * ds(definedon=mesh.Boundaries("top")), mesh)
    g.Set(CoefficientFunction((0, -0.03)),
          definedon=mesh.Boundaries("bottom"))
    top_after = Integrate(g[1] * ds(definedon=mesh.Boundaries("top")), mesh)
    bot_after = Integrate(g[1] * ds(definedon=mesh.Boundaries("bottom")),
                          mesh)
    print(f"second_set_zeroed_the_first_boundary={top_after == 0.0}")
    print("second_set_kept_its_own_boundary="
          f"{abs(bot_after + 0.03) < 1e-12}")
    if not (abs(top_before - 0.03) < 1e-12 and top_after == 0.0):
        print(f"FAIL: Set(definedon=...) no longer zeroes the other DOFs "
              f"(top {top_before:.6g} -> {top_after:.6g})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
