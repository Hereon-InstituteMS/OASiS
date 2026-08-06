"""Tier-2: without a tension/compression split a compressive load nucleates
exactly as much damage as tension - and the shipped vol-dev psi_plus does
NOT get max(d) below 1e-3, only a spectral split does.

Claim: ngsolve phase_field#5 - the Miehe energy split prevents crack growth
under compression.  Without it a compressive boundary load nucleates
spurious damage; with it compression yields d ~ 0, and a uniaxial-compression
sanity test should show max(d) < 1e-3.

Loading used here is CONFINED uniaxial: VectorH1 with dirichletx='left|right'
and dirichlety='bottom|top', u_y = +-0.03 on top, so the strain state is
exactly diag(0, +-0.03) and compression has no transverse tension to leak
through.

Wrong variant A: the shipped elastic step
  `gfu.vec.data = a_u.mat.Inverse(Vu.FreeDofs()) * f_u.vec`, which wipes the
  Dirichlet entries.  Then u == 0 and EVERY energy - split or not - reports
  max(d) = 0 under compression, so the sanity test passes while measuring
  nothing.
Wrong variant B: homogenise the elastic step, then drive damage with the
  full elastic energy (no split).
Right variant  : the spectral (principal-strain) Miehe split.

Observed on NGSolve 6.2.2604, unit_square maxh=0.15 (ne=96, VectorH1 order 1
ndof=126, 94 free; H1 ndof=63), E=1, nu=0.3, Gc=1e-3, l0=0.1, |u_y|=0.03
(2026-08-06), max(d):
                          tension        compression
  no split               1.080617e-01   1.080617e-01   (identical)
  vol-dev (as shipped)   1.304348e-01   4.411765e-02
  spectral               1.080617e-01   3.909618e-32

CORRECTION to the claim: the psi_plus shipped in phase_field_fracture_2d is
a volumetric/deviatoric split, and under confined compression it still
leaves max(d) = 4.4e-02, forty times the quoted 1e-3 bound.  Only the
spectral split reaches d ~ 0.  It does not disturb pure tension: the
spectral and no-split tensile answers agree exactly.
"""
from __future__ import annotations

import sys

from netgen.geom2d import unit_square
from ngsolve import (BilinearForm, CoefficientFunction, GridFunction, Grad,
                     H1, Id, IfPos, InnerProduct, LinearForm, Mesh, Trace,
                     VectorH1, dx, grad, sqrt)

E_MOD, NU = 1.0, 0.3
MU = E_MOD / (2 * (1 + NU))
LAM = E_MOD * NU / ((1 + NU) * (1 - 2 * NU))
GC, L0, KRES = 1e-3, 0.1, 1e-10
DISP = 0.03

mesh = Mesh(unit_square.GenerateMesh(maxh=0.15))
Vu = VectorH1(mesh, order=1, dirichletx="left|right",
              dirichlety="bottom|top")
Vd = H1(mesh, order=1)


def strain(w):
    return 0.5 * (Grad(w) + Grad(w).trans)


def psi_full(w):
    e = strain(w)
    return 0.5 * LAM * Trace(e) ** 2 + MU * InnerProduct(e, e)


def psi_voldev(w):
    """Exactly the psi_plus shipped in phase_field_fracture_2d."""
    e = strain(w)
    tr = Trace(e)
    abstr = IfPos(tr, tr, -tr)
    return (0.5 * LAM * 0.5 * (tr + abstr) ** 2
            + MU * InnerProduct(e, e) - MU / 3 * tr ** 2)


def psi_spectral(w):
    """Miehe spectral split: only positive principal strains drive damage."""
    e = strain(w)
    tr = Trace(e)
    det = e[0, 0] * e[1, 1] - e[0, 1] * e[1, 0]
    q = tr * tr - 4 * det
    disc = sqrt(IfPos(q, q, 0))
    e1, e2 = 0.5 * (tr + disc), 0.5 * (tr - disc)
    p1, p2 = IfPos(e1, e1, 0), IfPos(e2, e2, 0)
    return 0.5 * LAM * (p1 + p2) ** 2 + MU * (p1 * p1 + p2 * p2)


def run(psi, disp, homogenize, maxit=40):
    u, v = Vu.TnT()
    d, phi = Vd.TnT()
    gfu, gfd = GridFunction(Vu), GridFunction(Vd)
    gfd.Set(0)
    for _ in range(maxit):
        gfu.Set(CoefficientFunction((0, disp)),
                definedon=mesh.Boundaries("top"))
        a = BilinearForm(Vu)
        a += ((1 - gfd) ** 2 + KRES) * InnerProduct(
            2 * MU * strain(u) + LAM * Trace(strain(u)) * Id(2),
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
        delta = (gnew.vec - gfd.vec).Norm()
        gfd.vec.data = gnew.vec
        if delta < 1e-11:
            break
    return max(gfd.vec)


def main() -> int:
    ok = True
    nfree = sum(bool(b) for b in Vu.FreeDofs())
    print(f"mesh_ne={mesh.ne} ndof_u={Vu.ndof} nfree_u={nfree} "
          f"ndof_d={Vd.ndof} vector_space_class={type(Vu).__name__}")

    # ---- WRONG A: the shipped elastic step makes the test vacuous -------
    vacuous = {name: (run(psi, -DISP, False), run(psi, DISP, False))
               for name, psi in (("nosplit", psi_full),
                                 ("voldev", psi_voldev),
                                 ("spectral", psi_spectral))}
    all_zero = all(c < 1e-3 and t < 1e-3 for c, t in vacuous.values())
    print(f"shipped_elastic_step_dmax={ {k: f'{c:.3e}/{t:.3e}' for k, (c, t) in vacuous.items()} }")
    print(f"shipped_elastic_step_passes_every_split_vacuously={all_zero}")
    if not all_zero:
        print(f"FAIL: the shipped elastic step produced damage; the vacuous "
              f"sanity check is gone: {vacuous}", file=sys.stderr)
        ok = False

    # ---- WRONG B: homogenised, no split ---------------------------------
    ns_c = run(psi_full, -DISP, True)
    ns_t = run(psi_full, DISP, True)
    print(f"nosplit_compression_dmax={ns_c:.6e} tension_dmax={ns_t:.6e}")
    print(f"nosplit_compression_dmax_gt_1em3={ns_c > 1e-3}")
    print(f"nosplit_compression_equals_tension={abs(ns_c - ns_t) < 1e-12}")
    if not ns_c > 1e-3:
        print(f"FAIL: without a split, compression produced only "
              f"d_max={ns_c:.3e}; the spurious-damage pitfall is absent",
              file=sys.stderr)
        ok = False
    if abs(ns_c - ns_t) >= 1e-12:
        print(f"FAIL: the unsplit energy distinguished tension from "
              f"compression ({ns_t:.6e} vs {ns_c:.6e})", file=sys.stderr)
        ok = False

    # ---- the shipped vol-dev split does not reach the quoted 1e-3 -------
    vd_c = run(psi_voldev, -DISP, True)
    vd_t = run(psi_voldev, DISP, True)
    print(f"voldev_compression_dmax={vd_c:.6e} tension_dmax={vd_t:.6e}")
    print(f"voldev_reduces_compression_vs_nosplit={vd_c < ns_c}")
    print(f"voldev_compression_still_above_1em3={vd_c > 1e-3}")
    if not (vd_c < ns_c and vd_c > 1e-3):
        print(f"FAIL: the shipped vol-dev psi_plus behaved differently than "
              f"recorded (compression d_max={vd_c:.3e})", file=sys.stderr)
        ok = False

    # ---- RIGHT: spectral split ------------------------------------------
    sp_c = run(psi_spectral, -DISP, True)
    sp_t = run(psi_spectral, DISP, True)
    print(f"spectral_compression_dmax={sp_c:.6e} tension_dmax={sp_t:.6e}")
    print(f"spectral_compression_dmax_lt_1em3={sp_c < 1e-3}")
    print(f"spectral_tension_dmax_gt_1em3={sp_t > 1e-3}")
    print(f"spectral_tension_equals_nosplit_tension={abs(sp_t - ns_t) < 1e-12}")
    if not (sp_c < 1e-3 and sp_t > 1e-3):
        print(f"FAIL: the spectral split did not separate tension from "
              f"compression (C {sp_c:.3e}, T {sp_t:.3e})", file=sys.stderr)
        ok = False
    if abs(sp_t - ns_t) >= 1e-12:
        print(f"FAIL: the spectral split changed the pure-tension answer "
              f"({sp_t:.6e} vs {ns_t:.6e})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
