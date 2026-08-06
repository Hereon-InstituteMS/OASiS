"""Tier-2: grad(.).Trace() on a surface mesh is ALREADY the tangential
gradient - projecting it by hand is a no-op that costs ~2x.

Claim: ngsolve surface_pde#1 - grad on a surface mesh automatically produces
the TANGENTIAL (surface) gradient.  Manually projecting with (I - n n^T)
applies the projection twice and yields a numerically identical answer at
about double the assembly cost.  The .Trace() wrapper is required for
trial/test functions.

Wrong variant: assemble InnerProduct(P*grad(u).Trace(), P*grad(v).Trace())*ds
  with P = Id(3) - OuterProduct(n, n), and compare it against the plain
  grad(u).Trace()*grad(v).Trace()*ds form - matrix, solution, and cost.

Observed on NGSolve 6.2.2604, OCC sphere maxh=0.4, Curve(4), H1 order 3 on
mesh.Boundaries('.*'), gfu.Set(x*y+z) (2026-08-06):
  Integrate((grad(gfu).Trace()*n)**2*ds) = 1.54e-31   (|grad_S u|^2 = 13.41)
  Integrate(|P grad_S u - grad_S u|^2 ds) = 1.98e-31
  matrix difference / matrix norm         = 1.01e-16
  solution relative difference            = 4.98e-08
  assemble, min of 9: plain 1.24 ms, projected 2.40 ms -> ratio 1.94
  omitting .Trace() on a trial function raises NgException
    'Trialfunction does not support BND-forms, maybe a Trace() operator is
     missing, type = grad'
Refinement to the claim: .Trace() is required for PROXY (trial/test)
functions only.  On a GridFunction, Integrate(grad(gfu)*grad(gfu)*ds)
returns the same 13.405234462975978 as the .Trace() form, no exception.
"""
from __future__ import annotations

import math
import sys
import time

from netgen.occ import OCCGeometry, Pnt, Sphere
from ngsolve import (BilinearForm, GridFunction, H1, Id, Integrate,
                     InnerProduct, LinearForm, Mesh, OuterProduct, ds, grad,
                     specialcf, x, y, z)


def main() -> int:
    ok = True

    geo = OCCGeometry(Sphere(Pnt(0, 0, 0), r=1.0))
    mesh = Mesh(geo.GenerateMesh(maxh=0.4))
    mesh.Curve(4)
    surf = mesh.Boundaries(".*")
    fes = H1(mesh, order=3, definedon=surf)
    u, v = fes.TnT()
    nvec = specialcf.normal(3)
    proj = Id(3) - OuterProduct(nvec, nvec)
    print(f"surface_ndof={fes.ndof} normal_cf={type(nvec).__name__}")

    # ---- the surface gradient is already tangential ---------------------
    gfu = GridFunction(fes)
    gfu.Set(x * y + z, definedon=surf)
    gt = grad(gfu).Trace()
    normal_sq = Integrate((gt * nvec) ** 2 * ds, mesh)
    tang_sq = Integrate(gt * gt * ds, mesh)
    print(f"normal_component_ratio_lt_1em25={normal_sq / tang_sq < 1e-25}")
    if not normal_sq / tang_sq < 1e-25:
        print(f"FAIL: grad(gfu).Trace() has a normal component, ratio="
              f"{normal_sq / tang_sq:.3e}", file=sys.stderr)
        ok = False

    # ---- WRONG: project it again by hand --------------------------------
    reproj_sq = Integrate(InnerProduct(proj * gt - gt, proj * gt - gt) * ds,
                          mesh)
    print(f"manual_projection_is_a_noop={reproj_sq / tang_sq < 1e-25}")
    if not reproj_sq / tang_sq < 1e-25:
        print(f"FAIL: (I - n n^T) changed the surface gradient, relative "
              f"change {reproj_sq / tang_sq:.3e}", file=sys.stderr)
        ok = False

    t_plain, t_proj = [], []
    a_plain = a_proj = None
    for _ in range(9):
        a_plain = BilinearForm(fes)
        a_plain += grad(u).Trace() * grad(v).Trace() * ds
        t0 = time.perf_counter()
        a_plain.Assemble()
        t_plain.append(time.perf_counter() - t0)

        a_proj = BilinearForm(fes)
        a_proj += InnerProduct(proj * grad(u).Trace(),
                               proj * grad(v).Trace()) * ds
        t0 = time.perf_counter()
        a_proj.Assemble()
        t_proj.append(time.perf_counter() - t0)
    ratio = min(t_proj) / min(t_plain)
    print(f"same_sparsity={a_proj.mat.nze == a_plain.mat.nze}")
    print(f"proj_assemble_time_ratio_ge_1p25={ratio >= 1.25}")
    if not ratio >= 1.25:
        print(f"FAIL: the redundant projection cost nothing extra "
              f"(ratio {ratio:.2f}); the 2x-cost pitfall is absent",
              file=sys.stderr)
        ok = False

    diff = a_plain.mat.CreateMatrix()
    diff.AsVector().data = a_plain.mat.AsVector() - a_proj.mat.AsVector()
    rel_mat = diff.AsVector().Norm() / a_plain.mat.AsVector().Norm()
    print(f"proj_matrix_identical_to_plain={rel_mat < 1e-12}")
    if not rel_mat < 1e-12:
        print(f"FAIL: the projected form is not the plain form, relative "
              f"matrix difference {rel_mat:.3e}", file=sys.stderr)
        ok = False

    # ---- both forms solve the same Laplace-Beltrami problem -------------
    rhs = LinearForm((6 * z * z - 2) * v.Trace() * ds)
    rhs.Assemble()
    reg = BilinearForm(fes)
    reg += 1e-8 * u.Trace() * v.Trace() * ds
    reg.Assemble()
    sols = []
    for form in (a_plain, a_proj):
        mat = form.mat.CreateMatrix()
        mat.AsVector().data = form.mat.AsVector() + reg.mat.AsVector()
        gf = GridFunction(fes)
        gf.vec.data = mat.Inverse(fes.FreeDofs(), inverse="umfpack") * rhs.vec
        sols.append(gf)
    num = math.sqrt(Integrate((sols[0] - sols[1]) ** 2 * ds, mesh))
    den = math.sqrt(Integrate(sols[0] ** 2 * ds, mesh))
    print(f"proj_solution_identical_to_plain={num / den < 1e-5}")
    if not num / den < 1e-5:
        print(f"FAIL: the projected form solved a different problem, "
              f"relative difference {num / den:.3e}", file=sys.stderr)
        ok = False

    # ---- .Trace() is required for PROXIES, not for GridFunctions --------
    msg = ""
    try:
        bad = BilinearForm(fes)
        bad += grad(u) * grad(v) * ds
        bad.Assemble()
    except Exception as exc:                       # noqa: BLE001
        msg = str(exc)
    print(f"proxy_without_trace_raised={bool(msg)}")
    print(f"proxy_without_trace_msg={msg!r}")
    if "does not support BND-forms" not in msg:
        print(f"FAIL: omitting .Trace() on a proxy did not raise the "
              f"documented NgException; got {msg!r}", file=sys.stderr)
        ok = False

    gf_no_trace = Integrate(grad(gfu) * grad(gfu) * ds, mesh)
    print("gridfunction_grad_without_trace_matches_traced="
          f"{abs(gf_no_trace - tang_sq) < 1e-12 * tang_sq}")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
