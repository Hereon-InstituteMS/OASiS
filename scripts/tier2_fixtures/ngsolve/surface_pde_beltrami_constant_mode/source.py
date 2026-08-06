"""Tier-2: Laplace-Beltrami has kernel = constants and NEITHER failure mode
is loud.

Claim: ngsolve surface_pde#2 — on the sphere the pure surface-stiffness
matrix is singular (constants are in the kernel).  Dropping the regulariser
does NOT raise: the direct solver returns a solution with |u|_inf ~ 1e9.
Adding the shipped 1e-8*u*v*ds regulariser makes the solve stable and gets
the SHAPE right, but leaves an arbitrary constant offset, which is why the
shipped surface_pde_3d template prints max=3387.7 for a field whose true
range is [-1/3, 2/3].

Wrong variant A: assemble grad_S(u)*grad_S(v)*ds with no regularisation and
  invert on fes.FreeDofs() with umfpack, then with sparsecholesky.  Neither
  raises, neither prints anything, both return garbage.
Wrong variant B: solve the regularised system and report min/max of the raw
  coefficient vector, exactly as the shipped template does.
Right variant : subtract the surface mean (or pin one DOF) before reporting
  anything -> the field matches the analytic Y_2^0 = z^2 - 1/3.

Observed on NGSolve 6.2.2604 / netgen OCC sphere maxh=0.4, Curve(4),
H1 order 3 restricted to mesh.Boundaries(".*"), f = 6 z^2 - 2 (2026-08-06):
  umfpack, no regulariser        : no exception, |u|_inf = 6.14e+09
  sparsecholesky, no regulariser : no exception, |u|_inf = 2.69e+09
  1e-8 regulariser               : max(gfu.vec) = 171.49, mean = 170.82,
                                   L2 error vs z^2-1/3 = 6.1e+02
  after mean removal             : L2 error = 2.6e-04
  pin one DOF + mean removal     : L2 error = 2.6e-04
Neither solver emitted a single character on stdout/stderr.  `KSPSolve:
DIVERGED_BREAKDOWN` and `DIVERGED_INDEFINITE_PC` are PETSc strings; NGSolve
never produced them here.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

from netgen.occ import OCCGeometry, Pnt, Sphere
from ngsolve import (BilinearForm, BitArray, CoefficientFunction, GridFunction,
                     H1, Integrate, LinearForm, Mesh, ds, grad, z)
import ngsolve


def capture_fds(fn):
    """Run fn() with OS-level fds 1 and 2 redirected; return (value, text).

    Needed because NGSolve/UMFPACK chatter is written by C++ to fd 1/2 and
    would not be seen by contextlib.redirect_stdout.
    """
    with tempfile.TemporaryFile(mode="w+") as tmp:
        so, se = os.dup(1), os.dup(2)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(tmp.fileno(), 1)
            os.dup2(tmp.fileno(), 2)
            value = fn()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(so, 1)
            os.dup2(se, 2)
            os.close(so)
            os.close(se)
        tmp.seek(0)
        return value, tmp.read()


def main() -> int:
    ok = True
    print(f"ngsolve_version={ngsolve.__version__}")

    geo = OCCGeometry(Sphere(Pnt(0, 0, 0), r=1.0))
    mesh = Mesh(geo.GenerateMesh(maxh=0.4))
    mesh.Curve(4)
    surf = mesh.Boundaries(".*")
    fes = H1(mesh, order=3, definedon=surf)
    u, v = fes.TnT()
    print(f"mesh_dim={mesh.dim} surface_ndof={fes.ndof}")

    rhs = LinearForm((6 * z * z - 2) * v.Trace() * ds)
    rhs.Assemble()

    a_sing = BilinearForm(fes)
    a_sing += grad(u).Trace() * grad(v).Trace() * ds
    a_sing.Assemble()

    a_reg = BilinearForm(fes)
    a_reg += (grad(u).Trace() * grad(v).Trace()
              + 1e-8 * u.Trace() * v.Trace()) * ds
    a_reg.Assemble()

    # ---- WRONG A: no regularisation at all -------------------------------
    def solve_singular(kind):
        inv = a_sing.mat.Inverse(fes.FreeDofs(), inverse=kind)
        gf = GridFunction(fes)
        gf.vec.data = inv * rhs.vec
        return type(inv).__name__, max(abs(float(q)) for q in gf.vec)

    chatter = ""
    linf = {}
    inv_class = {}
    for kind in ("umfpack", "sparsecholesky"):
        raised = ""
        try:
            (cls, li), txt = capture_fds(lambda: solve_singular(kind))
        except Exception as exc:            # noqa: BLE001 - want the text
            raised, cls, li, txt = f"{type(exc).__name__}: {exc}", "", 0.0, ""
        chatter += txt
        linf[kind] = li
        inv_class[kind] = cls
        print(f"noreg_{kind}_raised={bool(raised)}")
        if raised:
            print(f"noreg_{kind}_exception={raised!r}")
            print(f"FAIL: {kind} raised on the singular Laplace-Beltrami "
                  f"matrix; the claim says it returns silently",
                  file=sys.stderr)
            ok = False
        else:
            print(f"noreg_{kind}_linf_gt_1e6={li > 1e6}")
            if li <= 1e6:
                print(f"FAIL: {kind} without a regulariser gave |u|_inf="
                      f"{li:.3e}, no constant-mode blow-up", file=sys.stderr)
                ok = False

    print("noreg_sparsecholesky_inverse_class="
          f"{inv_class['sparsecholesky']}")
    print(f"noreg_solver_printed_nothing={chatter.strip() == ''}")
    print("petsc_string_KSPSolve_DIVERGED_BREAKDOWN_emitted="
          f"{'DIVERGED_BREAKDOWN' in chatter}")
    print("petsc_string_DIVERGED_INDEFINITE_PC_emitted="
          f"{'DIVERGED_INDEFINITE_PC' in chatter}")
    if "DIVERGED" in chatter:
        print("FAIL: NGSolve emitted a PETSc-style DIVERGED string",
              file=sys.stderr)
        ok = False

    # ---- WRONG B: regularised solve, reported the way the template does --
    gf_reg = GridFunction(fes)
    gf_reg.vec.data = a_reg.mat.Inverse(fes.FreeDofs(),
                                        inverse="umfpack") * rhs.vec
    area = Integrate(CoefficientFunction(1) * ds, mesh)
    u_ex = z * z - 1.0 / 3.0
    mean_reg = Integrate(gf_reg * ds, mesh) / area
    mean_ex = Integrate(u_ex * ds, mesh) / area

    raw_max = max(gf_reg.vec)
    raw_min = min(gf_reg.vec)
    print(f"reg_rawvec_max_gt_100={raw_max > 100.0}")
    print(f"reg_rawvec_max_above_true_max_by_gt_100={raw_max - 2/3 > 100.0}")
    print(f"reg_rawvec_min_below_true_min={raw_min < -1/3}")
    if not (raw_max > 100.0):
        print(f"FAIL: raw coefficient max is {raw_max:.4f}; the "
              f"constant-mode offset the shipped template reports is gone",
              file=sys.stderr)
        ok = False

    l2_raw = math.sqrt(Integrate((gf_reg - u_ex) ** 2 * ds, mesh))
    print(f"reg_l2err_without_mean_removal_gt_1={l2_raw > 1.0}")
    if l2_raw <= 1.0:
        print(f"FAIL: L2 error without mean removal is {l2_raw:.3e}; the "
              f"constant offset did not appear", file=sys.stderr)
        ok = False

    # ---- RIGHT: remove the mean ------------------------------------------
    l2_shift = math.sqrt(
        Integrate((gf_reg - mean_reg - (u_ex - mean_ex)) ** 2 * ds, mesh))
    print(f"meanremoved_l2err_lt_1em3={l2_shift < 1e-3}")
    print(f"meanremoved_beats_raw_by_gt_1e4="
          f"{l2_raw / max(l2_shift, 1e-300) > 1e4}")
    if not l2_shift < 1e-3:
        print(f"FAIL: after mean removal the L2 error is still "
              f"{l2_shift:.3e}", file=sys.stderr)
        ok = False

    # nodal range of the mean-shifted field, against the P1 interpolant of
    # the analytic solution on the same mesh (min/max of a hierarchical
    # order-3 coefficient vector are NOT field values, which is the other
    # half of the shipped template's mistake).
    p1 = H1(mesh, order=1, definedon=surf)
    g_num = GridFunction(p1)
    g_num.Set(gf_reg - mean_reg, definedon=surf)
    g_ex = GridFunction(p1)
    g_ex.Set(u_ex, definedon=surf)
    dmax = abs(max(g_num.vec) - max(g_ex.vec))
    dmin = abs(min(g_num.vec) - min(g_ex.vec))
    print(f"meanremoved_p1_range_matches_analytic={max(dmax, dmin) < 1e-2}")
    print(f"meanremoved_p1_max_below_1={max(g_num.vec) < 1.0}")
    if not (max(dmax, dmin) < 1e-2 and max(g_num.vec) < 1.0):
        print(f"FAIL: mean-removed nodal range off by "
              f"{max(dmax, dmin):.3e}", file=sys.stderr)
        ok = False

    # ---- RIGHT (alternative): pin one free DOF ---------------------------
    pin = BitArray(fes.FreeDofs())
    first_free = next(i for i in range(len(pin)) if pin[i])
    pin[first_free] = False
    gf_pin = GridFunction(fes)
    gf_pin.vec.data = a_sing.mat.Inverse(pin, inverse="umfpack") * rhs.vec
    mean_pin = Integrate(gf_pin * ds, mesh) / area
    l2_pin = math.sqrt(
        Integrate((gf_pin - mean_pin - (u_ex - mean_ex)) ** 2 * ds, mesh))
    print(f"pinned_dof_index={first_free}")
    print(f"pinned_plus_meanremoved_l2err_lt_1em3={l2_pin < 1e-3}")
    if not l2_pin < 1e-3:
        print(f"FAIL: pinning one DOF then removing the mean left L2 error "
              f"{l2_pin:.3e}", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
