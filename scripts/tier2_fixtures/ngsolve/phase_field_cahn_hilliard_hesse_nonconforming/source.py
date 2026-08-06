"""Tier-2: a single-H1 fourth-order form built from Operator('hesse')
assembles cleanly and is rank-deficient; the mixed H1 x H1 formulation
converges.

Claim: ngsolve phase_field#7 - for Cahn-Hilliard use a mixed H1 x H1
formulation.  NGSolve will not stop you doing it the wrong way: the
single-H1 Hessian form ASSEMBLES with no exception at all, and neither
`NotImplementedError: H2 conformity required for biharmonic operator` nor
`coefficient not in BilinearForm space` exists in NGSolve 6.2.2604.

Wrong variant: BilinearForm += InnerProduct(u.Operator('hesse'),
  v.Operator('hesse'))*dx on H1(mesh, order=2).
Right variant : mixed (c, mu) on H1 x H1 - here the Ciarlet-Raviart pair
  mu = -Lap c, -Lap mu = f, against the simply-supported biharmonic MMS
  u = sin(pi x) sin(pi y), f = 4 pi^4 u (u = 0 and Lap u = 0 on the
  boundary, so the pair is consistent).

Observed on NGSolve 6.2.2604, unit_square (2026-08-06):
  maxh=0.2, order 2, no dirichlet: ne=52, ndof=125, the hesse form assembles
    silently with nnz=1277.  (The catalog's nnz=1323 does not reproduce on
    this netgen build.)
  u.Operator('biharmonic') raises
    NgException : Operator "biharmonic" does not exist for
    H1HighOrderFESpace!
  The discrete kernel is exhibited directly: a P1 hat function embedded in
    the order-2 space has element-wise zero Hessian, so its hesse energy is
    1.8e-27 while its H1 energy is 3.55 and its L2 norm^2 is 2.24e-02.
  Consequently, with dirichlet='.*' the matrix is singular:
    umfpack        -> NgException: UmfpackInverse: Numeric factorization
                      failed.
    sparsecholesky -> returns WITHOUT raising, solution is NaN
  Mixed H1 x H1, order 2: L2 error 1.741e-03 -> 1.724e-04 -> 2.045e-05
    at maxh 0.2 / 0.1 / 0.05 (EOC 3.34, 3.08).

Mutation control.  T2_MUTATE=1 applies the documented fix where the pathology
bites: the fourth-order system handed to umfpack and to sparsecholesky is built
from the mixed Ciarlet-Raviart pair (mixed_form) instead of the rank-deficient
single-H1 hesse form.  The matrix is then regular, so `hesse_umfpack_raised`
and the literal message `UmfpackInverse: Numeric factorization failed.`
disappear from the output and the fixture goes red.

Measured while doing that, and worth knowing: `hesse_sparsecholesky_solution_is_nan`
SURVIVES the mutation - sparsecholesky returns NaN on the regular mixed system
too, because that system is indefinite and Cholesky does not apply to it.  So
that one line is evidence about using the wrong factoriser, not about the hesse
form's singularity.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import math
import os
import sys

from netgen.geom2d import unit_square
from ngsolve import (BilinearForm, GridFunction, H1, Integrate, InnerProduct,
                     LinearForm, Mesh, dx, grad, sin, x, y)

PI = math.pi
U_EX = sin(PI * x) * sin(PI * y)
F_EX = 4 * PI ** 4 * U_EX

# Mutation control: solve the fourth-order system with the documented fix (the
# mixed Ciarlet-Raviart pair) instead of the rank-deficient single-H1 hesse
# form, removing the singularity this fixture measures.
MUTATE = os.environ.get("T2_MUTATE") == "1"


def hesse_form(mesh, dirichlet=None):
    fes = (H1(mesh, order=2) if dirichlet is None
           else H1(mesh, order=2, dirichlet=dirichlet))
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += InnerProduct(u.Operator("hesse"), v.Operator("hesse")) * dx
    a.Assemble()
    return fes, a


def mixed_form(mesh):
    """The right variant: the Ciarlet-Raviart pair mu = -Lap c, -Lap mu = f."""
    V = H1(mesh, order=2, dirichlet=".*")
    X = V * V
    (c, mu), (q, p) = X.TnT()
    a = BilinearForm(X)
    a += (mu * p - grad(c) * grad(p)) * dx
    a += grad(mu) * grad(q) * dx
    a.Assemble()
    f = LinearForm(X)
    f += F_EX * q * dx
    f.Assemble()
    return X, a, f


def mixed_l2_error(maxh):
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
    X, a, f = mixed_form(mesh)
    gf = GridFunction(X)
    gf.vec.data = a.mat.Inverse(X.FreeDofs()) * f.vec
    err = math.sqrt(Integrate((gf.components[0] - U_EX) ** 2, mesh))
    return err, type(X).__name__, X.ndof


def main() -> int:
    ok = True
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))

    # ---- WRONG: the single-H1 Hessian form assembles in silence ---------
    exc = ""
    try:
        fes, a = hesse_form(mesh)
        nnz = a.mat.nze
    except Exception as e:                         # noqa: BLE001
        exc, nnz = f"{type(e).__name__}: {e}", -1
    print(f"hesse_ne={mesh.ne} hesse_ndof={fes.ndof if not exc else -1} "
          f"hesse_nnz={nnz}")
    print(f"hesse_form_assembled_without_exception={not exc}")
    print(f"hesse_form_nnz_gt_0={nnz > 0}")
    print(f"hesse_assembly_exception={exc!r}")
    if exc or nnz <= 0:
        print(f"FAIL: the single-H1 hesse form did not assemble silently "
              f"({exc!r}, nnz={nnz})", file=sys.stderr)
        ok = False

    # the two catalog strings, and what NGSolve actually says
    u, v = fes.TnT()
    bih = ""
    try:
        u.Operator("biharmonic")
    except Exception as e:                         # noqa: BLE001
        bih = str(e)
    print(f"biharmonic_operator_raised={bool(bih)}")
    print(f"biharmonic_msg={bih!r}")
    seen = exc + bih
    print("fabricated_h2_conformity_string_emitted="
          f"{'H2 conformity required' in seen}")
    print("fabricated_coefficient_not_in_space_string_emitted="
          f"{'coefficient not in BilinearForm space' in seen}")
    if "does not exist for H1HighOrderFESpace" not in bih:
        print(f"FAIL: Operator('biharmonic') did not raise the documented "
              f"NgException; got {bih!r}", file=sys.stderr)
        ok = False
    if "H2 conformity required" in seen:
        print("FAIL: the fabricated H2-conformity string appeared",
              file=sys.stderr)
        ok = False

    # ---- the discrete kernel, exhibited ---------------------------------
    v1 = H1(mesh, order=1)
    v1d = H1(mesh, order=1, dirichlet=".*")
    free1 = v1d.FreeDofs()
    hat = GridFunction(v1)
    hat.vec[:] = 0
    hat.vec[next(i for i in range(v1d.ndof) if free1[i])] = 1.0
    emb = GridFunction(fes)
    emb.Set(hat)
    hes = emb.Operator("hesse")
    e_hesse = Integrate(InnerProduct(hes, hes), mesh)
    e_h1 = Integrate(grad(emb) * grad(emb), mesh)
    e_l2 = Integrate(emb * emb, mesh)
    print(f"p1_hat_is_not_the_zero_function={e_l2 > 1e-4}")
    print(f"p1_hat_has_positive_h1_energy={e_h1 > 1.0}")
    print(f"p1_hat_has_zero_hesse_energy={e_hesse / e_h1 < 1e-20}")
    if not (e_l2 > 1e-4 and e_h1 > 1.0 and e_hesse / e_h1 < 1e-20):
        print(f"FAIL: no discrete kernel found - hesse energy {e_hesse:.3e}, "
              f"H1 energy {e_h1:.4f}, L2^2 {e_l2:.3e}", file=sys.stderr)
        ok = False

    # ---- and what happens when you try to solve with it -----------------
    if MUTATE:
        # the documented fix, applied to the system that is actually solved
        fes_d, a_d, rhs = mixed_form(mesh)
    else:
        fes_d, a_d = hesse_form(mesh, dirichlet=".*")
        _, vd = fes_d.TnT()
        rhs = LinearForm(F_EX * vd * dx)
        rhs.Assemble()
    results = {}
    for solver in ("umfpack", "sparsecholesky"):
        try:
            gf = GridFunction(fes_d)
            gf.vec.data = a_d.mat.Inverse(fes_d.FreeDofs(),
                                          inverse=solver) * rhs.vec
            bad = any(math.isnan(float(q)) for q in gf.vec)
            results[solver] = ("", bad)
        except Exception as e:                     # noqa: BLE001
            results[solver] = (f"{type(e).__name__}: {e}", False)
    print(f"hesse_umfpack_raised={bool(results['umfpack'][0])}")
    print(f"hesse_umfpack_msg={results['umfpack'][0]!r}")
    print(f"hesse_sparsecholesky_raised={bool(results['sparsecholesky'][0])}")
    print("hesse_sparsecholesky_solution_is_nan="
          f"{results['sparsecholesky'][1]}")
    if "Numeric factorization failed" not in results["umfpack"][0]:
        print(f"FAIL: inverting the C0 hesse matrix with umfpack did not "
              f"fail as recorded: {results['umfpack']}", file=sys.stderr)
        ok = False
    if results["sparsecholesky"][0] or not results["sparsecholesky"][1]:
        print(f"FAIL: sparsecholesky did not silently return NaN: "
              f"{results['sparsecholesky']}", file=sys.stderr)
        ok = False

    # ---- RIGHT: mixed H1 x H1 -------------------------------------------
    errs = []
    cls = ""
    for maxh in (0.2, 0.1, 0.05):
        e, cls, nd = mixed_l2_error(maxh)
        errs.append(e)
        print(f"mixed_maxh={maxh} ndof={nd} l2err={e:.6e}")
    eocs = [math.log(errs[i] / errs[i + 1]) / math.log(2.0)
            for i in range(len(errs) - 1)]
    print(f"ch_space_class={cls}")
    print(f"mixed_l2err_decreases={errs[0] > errs[1] > errs[2]}")
    print(f"mixed_eoc_ge_2={all(r >= 2.0 for r in eocs)}")
    print(f"mixed_final_l2err_lt_1em4={errs[-1] < 1e-4}")
    if not (errs[0] > errs[1] > errs[2] and all(r >= 2.0 for r in eocs)
            and errs[-1] < 1e-4):
        print(f"FAIL: the mixed formulation did not converge "
              f"(errors {errs}, EOC {eocs})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
