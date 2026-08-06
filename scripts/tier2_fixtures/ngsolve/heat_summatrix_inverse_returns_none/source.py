"""Tier-2: m.mat + dt*a.mat gives a SumMatrix whose Inverse returns None.

Claim: NGSolve heat#1 -- the implicit-Euler operator must be built as
mstar = m.mat.CreateMatrix(); mstar.AsVector().data = m.mat.AsVector() +
dt*a.mat.AsVector(). The shortcut m.mat + dt*a.mat is a SILENT trap: it yields a
SumMatrix which DOES have .Inverse, so a hasattr guard passes, but the call
prints 'BaseMatrix::InverseMatrix not available' and returns None.

Wrong variant: the shortcut. Right variant: CreateMatrix + AsVector().data.
The fixture follows the failure past where the claim stops. The claim says the
next "* f.vec" dies on NoneType; measured, it does not -- NGSolve returns a lazy
DynamicVectorExpression and the error only appears at the ASSIGNMENT, as
NgException('DynamicBaseScalVec AssignTo not available'). That is one more line
of distance between the mistake and its symptom than the claim states.
"""
from __future__ import annotations

import sys

from netgen.geom2d import unit_square
from ngsolve import BilinearForm, GridFunction, H1, LinearForm, Mesh, dx, grad

DT = 0.01


def main() -> int:
    ok = True
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    fes = H1(mesh, order=1, dirichlet="left|right|top|bottom")
    n_free = sum(1 for b in fes.FreeDofs() if b)
    print(f"n_free_dofs={n_free}")
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += grad(u) * grad(v) * dx
    a.Assemble()
    m = BilinearForm(fes)
    m += u * v * dx
    m.Assemble()
    f = LinearForm(fes)
    f += 1 * v * dx
    f.Assemble()

    # --- WRONG variant: the operator-sum shortcut ------------------------
    shortcut = m.mat + DT * a.mat
    print(f"sum_type_name={type(shortcut).__name__}")
    print(f"sum_hasattr_inverse={hasattr(shortcut, 'Inverse')}")
    raised = ""
    inv = "unset"
    try:
        inv = shortcut.Inverse(fes.FreeDofs())
    except Exception as exc:                      # noqa: BLE001
        raised = f"{type(exc).__name__}: {exc}"
    print(f"sum_inverse_did_not_raise={not raised}")
    print(f"sum_inverse_returns_none={inv is None}")
    if raised or inv is not None:
        print(f"FAIL: SumMatrix.Inverse behaved differently -- raised "
              f"{raised!r}, returned {inv!r}", file=sys.stderr)
        ok = False

    # Where the user actually sees it. The claim says the next "* f.vec" dies
    # on NoneType. Measured, it does NOT: NGSolve's vector accepts the None on
    # the left and returns a lazy DynamicVectorExpression. The failure surfaces
    # one step further on, at the ASSIGNMENT, as
    # NgException('DynamicBaseScalVec AssignTo not available').
    product = None
    mul_raised = ""
    try:
        product = inv * f.vec
    except Exception as exc:                      # noqa: BLE001
        mul_raised = f"{type(exc).__name__}: {exc}"
    print(f"multiply_raised_nothing={not mul_raised}")
    print(f"product_type={type(product).__name__}")
    assign_msg = ""
    try:
        GridFunction(fes).vec.data = product
    except Exception as exc:                      # noqa: BLE001
        assign_msg = f"{type(exc).__name__}: {exc}"
    print(f"assignment_raises={bool(assign_msg)}")
    print(f"assignment_msg={assign_msg[:110]!r}")
    if mul_raised:
        print(f"FAIL: the multiply raised {mul_raised!r}; the claim's "
              f"'dies on NoneType' would then be right and this correction "
              f"stale", file=sys.stderr)
        ok = False
    if "AssignTo not available" not in assign_msg:
        print(f"FAIL: the assignment did not fail as measured; got "
              f"{assign_msg!r}", file=sys.stderr)
        ok = False

    # --- RIGHT variant ----------------------------------------------------
    mstar = m.mat.CreateMatrix()
    mstar.AsVector().data = m.mat.AsVector() + DT * a.mat.AsVector()
    print(f"correct_type_name={type(mstar).__name__}")
    good_inv = mstar.Inverse(fes.FreeDofs())
    gfu = GridFunction(fes)
    gfu.vec.data = good_inv * f.vec
    usable = good_inv is not None and max(abs(val) for val in gfu.vec) > 0.0
    print(f"correct_inverse_is_usable={usable}")
    if not usable:
        print("FAIL: the CreateMatrix form did not produce a usable inverse",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
