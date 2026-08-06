"""Tier-2: skipping the residual step silently discards the Dirichlet data.

Claim: NGSolve heat#3 -- the two-step pattern is Set the boundary DOFs, then
solve the RESIDUAL system on FreeDofs. Replacing the second step with a plain
assignment overwrites the boundary values, silently.

The exact solution u = x^2 - y^2 is harmonic and lies in H1(order=2), so the
correct answer is exact to round-off and any O(1) error is unambiguous.

Wrong variant: gfu.vec.data = Inverse * f.vec.
Right variant: gfu.vec.data += Inverse * (f.vec - a.mat * gfu.vec).
"""
from __future__ import annotations

import sys

from netgen.geom2d import unit_square
from ngsolve import (
    BilinearForm,
    GridFunction,
    H1,
    Integrate,
    LinearForm,
    Mesh,
    dx,
    grad,
    x,
    y,
)


def main() -> int:
    ok = True
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    fes = H1(mesh, order=2, dirichlet=".*")
    print(f"space_type={fes.type}")
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += grad(u) * grad(v) * dx
    a.Assemble()
    f = LinearForm(fes)
    f += 0 * v * dx
    f.Assemble()
    exact = x * x - y * y

    errors = {}
    boundary_max = {}
    raised = {}
    for label in ("two_step", "one_step"):
        gfu = GridFunction(fes)
        gfu.Set(exact, definedon=mesh.Boundaries(".*"))
        set_max = max(abs(val) for val in gfu.vec)
        try:
            if label == "two_step":
                gfu.vec.data += a.mat.Inverse(fes.FreeDofs()) * (
                    f.vec - a.mat * gfu.vec)
            else:
                gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec
            raised[label] = ""
        except Exception as exc:                  # noqa: BLE001
            raised[label] = f"{type(exc).__name__}: {exc}"
        errors[label] = float(Integrate((gfu - exact) ** 2, mesh)) ** 0.5
        boundary_max[label] = max(abs(val) for val in gfu.vec)
        print(f"{label}_l2_error={errors[label]:.3e}")
        print(f"{label}_after_set_max={set_max:.6f}")
        print(f"{label}_final_max={boundary_max[label]:.6e}")

    print(f"two_step_error_at_machine_zero={errors['two_step'] < 1e-12}")
    print(f"one_step_error_order_one={errors['one_step'] > 1e-2}")
    print(f"one_step_raised_nothing={not raised['one_step']}")
    print(f"one_step_boundary_values_lost="
          f"{boundary_max['one_step'] < 1e-12}")
    ratio = errors["one_step"] / max(errors["two_step"], 1e-300)
    print(f"error_ratio_over_1e10={ratio > 1e10}")

    if errors["two_step"] >= 1e-12:
        print(f"FAIL: the two-step pattern is not exact: "
              f"{errors['two_step']!r}", file=sys.stderr)
        ok = False
    if errors["one_step"] <= 1e-2 or raised["one_step"]:
        print(f"FAIL: the one-step assignment did not fail silently "
              f"(error {errors['one_step']!r}, raised {raised['one_step']!r})",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
