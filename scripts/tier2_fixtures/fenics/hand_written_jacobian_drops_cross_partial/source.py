"""Tier-2 for fenics reaction_diffusion#5: do NOT supply J to NonlinearProblem
and do NOT reach for ufl.variable / ufl.diff -- dolfinx differentiates the
residual exactly. The real risk is the hand-written Jacobian: drop one
cross-species partial and Newton degrades to a linear iteration.

Two-species 2A <-> B on a 16x16 unit square, P1 x P1, one backward-Euler step,
no-flux boundaries (so no Dirichlet rows blur the matrix comparison).

Observed on dolfinx 0.10.0:
  * dolfinx.fem.petsc.NonlinearProblem builds its Jacobian with
    derivative_block(F, u), whose docstring says "This is identical to calling
    `ufl.derivative` directly", and the two matrices assembled at the same
    state differ by exactly 0.000e+00.
  * ufl.diff(k1*a*a*b - k2*b, a) on a species pulled out with ufl.split raises
    ValueError: Expecting a Variable or SpatialCoordinate in diff. Wrapping the
    species in ufl.variable makes it compile, but the result is a
    VariableDerivative -- a scalar expression, not a bilinear form -- so it
    still has to be assembled by hand.
  * A hand Jacobian that drops the -k2*dB cross partial still solves, but takes
    more Newton iterations with a linear residual ratio instead of the 3
    quadratic ones of the automatic Jacobian.

Mutation control: T2_MUTATE=1 passes the COMPLETE hand Jacobian and skips the
ufl.diff attempt.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import basix.ufl  # noqa: E402
import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

N, DT, D, K1, K2 = 16, 0.05, 0.01, 1.0, 1.0


def build():
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P1, P1]))
    w = dolfinx.fem.Function(W)
    w_n = dolfinx.fem.Function(W)
    w_n.sub(0).interpolate(lambda x: 1.0 + 0.5 * np.sin(2 * np.pi * x[0]))
    w_n.sub(1).interpolate(lambda x: np.full_like(x[0], 0.2))
    w_n.x.scatter_forward()
    w.x.array[:] = w_n.x.array
    A, B = ufl.split(w)
    An, Bn = ufl.split(w_n)
    va, vb = ufl.TestFunctions(W)
    r = K1 * A * A - K2 * B
    F = (((A - An) / DT) * va * ufl.dx
         + D * ufl.dot(ufl.grad(A), ufl.grad(va)) * ufl.dx
         + 2 * r * va * ufl.dx
         + ((B - Bn) / DT) * vb * ufl.dx
         + D * ufl.dot(ufl.grad(B), ufl.grad(vb)) * ufl.dx
         - r * vb * ufl.dx)
    return msh, W, w, F, (A, B, va, vb)


def hand_jacobian(W, parts, full: bool):
    A, B, va, vb = parts
    du = ufl.TrialFunction(W)
    dA, dB = ufl.split(du)
    dr = 2 * K1 * A * dA - K2 * dB if full else 2 * K1 * A * dA
    return ((dA / DT) * va * ufl.dx
            + D * ufl.dot(ufl.grad(dA), ufl.grad(va)) * ufl.dx
            + 2 * dr * va * ufl.dx
            + (dB / DT) * vb * ufl.dx
            + D * ufl.dot(ufl.grad(dB), ufl.grad(vb)) * ufl.dx
            - dr * vb * ufl.dx)


def solve(J, tag):
    msh, W, w, F, parts = build()
    kwargs = {}
    if J is not None:
        kwargs["J"] = hand_jacobian(W, parts, full=J)
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, w, petsc_options_prefix=f"t2_rd5_{tag}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "snes_rtol": 1.0e-12, "snes_atol": 1.0e-14},
        **kwargs)
    prob.solver.setConvergenceHistory()
    prob.solve()
    w.x.scatter_forward()
    hist = np.array(prob.solver.getConvergenceHistory()[0], dtype=float)
    return (prob.solver.getIterationNumber(), prob.solver.getConvergedReason(),
            hist, w.x.array.copy())


def main() -> int:
    # 1) the automatic Jacobian IS ufl.derivative
    msh, W, w, F, _ = build()
    Ja = dolfinx.fem.petsc.assemble_matrix(
        dolfinx.fem.form(dolfinx.fem.petsc.derivative_block(F, w)))
    Ja.assemble()
    Ju = dolfinx.fem.petsc.assemble_matrix(
        dolfinx.fem.form(ufl.derivative(F, w)))
    Ju.assemble()
    da = Ja.copy().convert("dense").getDenseArray()
    du_ = Ju.copy().convert("dense").getDenseArray()
    diff = float(np.abs(da - du_).max())
    scale = float(np.abs(da).max())
    doc = dolfinx.fem.petsc.derivative_block.__doc__ or ""
    print("derivative_block docstring says: "
          + " ".join(doc.split())[-120:])
    print(f"auto_jacobian_inf_norm={scale:.3e} "
          f"auto_minus_ufl_derivative_inf_norm={diff:.3e}")
    exact = diff == 0.0 and scale > 0.0
    print(f"auto_jacobian_equals_ufl_derivative_exactly={exact}")

    # 2) ufl.diff on a species taken out of the mixed Function
    if not MUTATE:
        A, B = ufl.split(w)
        try:
            ufl.diff(K1 * A * A * B - K2 * B, A)
            diff_raised = False
        except ValueError as exc:
            diff_raised = True
            print(f"ufl_diff_error: {type(exc).__name__}: {exc}")
        Av = ufl.variable(A)
        expr = ufl.diff(K1 * Av * Av * B - K2 * B, Av)
        print(f"ufl_variable_diff_type={type(expr).__name__}")
        print(f"ufl_diff_on_a_split_species_raises={diff_raised}")
        print(f"ufl_variable_diff_is_not_a_bilinear_form="
              f"{not isinstance(expr, ufl.Form)}")
    else:
        diff_raised = True

    # 3) hand Jacobian with and without the cross-species partial
    its_a, rea_a, hist_a, u_a = solve(None, "auto")
    its_h, rea_h, hist_h, u_h = solve(MUTATE, "hand")
    print(f"auto_jacobian: iterations={its_a} reason={rea_a} history="
          + " ".join(f"{r:.3e}" for r in hist_a))
    print(f"hand_jacobian: iterations={its_h} reason={rea_h} history="
          + " ".join(f"{r:.3e}" for r in hist_h))
    same = bool(np.max(np.abs(u_a - u_h)) < 1e-9)
    more = its_h > its_a
    ratios = hist_h[1:] / hist_h[:-1] if len(hist_h) > 2 else np.array([1.0])
    linear = bool(len(hist_h) > 4 and ratios[1:].max() > 0.01)
    print(f"hand_jacobian_reaches_the_same_solution={same}")
    print(f"dropping_a_cross_partial_costs_more_iterations={more}")
    print(f"dropping_a_cross_partial_makes_the_rate_linear={linear}")

    if exact and diff_raised and more and linear and same:
        print("VERDICT=let_dolfinx_differentiate_the_residual")
        return 0
    print("VERDICT=hand_jacobian_was_as_good")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
