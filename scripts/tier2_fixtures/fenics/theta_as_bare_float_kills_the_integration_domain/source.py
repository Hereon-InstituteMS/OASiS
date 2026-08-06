"""Tier-2 for fenics time_dependent_heat#6: in a theta-scheme, never multiply a
form integrand by a bare Python float that can evaluate to exactly 0.0. UFL folds
the product to Zero and the resulting integral has no domain.

Wrong variant: `(1.0 - theta) * ufl.dot(ufl.grad(T_n), ufl.grad(v)) * ufl.dx`
with theta = 1.0 (backward Euler), which raises
`ValueError: This integral is missing an integration domain.` while the very
same expression is fine for theta = 0.5.
Right variant: wrap the weight as `fem.Constant(msh, 1.0 - theta)`. It builds
for every theta including 1.0, it gives the same answer as an explicitly
hand-written backward-Euler form, and theta can then be changed in place without
rebuilding the form.

Observed: the ValueError is raised at form-construction time (at the `* ufl.dx`,
before `fem.form` is ever called), and the Constant-weighted theta = 1.0 form
matches the hand-written backward-Euler solution to machine precision.

Mutation control: T2_MUTATE=1 builds the theta = 1.0 weight as a fem.Constant,
so nothing is raised.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

N, DT, NSTEP = 16, 0.01, 5


def setup():
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    left = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 0.0))
    dofs = dolfinx.fem.locate_dofs_topological(V, fdim, left)
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 1.0), dofs, V)
    return msh, V, bc


def build_theta_form(theta, as_constant: bool):
    """Return (T_n, a, L) or raise whatever UFL raises."""
    msh, V, bc = setup()
    T_n = dolfinx.fem.Function(V)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dt = dolfinx.fem.Constant(msh, DT)
    w_new = dolfinx.fem.Constant(msh, float(theta)) if as_constant else theta
    w_old = (dolfinx.fem.Constant(msh, float(1.0 - theta)) if as_constant
             else (1.0 - theta))
    a = (u / dt) * v * ufl.dx + w_new * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ((T_n / dt) * v * ufl.dx
         - w_old * ufl.dot(ufl.grad(T_n), ufl.grad(v)) * ufl.dx)
    return msh, V, bc, T_n, a, L


def march(msh, V, bc, T_n, a, L):
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], u=dolfinx.fem.Function(V),
        petsc_options_prefix="t2_tdh6_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    for _ in range(NSTEP):
        T_h = prob.solve()
        T_n.x.array[:] = T_h.x.array
    return T_n.x.array.copy()


def backward_euler_reference():
    msh, V, bc = setup()
    T_n = dolfinx.fem.Function(V)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dt = dolfinx.fem.Constant(msh, DT)
    a = (u / dt) * v * ufl.dx + ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (T_n / dt) * v * ufl.dx
    return march(msh, V, bc, T_n, a, L)


def main() -> int:
    as_constant = MUTATE
    err = "none"
    try:
        parts = build_theta_form(1.0, as_constant=as_constant)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        parts = None
    print(f"theta_1p0_weight_is_a_constant={as_constant}")
    print(f"theta_1p0_bare_float_error={err}")

    err_half = "none"
    try:
        build_theta_form(0.5, as_constant=False)
    except Exception as exc:  # noqa: BLE001
        err_half = f"{type(exc).__name__}: {exc}"
    print(f"theta_0p5_bare_float_error={err_half}")

    ref = backward_euler_reference()
    msh, V, bc, T_n, a, L = build_theta_form(1.0, as_constant=True)
    got = march(msh, V, bc, T_n, a, L)
    diff = float(np.max(np.abs(got - ref)))
    print(f"constant_weight_theta_1p0_vs_handwritten_be_maxdiff={diff:.3e}")
    print(f"constant_weight_builds_and_matches_backward_euler={diff < 1e-12}")
    print(f"bare_float_theta_0p5_builds_fine={err_half == 'none'}")
    print(f"bare_float_theta_1p0_raised_value_error="
          f"{err.startswith('ValueError')}")

    if (err.startswith("ValueError")
            and "missing an integration domain" in err
            and err_half == "none" and diff < 1e-12 and parts is None):
        print("VERDICT=bare_float_theta_1p0_has_no_integration_domain")
        return 0
    print("VERDICT=bare_float_weight_was_accepted")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
