"""Tier-2 for fenics nonlinear_pde#3: D = 1 + u**q with a non-integer q is only
defined while u >= 0. With a source that drives u negative the power leaves the
reals and Newton dies at iteration 0.

Wrong variant: D = 1 + u**1.5 with f = -1 and homogeneous Dirichlet data, so the
solution must be negative everywhere.
Right variants: an even integer exponent (q = 2), or abs(u)**q.

Observed on dolfinx 0.10.0:
  * q = 2.0 with the negative source converges (reason 3, u in [-7.33e-02, 0]).
  * q = 1.5 with the same source stops at iteration 0 and leaves u at exactly
    zero. The converged reason depends on the line search, NOT on the exponent:
    the default 'bt' and 'basic' report -4 (DIVERGED_FNORM_NAN) while 'l2'
    reports -6 (DIVERGED_LINE_SEARCH). The claim quotes only -6, which is the
    'l2' answer; on the default line search the code is -4.
  * abs(u)**1.5 converges on the same problem.

Mutation control: T2_MUTATE=1 replaces u**q by abs(u)**q in the as-written form,
so both line searches converge and the frozen zero field disappears.
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

N, Q, SOURCE = 16, 1.5, -1.0
USE_ABS_AS_WRITTEN = MUTATE


def run(tag: str, q: float, use_abs: bool, linesearch: str):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    dofs = dolfinx.fem.locate_dofs_topological(V, msh.topology.dim - 1, facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 0.0), dofs, V)
    u = dolfinx.fem.Function(V)  # starts at zero
    v = ufl.TestFunction(V)
    base = abs(u) if use_abs else u
    D = 1.0 + base ** q
    F = (D * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - dolfinx.fem.Constant(msh, SOURCE) * v * ufl.dx)
    problem = dolfinx.fem.petsc.NonlinearProblem(
        F, u, bcs=[bc], petsc_options_prefix=f"t2_np3_{tag}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "snes_max_it": 30,
                       "snes_linesearch_type": linesearch})
    problem.solve()
    arr = u.x.array.copy()
    r = problem.solver.getConvergedReason()
    it = problem.solver.getIterationNumber()
    print(f"{tag}: q={q} abs={use_abs} linesearch={linesearch} reason={r} "
          f"iterations={it} u_range=[{float(arr.min()):.4e}, "
          f"{float(arr.max()):.4e}] all_zero={bool(np.all(arr == 0.0))}")
    return r, it, arr


def main() -> int:
    print(f"source={SOURCE:g} q_as_written={Q} abs_as_written="
          f"{USE_ABS_AS_WRITTEN}")
    r_bt, it_bt, a_bt = run("as_written_bt", Q, USE_ABS_AS_WRITTEN, "bt")
    r_l2, it_l2, a_l2 = run("as_written_l2", Q, USE_ABS_AS_WRITTEN, "l2")
    r_int, it_int, a_int = run("even_integer_power", 2.0, False, "bt")
    r_abs, it_abs, a_abs = run("abs_of_u", Q, True, "bt")

    dead = (r_bt < 0 and r_l2 < 0 and it_bt == 0 and it_l2 == 0)
    frozen = bool(np.all(a_bt == 0.0) and np.all(a_l2 == 0.0))
    print(f"non_integer_power_dies_at_iteration_zero={dead}")
    print(f"solution_left_exactly_at_zero={frozen}")
    print(f"default_line_search_reason_is_fnorm_nan={r_bt == -4}")
    print(f"l2_line_search_reason_is_line_search={r_l2 == -6}")

    int_ok = r_int > 0 and float(a_int.min()) < -1.0e-3
    abs_ok = r_abs > 0 and float(a_abs.min()) < -1.0e-3
    print(f"even_integer_power_converges_negative={int_ok}")
    print(f"abs_of_u_converges_negative={abs_ok}")

    if dead and frozen and int_ok and abs_ok:
        print("VERDICT=non_integer_power_dies_when_the_load_pushes_u_negative")
        return 0
    print("VERDICT=non_integer_power_was_fine")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
