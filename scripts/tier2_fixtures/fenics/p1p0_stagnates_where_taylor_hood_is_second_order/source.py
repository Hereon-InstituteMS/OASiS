"""Tier-2 for fenics nearly_incompressible_elasticity#2: Taylor-Hood (P2/P1)
satisfies inf-sup and P1/P0 does not.

FINDING, and the reason this fixture pins what it pins: the claim says a
convergence-rate test with P1/P0 "stagnates at first-order in displacement". It
does not stagnate - there is no rate to measure. The bc-applied P1/DG0
saddle-point matrix is genuinely singular (its numerical null dimension grows
with refinement), the MUMPS direct solve reports converged reason -11 and the
returned velocity is not finite. The claim understates the failure.

Manufactured Stokes solution on the unit square, u = curl of
sin^2(pi x) sin^2(pi y) (divergence free, zero on the boundary),
p = cos(pi x) sin(pi y), f = -div grad u + grad p, Dirichlet velocity on every
boundary facet and one pressure dof pinned. P2/P1 solves at every level and its
L2 velocity error falls at third order; the tested P1/DG0 pair does not solve at
all.

Mutation control: T2_MUTATE=1 runs the study with P2/P1 as the tested pair,
which solves, stays finite and has null dimension 1 at every level.
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

LEVELS = (4, 8, 16)


def setup(n: int, k_vel: int, k_pre: int, family: str):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    gdim, tdim = msh.geometry.dim, msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    sx, sy = ufl.sin(pi * x[0]), ufl.sin(pi * x[1])
    cx, cy = ufl.cos(pi * x[0]), ufl.cos(pi * x[1])
    ue = ufl.as_vector((2 * pi * sx ** 2 * sy * cy,
                        -2 * pi * sx * cx * sy ** 2))
    pe = cx * sy
    fe = -ufl.div(ufl.grad(ue)) + ufl.grad(pe)

    V = basix.ufl.element("Lagrange", msh.basix_cell(), k_vel, shape=(gdim,))
    Q = basix.ufl.element(family, msh.basix_cell(), k_pre)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([V, Q]))
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v))
         - p * ufl.div(v) - q * ufl.div(u)) * ufl.dx
    L = ufl.inner(fe, v) * ufl.dx

    V0, _ = W.sub(0).collapse()
    ubc = dolfinx.fem.Function(V0)
    ubc.interpolate(dolfinx.fem.Expression(
        ue, V0.element.interpolation_points))
    dofs = dolfinx.fem.locate_dofs_topological(
        (W.sub(0), V0), tdim - 1,
        dolfinx.mesh.exterior_facet_indices(msh.topology))
    bcs = [dolfinx.fem.dirichletbc(ubc, dofs, W.sub(0))]
    Q0, pmap = W.sub(1).collapse()
    pin = dolfinx.fem.Function(Q0)
    bcs.append(dolfinx.fem.dirichletbc(
        pin, [np.array([pmap[0]], dtype=np.int32),
              np.array([0], dtype=np.int32)], W.sub(1)))
    return msh, W, a, L, bcs, ue


def solve_level(n: int, k_vel: int, k_pre: int, family: str, tag: str):
    _, W, a, L, bcs, ue = setup(n, k_vel, k_pre, family)
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=bcs, petsc_options_prefix=f"t2_nie2_{tag}{n}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})
    wh = prob.solve()
    reason = prob.solver.getConvergedReason()
    err = dolfinx.fem.assemble_scalar(dolfinx.fem.form(
        ufl.inner(wh.sub(0) - ue, wh.sub(0) - ue) * ufl.dx))
    return reason, float(np.sqrt(abs(err)))


def null_dimension(n: int, k_vel: int, k_pre: int, family: str) -> int:
    _, W, a, _, bcs, _ = setup(n, k_vel, k_pre, family)
    A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a), bcs=bcs)
    A.assemble()
    s = np.linalg.svd(A.convert("dense").getDenseArray().copy(),
                      compute_uv=False)
    return int((s / s[0] < 1e-10).sum())


def main() -> int:
    tested = (2, 1, "Lagrange", "th") if MUTATE else (1, 0, "DG", "p1p0")
    print(f"tested_pair=P{tested[0]}/"
          f"{'DG0' if tested[2] == 'DG' else 'P' + str(tested[1])}")
    r_test, e_test, r_th, e_th = [], [], [], []
    for n in LEVELS:
        a, b = solve_level(n, *tested)
        c, d = solve_level(n, 2, 1, "Lagrange", "ref")
        r_test.append(a)
        e_test.append(b)
        r_th.append(c)
        e_th.append(d)
        print(f"mesh={n}x{n} tested_reason={a} tested_L2={b:.6e} "
              f"taylor_hood_reason={c} taylor_hood_L2={d:.6e}")
    nulls = [null_dimension(n, *tested[:3]) for n in (4, 8)]
    print(f"tested_null_dimension_at_4x4_and_8x8={nulls}")
    th_rates = [float(np.log(e_th[i] / e_th[i + 1]) / np.log(2.0))
                for i in range(len(e_th) - 1)]
    print(f"taylor_hood_rates={[round(r, 3) for r in th_rates]}")

    test_fails = all(r < 0 for r in r_test)
    test_nonfinite = all(not np.isfinite(e) for e in e_test)
    th_ok = all(r > 0 for r in r_th) and min(th_rates) > 2.0
    nulls_grow = nulls[1] > nulls[0] > 1
    print(f"p1p0_direct_solve_reason_is_negative={test_fails}")
    print(f"p1p0_velocity_error_is_not_finite={test_nonfinite}")
    print(f"p1p0_null_dimension_grows_with_refinement={nulls_grow}")
    print(f"taylor_hood_solves_and_beats_second_order={th_ok}")
    print(f"first_order_stagnation_observed={not test_fails}")
    if test_fails and test_nonfinite and nulls_grow and th_ok:
        print("VERDICT=p1p0_system_is_singular_taylor_hood_converges")
        return 0
    print("VERDICT=tested_pair_solved_like_taylor_hood")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
