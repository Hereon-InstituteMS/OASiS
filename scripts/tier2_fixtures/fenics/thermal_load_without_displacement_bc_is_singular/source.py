"""Tier-2 for fenics thermal_structural#5: a thermal load is self-equilibrated,
so a model with no displacement Dirichlet condition has a singular stiffness
matrix -- and the default direct solver does not say so.

Measured here, on a thermally loaded square/cube with SI steel constants:

* the dense spectrum of the assembled operator holds exactly 3 (2D) or 6 (3D)
  eigenvalues below 1e-8 times the largest, the rigid translations and rotations;
  adding a minimal constraint set (3 scalar constraints in 2D, 6 in 3D) removes
  all of them;
* {"ksp_type": "preonly", "pc_type": "lu"} raises nothing, prints nothing, and
  getConvergedReason() returns 4 (CONVERGED_ITS) on the singular system;
* for the self-equilibrated thermal load the returned field is finite and the
  true residual is at round-off, so nothing looks wrong; for a load with a net
  resultant (gravity) the same silent path returns max|u| of order 1e7 m with a
  relative true residual around 1e-4, eleven orders worse than the constrained
  solve;
* an iterative solver is the one that complains: ksp_type cg with
  "ksp_error_if_not_converged": True raises petsc4py.PETSc.Error: error code 91.

Two parts of the inherited claim did NOT reproduce and the fixture pins what is
actually observed instead: the gravity residual is around 1e-4, not between 1 and
100; and CG does not report -4 (DIVERGED_DTOL) -- on the self-equilibrated load
plain CG converges (reason 2), and on gravity it stops with
"Diverged due to indefinite matrix" (reason -10).

Mutation control: T2_MUTATE=1 adds the minimal constraint set (u_x and u_y
pinned at one corner, u_y at a second corner in 2D), which removes the kernel.
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

E, NU, ALPHA, DT = 210e9, 0.3, 1.2e-5, 100.0
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
BETA = (3 * LAM + 2 * MU) * ALPHA


def minimal_bcs(msh, V, d):
    """3 scalar constraints in 2D, 6 in 3D: enough to kill rigid motion."""
    pts = [((0.0,) * d, tuple(range(d)))]
    if d == 2:
        pts.append(((1.0, 0.0), (1,)))
    else:
        pts.append(((1.0, 0.0, 0.0), (1, 2)))
        pts.append(((0.0, 1.0, 0.0), (2,)))
    bcs = []
    for point, comps in pts:
        for k in comps:
            Vk, _ = V.sub(k).collapse()
            dofs = dolfinx.fem.locate_dofs_geometrical(
                (V.sub(k), Vk),
                lambda x, p=point: np.logical_and.reduce(
                    [np.isclose(x[i], p[i]) for i in range(len(p))]))
            bcs.append(dolfinx.fem.dirichletbc(
                dolfinx.fem.Function(Vk), dofs, V.sub(k)))
    return bcs


def build(d, n, constrained, gravity=False):
    if d == 2:
        msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    else:
        msh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (d,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    eps = lambda w: ufl.sym(ufl.grad(w))  # noqa: E731
    a = ufl.inner(2 * MU * eps(u) + LAM * ufl.tr(eps(u)) * ufl.Identity(d),
                  eps(v)) * ufl.dx
    L = BETA * DT * ufl.div(v) * ufl.dx
    if gravity:
        g = (0.0, -7800.0 * 9.81) if d == 2 else (0.0, 0.0, -7800.0 * 9.81)
        L = L + ufl.inner(dolfinx.fem.Constant(msh, g), v) * ufl.dx
    bcs = minimal_bcs(msh, V, d) if constrained else []
    return msh, V, a, L, bcs


def near_null_count(d, n, constrained):
    _, _, a, _, bcs = build(d, n, constrained)
    A = dolfinx.fem.petsc.assemble_matrix(
        dolfinx.fem.form(a), bcs=bcs, diag=E)
    A.assemble()
    w = np.linalg.eigvalsh(A.convert("dense").getDenseArray())
    return int(np.sum(w < 1e-8 * w.max())), w.shape[0]


def solve(d, n, opts, prefix, constrained, gravity):
    _, _, a, L, bcs = build(d, n, constrained, gravity)
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=bcs, petsc_options_prefix=prefix, petsc_options=opts)
    raised = ""
    uh = None
    try:
        uh = prob.solve()
        if isinstance(uh, tuple):
            uh = uh[0]
    except BaseException as exc:  # noqa: BLE001
        raised = f"{type(exc).__module__}.{type(exc).__name__}: {exc}"
    reason = prob.solver.getConvergedReason()
    mx, rel = float("nan"), float("nan")
    if uh is not None:
        mx = float(np.max(np.abs(uh.x.array)))
        A = dolfinx.fem.petsc.assemble_matrix(
            dolfinx.fem.form(a), bcs=bcs, diag=E)
        A.assemble()
        b = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(L))
        if bcs:
            # zero Dirichlet data: the constrained rows of b must match the
            # diag entries put into A, otherwise the residual is meaningless
            dolfinx.fem.set_bc(b.array, bcs)
        r = A.createVecLeft()
        A.mult(uh.x.petsc_vec, r)
        r.axpy(-1.0, b)
        rel = r.norm() / b.norm()
    return reason, mx, rel, raised.splitlines()[0] if raised else ""


def main() -> int:
    constrained = MUTATE
    n2, tot2 = near_null_count(2, 4, constrained)
    n3, tot3 = near_null_count(3, 3, constrained)
    print(f"minimal_constraints_applied={constrained}")
    print(f"near_null_modes_2d={n2} of {tot2} dofs")
    print(f"near_null_modes_3d={n3} of {tot3} dofs")
    print(f"two_d_has_exactly_three_rigid_modes={n2 == 3}")
    print(f"three_d_has_exactly_six_rigid_modes={n3 == 6}")

    lu = {"ksp_type": "preonly", "pc_type": "lu"}
    r_t, mx_t, rel_t, exc_t = solve(2, 8, lu, "t2_ts5a_", constrained, False)
    print(f"lu_thermal_reason={r_t} max_abs_u={mx_t:.6e} "
          f"rel_true_residual={rel_t:.3e} raised={exc_t!r}")
    quiet = (r_t == 4) and (exc_t == "") and np.isfinite(mx_t)
    print(f"lu_returns_converged_its_and_says_nothing={quiet}")

    r_g, mx_g, rel_g, _ = solve(2, 8, lu, "t2_ts5b_", constrained, True)
    r_ref, mx_ref, rel_ref, _ = solve(2, 8, lu, "t2_ts5c_", True, True)
    print(f"lu_gravity_reason={r_g} max_abs_u={mx_g:.6e} "
          f"rel_true_residual={rel_g:.3e}")
    print(f"constrained_gravity_reference max_abs_u={mx_ref:.6e} "
          f"rel_true_residual={rel_ref:.3e}")
    garbage = (r_g == 4 and mx_g > 1e6 and rel_g > 1e3 * max(rel_ref, 1e-16))
    print(f"net_load_gives_huge_displacement_and_bad_residual={garbage}")

    cg = {"ksp_type": "cg", "pc_type": "jacobi",
          "ksp_error_if_not_converged": True}
    r_c, _, _, exc_c = solve(2, 8, cg, "t2_ts5d_", constrained, True)
    print(f"cg_gravity_reason={r_c} raised={exc_c!r}")
    complained = "error code 91" in exc_c
    print(f"cg_with_error_if_not_converged_raises_petsc_error_91={complained}")

    if n2 == 3 and n3 == 6 and quiet and garbage and complained:
        print("VERDICT=unconstrained_thermal_model_is_singular_and_lu_is_quiet")
        return 0
    print("VERDICT=stiffness_was_not_singular")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
