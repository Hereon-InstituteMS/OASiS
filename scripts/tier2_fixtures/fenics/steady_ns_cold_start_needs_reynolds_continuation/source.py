"""Tier-2 for fenics navier_stokes#6: on a steady lid-driven cavity, Newton
started from a zero initial guess stops working at high Reynolds number, and
refining the mesh does not rescue it. The cure is continuation in Re, keeping the
previous solution Function as the initial guess and changing only nu.value.

Taylor-Hood P2/P1, lid speed 1, pressure pinned at one corner dof, an undamped
Newton (snes_linesearch_type "basic") with a 60-iteration budget and MUMPS as the
factorisation package -- PETSc's own LU stops the same saddle-point solve at
DIVERGED_LINEAR_SOLVE on iteration 0.

Observed cold-starting Re = 1000: PETSc prints
"Nonlinear <prefix> solve did not converge due to DIVERGED_DTOL iterations 55",
getConvergedReason() is -9, and the velocity Function ends up holding max|u| in
the 1e3-1e4 range against an imposed lid speed of 1. Doubling the mesh from
12x12 to 24x24 does not rescue it -- it diverges the same way (the blow-up
magnitude is not monotone in mesh size, so the inherited "refining makes the
blow-up larger" wording is not reproduced; what reproduces is that refining does
not help). Stepping Re through 100, 200, 400, 600, 800, 1000 with the previous
solution retained converges at every stage in 4-5 Newton iterations with max|u|
staying at exactly 1.0000000000.

Mutation control: T2_MUTATE=1 reaches Re = 1000 by continuation instead of a
cold start, on both meshes.
"""
from __future__ import annotations

import os
import tempfile

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import basix.ufl  # noqa: E402
import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402
from petsc4py import PETSc  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

LID = 1.0
TARGET_RE = 1000.0
LADDER = (100.0, 200.0, 400.0, 600.0, 800.0, 1000.0)
REASONS = {v: k for k, v in PETSc.SNES.ConvergedReason.__dict__.items()
           if isinstance(v, int)}


def cavity(n, prefix):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    d = 2
    el_u = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(d,))
    el_p = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([el_u, el_p]))
    w = dolfinx.fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    nu = dolfinx.fem.Constant(msh, 1.0 / LADDER[0])
    res = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
           + ufl.inner(ufl.dot(u, ufl.nabla_grad(u)), v) * ufl.dx
           - p * ufl.div(v) * ufl.dx - q * ufl.div(u) * ufl.dx)

    msh.topology.create_connectivity(d - 1, d)
    V0, v_map = W.sub(0).collapse()
    lid = dolfinx.fem.Function(V0)
    lid.interpolate(lambda x: np.vstack(
        [np.isclose(x[1], 1.0) * LID, np.zeros_like(x[0])]))
    facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    bcs = [dolfinx.fem.dirichletbc(
        lid,
        dolfinx.fem.locate_dofs_topological((W.sub(0), V0), d - 1, facets),
        W.sub(0))]
    Q0, _ = W.sub(1).collapse()
    corner = dolfinx.fem.locate_dofs_geometrical(
        (W.sub(1), Q0),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    bcs.append(dolfinx.fem.dirichletbc(
        dolfinx.fem.Function(Q0), corner, W.sub(1)))

    problem = dolfinx.fem.petsc.NonlinearProblem(
        res, w, bcs=bcs, petsc_options_prefix=prefix,
        petsc_options={"snes_type": "newtonls",
                       "snes_linesearch_type": "basic",
                       "snes_max_it": 60, "snes_rtol": 1e-8,
                       "snes_converged_reason": "",
                       "ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})
    return w, nu, problem, np.array(v_map, dtype=np.int32)


def max_speed(w, v_map):
    return float(np.max(np.abs(w.x.array[v_map])))


def reach_target(n, prefix, continuation):
    w, nu, problem, v_map = cavity(n, prefix)
    ladder = LADDER if continuation else (TARGET_RE,)
    reasons, its, speeds = [], [], []
    for re in ladder:
        nu.value = 1.0 / re
        problem.solve()
        reasons.append(problem.solver.getConvergedReason())
        its.append(problem.solver.getIterationNumber())
        speeds.append(max_speed(w, v_map))
    return reasons, its, speeds


def main() -> int:
    mode = "continuation" if MUTATE else "cold_start"
    print(f"path_to_Re_{int(TARGET_RE)}_under_test={mode}")
    out = {}
    for n in (12, 24):
        reasons, its, speeds = reach_target(n, f"t2_ns6_{n}_", MUTATE)
        names = [REASONS.get(r, str(r)) for r in reasons]
        out[n] = (reasons, speeds)
        print(f"mesh={n}x{n} reasons={reasons} names={names} "
              f"iterations={its} final_max_speed={speeds[-1]:.6e}")

    diverged12 = out[12][0][-1] < 0
    diverged24 = out[24][0][-1] < 0
    print(f"cold_start_reason_name={REASONS.get(out[12][0][-1])}")
    print(f"cold_start_diverged={diverged12}")
    print(f"refining_the_mesh_does_not_rescue_the_cold_start="
          f"{diverged12 and diverged24}")
    blown = out[12][1][-1] > 100.0 * LID and out[24][1][-1] > 100.0 * LID
    print(f"cold_start_velocity_blows_past_the_lid_speed={blown}")

    reasons, its, speeds = reach_target(12, "t2_ns6_cont_", True)
    print(f"continuation_ladder={[int(r) for r in LADDER]} reasons={reasons} "
          f"iterations={its}")
    print(f"continuation_max_speeds={[f'{s:.10f}' for s in speeds]}")
    every = all(r > 0 for r in reasons)
    handful = all(i <= 10 for i in its)
    at_lid = all(abs(s - LID) < 1e-9 for s in speeds)
    print(f"continuation_converges_at_every_stage={every}")
    print(f"continuation_needs_only_a_handful_of_newton_steps={handful}")
    print(f"continuation_keeps_max_velocity_at_the_lid_speed={at_lid}")

    if diverged12 and diverged24 and blown and every and handful and at_lid:
        print("VERDICT=cold_start_fails_reynolds_continuation_works")
        return 0
    print("VERDICT=cold_start_was_fine")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
