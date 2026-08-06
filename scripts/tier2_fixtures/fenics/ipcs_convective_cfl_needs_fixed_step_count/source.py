"""Tier-2 for fenics navier_stokes#8: an IPCS splitting that treats convection
EXPLICITLY obeys a convective CFL limit, it blows up with no solver complaint
whatsoever, and a short test run hides the blow-up because a large dt reaches the
chosen end time in only a handful of steps.

Channel on the unit square meshed 8x8 (h = 0.125), P2/P1, nu = 0.005, parabolic
inflow with peak speed 1, p = 0 at the outflow. Three linear steps per time step
(tentative velocity with dot(u_n, nabla_grad(u_n)) taken at the old level,
pressure Poisson, velocity correction), each solved with a direct LU
LinearProblem.

Observed: judged by a FIXED NUMBER OF STEPS (60), dt = 0.05 drives max|u| from
1.0 past 1e2 at step 24 and on to 1e12 within a few more steps, while all three
LinearProblem.solver objects report CONVERGED_ITS (reason 4) on every single
step. The identical dt looks perfectly healthy when the loop is stopped at a
fixed end time of 0.6, because that is only 12 steps and max|u| is then 1.14.
dt = 0.02 survives all 60 steps with max|u| at 1.0, so the threshold lies between
0.02 and 0.05 -- well below h itself.

Mutation control: T2_MUTATE=1 runs the same fixed step count at dt = 0.02, below
the threshold, and nothing blows up.
"""
from __future__ import annotations

import os
import tempfile

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

NU = 0.005
N_CELLS = 8
H = 1.0 / N_CELLS
N_STEPS = 60
END_TIME = 0.6
DT_TEST = 0.02 if MUTATE else 0.05
DT_STABLE = 0.02
BLOWN = 1.0e6


def ipcs(dt, n_steps, prefix):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N_CELLS, N_CELLS)
    d = 2
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 2, (d,)))
    Q = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)
    u_n, p_n = dolfinx.fem.Function(V), dolfinx.fem.Function(Q)
    u_star, p_new = dolfinx.fem.Function(V), dolfinx.fem.Function(Q)
    k = dolfinx.fem.Constant(msh, dt)

    msh.topology.create_connectivity(d - 1, d)
    inflow = dolfinx.mesh.locate_entities_boundary(
        msh, d - 1, lambda x: np.isclose(x[0], 0.0))
    walls = dolfinx.mesh.locate_entities_boundary(
        msh, d - 1, lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    outflow = dolfinx.mesh.locate_entities_boundary(
        msh, d - 1, lambda x: np.isclose(x[0], 1.0))
    g = dolfinx.fem.Function(V)
    g.interpolate(lambda x: np.vstack(
        [4.0 * x[1] * (1.0 - x[1]), np.zeros_like(x[0])]))
    bcu = [dolfinx.fem.dirichletbc(
        g, dolfinx.fem.locate_dofs_topological(V, d - 1, inflow)),
        dolfinx.fem.dirichletbc(
            np.zeros(d),
            dolfinx.fem.locate_dofs_topological(V, d - 1, walls), V)]
    bcp = [dolfinx.fem.dirichletbc(
        dolfinx.fem.Constant(msh, 0.0),
        dolfinx.fem.locate_dofs_topological(Q, d - 1, outflow), Q)]

    a1 = ((1 / k) * ufl.dot(u, v) * ufl.dx
          + NU * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    l1 = ((1 / k) * ufl.dot(u_n, v) * ufl.dx
          - ufl.dot(ufl.dot(u_n, ufl.nabla_grad(u_n)), v) * ufl.dx
          - ufl.dot(ufl.grad(p_n), v) * ufl.dx)
    a2 = ufl.dot(ufl.grad(p), ufl.grad(q)) * ufl.dx
    l2 = (ufl.dot(ufl.grad(p_n), ufl.grad(q)) * ufl.dx
          - (1 / k) * ufl.div(u_star) * q * ufl.dx)
    a3 = ufl.dot(u, v) * ufl.dx
    l3 = (ufl.dot(u_star, v) * ufl.dx
          - k * ufl.dot(ufl.grad(p_new - p_n), v) * ufl.dx)
    opts = {"ksp_type": "preonly", "pc_type": "lu"}
    step1 = dolfinx.fem.petsc.LinearProblem(
        a1, l1, bcs=bcu, u=u_star, petsc_options_prefix=prefix + "1_",
        petsc_options=opts)
    step2 = dolfinx.fem.petsc.LinearProblem(
        a2, l2, bcs=bcp, u=p_new, petsc_options_prefix=prefix + "2_",
        petsc_options=opts)
    step3 = dolfinx.fem.petsc.LinearProblem(
        a3, l3, bcs=[], u=u_n, petsc_options_prefix=prefix + "3_",
        petsc_options=opts)

    reasons, first_cross, peak, done = set(), None, 0.0, 0
    for i in range(n_steps):
        step1.solve()
        reasons.add(step1.solver.getConvergedReason())
        step2.solve()
        reasons.add(step2.solver.getConvergedReason())
        step3.solve()
        reasons.add(step3.solver.getConvergedReason())
        p_n.x.array[:] = p_new.x.array
        speed = float(np.max(np.abs(u_n.x.array)))
        done = i + 1
        if not np.isfinite(speed) or speed > BLOWN:
            peak = speed
            first_cross = first_cross or done
            break
        peak = speed
        if first_cross is None and speed > 1.0e2:
            first_cross = done
    return peak, first_cross, sorted(reasons), done


def main() -> int:
    print(f"h={H} nu={NU} inflow_peak_speed=1.0 fixed_step_count={N_STEPS}")
    print(f"dt_under_test={DT_TEST}")

    peak_a, cross_a, reasons_a, done_a = ipcs(DT_TEST, N_STEPS, "t2_ns8a_")
    print(f"fixed_step_count_run dt={DT_TEST} steps_executed={done_a} "
          f"peak_max_speed={peak_a:.4e} first_step_above_1e2={cross_a} "
          f"ksp_converged_reasons={reasons_a}")
    blew = peak_a > BLOWN
    quiet = reasons_a == [4]
    print(f"fixed_step_count_run_blows_up={blew}")
    print(f"every_ksp_reported_converged_its={quiet}")

    n_short = max(1, int(round(END_TIME / DT_TEST)))
    peak_b, cross_b, _, done_b = ipcs(DT_TEST, n_short, "t2_ns8b_")
    print(f"fixed_end_time_run dt={DT_TEST} end_time={END_TIME} "
          f"steps={done_b} peak_max_speed={peak_b:.4e}")
    hidden = peak_b < 2.0 and cross_b is None
    print(f"fixed_end_time_run_looks_healthy={hidden}")

    peak_c, cross_c, _, done_c = ipcs(DT_STABLE, N_STEPS, "t2_ns8c_")
    print(f"below_threshold_run dt={DT_STABLE} steps={done_c} "
          f"peak_max_speed={peak_c:.6f}")
    survives = peak_c < 1.01 and cross_c is None and done_c == N_STEPS
    print(f"smaller_dt_survives_the_same_step_count={survives}")
    print(f"stability_threshold_is_well_below_h={DT_STABLE < 0.25 * H / 1.0}")

    if blew and quiet and hidden and survives:
        print("VERDICT=explicit_convection_cfl_shows_up_only_at_fixed_step_count")
        return 0
    print("VERDICT=no_cfl_limit_seen")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
