"""Tier-2 for fenics cahn_hilliard#2: wrap the concentration in ufl.variable()
and let ufl.diff produce dfdc. ufl.diff is exact -- it agrees with a hand-coded
200*c*(1-c)*(1-2*c) at round-off. The danger of hand-coding is NOT a loss of
Newton convergence, because dolfinx builds the Jacobian from whatever residual
you wrote: you silently solve a different physical problem and the solver output
looks BETTER than the correct one.

Wrong variant: the frequently repeated 12*c*(c-1)*(2*c-1), which is the
derivative of 6*c^2*(1-c)^2, i.e. exactly 0.06 times too small.

Observed on dolfinx 0.10.0 (unit square 24x24, P1 x P1 mixed, lmbda = 1e-2,
M = 1, theta = 0.5, dt = 5e-6, 30 steps): the corrupted run converges in
exactly 2 Newton iterations on every step with zero failures, while the correct
run needs about 17; and the corrupted concentration never leaves the immediate
neighbourhood of its initial mean while the correct one separates out past
[0, 1]. Low, flat Newton counts in the separation phase are a warning sign.

Mutation control: T2_MUTATE=1 gives the "corrupted" run the correct derivative,
so it separates and its Newton count rises.
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

N, LMBDA, MOB, THETA, DT, STEPS = 24, 1.0e-2, 1.0, 0.5, 5.0e-6, 30


def dfdc_of(c, kind: str):
    if kind == "ufl_diff":
        cv = ufl.variable(c)
        return ufl.diff(100.0 * cv**2 * (1 - cv) ** 2, cv)
    if kind == "hand":
        return 200.0 * c * (1 - c) * (1 - 2 * c)
    if kind == "quoted_12c":
        return 12.0 * c * (c - 1) * (2 * c - 1)
    raise ValueError(kind)


def scalar(expr) -> float:
    return float(dolfinx.fem.assemble_scalar(dolfinx.fem.form(expr)))


def march(tag: str, kind: str):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    ME = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P1, P1]))
    u, u0 = dolfinx.fem.Function(ME), dolfinx.fem.Function(ME)
    rng = np.random.default_rng(7)
    u.sub(0).interpolate(lambda x: 0.63 + 0.02 * (0.5 - rng.random(x.shape[1])))
    u.sub(1).interpolate(lambda x: np.zeros(x.shape[1]))
    u.x.scatter_forward()
    u0.x.array[:] = u.x.array
    q, v = ufl.TestFunctions(ME)
    c, mu = ufl.split(u)
    c0, mu0 = ufl.split(u0)
    mu_mid = (1.0 - THETA) * mu0 + THETA * mu
    F = ((c - c0) * q * ufl.dx
         + DT * MOB * ufl.dot(ufl.grad(mu_mid), ufl.grad(q)) * ufl.dx
         + mu * v * ufl.dx - dfdc_of(c, kind) * v * ufl.dx
         - LMBDA * ufl.dot(ufl.grad(c), ufl.grad(v)) * ufl.dx)
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, u, petsc_options_prefix=f"t2_ch2_{tag}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "snes_max_it": 40})
    _, cdofs = ME.sub(0).collapse()
    cdofs = np.asarray(cdofs, dtype=np.int32)
    its, reasons = [], []
    for _ in range(STEPS):
        u0.x.array[:] = u.x.array
        prob.solve()
        u.x.scatter_forward()
        its.append(prob.solver.getIterationNumber())
        reasons.append(prob.solver.getConvergedReason())
    c_end = u.x.array[cdofs]
    return dict(its=its, mean_it=float(np.mean(its)),
                all_converged=all(r > 0 for r in reasons),
                cmin=float(c_end.min()), cmax=float(c_end.max()))


def main() -> int:
    # 1) ufl.diff is exactly the analytic derivative
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    cf = dolfinx.fem.Function(V)
    rng = np.random.default_rng(3)
    cf.interpolate(lambda x: 0.63 + 0.4 * (0.5 - rng.random(x.shape[1])))
    cf.x.scatter_forward()
    d_ufl, d_hand = dfdc_of(cf, "ufl_diff"), dfdc_of(cf, "hand")
    l2_diff = np.sqrt(scalar((d_ufl - d_hand) ** 2 * ufl.dx))
    l2_ref = np.sqrt(scalar(d_ufl**2 * ufl.dx))
    print(f"l2_norm_of_dfdc={l2_ref:.6e} "
          f"l2_norm_of_ufl_diff_minus_hand_coded={l2_diff:.3e}")
    exact = l2_diff < 1.0e-12 * l2_ref
    print(f"ufl_diff_matches_the_hand_coded_derivative={exact}")

    # 2) the quoted 12*c*(c-1)*(2*c-1) is 0.06 times the correct derivative
    c03 = dolfinx.fem.Function(V)
    c03.x.array[:] = 0.3
    v_ok = scalar(dfdc_of(c03, "hand") * ufl.dx)
    v_bad = scalar(dfdc_of(c03, "quoted_12c") * ufl.dx)
    ratio = v_bad / v_ok
    print(f"at_c_0p3 correct_dfdc={v_ok:.6f} quoted_dfdc={v_bad:.6f} "
          f"ratio={ratio:.12f}")
    is_006 = abs(ratio - 0.06) < 1.0e-12
    print(f"the_quoted_12c_expression_is_0p06_times_the_correct_one={is_006}")

    # 3) the time loop with the correct and with the corrupted derivative
    good = march("good", "ufl_diff")
    bad = march("bad", "ufl_diff" if MUTATE else "quoted_12c")
    print(f"correct_run: mean_newton_its={good['mean_it']:.2f} "
          f"first_its={good['its'][:6]} "
          f"c_range=[{good['cmin']:.6f}, {good['cmax']:.6f}]")
    print(f"corrupted_run: mean_newton_its={bad['mean_it']:.2f} "
          f"first_its={bad['its'][:6]} "
          f"c_range=[{bad['cmin']:.6f}, {bad['cmax']:.6f}]")
    quiet = bad["all_converged"]
    cheaper = bad["mean_it"] < 0.5 * good["mean_it"]
    stuck = bad["cmin"] > 0.6 and bad["cmax"] < 0.66
    moved = good["cmin"] < -0.01 and good["cmax"] > 1.01
    print(f"corrupted_run_converged_every_step={quiet}")
    print(f"corrupted_run_needed_fewer_newton_iterations={cheaper}")
    print(f"corrupted_run_never_separated={stuck}")
    print(f"correct_run_separated={moved}")

    if exact and is_006 and quiet and cheaper and stuck and moved:
        print("VERDICT=flat_low_newton_counts_are_a_warning_not_a_success")
        return 0
    print("VERDICT=corrupted_derivative_was_caught_by_the_solver")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
