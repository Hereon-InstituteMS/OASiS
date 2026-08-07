"""Tier-2 for fenics cahn_hilliard#1: the initial condition must carry a random
perturbation. The usual explanation -- "c = 0.5 exactly is the unstable
symmetric mean, so it cannot separate" -- gets the right answer for the wrong
reason: ANY uniform initial concentration is frozen, and 0.5 is not special.

Unit square 24x24, P1 x P1 mixed, lmbda = 1e-2, M = 1, theta = 0.5,
dt = 5e-6, 30 backward-Euler/Crank-Nicolson steps, four initial conditions run
side by side: uniform 0.5, uniform 0.63 (the value usually recommended as the
"safe" one), 0.5 + 0.02*(0.5 - rng.random()) and 0.63 + the same noise.

Observed on dolfinx 0.10.0: uniform 0.5 stays at exactly 0.5 with standard
deviation 0.0, uniform 0.63 stays at 0.63 with standard deviation at round-off
(order 1e-16), while BOTH noisy runs separate out past [0, 1]. Every one of the
four runs converges at every step, so the solver reports nothing at all about
the frozen ones.

Mutation control: T2_MUTATE=1 adds the perturbation to the two "uniform" runs
as well, so nothing stays frozen.
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


def march(tag: str, mean: float, noise: float):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    ME = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P1, P1]))
    u, u0 = dolfinx.fem.Function(ME), dolfinx.fem.Function(ME)
    rng = np.random.default_rng(7)
    if noise > 0.0:
        u.sub(0).interpolate(
            lambda x: mean + noise * (0.5 - rng.random(x.shape[1])))
    else:
        u.sub(0).interpolate(lambda x: np.full(x.shape[1], mean))
    u.sub(1).interpolate(lambda x: np.zeros(x.shape[1]))
    u.x.scatter_forward()
    u0.x.array[:] = u.x.array
    q, v = ufl.TestFunctions(ME)
    c, mu = ufl.split(u)
    c0, mu0 = ufl.split(u0)
    cv = ufl.variable(c)
    dfdc = ufl.diff(100.0 * cv**2 * (1 - cv) ** 2, cv)
    mu_mid = (1.0 - THETA) * mu0 + THETA * mu
    F = ((c - c0) * q * ufl.dx
         + DT * MOB * ufl.dot(ufl.grad(mu_mid), ufl.grad(q)) * ufl.dx
         + mu * v * ufl.dx - dfdc * v * ufl.dx
         - LMBDA * ufl.dot(ufl.grad(c), ufl.grad(v)) * ufl.dx)
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, u, petsc_options_prefix=f"t2_ch1_{tag}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "snes_max_it": 40})
    _, cdofs = ME.sub(0).collapse()
    cdofs = np.asarray(cdofs, dtype=np.int32)
    reasons = []
    for _ in range(STEPS):
        u0.x.array[:] = u.x.array
        prob.solve()
        u.x.scatter_forward()
        reasons.append(prob.solver.getConvergedReason())
    c_end = u.x.array[cdofs]
    return dict(all_converged=all(r > 0 for r in reasons),
                cmin=float(c_end.min()), cmax=float(c_end.max()),
                cstd=float(c_end.std()))


def main() -> int:
    noise = 0.02
    cases = {
        "uniform_0.5": (0.5, noise if MUTATE else 0.0),
        "uniform_0.63": (0.63, noise if MUTATE else 0.0),
        "0.5_plus_noise": (0.5, noise),
        "0.63_plus_noise": (0.63, noise),
    }
    out = {}
    for i, (tag, (mean, nz)) in enumerate(cases.items()):
        r = march(f"k{i}", mean, nz)
        out[tag] = r
        print(f"{tag}: noise={nz} all_converged={r['all_converged']} "
              f"c_range=[{r['cmin']:.6f}, {r['cmax']:.6f}] "
              f"std={r['cstd']:.3e}")

    def frozen(r, mean):
        return (r["cstd"] < 1.0e-14
                and abs(r["cmin"] - mean) < 1.0e-12
                and abs(r["cmax"] - mean) < 1.0e-12)

    def separated(r):
        return r["cmin"] < -0.01 and r["cmax"] > 1.01

    f05 = frozen(out["uniform_0.5"], 0.5)
    f63 = frozen(out["uniform_0.63"], 0.63)
    s05 = separated(out["0.5_plus_noise"])
    s63 = separated(out["0.63_plus_noise"])
    conv = all(r["all_converged"] for r in out.values())
    print(f"uniform_0p5_never_separates={f05}")
    print(f"uniform_0p63_never_separates={f63}")
    print(f"mean_0p5_plus_noise_separates={s05}")
    print(f"mean_0p63_plus_noise_separates={s63}")
    print(f"zero_point_five_is_not_special={f05 and f63 and s05 and s63}")
    print(f"every_run_converged_so_the_solver_says_nothing={conv}")
    if f05 and f63 and s05 and s63 and conv:
        print("VERDICT=the_perturbation_not_the_mean_breaks_the_symmetry")
        return 0
    print("VERDICT=uniform_start_separated_after_all")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
