"""Tier-2 for fenics multiphase#2: Allen-Cahn does not conserve the phase
volume. int(phi) dx drifts monotonically because the curvature flow shrinks the
droplet, so int(phi) dx is NOT usable as a correctness check. The free energy
E(phi) = int eps/2 |grad phi|^2 + (phi^2-1)^2/(4 eps) dx is the Lyapunov
functional that IS usable, and it does hold at every step.

Same droplet (r = 0.25, eps = 3h) on a 32x32 unit square, 25 backward Euler
steps of dt = 1e-3, every step converged. Allen-Cahn moves int(phi) dx by
several percent, monotonically, while the energy falls at every single step.
The claim's remedy - "if your application needs volume conservation, the model
must change to Cahn-Hilliard" - is the mutation: the same droplet under
Cahn-Hilliard holds int(phi) dx to a relative 1e-13, because taking q = 1 in
its first equation makes conservation an identity of the discrete form.

Mutation control: T2_MUTATE=1 runs Cahn-Hilliard; the volume drift vanishes
while the energy still decreases monotonically.
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

N, NSTEP, DT, EPS_OVER_H, R = 32, 25, 1e-3, 3.0, 0.25
OPTS = {"snes_type": "newtonls", "ksp_type": "preonly", "pc_type": "lu"}


def droplet(eps: float):
    def ic(x):
        d = R - np.sqrt((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2)
        return np.tanh(d / (eps * np.sqrt(2.0)))
    return ic


def allen_cahn():
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    eps = EPS_OVER_H / N
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    phi, phi_n = dolfinx.fem.Function(V), dolfinx.fem.Function(V)
    phi.interpolate(droplet(eps))
    phi_n.interpolate(droplet(eps))
    v = ufl.TestFunction(V)
    dt_c = dolfinx.fem.Constant(msh, DT)
    eps_c = dolfinx.fem.Constant(msh, eps)
    F = ((phi - phi_n) / dt_c * v * ufl.dx
         + eps_c * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx
         + (1.0 / eps_c) * (phi ** 3 - phi) * v * ufl.dx)
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, phi, petsc_options_prefix="t2_mp2_ac_", petsc_options=OPTS)
    return prob, phi, phi_n, eps_c, phi


def cahn_hilliard():
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    eps = EPS_OVER_H / N
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P1, P1]))
    w, w0 = dolfinx.fem.Function(W), dolfinx.fem.Function(W)
    c, mu = ufl.split(w)
    c0, _ = ufl.split(w0)
    q, v = ufl.TestFunctions(W)
    W0, cmap = W.sub(0).collapse()
    seed = dolfinx.fem.Function(W0)
    seed.interpolate(droplet(eps))
    w.x.array[cmap] = seed.x.array
    w0.x.array[cmap] = seed.x.array
    dt_c = dolfinx.fem.Constant(msh, DT)
    eps_c = dolfinx.fem.Constant(msh, eps)
    F = ((c - c0) / dt_c * q * ufl.dx
         + ufl.dot(ufl.grad(mu), ufl.grad(q)) * ufl.dx
         + mu * v * ufl.dx
         - (1.0 / eps_c) * (c ** 3 - c) * v * ufl.dx
         - eps_c * ufl.dot(ufl.grad(c), ufl.grad(v)) * ufl.dx)
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, w, petsc_options_prefix="t2_mp2_ch_", petsc_options=OPTS)
    return prob, w, w0, eps_c, c


def history(model: str):
    build = cahn_hilliard if model == "cahn_hilliard" else allen_cahn
    prob, state, state_n, eps_c, c = build()
    energy = dolfinx.fem.form(
        eps_c / 2 * ufl.dot(ufl.grad(c), ufl.grad(c)) * ufl.dx
        + (c ** 2 - 1) ** 2 / (4 * eps_c) * ufl.dx)
    volume = dolfinx.fem.form(c * ufl.dx)
    es = [float(dolfinx.fem.assemble_scalar(energy))]
    vs = [float(dolfinx.fem.assemble_scalar(volume))]
    reasons = []
    for _ in range(NSTEP):
        prob.solve()
        reasons.append(prob.solver.getConvergedReason())
        es.append(float(dolfinx.fem.assemble_scalar(energy)))
        vs.append(float(dolfinx.fem.assemble_scalar(volume)))
        state_n.x.array[:] = state.x.array
    return es, vs, reasons


def main() -> int:
    model = "cahn_hilliard" if MUTATE else "allen_cahn"
    es, vs, reasons = history(model)
    drift = abs(vs[-1] - vs[0]) / abs(vs[0])
    mono = all(es[i + 1] <= es[i] + 1e-14 for i in range(len(es) - 1))
    conv = all(r > 0 for r in reasons)
    steady = all(abs(vs[i + 1] - vs[i]) < 1e-12 for i in range(len(vs) - 1))
    monotone_drift = steady or all(
        (vs[i + 1] - vs[i]) * (vs[1] - vs[0]) > 0 for i in range(len(vs) - 1))
    print(f"model={model}")
    print(f"volume_first={vs[0]:.6e} volume_last={vs[-1]:.6e} "
          f"relative_drift_percent={drift * 100:.4e}")
    print(f"energy_first={es[0]:.6f} energy_last={es[-1]:.6f}")
    print(f"every_step_converged={conv}")
    print(f"volume_drift_exceeds_one_percent={drift > 0.01}")
    print(f"volume_drift_is_monotone={monotone_drift}")
    print(f"energy_is_non_increasing_at_every_step={mono}")
    if conv and mono and drift > 0.01 and monotone_drift:
        print("VERDICT=volume_drifts_while_the_energy_is_a_lyapunov_functional")
        return 0
    print("VERDICT=volume_was_conserved")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
