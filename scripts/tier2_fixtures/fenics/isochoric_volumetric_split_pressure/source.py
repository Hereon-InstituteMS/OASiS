"""Tier-2 for fenics hyperelasticity#5: the isochoric/volumetric split
F = F_iso*F_vol with F_vol = J^(1/d)*I and W = W_iso(F_iso) + kappa/2*(J-1)^2
gives a clean, bounded post-processed pressure dU/dJ = kappa*(J-1) -- but the
inherited signal for the OTHER half of the claim does NOT reproduce.

Claimed: "without the split, the discrete pressure Function at Gauss points
oscillates wildly element-to-element", implying the split removes the
oscillation. Measured here on a nearly incompressible block (mu = 1,
kappa = 1e4, 5 % stretch applied in 4 increments, every SNES solve converging
with reason 3), reading the hydrostatic pressure -tr(sigma)/d into a DG0 space:

* pure-displacement P1 WITH the split: the element pressures span about 43 times
  their mean;
* the same P1 model with the split switched off (coupled mu/2*(tr C - d) -
  mu*ln J + lambda/2*(ln J)^2, same bulk modulus): the pressures are the same to
  better than 1 % in standard deviation -- the split changes nothing about the
  oscillation;
* a mixed (u, p) P2/P1 space with the identical isochoric energy: the span drops
  to about 1.2 times the mean and the standard deviation falls by three orders
  of magnitude.

So the oscillation is a property of the pure-displacement space, not of the
constitutive split, and the split alone is not the cure. The bounded-dU/dJ half
of the claim does hold and is checked too.

Mutation control: T2_MUTATE=1 makes the mixed (u, p) space the model under test,
which is what actually removes the oscillation.
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

MU, KAPPA = 1.0, 1.0e4
STRETCH, N_STEPS, N_CELLS = 0.05, 4, 12
OPTS = {"snes_type": "newtonls", "ksp_type": "preonly", "pc_type": "lu",
        "snes_max_it": 40, "snes_rtol": 1e-9}


def _ramp(problem, g, n_steps):
    reasons = []
    for i in range(n_steps):
        s = STRETCH * (i + 1) / n_steps
        g.interpolate(lambda x, s=s: np.vstack(
            [s * np.ones_like(x[0]), np.zeros_like(x[0])]))
        problem.solve()
        reasons.append(problem.solver.getConvergedReason())
    return reasons


def pure_displacement(split, prefix):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N_CELLS, N_CELLS)
    d = 2
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (d,)))
    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)
    f_var = ufl.variable(ufl.Identity(d) + ufl.grad(u))
    jac = ufl.det(f_var)
    right_c = f_var.T * f_var
    if split:
        psi = (MU / 2) * (jac ** (-2.0 / d) * ufl.tr(right_c) - d) \
            + (KAPPA / 2) * (jac - 1) ** 2
    else:
        lam = KAPPA - 2 * MU / d
        psi = (MU / 2) * (ufl.tr(right_c) - d) - MU * ufl.ln(jac) \
            + (lam / 2) * ufl.ln(jac) ** 2
    piola = ufl.diff(psi, f_var)
    res = ufl.inner(piola, ufl.grad(v)) * ufl.dx

    msh.topology.create_connectivity(d - 1, d)
    left = dolfinx.mesh.locate_entities_boundary(
        msh, d - 1, lambda x: np.isclose(x[0], 0.0))
    right = dolfinx.mesh.locate_entities_boundary(
        msh, d - 1, lambda x: np.isclose(x[0], 1.0))
    g = dolfinx.fem.Function(V)
    bcs = [dolfinx.fem.dirichletbc(
        np.zeros(d),
        dolfinx.fem.locate_dofs_topological(V, d - 1, left), V),
        dolfinx.fem.dirichletbc(
            g, dolfinx.fem.locate_dofs_topological(V, d - 1, right))]
    problem = dolfinx.fem.petsc.NonlinearProblem(
        res, u, bcs=bcs, petsc_options_prefix=prefix, petsc_options=OPTS)
    reasons = _ramp(problem, g, N_STEPS)

    Q = dolfinx.fem.functionspace(msh, ("DG", 0))
    sigma = (1 / jac) * piola * f_var.T
    ph = dolfinx.fem.Function(Q)
    ph.interpolate(dolfinx.fem.Expression(-ufl.tr(sigma) / d,
                                          Q.element.interpolation_points))
    volumetric = None
    if split:
        pv = dolfinx.fem.Function(Q)
        pv.interpolate(dolfinx.fem.Expression(
            KAPPA * (jac - 1), Q.element.interpolation_points))
        volumetric = (float(dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(KAPPA * (jac - 1) * ufl.dx))),
            float(np.max(np.abs(pv.x.array))))
    return reasons, np.array(ph.x.array), volumetric


def mixed_up(prefix):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N_CELLS, N_CELLS)
    d = 2
    el_u = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(d,))
    el_p = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([el_u, el_p]))
    w = dolfinx.fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    f_var = ufl.variable(ufl.Identity(d) + ufl.grad(u))
    jac = ufl.det(f_var)
    psi_iso = (MU / 2) * (jac ** (-2.0 / d) * ufl.tr(f_var.T * f_var) - d)
    piola_iso = ufl.diff(psi_iso, f_var)
    res = (ufl.inner(piola_iso + p * jac * ufl.inv(f_var).T,
                     ufl.grad(v)) * ufl.dx
           + q * ((jac - 1) - p / KAPPA) * ufl.dx)

    msh.topology.create_connectivity(d - 1, d)
    left = dolfinx.mesh.locate_entities_boundary(
        msh, d - 1, lambda x: np.isclose(x[0], 0.0))
    right = dolfinx.mesh.locate_entities_boundary(
        msh, d - 1, lambda x: np.isclose(x[0], 1.0))
    V0, _ = W.sub(0).collapse()
    g = dolfinx.fem.Function(V0)
    bcs = [dolfinx.fem.dirichletbc(
        dolfinx.fem.Function(V0),
        dolfinx.fem.locate_dofs_topological((W.sub(0), V0), d - 1, left),
        W.sub(0)),
        dolfinx.fem.dirichletbc(
            g,
            dolfinx.fem.locate_dofs_topological((W.sub(0), V0), d - 1, right),
            W.sub(0))]
    problem = dolfinx.fem.petsc.NonlinearProblem(
        res, w, bcs=bcs, petsc_options_prefix=prefix, petsc_options=OPTS)
    reasons = _ramp(problem, g, N_STEPS)

    Q = dolfinx.fem.functionspace(msh, ("DG", 0))
    ph = dolfinx.fem.Function(Q)
    ph.interpolate(dolfinx.fem.Expression(-p, Q.element.interpolation_points))
    return reasons, np.array(ph.x.array)


def stats(name, reasons, arr):
    span = float(arr.max() - arr.min())
    mean = float(arr.mean())
    std = float(arr.std())
    print(f"{name} reasons={reasons} p_mean={mean:+.4e} p_std={std:.4e} "
          f"span={span:.4e} span_over_abs_mean={span / abs(mean):.3f}")
    return span / abs(mean), std, all(r > 0 for r in reasons)


def main() -> int:
    r_s, p_s, vol = pure_displacement(True, "t2_hy5a_")
    ratio_s, std_s, ok_s = stats("pure_displacement_P1_with_split", r_s, p_s)
    r_n, p_n, _ = pure_displacement(False, "t2_hy5b_")
    ratio_n, std_n, ok_n = stats("pure_displacement_P1_no_split", r_n, p_n)
    r_m, p_m = mixed_up("t2_hy5c_")
    ratio_m, std_m, ok_m = stats("mixed_u_p_P2P1_with_split", r_m, p_m)

    integral, peak = vol
    print(f"dUdJ_integral={integral:.6e} dUdJ_max_abs={peak:.6e}")
    bounded = np.isfinite(integral) and np.isfinite(peak) and peak < 1e3 * KAPPA
    print(f"split_volumetric_pressure_dUdJ_is_bounded={bounded}")
    print(f"every_solve_converged={ok_s and ok_n and ok_m}")

    ratio_test = ratio_m if MUTATE else ratio_s
    print(f"model_under_test={'mixed_u_p' if MUTATE else 'pure_displacement'} "
          f"span_over_abs_mean={ratio_test:.3f}")
    oscillates = ratio_test > 10.0
    print(f"pressure_under_test_oscillates_element_to_element={oscillates}")

    split_gap = abs(std_s - std_n) / std_s
    print(f"std_relative_gap_between_split_and_no_split={split_gap:.4e}")
    print(f"switching_the_split_off_changes_the_oscillation_by_under_one_percent="
          f"{split_gap < 0.01}")
    fixed = ratio_m < 3.0 and std_m < 1e-2 * std_s
    print(f"mixed_u_p_space_removes_the_oscillation={fixed}")

    if (oscillates and split_gap < 0.01 and fixed and bounded
            and ok_s and ok_n and ok_m):
        print("VERDICT=split_alone_does_not_smooth_the_pressure_mixed_space_does")
        return 0
    print("VERDICT=pressure_field_was_smooth_without_a_mixed_space")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
