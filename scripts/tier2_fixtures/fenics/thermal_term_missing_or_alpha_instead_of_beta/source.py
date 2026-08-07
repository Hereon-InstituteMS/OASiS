"""Tier-2 for fenics thermal_structural#0: temperature enters the elasticity
form only through the thermal term, and that term must carry the thermal stress
modulus beta = (3*lam + 2*mu)*alpha, not alpha.

Two wrong variants are run against the same clamped-left square with
T = 300 + 100*x and T_ref = 300, SI steel constants.
(a) Thermal term omitted: nothing is raised and the returned displacement is
    BIT-EXACTLY zero -- np.all(uh.x.array == 0.0) is True -- but only because
    nothing else loads the body. Adding gravity to the same omitted-term model
    gives a non-zero displacement that is silently just the isothermal answer,
    so "zero displacement" is a tell only for a thermally-loaded-only model.
(b) alpha used where (3*lam + 2*mu)*alpha belongs: nothing is raised and
    max|u| falls by exactly the factor 3*lam + 2*mu (5.25e11 for these
    constants), i.e. to the 1e-15 range, which reads as zero on any plot.

Mutation control: T2_MUTATE=1 puts beta*(T - T_ref)*div(v)*dx into L in both
variants, so both become the correct model.
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

E, NU, ALPHA = 210e9, 0.3, 1.2e-5
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
BETA = (3 * LAM + 2 * MU) * ALPHA


def solve(mode: str, gravity: bool = False):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 12, 12)
    d = msh.geometry.dim
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (d,)))
    S = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    T = dolfinx.fem.Function(S)
    T.interpolate(lambda x: 300.0 + 100.0 * x[0])
    t_ref = dolfinx.fem.Constant(msh, 300.0)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    eps = lambda w: ufl.sym(ufl.grad(w))  # noqa: E731
    a = ufl.inner(2 * MU * eps(u) + LAM * ufl.tr(eps(u)) * ufl.Identity(d),
                  eps(v)) * ufl.dx
    L = ufl.inner(dolfinx.fem.Constant(msh, (0.0,) * d), v) * ufl.dx
    if gravity:
        L = L + ufl.inner(dolfinx.fem.Constant(msh, (0.0, -7800.0 * 9.81)),
                          v) * ufl.dx
    if mode == "correct":
        L = L + BETA * (T - t_ref) * ufl.div(v) * ufl.dx
    elif mode == "alpha_only":
        L = L + ALPHA * (T - t_ref) * ufl.div(v) * ufl.dx
    elif mode != "omitted":
        raise ValueError(mode)

    msh.topology.create_connectivity(d - 1, d)
    left = dolfinx.mesh.locate_entities_boundary(
        msh, d - 1, lambda x: np.isclose(x[0], 0.0))
    bc = dolfinx.fem.dirichletbc(
        np.zeros(d), dolfinx.fem.locate_dofs_topological(V, d - 1, left), V)
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix="t2_ts0_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = prob.solve()
    if isinstance(uh, tuple):
        uh = uh[0]
    arr = np.array(uh.x.array)
    return float(np.max(np.abs(arr))), bool(np.all(arr == 0.0))


def main() -> int:
    mode_a = "correct" if MUTATE else "omitted"
    mode_b = "correct" if MUTATE else "alpha_only"

    ref, _ = solve("correct")
    ma, zero_a = solve(mode_a)
    mb, _ = solve(mode_b)
    grav_wrong, _ = solve(mode_a, gravity=True)
    grav_right, _ = solve("correct", gravity=True)

    print(f"correct_max_abs_u={ref:.6e}")
    print(f"variant_a_mode={mode_a} max_abs_u={ma:.6e}")
    print(f"variant_b_mode={mode_b} max_abs_u={mb:.6e}")
    print(f"gravity_wrong_max_abs_u={grav_wrong:.6e} "
          f"gravity_correct_max_abs_u={grav_right:.6e}")

    exact_zero = zero_a and ref > 1e-6
    print(f"omitted_thermal_term_is_bit_exactly_zero={exact_zero}")

    shrink = ref / mb if mb > 0.0 else float("inf")
    print(f"correct_over_alpha_only_ratio={shrink:.6e}")
    ok_shrink = abs(shrink / (3 * LAM + 2 * MU) - 1.0) < 1e-6
    print(f"alpha_only_is_smaller_by_3lam_plus_2mu={ok_shrink}")
    print(f"alpha_only_reads_as_zero={mb < 1e-12}")

    hidden = grav_wrong > 0.0 and abs(grav_wrong / grav_right - 1.0) > 0.5
    print(f"body_force_hides_the_missing_thermal_term={hidden}")

    if exact_zero and ok_shrink and mb < 1e-12 and hidden:
        print("VERDICT=thermal_term_must_carry_3lam_plus_2mu_times_alpha")
        return 0
    print("VERDICT=thermal_term_errors_were_visible")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
