"""Tier-2 for fenics thermal_structural#7: two-way thermo-mechanical coupling
needs a real feedback term before Picard iteration means anything, and once it is
there the size of the effect -- not any error message -- is what matters.

The feedback used here is conduction on the DEFORMED configuration pulled back to
the reference mesh: F = Identity(d) + grad(u), J = det(F),
a_T = kappa*J*inner(inv(F).T*grad(T), inv(F).T*grad(s))*dx, alternating with a
thermo-elastic solve for u. Nothing raises either way. Measured on a clamped
square, kappa = 45 W/m/K, steel constants:

* the relative change norm(T_new - T_old)/norm(T_new), computed with
  fem.assemble_scalar, falls geometrically -- roughly 1.6e-1, 4.1e-5, 5.2e-9,
  4.5e-13 at a thermal strain of 1.2e-3;
* the relative error of the single non-iterated pass, measured against the
  converged loop, is proportional to the thermal strain alpha*(T - T_ref) and is
  about 0.19 times it -- 2.3e-4 at alpha*dT = 1.2e-3 and 5.7e-3 at
  alpha*dT = 3.0e-2;
* so for metals one pass is already good to several digits, and the loop only
  pays for itself when the thermal strain reaches a few percent. The older
  "one-shot error of order alpha*dT*L" wording predicts a relative error of order
  one and overstates the measurement by roughly the inverse of the thermal
  strain.

Mutation control: T2_MUTATE=1 runs the full Picard loop for the model under test
instead of a single pass, and the measured one-shot error collapses to round-off.
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

E, NU, KAPPA, T_REF = 210e9, 0.3, 45.0, 300.0
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
N_CONVERGED = 6


def coupled(alpha, d_temp, n_passes, tag):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 12, 12)
    d = 2
    beta = (3 * LAM + 2 * MU) * alpha
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (d,)))
    S = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    temp = dolfinx.fem.Function(S)
    t_old = dolfinx.fem.Function(S)
    temp.x.array[:] = T_REF

    msh.topology.create_connectivity(d - 1, d)
    left = dolfinx.mesh.locate_entities_boundary(
        msh, d - 1, lambda x: np.isclose(x[0], 0.0))
    right = dolfinx.mesh.locate_entities_boundary(
        msh, d - 1, lambda x: np.isclose(x[0], 1.0))
    bc_t = [
        dolfinx.fem.dirichletbc(
            dolfinx.fem.Constant(msh, T_REF + d_temp),
            dolfinx.fem.locate_dofs_topological(S, d - 1, left), S),
        dolfinx.fem.dirichletbc(
            dolfinx.fem.Constant(msh, T_REF),
            dolfinx.fem.locate_dofs_topological(S, d - 1, right), S)]
    bc_u = [dolfinx.fem.dirichletbc(
        np.zeros(d),
        dolfinx.fem.locate_dofs_topological(V, d - 1, left), V)]

    t_trial, s = ufl.TrialFunction(S), ufl.TestFunction(S)
    f_def = ufl.Identity(d) + ufl.grad(u)
    jac = ufl.det(f_def)
    f_inv = ufl.inv(f_def)
    a_t = KAPPA * jac * ufl.inner(f_inv.T * ufl.grad(t_trial),
                                  f_inv.T * ufl.grad(s)) * ufl.dx
    l_t = dolfinx.fem.Constant(msh, 0.0) * s * ufl.dx

    u_trial, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    eps = lambda w: ufl.sym(ufl.grad(w))  # noqa: E731
    a_u = ufl.inner(2 * MU * eps(u_trial)
                    + LAM * ufl.tr(eps(u_trial)) * ufl.Identity(d),
                    eps(v)) * ufl.dx
    l_u = beta * (temp - T_REF) * ufl.div(v) * ufl.dx

    changes = []
    for k in range(n_passes):
        t_old.x.array[:] = temp.x.array
        p_t = dolfinx.fem.petsc.LinearProblem(
            a_t, l_t, bcs=bc_t, petsc_options_prefix=f"{tag}T{k}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        th = p_t.solve()
        if isinstance(th, tuple):
            th = th[0]
        temp.x.array[:] = th.x.array
        num = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form((temp - t_old) ** 2 * ufl.dx))
        den = dolfinx.fem.assemble_scalar(dolfinx.fem.form(temp ** 2 * ufl.dx))
        changes.append(float(np.sqrt(num / den)))
        p_u = dolfinx.fem.petsc.LinearProblem(
            a_u, l_u, bcs=bc_u, petsc_options_prefix=f"{tag}U{k}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = p_u.solve()
        if isinstance(uh, tuple):
            uh = uh[0]
        u.x.array[:] = uh.x.array
    return np.array(u.x.array), changes


def main() -> int:
    passes = N_CONVERGED if MUTATE else 1
    print(f"passes_of_the_model_under_test={passes}")
    ratios = {}
    geometric = True
    for alpha, d_temp in ((1.2e-5, 100.0), (1.2e-5, 2500.0)):
        strain = alpha * d_temp
        tag = f"t2_ts7_{int(strain * 1e6)}_"
        u_test, _ = coupled(alpha, d_temp, passes, tag + "a")
        u_conv, changes = coupled(alpha, d_temp, N_CONVERGED, tag + "b")
        err = float(np.linalg.norm(u_test - u_conv)
                    / np.linalg.norm(u_conv))
        ratios[strain] = err / strain
        print(f"thermal_strain={strain:.3e} one_shot_relative_error={err:.3e} "
              f"error_over_thermal_strain={err / strain:.4f}")
        print(f"  picard_relative_changes="
              f"{[f'{c:.2e}' for c in changes[:4]]}")
        for a, b in zip(changes[:3], changes[1:4]):
            geometric &= b < 0.5 * a

    keys = sorted(ratios)
    in_band = all(0.05 < ratios[k] < 0.5 for k in keys)
    steady = abs(ratios[keys[1]] / ratios[keys[0]] - 1.0) < 0.5
    print(f"picard_change_falls_geometrically={geometric}")
    print(f"one_shot_error_is_about_a_tenth_of_the_thermal_strain={in_band}")
    print(f"error_over_strain_is_the_same_at_both_load_levels={steady}")

    metal_err = ratios[keys[0]] * keys[0]
    print(f"metal_case_one_shot_relative_error={metal_err:.3e}")
    cheap = metal_err < 1e-3
    print(f"single_pass_is_good_to_several_digits_for_metals={cheap}")
    print(f"older_order_alpha_dT_L_estimate_would_be_relative_order_one="
          f"{metal_err < 0.05}")

    if geometric and in_band and steady and cheap:
        print("VERDICT=one_shot_error_scales_with_the_thermal_strain")
        return 0
    print("VERDICT=one_shot_error_does_not_scale_with_the_thermal_strain")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
