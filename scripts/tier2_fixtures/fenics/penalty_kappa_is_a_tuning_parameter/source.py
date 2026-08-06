"""Tier-2 for fenics nearly_incompressible_elasticity#3: the penalty method
(large kappa) is an alternative to the mixed formulation but introduces
parameter sensitivity - too small and the volumetric constraint is not enforced
(det(F) - 1 away from 0), too large and the Jacobian becomes ill-conditioned.
The mixed (u, p) method is parameter-free.

A plane-strain Neo-Hookean square is stretched 20% by Dirichlet data, with
psi = mu/2 (I_C - 2 - 2 ln J) + kappa/2 (J-1)^2 and P = dpsi/dF, solved with
SNES/LU for kappa/mu = 1e0 ... 1e14. For each kappa the fixture reports the
volumetric error sqrt(mean (J-1)^2), the SNES converged reason and the
condition number of the assembled Jacobian at the solution.

Observed: 14.4% volumetric error at kappa/mu = 1, falling to a P1 floor of about
2.9%; condition number rising from 1.9e2 to 9.6e14; SNES line search failing
(reason -6) at kappa/mu = 1e14. Note the failure is NOT monotone in kappa - the
line search also failed at kappa/mu = 1e4 while 1e8 and 1e12 converged - so a
single successful run at one kappa says nothing about the next one.

Mutation control: T2_MUTATE=1 solves the same problem with the parameter-free
mixed (u, p) Taylor-Hood formulation, which has no kappa to tune.
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

MU = 1.0
STRETCH = 0.2
N = 8
KAPPAS = (1.0e0, 1.0e2, 1.0e4, 1.0e8, 1.0e12, 1.0e14)


def grid():
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    d = msh.geometry.dim
    msh.topology.create_connectivity(d - 1, d)
    left = dolfinx.mesh.locate_entities_boundary(
        msh, d - 1, lambda x: np.isclose(x[0], 0.0))
    right = dolfinx.mesh.locate_entities_boundary(
        msh, d - 1, lambda x: np.isclose(x[0], 1.0))
    return msh, d, left, right


def volumetric_error(jac_expr, msh) -> float:
    area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(
        dolfinx.fem.Constant(msh, 1.0) * ufl.dx))
    dev = dolfinx.fem.assemble_scalar(dolfinx.fem.form(
        (jac_expr - 1.0) ** 2 * ufl.dx))
    return float(np.sqrt(abs(dev) / area))


def condition_number(form, bcs) -> float:
    A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(form), bcs=bcs)
    A.assemble()
    return float(np.linalg.cond(A.convert("dense").getDenseArray().copy()))


def penalty_run(kappa: float, tag: str):
    msh, d, left, right = grid()
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (d,)))
    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)
    fv = ufl.variable(ufl.Identity(d) + ufl.grad(u))
    jj = ufl.det(fv)
    psi = (MU / 2) * (ufl.tr(fv.T * fv) - d - 2 * ufl.ln(jj)) \
        + (kappa / 2) * (jj - 1.0) ** 2
    res = ufl.inner(ufl.diff(psi, fv), ufl.grad(v)) * ufl.dx
    bcs = [
        dolfinx.fem.dirichletbc(
            np.zeros(d),
            dolfinx.fem.locate_dofs_topological(V, d - 1, left), V),
        dolfinx.fem.dirichletbc(
            np.array([STRETCH, 0.0]),
            dolfinx.fem.locate_dofs_topological(V, d - 1, right), V),
    ]
    prob = dolfinx.fem.petsc.NonlinearProblem(
        res, u, bcs=bcs, petsc_options_prefix=tag,
        petsc_options={"snes_type": "newtonls", "ksp_type": "preonly",
                       "pc_type": "lu", "snes_max_it": 40,
                       "snes_rtol": 1e-9})
    raised = ""
    try:
        prob.solve()
    except Exception as exc:  # noqa: BLE001
        raised = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    reason = prob.solver.getConvergedReason()
    err = volumetric_error(ufl.det(ufl.Identity(d) + ufl.grad(u)), msh)
    cond = condition_number(ufl.derivative(res, u), bcs)
    return reason, err, cond, raised


def mixed_run(tag: str):
    msh, d, left, right = grid()
    P2 = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(d,))
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P2, P1]))
    w = dolfinx.fem.Function(W)
    u, p = ufl.split(w)
    fv = ufl.variable(ufl.Identity(d) + ufl.grad(u))
    jj = ufl.det(fv)
    energy = ((MU / 2) * (ufl.tr(fv.T * fv) - d - 2 * ufl.ln(jj))
              + p * (jj - 1.0)) * ufl.dx
    res = ufl.derivative(energy, w, ufl.TestFunction(W))
    V0, _ = W.sub(0).collapse()
    clamp = dolfinx.fem.Function(V0)
    pull = dolfinx.fem.Function(V0)
    pull.x.array[0::2] = STRETCH
    bcs = [
        dolfinx.fem.dirichletbc(
            clamp,
            dolfinx.fem.locate_dofs_topological(
                (W.sub(0), V0), d - 1, left), W.sub(0)),
        dolfinx.fem.dirichletbc(
            pull,
            dolfinx.fem.locate_dofs_topological(
                (W.sub(0), V0), d - 1, right), W.sub(0)),
    ]
    prob = dolfinx.fem.petsc.NonlinearProblem(
        res, w, bcs=bcs, petsc_options_prefix=tag,
        petsc_options={"snes_type": "newtonls", "ksp_type": "preonly",
                       "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
                       "snes_max_it": 40, "snes_rtol": 1e-9})
    raised = ""
    try:
        prob.solve()
    except Exception as exc:  # noqa: BLE001
        raised = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    reason = prob.solver.getConvergedReason()
    uh = w.sub(0)
    err = volumetric_error(ufl.det(ufl.Identity(d) + ufl.grad(uh)), msh)
    cond = condition_number(ufl.derivative(res, w), bcs)
    return reason, err, cond, raised


def main() -> int:
    if MUTATE:
        print("formulation=mixed_u_p_taylor_hood")
        reason, err, cond, raised = mixed_run("t2_nie3_mx_")
        print(f"mixed_reason={reason} volumetric_error={err:.6e} "
              f"condition_number={cond:.3e} raised={raised!r}")
        print(f"volume_error_exceeds_1_percent_somewhere={err > 1e-2}")
        print(f"condition_number_exceeds_1e12_somewhere={cond > 1e12}")
        print(f"newton_failed_at_some_setting={reason < 0 or bool(raised)}")
        print(f"parameter_sensitivity_observed="
              f"{(err > 1e-2) and (cond > 1e12)}")
        print("VERDICT=mixed_formulation_needs_no_penalty")
        return 0

    print("formulation=penalty")
    rows = []
    for kappa in KAPPAS:
        reason, err, cond, raised = penalty_run(
            kappa, f"t2_nie3_k{kappa:.0e}_")
        rows.append((kappa, reason, err, cond, raised))
        print(f"kappa_over_mu={kappa:.0e} snes_reason={reason} "
              f"volumetric_error={err:.6e} condition_number={cond:.3e} "
              f"raised={raised!r}")
    loose = rows[0]
    tight = rows[-1]
    loose_bad = loose[2] > 1e-2
    tight_bad = tight[3] > 1e12
    stalled_tight = tight[1] < 0 or bool(tight[4])
    stalled_any = any(r[1] < 0 or r[4] for r in rows)
    print(f"volume_error_exceeds_1_percent_somewhere={loose_bad}")
    print(f"condition_number_exceeds_1e12_somewhere={tight_bad}")
    print(f"newton_failed_at_some_setting={stalled_any}")
    print(f"newton_failed_at_the_largest_penalty={stalled_tight}")
    print(f"parameter_sensitivity_observed={loose_bad and tight_bad}")
    if loose_bad and tight_bad and stalled_any:
        print("VERDICT=penalty_kappa_trades_constraint_error_for_conditioning")
        return 0
    print("VERDICT=penalty_is_insensitive_to_kappa")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
