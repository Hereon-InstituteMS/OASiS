"""Tier-2 for fenics cahn_hilliard#4: build the residual from ufl.split(u) and
ufl.split(u0), NEVER from u.sub(0)/u.sub(1). The u.sub(i) spelling is caught by
nothing -- dolfinx.fem.form() succeeds and NonlinearProblem is constructed --
and the damage only surfaces at solve time.

Root cause, measured: u.sub(0) is a separate UFL Coefficient, so the residual
does not depend on u at all and ufl.derivative(F, u) assembles to a matrix of
norm exactly 0.0 (the ufl.split residual gives a nonzero one).

Observed signal on dolfinx 0.10.0 / PETSc 3.24.5: solve() RAISES
`petsc4py.PETSc.Error: error code 73` out of PCSetUp_LU /
MatLUFactorSymbolic_SeqAIJ with the text "Matrix is missing diagonal entry 0",
and the concentration is left exactly at its initial values. NOTE the knowledge
text quotes "DIVERGED_PC_FAILED", "PC failed due to FACTOR_OTHER" and
"DIVERGED_LINEAR_SOLVE" with SNES reason -3 / KSP reason -11; none of those
appear on this install -- the all-zero tangent fails in the SYMBOLIC
factorisation and PETSc raises instead of reporting a diverged reason. The
fixture pins the behaviour that actually occurs.

Mutation control: T2_MUTATE=1 builds the same residual with ufl.split, so the
Jacobian is nonzero and the solve converges.
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

N, LMBDA, MOB, THETA, DT = 24, 1.0e-2, 1.0, 0.5, 5.0e-6


def build(tag: str, use_sub: bool):
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
    if use_sub:
        c, mu = u.sub(0), u.sub(1)
        c0, mu0 = u0.sub(0), u0.sub(1)
    else:
        c, mu = ufl.split(u)
        c0, mu0 = ufl.split(u0)
    cv = ufl.variable(c)
    dfdc = ufl.diff(100.0 * cv**2 * (1 - cv) ** 2, cv)
    mu_mid = (1.0 - THETA) * mu0 + THETA * mu
    F = ((c - c0) * q * ufl.dx
         + DT * MOB * ufl.dot(ufl.grad(mu_mid), ufl.grad(q)) * ufl.dx
         + mu * v * ufl.dx - dfdc * v * ufl.dx
         - LMBDA * ufl.dot(ufl.grad(c), ufl.grad(v)) * ufl.dx)
    return msh, ME, u, u0, F, tag


def jacobian_norm(F, u) -> float:
    J = dolfinx.fem.petsc.assemble_matrix(
        dolfinx.fem.form(ufl.derivative(F, u)))
    J.assemble()
    return float(J.norm())


def try_step(msh, ME, u, u0, F, tag):
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, u, petsc_options_prefix=f"t2_ch4_{tag}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "snes_max_it": 30})
    print(f"{tag}: nonlinearproblem_constructed=True")
    _, cdofs = ME.sub(0).collapse()
    cdofs = np.asarray(cdofs, dtype=np.int32)
    before = u.x.array[cdofs].copy()
    raised = ""
    reason = None
    try:
        u0.x.array[:] = u.x.array
        prob.solve()
        u.x.scatter_forward()
        reason = prob.solver.getConvergedReason()
    except Exception as exc:
        raised = f"{type(exc).__module__}.{type(exc).__name__}: {exc}"
    moved = float(np.abs(u.x.array[cdofs] - before).max())
    return raised, reason, moved


def main() -> int:
    wrong = build("u_sub", use_sub=not MUTATE)
    print("u_sub_residual: compiling with dolfinx.fem.form ...")
    dolfinx.fem.form(wrong[4])
    print("fem_form_accepted_the_u_sub_residual=True")
    n_wrong = jacobian_norm(wrong[4], wrong[2])
    right = build("split", use_sub=False)
    n_right = jacobian_norm(right[4], right[2])
    print(f"jacobian_norm_u_sub={n_wrong:.3e} "
          f"jacobian_norm_ufl_split={n_right:.6f}")
    zero_J = n_wrong == 0.0
    good_J = n_right > 0.0
    print(f"u_sub_jacobian_is_exactly_zero={zero_J}")
    print(f"ufl_split_jacobian_is_nonzero={good_J}")

    raised_w, reason_w, moved_w = try_step(*wrong)
    print(f"u_sub_solve_raised={raised_w if raised_w else False}")
    print(f"u_sub_field_max_change={moved_w:.3e}")
    raised_r, reason_r, moved_r = try_step(*right)
    print(f"ufl_split_solve_raised={raised_r if raised_r else False}")
    print(f"ufl_split_reason={reason_r} field_max_change={moved_r:.3e}")

    lower = raised_w.lower()
    claimed = ("diverged_pc_failed" in lower or "factor_other" in lower
               or "diverged_linear_solve" in lower)
    print(f"claimed_diverged_pc_failed_text_present={claimed}")
    hard = ("error code 73" in lower
            and "missing diagonal entry" in lower)
    print(f"u_sub_solve_raised_a_hard_petsc_error={hard}")
    print(f"u_sub_left_the_concentration_untouched={moved_w == 0.0}")
    ok_right = (not raised_r) and reason_r is not None and reason_r > 0 \
        and moved_r > 0.0
    print(f"ufl_split_step_converged_and_moved_the_field={ok_right}")

    if zero_J and good_J and hard and moved_w == 0.0 and ok_right \
            and not claimed:
        print("VERDICT=u_sub_residual_compiles_then_dies_with_a_zero_tangent")
        return 0
    print("VERDICT=u_sub_residual_was_fine")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
