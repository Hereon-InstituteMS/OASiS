"""Tier-2 for fenics fracture#7: neither problem.solve() raises on failure, so an
unchecked staggered loop will happily iterate on a diverged displacement field.
The loop keeps running and prints plausible damage values while SNES has been
returning a negative converged reason for every sweep. Put
`assert pu.solver.getConvergedReason() > 0` immediately after the displacement
solve.

Wrong variant: the staggered loop with no check on the solver status. Right
variant (used by the mutant): the same loop with the assertion after the
displacement solve.

Force-controlled staggered AT2 phase-field on a 16x16 unit square, P1 displacement
and P1 damage, E = 210, nu = 0.3, Gc = 2.7e-3, l0 = 2h = 0.125, k_res = 0, a
pre-crack seeded as a large history value along y = 0.5 for x < 0.5, the bottom
edge clamped and a vertical traction ramped over twelve steps on the top edge,
four staggered sweeps per step, SNES + MUMPS LU for the displacement step and
MUMPS LU for the damage step. Under force control the crack runs unstably, the
degraded stiffness of the separated specimen becomes singular, and the
displacement problem stops being solvable.

Observed on dolfinx 0.10.0 / PETSc 3.24.5: the first two load steps converge with
SNES reason 3, the third turns at its fourth sweep (reason -5, DIVERGED_MAX_IT),
and the remaining nine load steps report a negative reason (-5 / -6,
DIVERGED_LINE_SEARCH) on every one of their four sweeps. NOTHING is raised by
either solve(), the damage solve keeps returning KSP reason 4, and the printed
damage stays entirely plausible -- max(d) = 1.0000, min(d) around 8e-02, every
value finite -- while the displacement field it was computed from reaches 1e+27 to
1e+28. IMPORTANT, and contrary to the second half of the claim: the regularised
surface energy is non-decreasing at every one of these steps (2.2390e-03 rising
monotonically to 8.7745e-03), so the cheap reference-free check the claim
recommends does NOT catch this failure; only the converged reason does. Exact
reason codes and the step at which the divergence starts move a little between
runs (MUMPS on a nearly singular operator), so the fixture asserts only that
whole load steps diverge, that nothing raises and that the damage still looks
sane.

Mutation control: T2_MUTATE=1 runs the same loop with
`assert pu.solver.getConvergedReason() > 0` after the displacement solve; the
AssertionError fires at the first diverged sweep, the loop stops there, and the
tokens that describe running to the end go missing.
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

from dolfinx import fem, mesh  # noqa: E402

DTYPE = dolfinx.default_scalar_type
E, NU, GC = 210.0, 0.3, 2.7e-3
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))
N, DEGREE, K_RES = 16, 1, 0.0
SWEEPS = 4
TRACTIONS = np.linspace(0.05, 1.2, 12)


def build():
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N)
    h = 1.0 / N
    l0 = 2.0 * h
    V = fem.functionspace(msh, ("Lagrange", DEGREE, (2,)))
    D = fem.functionspace(msh, ("Lagrange", DEGREE))
    Q = fem.functionspace(msh, ("DG", 0))
    u, d = fem.Function(V), fem.Function(D)
    H, Hn = fem.Function(Q), fem.Function(Q)
    msh.topology.create_connectivity(2, 0)
    conn = msh.topology.connectivity(2, 0)
    nc = msh.topology.index_map(2).size_local
    mids = np.array([msh.geometry.x[conn.links(c)].mean(axis=0)
                     for c in range(nc)])
    seed = np.where((np.abs(mids[:, 1] - 0.5) < 0.55 * h)
                    & (mids[:, 0] < 0.5))[0]
    H.x.array[:] = 0.0
    H.x.array[seed] = 1e3 * GC / (2 * l0)

    msh.topology.create_connectivity(1, 2)
    top = mesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[1], 1.0))
    bot = mesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[1], 0.0))
    ft = mesh.meshtags(msh, 1, np.sort(top), np.full(len(top), 1, np.int32))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
    trac = fem.Constant(msh, np.zeros(2, dtype=DTYPE))

    v = ufl.TestFunction(V)
    g = (1 - d) ** 2 + K_RES
    e = ufl.sym(ufl.grad(u))
    sig = LAM * ufl.tr(e) * ufl.Identity(2) + 2 * MU * e
    F = (g * ufl.inner(sig, ufl.sym(ufl.grad(v))) * ufl.dx
         - ufl.dot(trac, v) * ds(1))
    u_bot = fem.Function(V)
    bcs_u = [fem.dirichletbc(u_bot, fem.locate_dofs_topological(V, 1, bot))]
    pu = dolfinx.fem.petsc.NonlinearProblem(
        F, u, bcs=bcs_u, petsc_options_prefix="t2f7u_",
        petsc_options={"snes_type": "newtonls", "ksp_type": "preonly",
                       "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
    w, dd = ufl.TestFunction(D), ufl.TrialFunction(D)
    a_d = ((2.0 * H + GC / l0) * dd * w
           + GC * l0 * ufl.dot(ufl.grad(dd), ufl.grad(w))) * ufl.dx
    L_d = 2.0 * H * w * ufl.dx
    pd = dolfinx.fem.petsc.LinearProblem(
        a_d, L_d, bcs=[], u=d, petsc_options_prefix="t2f7d_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})
    eu = ufl.sym(ufl.grad(u))
    psi_expr = fem.Expression(0.5 * LAM * ufl.tr(eu) ** 2 + MU * ufl.inner(eu, eu),
                              Q.element.interpolation_points)
    surf = fem.form(GC / (2 * l0) * (d ** 2 + l0 ** 2
                                     * ufl.dot(ufl.grad(d), ufl.grad(d))) * ufl.dx)
    return dict(u=u, d=d, H=H, Hn=Hn, pu=pu, pd=pd, psi=psi_expr, surf=surf,
                trac=trac)


def main() -> int:
    m = build()
    m["pd"].solve()
    if MUTATE:
        print("mutation=loop_asserts_the_displacement_converged_reason")
    raised_u, raised_d = "", ""
    guard_msg = ""
    rows = []
    first_bad = None
    for i, t in enumerate(TRACTIONS):
        m["trac"].value[1] = t
        reasons, d_reasons = [], []
        for k in range(SWEEPS):
            try:
                m["pu"].solve()
            except Exception as exc:  # noqa: BLE001
                raised_u = f"{type(exc).__name__}: {str(exc)[:70]}"
            r = int(m["pu"].solver.getConvergedReason())
            reasons.append(r)
            if r <= 0 and first_bad is None:
                first_bad = (i + 1, k + 1)
            if MUTATE:
                # the guard the claim asks for
                try:
                    assert r > 0, ("displacement solve diverged, SNES reason "
                                   f"{r}")
                except AssertionError as exc:
                    guard_msg = f"AssertionError: {exc}"
                    print(f"guarded_loop_stopped_at_step={i + 1} sweep={k + 1} "
                          f"-> {guard_msg}")
                    break
            m["Hn"].interpolate(m["psi"])
            m["H"].x.array[:] = np.maximum(m["H"].x.array, m["Hn"].x.array)
            try:
                m["pd"].solve()
            except Exception as exc:  # noqa: BLE001
                raised_d = f"{type(exc).__name__}: {str(exc)[:70]}"
            d_reasons.append(int(m["pd"].solver.getConvergedReason()))
        rows.append(dict(t=float(t), reasons=reasons, d_reasons=d_reasons,
                         dmax=float(m["d"].x.array.max()),
                         dmin=float(m["d"].x.array.min()),
                         surf=float(fem.assemble_scalar(m["surf"])),
                         umax=float(np.abs(m["u"].x.array).max()),
                         finite=bool(np.all(np.isfinite(m["d"].x.array)))))
        print(f"step={i + 1:2d} traction={t:.3f} displacement_reasons={reasons} "
              f"damage_reasons={d_reasons} max_d={rows[-1]['dmax']:.4f} "
              f"min_d={rows[-1]['dmin']:.2e} surface_energy={rows[-1]['surf']:.4e} "
              f"max_abs_u={rows[-1]['umax']:.3e}")
        if guard_msg:
            break

    if MUTATE:
        print(f"guard_fired={bool(guard_msg)}")
        print("VERDICT=guarded_loop_stopped_on_the_first_diverged_solve")
        return 0

    bad_steps = [r for r in rows if r["reasons"] and all(x <= 0 for x in r["reasons"])]
    surfs = [r["surf"] for r in rows]
    plausible = all(0.0 <= r["dmin"] and r["dmax"] <= 1.05 and r["finite"]
                    for r in bad_steps)
    garbage_u = any(r["umax"] > 1e10 for r in bad_steps)
    non_decreasing = all(b >= a - 1e-15 for a, b in zip(surfs, surfs[1:]))
    print(f"first_step_and_sweep_with_a_negative_reason={first_bad}")
    print(f"number_of_load_steps_whose_every_sweep_diverged={len(bad_steps)}")
    print(f"displacement_solve_never_raised={raised_u == ''}")
    print(f"damage_solve_never_raised={raised_d == ''}")
    print(f"whole_sweep_diverged_on_at_least_three_load_steps={len(bad_steps) >= 3}")
    print(f"damage_values_stayed_plausible_while_the_solve_diverged={plausible}")
    print(f"displacement_field_exceeded_1e10_on_those_steps={garbage_u}")
    print(f"surface_energy_stayed_non_decreasing_so_that_check_missed_it="
          f"{non_decreasing}")
    if raised_u == "" and raised_d == "" and len(bad_steps) >= 3 and plausible \
            and garbage_u and first_bad is not None:
        print("VERDICT=unchecked_staggered_loop_ran_on_a_diverged_displacement_field")
        return 0
    print("VERDICT=the_staggered_loop_noticed_the_failure")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
