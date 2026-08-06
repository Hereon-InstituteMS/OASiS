"""Tier-2 for fenics fracture#0: irreversibility is not optional and its absence is
not a small effect. Enforce it with a history field updated by an element-wise
maximum after every displacement solve,
H.x.array[:] = np.maximum(H.x.array, Hn.x.array), where Hn holds the interpolated
tensile energy density.

Wrong variant: drive the damage problem with the CURRENT tensile energy density
(the seeded pre-crack is still kept as a floor, so only the accumulation is
dropped). Right variant: the element-wise maximum against the previous history.

Staggered AT2 phase-field on a 20x20 unit square, P1 displacement and P1 damage,
E = 210, nu = 0.3, Gc = 2.7e-3, l0 = 2h = 0.1, a pre-crack seeded as a large
history value on the cells of a band at y = 0.5 with x < 0.5, displacement control
on the top edge: eight steps up to 6.4e-3 and eight steps back down to zero.

Observed on dolfinx 0.10.0: WITHOUT the history update the worst per-step change
of a single nodal damage value on the unloading path is -0.8084, i.e. a node goes
from fully damaged to nearly undamaged inside one load step, and the damage
integrated over the body falls from a peak of 0.27063 to 0.16559 -- it loses
38.81% of its peak, "roughly forty percent" -- by the time the load is back at
zero, so the crack visibly heals. WITH the history update the same integral loses
exactly nothing during unloading (peak 0.32380, final 0.32380, fraction 0.0000).
The claim's caveat also reproduces: even with the history field the worst nodal
change while the load is still INCREASING is -0.0021, the crack redistributing
rather than healing.

Mutation control: T2_MUTATE=1 gives the run under test the history update, so the
healing tokens go False.
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
N = 20
K_RES = 1e-6
U_MAX = 6.4e-3
N_UP = 8


def build(n: int):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n)
    h = 1.0 / n
    l0 = 2.0 * h
    V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    D = fem.functionspace(msh, ("Lagrange", 1))
    Q = fem.functionspace(msh, ("DG", 0))
    u, d = fem.Function(V), fem.Function(D)
    H, Hn, H_seed = fem.Function(Q), fem.Function(Q), fem.Function(Q)

    msh.topology.create_connectivity(2, 0)
    conn = msh.topology.connectivity(2, 0)
    nc = msh.topology.index_map(2).size_local
    mids = np.array([msh.geometry.x[conn.links(c)].mean(axis=0)
                     for c in range(nc)])
    seed = (np.abs(mids[:, 1] - 0.5) < 0.55 * h) & (mids[:, 0] < 0.5)
    H_seed.x.array[:] = 0.0
    H_seed.x.array[np.where(seed)[0]] = 1e3 * GC / (2 * l0)
    H.x.array[:] = H_seed.x.array

    v, du = ufl.TestFunction(V), ufl.TrialFunction(V)
    g = (1 - d) ** 2 + K_RES
    e = ufl.sym(ufl.grad(du))
    sig = LAM * ufl.tr(e) * ufl.Identity(2) + 2 * MU * e
    a_u = g * ufl.inner(sig, ufl.sym(ufl.grad(v))) * ufl.dx
    L_u = ufl.inner(fem.Constant(msh, np.zeros(2, dtype=DTYPE)), v) * ufl.dx
    msh.topology.create_connectivity(1, 2)
    top = mesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[1], 1.0))
    bot = mesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[1], 0.0))
    u_top, u_bot = fem.Function(V), fem.Function(V)
    bcs_u = [fem.dirichletbc(u_top, fem.locate_dofs_topological(V, 1, top)),
             fem.dirichletbc(u_bot, fem.locate_dofs_topological(V, 1, bot))]

    w, dd = ufl.TestFunction(D), ufl.TrialFunction(D)
    a_d = ((2.0 * H + GC / l0) * dd * w
           + GC * l0 * ufl.dot(ufl.grad(dd), ufl.grad(w))) * ufl.dx
    L_d = 2.0 * H * w * ufl.dx

    pu = dolfinx.fem.petsc.LinearProblem(
        a_u, L_u, bcs=bcs_u, u=u, petsc_options_prefix="t2f0u_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})
    pd = dolfinx.fem.petsc.LinearProblem(
        a_d, L_d, bcs=[], u=d, petsc_options_prefix="t2f0d_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})
    eu = ufl.sym(ufl.grad(u))
    psi = 0.5 * LAM * ufl.tr(eu) ** 2 + MU * ufl.inner(eu, eu)
    psi_expr = fem.Expression(psi, Q.element.interpolation_points)
    dint = fem.form(d * ufl.dx)
    surf = fem.form(GC / (2 * l0) * (d ** 2 + l0 ** 2
                                     * ufl.dot(ufl.grad(d), ufl.grad(d))) * ufl.dx)
    return dict(u=u, d=d, H=H, Hn=Hn, H_seed=H_seed, pu=pu, pd=pd,
                psi_expr=psi_expr, dint=dint, surf=surf, u_top=u_top)


def load_step(m, disp: float, history: bool, max_sweeps=400, tol=1e-4) -> int:
    m["u_top"].x.array[:] = 0.0
    m["u_top"].x.array[1::2] = disp
    prev = m["d"].x.array.copy()
    for k in range(max_sweeps):
        prev[:] = m["d"].x.array
        m["pu"].solve()
        m["Hn"].interpolate(m["psi_expr"])
        if history:
            m["H"].x.array[:] = np.maximum(m["H"].x.array, m["Hn"].x.array)
        else:
            # no accumulation: only the seeded pre-crack survives as a floor
            m["H"].x.array[:] = np.maximum(m["H_seed"].x.array, m["Hn"].x.array)
        m["pd"].solve()
        if np.max(np.abs(m["d"].x.array - prev)) < tol:
            return k + 1
    return max_sweeps


def run(history: bool) -> dict:
    m = build(N)
    m["pd"].solve()          # seed the pre-crack into d
    path = ([U_MAX * (i + 1) / N_UP for i in range(N_UP)]
            + [U_MAX * (N_UP - 1 - i) / N_UP for i in range(N_UP)])
    prev = m["d"].x.array.copy()
    worst_load, worst_unload = 0.0, 0.0
    peak, series = 0.0, []
    for i, disp in enumerate(path):
        load_step(m, disp, history)
        drop = float(np.min(m["d"].x.array - prev))
        prev = m["d"].x.array.copy()
        di = float(fem.assemble_scalar(m["dint"]))
        series.append(di)
        peak = max(peak, di)
        if i < N_UP:
            worst_load = min(worst_load, drop)
        else:
            worst_unload = min(worst_unload, drop)
    unloading = series[N_UP - 1:]
    lost = (peak - series[-1]) / peak
    lost_during_unload = max(0.0, (unloading[0] - min(unloading)) / unloading[0])
    return dict(peak=peak, final=series[-1], lost=lost,
                lost_during_unload=lost_during_unload,
                worst_load=worst_load, worst_unload=worst_unload,
                dmax=float(m["d"].x.array.max()),
                surf=float(fem.assemble_scalar(m["surf"])))


def show(tag: str, r: dict) -> None:
    print(f"{tag}: peak_integrated_damage={r['peak']:.5f} "
          f"integrated_damage_at_zero_load={r['final']:.5f} "
          f"fraction_lost={r['lost']:.4f}")
    print(f"{tag}: worst_nodal_step_change_while_loading={r['worst_load']:+.4f} "
          f"worst_nodal_step_change_while_unloading={r['worst_unload']:+.4f}")


def main() -> int:
    tested = run(history=MUTATE)
    show("under_test", tested)
    if MUTATE:
        print("mutation=run_under_test_uses_the_history_maximum")
    good = run(history=True)
    show("with_history", good)

    heals_node = tested["worst_unload"] <= -0.8
    heals_body = tested["lost"] > 0.3
    good_keeps = good["lost_during_unload"] < 1e-12
    caveat = -0.05 < good["worst_load"] < 0.0
    print(f"no_history_worst_nodal_step_change_is_near_minus_one={heals_node}")
    print(f"no_history_integrated_damage_lost_over_thirty_percent_of_its_peak="
          f"{heals_body}")
    print(f"with_history_integrated_damage_loses_nothing_while_unloading="
          f"{good_keeps}")
    print(f"with_history_small_nodal_decreases_still_occur_while_loading={caveat}")
    if heals_node and heals_body and good_keeps and caveat:
        print("VERDICT=without_the_history_field_the_crack_heals")
        return 0
    print("VERDICT=damage_stayed_irreversible_without_the_history_field")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
