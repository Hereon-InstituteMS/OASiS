"""Tier-2 for fenics fracture#6: seeding a pre-crack by writing a large value into
the history field makes the damage overshoot one slightly near the seed, which is
harmless but will trip a naive assertion that d <= 1. It is the AT2 gradient term
smoothing a step change in the driving force, not a solver failure. Check
min(d) >= 0 and that the regularised surface energy increases, rather than
asserting a hard upper bound of one.

Wrong variant: seed the pre-crack as a large history value and then assert
d <= 1. Right variant (used by the mutant): seed the pre-crack as a Dirichlet
condition d = 1 on the notch nodes, which cannot overshoot.

Staggered AT2 phase-field on a 20x20 unit square, P1 displacement and P1 damage,
E = 210, nu = 0.3, Gc = 2.7e-3, l0 = 2h = 0.1, pre-crack along y = 0.5 for
x < 0.5, three displacement-controlled load steps.

Observed on dolfinx 0.10.0: the damage subproblem is linear and unconstrained, so
the seeded step in the driving force is smoothed by the Gc*l0*grad(d).grad(w) term
and max(d) comes out at 1.0187 immediately after seeding and stays above one at
every load step, i.e. 1.87% over the bound; min(d) never goes negative and the
regularised surface energy increases at every step. The overshoot node sits on the
boundary of the seeded band, and with the history field enabled the node showing
the worst non-monotone decrease of d during loading is in the same region. Seeding
the same pre-crack with a Dirichlet condition instead gives max(d) = 1.0 exactly.

Mutation control: T2_MUTATE=1 seeds the pre-crack with a Dirichlet condition, so
max(d) never exceeds one and the overshoot tokens go False.
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
STEPS = (2.0e-3, 4.0e-3, 6.0e-3)


def build(dirichlet_seed: bool):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N)
    h = 1.0 / N
    l0 = 2.0 * h
    V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    D = fem.functionspace(msh, ("Lagrange", 1))
    Q = fem.functionspace(msh, ("DG", 0))
    u, d = fem.Function(V), fem.Function(D)
    H, Hn = fem.Function(Q), fem.Function(Q)

    msh.topology.create_connectivity(2, 0)
    conn = msh.topology.connectivity(2, 0)
    nc = msh.topology.index_map(2).size_local
    mids = np.array([msh.geometry.x[conn.links(c)].mean(axis=0)
                     for c in range(nc)])
    seed_cells = np.where((np.abs(mids[:, 1] - 0.5) < 0.55 * h)
                          & (mids[:, 0] < 0.5))[0].astype(np.int32)
    H.x.array[:] = 0.0
    bcs_d = []
    if dirichlet_seed:
        one = fem.Function(D)
        one.x.array[:] = 1.0
        seed_dofs = fem.locate_dofs_topological(D, 2, seed_cells)
        bcs_d = [fem.dirichletbc(one, seed_dofs)]
    else:
        H.x.array[seed_cells] = 1e3 * GC / (2 * l0)

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
        a_u, L_u, bcs=bcs_u, u=u, petsc_options_prefix="t2f6u_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})
    pd = dolfinx.fem.petsc.LinearProblem(
        a_d, L_d, bcs=bcs_d, u=d, petsc_options_prefix="t2f6d_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})
    eu = ufl.sym(ufl.grad(u))
    psi = 0.5 * LAM * ufl.tr(eu) ** 2 + MU * ufl.inner(eu, eu)
    psi_expr = fem.Expression(psi, Q.element.interpolation_points)
    surf = fem.form(GC / (2 * l0) * (d ** 2 + l0 ** 2
                                     * ufl.dot(ufl.grad(d), ufl.grad(d))) * ufl.dx)
    return dict(msh=msh, D=D, d=d, H=H, Hn=Hn, pu=pu, pd=pd, psi_expr=psi_expr,
                surf=surf, u_top=u_top, h=h, l0=l0)


def sweep(m, disp, max_sweeps=200, tol=1e-4) -> int:
    m["u_top"].x.array[:] = 0.0
    m["u_top"].x.array[1::2] = disp
    prev = m["d"].x.array.copy()
    for k in range(max_sweeps):
        prev[:] = m["d"].x.array
        m["pu"].solve()
        m["Hn"].interpolate(m["psi_expr"])
        m["H"].x.array[:] = np.maximum(m["H"].x.array, m["Hn"].x.array)
        m["pd"].solve()
        if np.max(np.abs(m["d"].x.array - prev)) < tol:
            return k + 1
    return max_sweeps


def main() -> int:
    m = build(dirichlet_seed=MUTATE)
    if MUTATE:
        print("mutation=precrack_seeded_with_a_dirichlet_condition_on_d")
    m["pd"].solve()
    coords = m["D"].tabulate_dof_coordinates()
    d0 = float(m["d"].x.array.max())
    node0 = int(np.argmax(m["d"].x.array))
    print(f"l0={m['l0']:.4f} h={m['h']:.4f}")
    print(f"after_seeding_max_d={d0:.4f} min_d={float(m['d'].x.array.min()):.3e} "
          f"argmax_at=({coords[node0][0]:.4f}, {coords[node0][1]:.4f})")

    maxima, surfs, worst_drop_node = [], [], None
    worst_drop = 0.0
    prev = m["d"].x.array.copy()
    for i, disp in enumerate(STEPS):
        sw = sweep(m, disp)
        diff = m["d"].x.array - prev
        j = int(np.argmin(diff))
        if diff[j] < worst_drop:
            worst_drop, worst_drop_node = float(diff[j]), j
        prev = m["d"].x.array.copy()
        dmax = float(m["d"].x.array.max())
        dmin = float(m["d"].x.array.min())
        gs = float(fem.assemble_scalar(m["surf"]))
        maxima.append(dmax)
        surfs.append(gs)
        print(f"step={i+1} u={disp:.5f} sweeps={sw:3d} max_d={dmax:.4f} "
              f"min_d={dmin:.3e} surface_energy={gs:.5e}")

    seed_edge = abs(coords[node0][0] - 0.5) < 1.5 * m["h"] or \
        abs(abs(coords[node0][1] - 0.5) - 0.5 * m["h"]) < 1.5 * m["h"]
    naive_assert = ""
    try:
        assert m["d"].x.array.max() <= 1.0, "damage exceeded one"
    except AssertionError as exc:
        naive_assert = f"AssertionError: {exc}"
        print(f"naive_upper_bound_check -> {naive_assert}")
    drop_near = False
    if worst_drop_node is not None:
        c = coords[worst_drop_node]
        print(f"worst_nonmonotone_nodal_decrease={worst_drop:+.5f} "
              f"at=({c[0]:.4f}, {c[1]:.4f})")
        drop_near = float(np.linalg.norm(c[:2] - coords[node0][:2])) < 4.0 * m["h"]

    over_seed = d0 > 1.0
    over_steps = all(x > 1.0 for x in maxima)
    small = (max(maxima + [d0]) - 1.0) < 0.05
    nonneg = float(m["d"].x.array.min()) >= 0.0
    increasing = all(b >= a for a, b in zip(surfs, surfs[1:]))
    print(f"max_d_exceeds_one_right_after_seeding={over_seed}")
    print(f"max_d_still_exceeds_one_at_every_load_step={over_steps}")
    print(f"overshoot_is_under_five_percent={small}")
    print(f"min_d_is_non_negative={nonneg}")
    print(f"regularised_surface_energy_increases_every_step={increasing}")
    print(f"overshoot_node_sits_on_the_seed_boundary={seed_edge}")
    print(f"naive_assert_d_le_one_fails={bool(naive_assert)}")
    print(f"worst_nonmonotone_decrease_is_in_the_overshoot_region={drop_near}")
    if over_seed and over_steps and small and nonneg and increasing \
            and seed_edge and naive_assert and drop_near:
        print("VERDICT=history_seeded_precrack_overshoots_one_harmlessly")
        return 0
    print("VERDICT=history_seeded_precrack_stayed_at_or_below_one")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
