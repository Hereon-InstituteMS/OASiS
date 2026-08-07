"""Tier-2 for fenics fracture#1: a mesh that is coarse compared with l0 makes the
model TOUGHER, not weaker. The previously stated signal -- that the computed
fracture energy "under-shoots Griffith's G_c * area by ~30-50% when h ~ l0" -- is
wrong in sign. Running the same specimen at a sequence of mesh sizes with l0 and
Gc held fixed, the energy dissipated in breaking it comes out ABOVE Gc times the
crack area at every resolution, the excess grows as the mesh is coarsened, and the
displacement at which the crack finally runs rises with coarsening too. The
convergence test is: divide the dissipated energy by Gc times the crack area and
refine until that ratio stops falling.

Wrong variant: report the h ~ l0 mesh as the answer. Right variant: refine to
h = l0/2 and check that the ratio is still falling.

Staggered AT2 phase-field, P1 displacement and P1 damage, E = 210, nu = 0.3,
Gc = 2.7e-3, l0 = 0.1 held FIXED while the mesh goes 8x8, 12x12, 20x20
(h/l0 = 1.25, 0.833, 0.5), pre-crack seeded as a large history value along y = 0.5
for x < 0.5 so the crack area created is the remaining ligament, 0.5. Sixteen
displacement-controlled steps to 1e-2. The dissipated energy is the rise of the
regularised surface energy from just after seeding to the end of the ramp.

Observed on dolfinx 0.10.0: the ratio dissipated / (Gc * 0.5) is 1.4006, 1.3201
and 1.2524 -- above one at every resolution, falling monotonically as the mesh is
refined, never approaching one from below. The displacement at which the
integrated damage takes its largest jump falls with refinement as well: 0.00750,
0.00688, 0.00625. At the finest mesh the ratio is still dropping, so this sequence
is not converged, which is exactly what the test is for.

Mutation control: T2_MUTATE=1 makes the mesh under test the fine 20x20 one, whose
overshoot is 25% rather than 40%, so the coarse-mesh token goes False.
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
L0, K_RES = 0.1, 1e-6
SIZES = (8, 12, 20)
CRACK_AREA = 0.5          # the ligament the crack has to break
STEPS = np.linspace(0.0, 1.0e-2, 17)[1:]


def build(n: int):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n)
    h = 1.0 / n
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
    seed = np.where((np.abs(mids[:, 1] - 0.5) < 0.55 * h)
                    & (mids[:, 0] < 0.5))[0]
    H.x.array[:] = 0.0
    H.x.array[seed] = 1e3 * GC / (2 * L0)

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
    a_d = ((2.0 * H + GC / L0) * dd * w
           + GC * L0 * ufl.dot(ufl.grad(dd), ufl.grad(w))) * ufl.dx
    L_d = 2.0 * H * w * ufl.dx
    pu = dolfinx.fem.petsc.LinearProblem(
        a_u, L_u, bcs=bcs_u, u=u, petsc_options_prefix=f"t2f1u_{n}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})
    pd = dolfinx.fem.petsc.LinearProblem(
        a_d, L_d, bcs=[], u=d, petsc_options_prefix=f"t2f1d_{n}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})
    eu = ufl.sym(ufl.grad(u))
    psi_expr = fem.Expression(0.5 * LAM * ufl.tr(eu) ** 2 + MU * ufl.inner(eu, eu),
                              Q.element.interpolation_points)
    surf = fem.form(GC / (2 * L0) * (d ** 2 + L0 ** 2
                                     * ufl.dot(ufl.grad(d), ufl.grad(d))) * ufl.dx)
    dint = fem.form(d * ufl.dx)
    return dict(d=d, H=H, Hn=Hn, pu=pu, pd=pd, psi=psi_expr, surf=surf,
                dint=dint, u_top=u_top, h=h)


def run(n: int) -> dict:
    m = build(n)
    m["pd"].solve()
    seeded = float(fem.assemble_scalar(m["surf"]))
    prev = float(fem.assemble_scalar(m["dint"]))
    best, u_crit = 0.0, 0.0
    for disp in STEPS:
        m["u_top"].x.array[:] = 0.0
        m["u_top"].x.array[1::2] = disp
        for _ in range(400):
            before = m["d"].x.array.copy()
            m["pu"].solve()
            m["Hn"].interpolate(m["psi"])
            m["H"].x.array[:] = np.maximum(m["H"].x.array, m["Hn"].x.array)
            m["pd"].solve()
            if np.max(np.abs(m["d"].x.array - before)) < 1e-4:
                break
        now = float(fem.assemble_scalar(m["dint"]))
        if now - prev > best:
            best, u_crit = now - prev, float(disp)
        prev = now
    dissipated = float(fem.assemble_scalar(m["surf"])) - seeded
    return dict(n=n, h=m["h"], dissipated=dissipated,
                ratio=dissipated / (GC * CRACK_AREA), u_crit=u_crit)


def main() -> int:
    under_test = SIZES[-1] if MUTATE else SIZES[0]
    if MUTATE:
        print(f"mutation=mesh_under_test_is_the_fine_{under_test}x{under_test}_one")
    rows = [run(n) for n in SIZES]
    for r in rows:
        print(f"n={r['n']:2d} h={r['h']:.4f} h_over_l0={r['h'] / L0:.3f} "
              f"dissipated_energy={r['dissipated']:.5e} "
              f"dissipated_over_Gc_times_crack_area={r['ratio']:.4f} "
              f"displacement_at_which_the_crack_runs={r['u_crit']:.5f}")

    ratios = [r["ratio"] for r in rows]
    ucrits = [r["u_crit"] for r in rows]
    tested = next(r for r in rows if r["n"] == under_test)
    above = all(x > 1.0 for x in ratios)
    falling = all(b < a for a, b in zip(ratios, ratios[1:]))
    never_under = min(ratios) > 1.0
    u_falls = all(b <= a for a, b in zip(ucrits, ucrits[1:])) and ucrits[-1] < ucrits[0]
    still_falling = ratios[-1] < ratios[-2]
    coarse_over = tested["ratio"] > 1.35
    print(f"mesh_under_test={under_test} its_ratio={tested['ratio']:.4f}")
    print(f"every_resolution_dissipates_more_than_gc_times_the_crack_area={above}")
    print(f"the_excess_falls_monotonically_under_refinement={falling}")
    print(f"no_resolution_undershoots_griffith={never_under}")
    print(f"the_crack_runs_at_a_smaller_displacement_as_the_mesh_refines={u_falls}")
    print(f"the_ratio_is_still_falling_at_the_finest_mesh={still_falling}")
    print(f"mesh_under_test_overshoots_griffith_by_over_thirty_five_percent="
          f"{coarse_over}")
    if above and falling and never_under and u_falls and still_falling \
            and coarse_over:
        print("VERDICT=a_coarse_mesh_dissipates_more_than_griffith_not_less")
        return 0
    print("VERDICT=the_coarse_mesh_undershot_griffith")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
