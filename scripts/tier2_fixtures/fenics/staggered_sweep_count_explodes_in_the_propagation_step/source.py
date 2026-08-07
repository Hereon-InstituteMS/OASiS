"""Tier-2 for fenics fracture#3: the staggered iteration count is not roughly
constant, it explodes in the one load step where the crack propagates. The number
of staggered sweeps needed to reach a fixed tolerance on d sits in the single
digits while the crack is dormant, jumps by one to two orders of magnitude in the
step in which the integrated damage takes its big jump, then drops straight back
to single digits. Set the staggered iteration cap generously: a cap of a few tens
will silently truncate exactly the step that matters and leave a half-propagated
crack.

Wrong variant: max_sweeps = 30. Right variant: max_sweeps = 2000.

Staggered AT2 phase-field on a 20x20 unit square, P1 displacement and P1 damage,
E = 210, nu = 0.3, Gc = 2.7e-3, l0 = 2h = 0.1, pre-crack seeded as a large history
value along y = 0.5 for x < 0.5, sixteen displacement-controlled steps to 8e-3,
sweeps stopped when max|d - d_prev| < 1e-4.

Observed on dolfinx 0.10.0 with the generous cap: 2, 3, 3, 3, 4, 5, 5, 7, 8, 11,
18, 63, 61, 7, 6, 5 sweeps -- a median of 5 while the crack is dormant, a factor
of about twelve at steps 12 and 13, which are exactly the steps where the
integrated damage takes its big jump (per-step growth 0.0156, 0.0237, 0.0489
against 0.0006 afterwards), then straight back to single digits. With the cap at
30 those two steps stop at exactly 30 sweeps and nothing is raised.

DISCREPANCY: the second half of the claim, that such a cap "will leave a
half-propagated crack", does NOT reproduce on this specimen. The truncated run's
regularised surface energy is 0.9943 of the uncapped one at the truncated step,
0.9048 one step later, and 1.0018 at the end of the ramp -- truncation delays the
propagation by about one load step, it does not lose the crack. The fixture pins
that too.

Mutation control: T2_MUTATE=1 gives the run under test the generous cap, so the
truncation tokens go False.
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
N, K_RES = 20, 1e-6
TOL = 1e-4
SMALL_CAP, BIG_CAP = 30, 2000
STEPS = np.linspace(0.0, 8e-3, 17)[1:]


def build():
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
    seed = np.where((np.abs(mids[:, 1] - 0.5) < 0.55 * h)
                    & (mids[:, 0] < 0.5))[0]
    H.x.array[:] = 0.0
    H.x.array[seed] = 1e3 * GC / (2 * l0)

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
        a_u, L_u, bcs=bcs_u, u=u, petsc_options_prefix="t2f3u_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})
    pd = dolfinx.fem.petsc.LinearProblem(
        a_d, L_d, bcs=[], u=d, petsc_options_prefix="t2f3d_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})
    eu = ufl.sym(ufl.grad(u))
    psi_expr = fem.Expression(0.5 * LAM * ufl.tr(eu) ** 2 + MU * ufl.inner(eu, eu),
                              Q.element.interpolation_points)
    surf = fem.form(GC / (2 * l0) * (d ** 2 + l0 ** 2
                                     * ufl.dot(ufl.grad(d), ufl.grad(d))) * ufl.dx)
    dint = fem.form(d * ufl.dx)
    return dict(d=d, H=H, Hn=Hn, pu=pu, pd=pd, psi=psi_expr, surf=surf,
                dint=dint, u_top=u_top)


def run(cap: int) -> dict:
    m = build()
    m["pd"].solve()
    sweeps, growth, energies, raised = [], [], [], ""
    prev = float(fem.assemble_scalar(m["dint"]))
    for disp in STEPS:
        m["u_top"].x.array[:] = 0.0
        m["u_top"].x.array[1::2] = disp
        used = cap
        for k in range(cap):
            before = m["d"].x.array.copy()
            try:
                m["pu"].solve()
                m["Hn"].interpolate(m["psi"])
                m["H"].x.array[:] = np.maximum(m["H"].x.array, m["Hn"].x.array)
                m["pd"].solve()
            except Exception as exc:  # noqa: BLE001
                raised = f"{type(exc).__name__}: {str(exc)[:60]}"
            if np.max(np.abs(m["d"].x.array - before)) < TOL:
                used = k + 1
                break
        sweeps.append(used)
        now = float(fem.assemble_scalar(m["dint"]))
        growth.append(now - prev)
        energies.append(float(fem.assemble_scalar(m["surf"])))
        prev = now
    return dict(sweeps=sweeps, growth=growth, raised=raised, energies=energies,
                surf=float(fem.assemble_scalar(m["surf"])),
                dint=prev)


def main() -> int:
    cap = BIG_CAP if MUTATE else SMALL_CAP
    if MUTATE:
        print("mutation=run_under_test_uses_the_generous_cap")
    generous = run(BIG_CAP)
    capped = run(cap)
    print(f"generous_cap={BIG_CAP} sweeps_per_load_step={generous['sweeps']}")
    print(f"integrated_damage_growth_per_step="
          f"{['%.4f' % x for x in generous['growth']]}")
    print(f"cap_under_test={cap} sweeps_per_load_step={capped['sweeps']}")
    print(f"final_surface_energy generous={generous['surf']:.5e} "
          f"under_test={capped['surf']:.5e} "
          f"ratio={capped['surf'] / generous['surf']:.4f}")
    print(f"raised_under_test={capped['raised']!r}")

    sw = generous["sweeps"]
    peak_i = int(np.argmax(sw))
    growth_i = int(np.argmax(generous["growth"]))
    dormant = [s for i, s in enumerate(sw) if abs(i - peak_i) > 1]
    single_digits = float(np.median(dormant)) < 10 and sw[0] < 10
    spike = sw[peak_i] >= 10 * float(np.median(dormant))
    same_step = abs(peak_i - growth_i) <= 1
    returns = sw[-1] < 10
    truncated = capped["sweeps"][peak_i] == cap
    at_step = capped["energies"][peak_i] / generous["energies"][peak_i]
    nxt = min(peak_i + 1, len(sw) - 1)
    at_next = capped["energies"][nxt] / generous["energies"][nxt]
    at_end = capped["surf"] / generous["surf"]
    print(f"peak_sweep_step={peak_i + 1} peak_sweeps={sw[peak_i]} "
          f"median_dormant_sweeps={float(np.median(dormant)):.1f} "
          f"largest_damage_growth_step={growth_i + 1}")
    print(f"surface_energy_ratio_at_the_truncated_step={at_step:.4f} "
          f"at_the_next_step={at_next:.4f} "
          f"at_the_end_of_the_ramp={at_end:.4f}")
    print(f"dormant_steps_stay_in_single_digits={single_digits}")
    print(f"propagation_step_needs_at_least_ten_times_the_dormant_count={spike}")
    print(f"the_spike_is_the_step_of_largest_damage_growth={same_step}")
    print(f"the_count_returns_to_single_digits_afterwards={returns}")
    print(f"a_cap_of_a_few_tens_truncates_exactly_that_step={truncated}")
    print(f"the_truncated_run_raised_nothing={capped['raised'] == ''}")
    # the half of the claim that does NOT reproduce on this specimen
    print(f"truncation_changed_the_crack_by_under_one_percent_at_that_step="
          f"{abs(at_step - 1.0) < 0.01}")
    print(f"the_truncated_run_catches_up_by_the_end_of_the_ramp="
          f"{abs(at_end - 1.0) < 0.02}")
    if single_digits and spike and same_step and returns and truncated \
            and capped["raised"] == "":
        print("VERDICT=sweep_count_explodes_in_the_propagation_step_and_a_small_"
              "cap_truncates_it")
        return 0
    print("VERDICT=sweep_count_stayed_roughly_constant")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
