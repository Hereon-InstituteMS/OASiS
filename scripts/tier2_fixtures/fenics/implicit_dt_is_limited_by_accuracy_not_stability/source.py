"""Tier-2 for fenics time_dependent_heat#9: backward Euler is implicit, so there
is NO CFL condition - dt is limited only by accuracy. Sizing dt from an
explicit-stability formula such as dt < h^2/(2*alpha) just multiplies the number
of steps for nothing, and conversely a dt so large that it smears the transient
produces a perfectly stable, perfectly converged, physically useless answer. Only
a probe history at two different dt values shows it.

Unit square 32x32 (h = 1/32, alpha = 1, so the explicit limit is
h^2/2 = 4.883e-04). T = 0 initially; the left wall is held at 1 for a pulse of
duration 0.01 and at 0 afterwards; end time 0.05. A point two cells from the wall
is probed every step with Function.eval against a bb_tree cell candidate, and the
peak of that probe history is compared against a fine reference at dt_exp/4.

Observed: every dt from dt_exp/4 up to 50*dt_exp is stable - finite, bounded by
1, KSP converged reason 4 at every step - so nothing enforces the explicit limit.
dt_exp needs 102 steps for a probe peak 3.2% off the reference while 5*dt_exp
needs 20 steps for 6.9%: the explicit-limit run buys accuracy that a five times
cheaper run already had. At 25*dt_exp and 50*dt_exp the wall pulse falls entirely
between two time levels, the field never leaves 0.000000 anywhere, the probe peak
is exactly 0.0 - 100% wrong - and every solve still reports converged reason 4.
Note also that the SMALLEST dt is the one that dips to -0.1290: small steps are
not automatically safer on a P1 mass matrix.

Mutation control: T2_MUTATE=1 selects 5*dt_exp as the checked step; its probe
peak error is 6.9%, far below the 50% threshold, so the smearing finding is lost.
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
import dolfinx.geometry  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

from petsc4py import PETSc  # noqa: E402

N, T_END, PULSE = 32, 0.05, 0.01
DT_EXP = (1.0 / N) ** 2 / 2.0          # explicit stability limit for alpha = 1
FACTORS = (0.25, 1.0, 5.0, 25.0, 50.0)
REF, LIMIT, CHEAP, HUGE = 0.25, 1.0, 5.0, 50.0


def march(dt_val: float):
    nstep = max(1, int(round(T_END / dt_val)))
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    T_n, T_h = dolfinx.fem.Function(V), dolfinx.fem.Function(V)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dt = dolfinx.fem.Constant(msh, dt_val)
    g = dolfinx.fem.Constant(msh, 0.0)
    a = (u / dt) * v * ufl.dx + ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (T_n / dt) * v * ufl.dx
    left = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 0.0))
    bcs = [dolfinx.fem.dirichletbc(
        g, dolfinx.fem.locate_dofs_topological(V, fdim, left), V)]
    a_f, L_f = dolfinx.fem.form(a), dolfinx.fem.form(L)
    A = dolfinx.fem.petsc.assemble_matrix(a_f, bcs=bcs)
    A.assemble()
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    b = dolfinx.fem.petsc.create_vector(V)

    point = np.array([[2.0 / N, 0.5, 0.0]], dtype=np.float64)
    tree = dolfinx.geometry.bb_tree(msh, tdim)
    cand = dolfinx.geometry.compute_collisions_points(tree, point)
    cell = dolfinx.geometry.compute_colliding_cells(msh, cand, point).links(0)[:1]

    reasons, hist, t, lo, hi = [], [], 0.0, 0.0, 0.0
    for _ in range(nstep):
        t += dt_val
        g.value = 1.0 if t <= PULSE + 1e-12 else 0.0
        with b.localForm() as loc:
            loc.set(0.0)
        dolfinx.fem.petsc.assemble_vector(b, L_f)
        dolfinx.fem.petsc.apply_lifting(b, [a_f], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, bcs)
        ksp.solve(b, T_h.x.petsc_vec)
        T_h.x.scatter_forward()
        reasons.append(ksp.getConvergedReason())
        hist.append(float(T_h.eval(point, cell)[0]))
        lo = min(lo, float(T_h.x.array.min()))
        hi = max(hi, float(T_h.x.array.max()))
        T_n.x.array[:] = T_h.x.array
    return {"nstep": nstep, "peak": max(hist), "reasons": set(reasons),
            "lo": lo, "hi": hi,
            "finite": bool(np.all(np.isfinite(T_h.x.array)))}


def main() -> int:
    runs = {f: march(f * DT_EXP) for f in FACTORS}
    for f in FACTORS:
        r = runs[f]
        print(f"dt_over_explicit_limit={f:g} dt={f * DT_EXP:.3e} "
              f"steps={r['nstep']} probe_peak={r['peak']:.6f} "
              f"range=[{r['lo']:.4f}, {r['hi']:.4f}] "
              f"reason_4_every_step={r['reasons'] == {4}} finite={r['finite']}")
    ref = runs[REF]["peak"]
    err = {f: abs(runs[f]["peak"] - ref) / ref for f in FACTORS}
    for f in FACTORS[1:]:
        print(f"probe_peak_error_at_{f:g}x_explicit_limit={err[f] * 100:.1f}percent")

    no_blowup = all(runs[f]["finite"] and runs[f]["reasons"] == {4}
                    and runs[f]["hi"] <= 1.0 + 1e-12 for f in FACTORS)
    cheaper = (runs[LIMIT]["nstep"] >= 5 * runs[CHEAP]["nstep"]
               and err[CHEAP] < 0.10)
    missed = runs[HUGE]["peak"] == 0.0 and err[HUGE] > 0.5
    small_dt_dips = runs[REF]["lo"] < -1e-3
    print(f"no_dt_blew_up_every_run_reported_reason_4={no_blowup}")
    print(f"explicit_limit_costs_five_times_the_steps_for_no_gain={cheaper}")
    print(f"huge_dt_misses_the_wall_pulse_entirely={missed}")
    print(f"smallest_dt_is_the_one_that_undershoots_zero={small_dt_dips}")

    sel = CHEAP if MUTATE else HUGE
    print(f"selected_dt_over_explicit_limit={sel:g} "
          f"selected_steps={runs[sel]['nstep']}")
    print(f"selected_probe_peak_error_exceeds_fifty_percent={err[sel] > 0.5}")
    print(f"selected_converged_and_bounded="
          f"{runs[sel]['reasons'] == {4} and runs[sel]['hi'] <= 1.0 + 1e-12}")

    if (no_blowup and cheaper and missed and err[sel] > 0.5
            and runs[sel]["reasons"] == {4}):
        print("VERDICT=implicit_dt_is_limited_by_accuracy_not_by_stability")
        return 0
    print("VERDICT=an_implicit_step_hit_a_stability_limit")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
