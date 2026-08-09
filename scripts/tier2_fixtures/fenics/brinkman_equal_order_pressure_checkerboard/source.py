"""Tier-2 for fenics stokes_darcy#3: the Brinkman/Stokes system is a saddle point,
so the velocity and pressure spaces must satisfy the LBB (inf-sup) condition; the
Darcy drag term does not rescue an unstable pair.

Wrong variant: equal-order P1/P1 velocity/pressure (and, differently wrong,
P1 velocity with DG0 pressure). Right variant: Taylor-Hood P2/P1 (or P3/P2).

Single-mesh Brinkman on the unit square: porous lower half with K = 1e-4, free
upper half, no-slip on the two horizontal walls, unit pressure imposed weakly at
the inlet x = 0 and zero at the outlet x = 1, MUMPS LU.

Observed on dolfinx 0.10.0 / PETSc 3.24.5: with P1/P1 nothing is raised,
getConvergedReason() returns 4, the inlet/outlet mass balance is machine zero
(1e-18 to 1e-16), and only the pressure is wrong -- the cellwise oscillation
measure ||p - cellmean(p)|| / ||p|| is 0.8135, 0.9185 and about 0.96 on 8x8, 16x16
and 32x32, i.e. almost all of the pressure energy is intra-cell oscillation, and
the peak |p| reaches 3.5, 11.9 and about 15 against the imposed inlet value of 1
(the 32x32 numbers wobble in the third digit between runs, MUMPS on a nearly
singular saddle point, which is why ranges and not values are asserted). The
stable pair gives 0.0510, 0.0255, 0.0128 -- a few percent, halving under refinement
-- with peak |p| of 1.0022 down to 1.0006. P1/DG0 fails loudly instead:
getConvergedReason() returns -11 and the solution is inf (at 8x8 it still solves,
so the loud failure is shown at 16x16). P3/P2 is clean like P2/P1.

Mutation control: T2_MUTATE=1 puts Taylor-Hood P2/P1 in the slot under test, so
the oscillation tokens go False.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import basix.ufl  # noqa: E402
import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dolfinx import fem, mesh  # noqa: E402
from petsc4py import PETSc  # noqa: E402

MU, KPERM, PIN = 1.0, 1e-4, 1.0
SIZES = (8, 16, 32)


def solve(n: int, vdeg: int, pdeg: int, pfam: str = "Lagrange"):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n)
    tdim = msh.topology.dim
    porous = mesh.locate_entities(msh, tdim, lambda x: x[1] <= 0.5 + 1e-12)
    marks = np.full(msh.topology.index_map(tdim).size_local, 1, dtype=np.int32)
    marks[porous] = 2
    ct = mesh.meshtags(msh, tdim, np.arange(len(marks), dtype=np.int32), marks)
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    inlet = mesh.locate_entities_boundary(msh, fdim,
                                          lambda x: np.isclose(x[0], 0.0))
    outlet = mesh.locate_entities_boundary(msh, fdim,
                                           lambda x: np.isclose(x[0], 1.0))
    wall = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    fi = np.concatenate([inlet, outlet])
    fv = np.concatenate([np.full(len(inlet), 1, np.int32),
                         np.full(len(outlet), 2, np.int32)])
    o = np.argsort(fi)
    ft = mesh.meshtags(msh, fdim, fi[o], fv[o])

    Ve = basix.ufl.element("Lagrange", msh.basix_cell(), vdeg, shape=(tdim,))
    Pe = basix.ufl.element(pfam, msh.basix_cell(), pdeg)
    W = fem.functionspace(msh, basix.ufl.mixed_element([Ve, Pe]))
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    dx = ufl.Measure("dx", domain=msh, subdomain_data=ct)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
    nvec = ufl.FacetNormal(msh)
    a = (MU * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
         + (MU / KPERM) * ufl.inner(u, v) * dx(2)
         - p * ufl.div(v) * dx - q * ufl.div(u) * dx)
    L = -PIN * ufl.dot(v, nvec) * ds(1)
    V0, _ = W.sub(0).collapse()
    zero = fem.Function(V0)
    bcs = [fem.dirichletbc(
        zero, fem.locate_dofs_topological((W.sub(0), V0), fdim, wall), W.sub(0))]

    af, Lf = fem.form(a), fem.form(L)
    A = dolfinx.fem.petsc.assemble_matrix(af, bcs=bcs)
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(Lf)
    dolfinx.fem.petsc.apply_lifting(b, [af], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs)
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    wh = fem.Function(W)
    raised = ""
    try:
        ksp.solve(b, wh.x.petsc_vec)
        wh.x.scatter_forward()
    except Exception as exc:  # noqa: BLE001
        raised = f"{type(exc).__name__}: {exc}"
    reason = int(ksp.getConvergedReason())
    uh, ph = wh.sub(0).collapse(), wh.sub(1).collapse()
    finite = bool(np.all(np.isfinite(ph.x.array))
                  and np.all(np.isfinite(uh.x.array)))
    if not finite:
        return dict(reason=reason, raised=raised, finite=False,
                    balance=float("nan"), osc=float("nan"),
                    pmax=float(np.abs(ph.x.array).max()))
    flux_in = float(fem.assemble_scalar(fem.form(ufl.dot(uh, nvec) * ds(1))))
    flux_out = float(fem.assemble_scalar(fem.form(ufl.dot(uh, nvec) * ds(2))))
    DG0 = fem.functionspace(msh, ("DG", 0))
    w0 = ufl.TestFunction(DG0)
    cell_int = fem.assemble_vector(fem.form(ph * w0 * ufl.dx)).array
    cell_vol = fem.assemble_vector(fem.form(w0 * ufl.dx)).array
    means = cell_int / cell_vol
    p_sq = float(fem.assemble_scalar(fem.form(ph * ph * ufl.dx)))
    osc_sq = p_sq - float(np.sum(cell_vol * means ** 2))
    return dict(reason=reason, raised=raised, finite=True,
                balance=flux_in + flux_out,
                osc=float(np.sqrt(max(osc_sq, 0.0) / p_sq)),
                pmax=float(np.abs(ph.x.array).max()))


def main() -> int:
    vd, pd = (2, 1) if MUTATE else (1, 1)
    if MUTATE:
        print("mutation=slot_under_test_is_taylor_hood_p2_p1")
    tested, stable = [], []
    for n in SIZES:
        t = solve(n, vd, pd)
        s = solve(n, 2, 1)
        tested.append(t)
        stable.append(s)
        print(f"n={n:2d} under_test=P{vd}/P{pd} reason={t['reason']} "
              f"raised={t['raised'][:40]!r} balance={t['balance']:.3e} "
              f"oscillation_ratio={t['osc']:.4f} peak_abs_p={t['pmax']:.4e}")
        print(f"n={n:2d} stable=P2/P1        reason={s['reason']} "
              f"balance={s['balance']:.3e} oscillation_ratio={s['osc']:.4f} "
              f"peak_abs_p={s['pmax']:.4e}")

    loud = solve(16, 1, 0, "DG")
    print(f"n=16 P1/DG0 reason={loud['reason']} finite={loud['finite']} "
          f"peak_abs_p={loud['pmax']:.3e} raised={loud['raised'][:40]!r}")
    hi = solve(8, 3, 2)
    print(f"n= 8 P3/P2 reason={hi['reason']} oscillation_ratio={hi['osc']:.4f} "
          f"peak_abs_p={hi['pmax']:.4e}")

    nothing_raised = all(t["raised"] == "" for t in tested)
    reason_four = all(t["reason"] == 4 for t in tested)
    balanced = all(abs(t["balance"]) < 1e-12 for t in tested)
    oscillating = all(0.7 <= t["osc"] <= 0.99 for t in tested)
    peaked = sum(t["pmax"] > 3.0 * PIN for t in tested) >= 2
    stable_small = all(s["osc"] < 0.1 for s in stable) and \
        stable[-1]["osc"] < 0.5 * stable[0]["osc"]
    stable_peak = all(abs(s["pmax"] - PIN) < 0.05 * PIN for s in stable)
    loud_fail = loud["reason"] == -11 and not loud["finite"]
    hi_clean = hi["reason"] == 4 and hi["osc"] < 0.1
    print(f"equal_order_raised_nothing={nothing_raised}")
    print(f"equal_order_converged_reason_is_four={reason_four}")
    print(f"equal_order_mass_balance_is_machine_zero={balanced}")
    print(f"equal_order_oscillation_ratio_between_0p7_and_0p99_everywhere="
          f"{oscillating}")
    print(f"equal_order_peak_pressure_far_above_the_imposed_inlet_value={peaked}")
    print(f"stable_pair_oscillation_is_a_few_percent_and_halves={stable_small}")
    print(f"stable_pair_peak_pressure_equals_the_inlet_value={stable_peak}")
    print(f"p1_dg0_fails_loudly_with_reason_minus_eleven_and_nonfinite="
          f"{loud_fail}")
    print(f"p3_p2_is_clean_like_p2_p1={hi_clean}")
    if nothing_raised and reason_four and balanced and oscillating and peaked \
            and stable_small and stable_peak and loud_fail and hi_clean:
        print("VERDICT=equal_order_brinkman_pressure_oscillates_silently")
        return 0
    print("VERDICT=equal_order_brinkman_pressure_was_fine")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
