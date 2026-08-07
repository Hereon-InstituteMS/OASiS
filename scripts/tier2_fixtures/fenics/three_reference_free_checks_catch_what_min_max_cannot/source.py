"""Tier-2 for fenics time_dependent_heat#10: the three reference-free checks that
belong in every transient heat run - (1) the discrete maximum principle, (2) the
Galerkin residual at the solution over the UNCONSTRAINED dofs, (3) the nodal
reactions summed over each Dirichlet wall, whose imbalance must shrink as steady
state is approached - and the fact that printing only min/max of T detects none of
the failures.

32x32 unit square, T = 1 on the left wall, T = 0 on the right, insulated top and
bottom, T = 0 initially, backward Euler dt = 0.01, manual assemble-once loop run to
t = 0.5. All three checks pass on the correct run: T stays in [0.000000, 1.000000],
the free-dof residual is at machine zero, and the wall reactions come out equal and
opposite to within a few percent with an imbalance that shrinks further by t = 1.0,
because the imbalance is the rate of energy storage and the run stops before steady
state. The same loop with apply_lifting omitted prints the IDENTICAL min/max history
- so check (1) is blind to it - while the Galerkin residual over the free dofs jumps
to order one.

Mutation control: T2_MUTATE=1 restores apply_lifting in the second run, so the
residual check no longer has anything to catch.
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

from petsc4py import PETSc  # noqa: E402

N, DT = 32, 0.01


def run(nstep: int, lift: bool):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    T_n, T_h = dolfinx.fem.Function(V), dolfinx.fem.Function(V)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dt_c = dolfinx.fem.Constant(msh, DT)
    a = (u / dt_c) * v * ufl.dx + ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (T_n / dt_c) * v * ufl.dx
    left = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 0.0))
    right = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 1.0))
    ldofs = dolfinx.fem.locate_dofs_topological(V, fdim, left)
    rdofs = dolfinx.fem.locate_dofs_topological(V, fdim, right)
    bcs = [dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 1.0), ldofs, V),
           dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 0.0), rdofs, V)]
    a_f, L_f = dolfinx.fem.form(a), dolfinx.fem.form(L)
    A = dolfinx.fem.petsc.assemble_matrix(a_f, bcs=bcs)
    A.assemble()
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    b = dolfinx.fem.petsc.create_vector(V)

    lines, lo, hi = [], 0.0, 0.0
    prev = T_n.x.array.copy()
    for _ in range(nstep):
        with b.localForm() as loc:
            loc.set(0.0)
        dolfinx.fem.petsc.assemble_vector(b, L_f)
        if lift:
            dolfinx.fem.petsc.apply_lifting(b, [a_f], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, bcs)
        ksp.solve(b, T_h.x.petsc_vec)
        T_h.x.scatter_forward()
        lo = min(lo, float(T_h.x.array.min()))
        hi = max(hi, float(T_h.x.array.max()))
        lines.append(f"T in [{T_h.x.array.min():.6f}, "
                     f"{T_h.x.array.max():.6f}]")
        prev = T_n.x.array.copy()
        T_n.x.array[:] = T_h.x.array

    # Residual of the last step: T_n must hold the value THAT step used.
    T_n.x.array[:] = prev
    F = ufl.replace(a - L, {u: T_h})
    r = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(F))
    r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    res = r.array.copy()
    free = np.setdiff1d(np.arange(res.size), np.concatenate([ldofs, rdofs]))
    return (lines, lo, hi, float(np.max(np.abs(res[free]))),
            float(np.sum(res[ldofs])), float(np.sum(res[rdofs])))


def main() -> int:
    lines, lo, hi, resid, react_l, react_r = run(50, lift=True)
    _, _, _, _, l100, r100 = run(100, lift=True)
    bad_lines, bad_lo, bad_hi, bad_resid, _, _ = run(50, lift=MUTATE)

    print(f"correct_range=[{lo:.6f}, {hi:.6f}] "
          f"free_dof_residual={resid:.3e}")
    print(f"wall_reactions={react_l:.6e} and {react_r:.6e} "
          f"imbalance_at_t0p5={react_l + react_r:.3e}")
    print(f"imbalance_at_t1p0={l100 + r100:.3e}")
    dmp = lo >= -1e-12 and hi <= 1.0 + 1e-12
    zero_res = resid < 1.5e-12
    opposite = abs(react_l + react_r) < 0.05 * abs(react_l)
    shrinking = abs(l100 + r100) < abs(react_l + react_r)
    print(f"discrete_maximum_principle_holds={dmp}")
    print(f"galerkin_residual_at_free_dofs_is_machine_zero={zero_res}")
    print(f"wall_reactions_are_equal_and_opposite={opposite}")
    print(f"reaction_imbalance_shrinks_towards_steady_state={shrinking}")

    print(f"second_run_lifting={MUTATE}")
    print(f"second_run_last_line={bad_lines[-1]}")
    same_print = bad_lines == lines
    print(f"second_run_prints_the_identical_min_max_history={same_print}")
    print(f"second_run_free_dof_residual={bad_resid:.3e}")
    caught = bad_resid > 1e-2
    print(f"residual_check_catches_the_second_run={caught}")
    print(f"min_max_check_catches_the_second_run="
          f"{not (bad_lo >= -1e-12 and bad_hi <= 1.0 + 1e-12)}")

    if dmp and zero_res and opposite and shrinking and same_print and caught:
        print("VERDICT=the_three_checks_hold_and_the_residual_catches_what_min_max_cannot")
        return 0
    print("VERDICT=checks_did_not_discriminate")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
