"""Tier-2 for fenics heat#4: Crank-Nicolson rings in TIME on a sharp transient
and a min/max check on the field will not see it.

T0 = 0, left wall jumps to 1 at t = 0+, other walls insulated. The nodal history
at the first free node is scored exactly as the claim describes: differences
d_n, count of sign reversals, alternation amplitude. Backward Euler must give
zero reversals; Crank-Nicolson must give many. The claim's two corrections are
checked too: the field never leaves [0, 1], so a range check is blind.

Mutation control: T2_MUTATE=1 runs theta = 1 (backward Euler) in the slot where
the pathology is expected; the reversals vanish.
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

NSTEP, DT, N = 20, 0.01, 40


def history(theta: float):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    T_n = dolfinx.fem.Function(V)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dt = dolfinx.fem.Constant(msh, DT)
    th = dolfinx.fem.Constant(msh, theta)
    # (1 - theta) has to be a Constant, not a Python float: at theta = 1 the
    # float 0.0 makes UFL fold the whole term to a domain-less Zero and the
    # form then raises "This integral is missing an integration domain."
    omth = dolfinx.fem.Constant(msh, 1.0 - theta)
    a = (u / dt) * v * ufl.dx + th * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ((T_n / dt) * v * ufl.dx
         - omth * ufl.dot(ufl.grad(T_n), ufl.grad(v)) * ufl.dx)
    left = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 0.0))
    dofs = dolfinx.fem.locate_dofs_topological(V, fdim, left)
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 1.0), dofs, V)
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix="t2_cn_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    coords = V.tabulate_dof_coordinates()
    free = np.setdiff1d(np.arange(coords.shape[0]), dofs)
    probe = int(free[np.argmin(coords[free, 0])])
    hist, lo, hi = [], [], []
    for _ in range(NSTEP):
        T_h = prob.solve()
        if isinstance(T_h, tuple):
            T_h = T_h[0]
        hist.append(float(T_h.x.array[probe]))
        lo.append(float(np.min(T_h.x.array)))
        hi.append(float(np.max(T_h.x.array)))
        T_n.x.array[:] = T_h.x.array
    d = np.diff(np.array(hist))
    rev = int(np.sum(d[:-1] * d[1:] < 0))
    amp = 0.0
    for i in range(len(d) - 1):
        if d[i] * d[i + 1] < 0:
            amp = max(amp, min(abs(d[i]), abs(d[i + 1])))
    return rev, amp, min(lo), max(hi)


def main() -> int:
    cn_rev, cn_amp, cn_lo, cn_hi = history(0.5 if not MUTATE else 1.0)
    be_rev, be_amp, _, _ = history(1.0)
    print(f"cn_sign_reversals={cn_rev} cn_alternation_amplitude={cn_amp:.5f}")
    print(f"be_sign_reversals={be_rev} be_alternation_amplitude={be_amp:.5f}")
    print(f"cn_field_min={cn_lo:.6e} cn_field_max={cn_hi:.6f}")
    rings = cn_rev >= 10 and cn_amp > 0.01
    be_clean = be_rev == 0 and be_amp == 0.0
    in_range = cn_lo >= -1e-9 and cn_hi <= 1.0 + 1e-9
    print(f"cn_rings_in_time={rings}")
    print(f"be_monotone={be_clean}")
    print(f"cn_field_stays_in_physical_range={in_range}")
    if rings and be_clean and in_range:
        print("VERDICT=cn_rings_in_time_while_field_range_looks_fine")
        return 0
    print("VERDICT=no_ringing_detected")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
