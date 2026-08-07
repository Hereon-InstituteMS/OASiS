"""Tier-2 for fenics maxwell#0: Maxwell needs H(curl) elements. Lagrange spaces
are silent at compile time -- ufl.curl is accepted on a vector Lagrange space and
fem.form builds it without a word -- and the failure is numerical: a Lagrange
field is continuous in ALL components, so it cannot carry the normal jump that a
material interface puts into E, and its error stops going away.

The test has an exact solution, so nothing has to be assumed. On the unit square
with eps = 1 for x < 1/2 and eps = 100 for x > 1/2, the field
E = (1/eps, 0) satisfies curl E = 0 exactly and
(curl E, curl v) + (eps E, v) = (f, v) for the constant f = (1, 0) with the
natural boundary condition, so it is the exact solution of the discretised
problem's continuous counterpart. Its tangential component is continuous and its
normal component jumps by a factor 100 at x = 1/2.

Observed on dolfinx 0.10.0, meshes 8x8 / 16x16 / 32x32 with the interface on
mesh lines: N1curl degree 1 reproduces the field to machine precision (relative
L2 error ~1e-15, because the exact field lies in the space), while vector
Lagrange degree 1 stalls at a relative L2 error of tens of percent that improves
only at about half an order per refinement -- it is bridging a jump across one
element, and no refinement fixes that.

FINDING against the claim text: the claim's stated observable, "convergence
against an analytic test (e.g. uniform B in a cavity) plateaus at ~10% error",
is only half right. The error does stay O(10 %) here, but it is not a flat
plateau: it decays slowly (measured order ~0.5). With a SMOOTH exact solution
vector Lagrange converges normally, so the plateau needs a material interface --
the claim's "uniform B in a cavity" example would not show it.

Mutation control: T2_MUTATE=1 puts N1curl in the space under test; the error
drops to machine zero at every level and the stall signal disappears.
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

LEVELS = (8, 16, 32)
EPS_RIGHT = 100.0


def element(msh, family: str):
    if family == "n1curl":
        return basix.ufl.element("N1curl", msh.basix_cell(), 1)
    return basix.ufl.element("Lagrange", msh.basix_cell(), 1, shape=(2,))


def solve(n: int, family: str) -> tuple[float, bool]:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    tdim = msh.topology.dim
    ncells = msh.topology.index_map(tdim).size_local
    mid = dolfinx.mesh.compute_midpoints(
        msh, tdim, np.arange(ncells, dtype=np.int32)).T
    DG0 = dolfinx.fem.functionspace(msh, ("DG", 0))
    eps = dolfinx.fem.Function(DG0)
    eps.x.array[:] = 1.0
    eps.x.array[mid[0] > 0.5] = EPS_RIGHT

    V = dolfinx.fem.functionspace(msh, element(msh, family))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    compiled = True
    try:
        dolfinx.fem.form(ufl.inner(ufl.curl(u), ufl.curl(v)) * ufl.dx)
    except Exception as exc:
        compiled = False
        print(f"curl_form_on_{family}_failed {type(exc).__name__}: {exc}")
    a = (ufl.inner(ufl.curl(u), ufl.curl(v))
         + eps * ufl.inner(u, v)) * ufl.dx
    f = ufl.as_vector((1.0, 0.0))
    L = ufl.inner(f, v) * ufl.dx
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[], petsc_options_prefix=f"t2_mx0_{family}_{n}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})
    eh = prob.solve()
    if isinstance(eh, tuple):
        eh = eh[0]
    assert prob.solver.getConvergedReason() > 0
    exact = ufl.as_vector((1.0 / eps, 0.0))
    num = dolfinx.fem.assemble_scalar(dolfinx.fem.form(
        ufl.inner(eh - exact, eh - exact) * ufl.dx))
    den = dolfinx.fem.assemble_scalar(dolfinx.fem.form(
        ufl.inner(exact, exact) * ufl.dx))
    return float(np.sqrt(abs(num) / abs(den))), compiled


def main() -> int:
    fam = "n1curl" if MUTATE else "lagrange"
    print(f"space_under_test={fam}_degree_1 eps_ratio={EPS_RIGHT:.0f}")
    errs, ref_errs, compiled = [], [], []
    for n in LEVELS:
        e, c = solve(n, fam)
        errs.append(e)
        compiled.append(c)
        r, _ = solve(n, "n1curl")
        ref_errs.append(r)
        print(f"level={n}x{n} relative_L2_error_under_test={e:.6e} "
              f"relative_L2_error_n1curl={r:.6e}")

    print(f"form_compiles_on_the_space_under_test={all(compiled)}")
    rates = [float(np.log2(errs[i] / errs[i + 1]))
             for i in range(len(errs) - 1)]
    print(f"observed_orders_under_test={[round(x, 3) for x in rates]}")
    n1curl_exact = all(r < 1e-10 for r in ref_errs)
    stalls = all(e > 0.05 for e in errs)
    slow = all(r < 1.0 for r in rates)
    print(f"n1curl_reproduces_the_exact_field_to_machine_precision="
          f"{n1curl_exact}")
    print(f"under_test_error_stays_above_5_percent={stalls}")
    print(f"under_test_order_is_below_one={slow}")

    if all(compiled) and n1curl_exact and stalls and slow:
        print("VERDICT=lagrange_cannot_carry_the_normal_jump_n1curl_is_exact")
        return 0
    print("VERDICT=space_under_test_converged_like_hcurl")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
