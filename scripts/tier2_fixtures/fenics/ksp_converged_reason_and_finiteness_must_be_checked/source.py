"""Tier-2 for fenics dg_methods#2: ALWAYS check ksp.getConvergedReason() and
the finiteness of the solution array — a DG solve can fail completely and still
leave the process looking healthy.

Wrong variant: the script that solves and prints. It is reproduced literally:
the fixture writes a small child script that assembles the DG advection
operator with the boundary term written as the raw dot(b, n)*u*v*ds (the defect
that makes the operator singular), solves with LU, prints the solution range
and stops. The child is then executed as a separate process so its exit status
is a real measurement and not an assertion.

Observed: the child prints "u: min=inf, max=inf" and exits with status 0. The
parent repeats the same solve in-process and finds KSPConvergedReason = -11
(KSP_DIVERGED_PC_FAILED) and every single dof non-finite. Nothing was raised,
nothing was logged, the exit status is that of a successful run.

Mutation control: T2_MUTATE=1 gives the child the outflow-restricted boundary
term, i.e. the correct form. The child then prints a finite range, the reason
is positive, and the "silent inf" expectations disappear.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402
from petsc4py import PETSc  # noqa: E402

from dolfinx import fem, mesh  # noqa: E402
from dolfinx.fem.petsc import assemble_matrix, assemble_vector  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

# (coefficient of b.n, coefficient of |b.n|) in the ds advection term
COEFFS = (0.5, 0.5) if MUTATE else (1.0, 0.0)

CHILD = '''
import numpy as np, ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
msh = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.triangle)
msh.topology.create_connectivity(1, 2)
V = fem.functionspace(msh, ("DG", 1))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
b = ufl.as_vector([1.0, 0.5])
n = ufl.FacetNormal(msh)
bn = ufl.dot(b, n)
up = ((bn("+") + abs(bn("+")))/2.0*u("+") + (bn("+") - abs(bn("+")))/2.0*u("-"))
c1 = fem.Constant(msh, {c1})
c2 = fem.Constant(msh, {c2})
a = (-ufl.inner(u*b, ufl.grad(v))*ufl.dx + up*ufl.jump(v)*ufl.dS
     + (c1*bn + c2*abs(bn))*u*v*ufl.ds)
f = fem.Constant(msh, 1.0)
A = assemble_matrix(fem.form(a)); A.assemble()
rhs = assemble_vector(fem.form(f*v*ufl.dx))
rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A); ksp.setType("preonly"); ksp.getPC().setType("lu")
uh = fem.Function(V)
ksp.solve(rhs, uh.x.petsc_vec)
uh.x.scatter_forward()
arr = uh.x.array
print(f"u: min={{arr.min():.6e}}, max={{arr.max():.6e}}")
'''


def main() -> int:
    # ---- the same solve in-process, with the checks the child omits ----
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.triangle)
    msh.topology.create_connectivity(1, 2)
    V = fem.functionspace(msh, ("DG", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    b = ufl.as_vector([1.0, 0.5])
    n = ufl.FacetNormal(msh)
    bn = ufl.dot(b, n)
    up = ((bn("+") + abs(bn("+"))) / 2.0 * u("+")
          + (bn("+") - abs(bn("+"))) / 2.0 * u("-"))
    c1 = fem.Constant(msh, COEFFS[0])
    c2 = fem.Constant(msh, COEFFS[1])
    a = (-ufl.inner(u * b, ufl.grad(v)) * ufl.dx
         + up * ufl.jump(v) * ufl.dS
         + (c1 * bn + c2 * abs(bn)) * u * v * ufl.ds)
    f = fem.Constant(msh, 1.0)
    A = assemble_matrix(fem.form(a))
    A.assemble()
    rhs = assemble_vector(fem.form(f * v * ufl.dx))
    rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    uh = fem.Function(V)
    raised = ""
    try:
        ksp.solve(rhs, uh.x.petsc_vec)
    except Exception as exc:                            # pragma: no cover
        raised = f"{type(exc).__name__}: {exc}"
    uh.x.scatter_forward()
    reason = ksp.getConvergedReason()
    arr = uh.x.array
    n_bad = int(np.sum(~np.isfinite(arr)))
    print(f"solve_raised_nothing={raised == ''}")
    print(f"ksp_converged_reason={reason} reason_is_diverged_pc_failed="
          f"{reason == -11}")
    print(f"nonfinite_dofs={n_bad}/{arr.size} all_dofs_nonfinite="
          f"{n_bad == arr.size}")
    print(f"isfinite_check_would_catch_it={not np.all(np.isfinite(arr))}")

    # ---- the unchecked script, as its own process ----------------------
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "unchecked.py")
        with open(path, "w") as fh:
            fh.write(CHILD.format(c1=COEFFS[0], c2=COEFFS[1]))
        res = subprocess.run([sys.executable, path], capture_output=True,
                             text=True, timeout=600)
    out = (res.stdout or "") + (res.stderr or "")
    tail = [ln for ln in out.splitlines() if ln.startswith("u: min=")]
    print(f"unchecked_script_exit_status={res.returncode}")
    print(f"unchecked_script_exit_status_is_zero={res.returncode == 0}")
    for ln in tail:
        print(f"unchecked_script_printed -> {ln}")
    printed_inf = any("min=inf" in ln for ln in tail)
    print(f"unchecked_script_printed_inf={printed_inf}")

    if (raised == "" and reason == -11 and n_bad == arr.size
            and res.returncode == 0 and printed_inf):
        print("VERDICT=silent_inf_with_exit_status_zero")
        return 0
    print("VERDICT=failure_was_not_silent")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
