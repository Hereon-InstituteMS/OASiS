"""Tier-2 for fenics matrix_free_poisson#9: `dolfinx.PETScKrylovSolver` does not
exist in dolfinx 0.10.0. If you want PETSc's CG instead of a hand-written loop,
create it directly through petsc4py (PETSc.KSP().create(comm)); if you want
dolfinx's high-level wrapper for the ASSEMBLED problem, that is
dolfinx.fem.petsc.LinearProblem, which requires the keyword-only argument
petsc_options_prefix.

Wrong variant: dolfinx.PETScKrylovSolver(...), and LinearProblem(a, L, bcs=...)
without petsc_options_prefix. Right variant: PETSc.KSP().create(msh.comm), or
LinearProblem(..., petsc_options_prefix="mf_").

Observed on dolfinx 0.10.0 / petsc4py 3.24.5:
"AttributeError: module 'dolfinx' has no attribute 'PETScKrylovSolver'", and no
module in the dolfinx package tree exports that name at all (the scan over
dolfinx, dolfinx.fem, dolfinx.fem.petsc, dolfinx.la, dolfinx.nls,
dolfinx.nls.petsc and dolfinx.cpp returns an empty list). Omitting the keyword
gives "TypeError: LinearProblem.__init__() missing 1 required keyword-only
argument: 'petsc_options_prefix'". Both correct routes are exercised here and
agree on the same solution to 1e-12.

"Traceback" is deliberately NOT in this fixture's forbid_in_output: the TypeError
leaves a half-constructed LinearProblem whose __del__ then raises
"AttributeError: 'LinearProblem' object has no attribute '_solver'", and Python
prints that as an ignored-exception traceback on stderr. That noise is dolfinx's,
not the fixture's.

Mutation control: T2_MUTATE=1 skips the two wrong calls and only uses
PETSc.KSP().create plus LinearProblem(..., petsc_options_prefix=...), so the
AttributeError and TypeError texts never appear.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import importlib  # noqa: E402

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dolfinx import fem, mesh  # noqa: E402
from petsc4py import PETSc  # noqa: E402

DTYPE = dolfinx.default_scalar_type
N = 16
MODULES = ["dolfinx", "dolfinx.fem", "dolfinx.fem.petsc", "dolfinx.la",
           "dolfinx.nls", "dolfinx.nls.petsc", "dolfinx.cpp"]


def main() -> int:
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V = fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = 10.0 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    facets = mesh.exterior_facet_indices(msh.topology)
    bdofs = fem.locate_dofs_topological(V, tdim - 1, facets)
    uD = fem.Function(V, dtype=DTYPE)
    uD.interpolate(lambda X: 0.5 * X[0])
    bc = fem.dirichletbc(uD, bdofs)

    attr_err = ""
    type_err = ""
    if MUTATE:
        print("mutation=only_petsc4py_ksp_and_prefixed_linearproblem_are_used")
    else:
        try:
            dolfinx.PETScKrylovSolver  # the wrong variant
            print("dolfinx_petsckrylovsolver_exists=True")
        except AttributeError as exc:
            attr_err = f"{type(exc).__name__}: {exc}"
            print(f"dolfinx.PETScKrylovSolver -> {attr_err}")
        try:
            dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc])
            print("linearproblem_without_prefix_constructed=True")
        except TypeError as exc:
            type_err = f"{type(exc).__name__}: {exc}"
            print(f"LinearProblem(a, L, bcs=[bc]) -> {type_err}")

    found = []
    for name in MODULES:
        try:
            m = importlib.import_module(name)
        except Exception:  # noqa: BLE001
            continue
        found += [f"{name}.{n}" for n in dir(m) if "krylov" in n.lower()]
    print(f"names_containing_krylov_in_dolfinx_tree={found}")

    # correct route 1: petsc4py CG on the assembled operator
    af, Lf = fem.form(a), fem.form(L)
    A = dolfinx.fem.petsc.assemble_matrix(af, bcs=[bc])
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(Lf)
    dolfinx.fem.petsc.apply_lifting(b, [af], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, [bc])
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("jacobi")
    ksp.setTolerances(rtol=1e-12)
    uh = fem.Function(V, dtype=DTYPE)
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    reason = int(ksp.getConvergedReason())
    print(f"petsc4py_ksp_type={ksp.getType()} converged_reason={reason} "
          f"iterations={ksp.getIterationNumber()}")

    # correct route 2: LinearProblem WITH the keyword-only prefix
    lp = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix="t2mf_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    out = lp.solve()
    uh2 = out[0] if isinstance(out, tuple) else out
    diff = float(np.abs(uh.x.array - uh2.x.array).max())
    print(f"ksp_vs_linearproblem_max_diff={diff:.3e}")

    print(f"petsckrylovsolver_raises_attributeerror={bool(attr_err)}")
    print(f"no_krylov_name_anywhere_in_dolfinx={found == []}")
    print(f"linearproblem_without_prefix_raises_typeerror={bool(type_err)}")
    print(f"petsc4py_ksp_converged={reason > 0}")
    print(f"prefixed_linearproblem_agrees_with_petsc4py_ksp={diff < 1e-12}")
    if attr_err and type_err and found == [] and reason > 0 and diff < 1e-12:
        print("VERDICT=no_petsckrylovsolver_and_linearproblem_needs_the_prefix")
        return 0
    print("VERDICT=petsckrylovsolver_or_prefixless_linearproblem_worked")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
