"""Tier-2 for fenics heat#0: an insulated wall is a NATURAL condition. Writing
T = 0 there solves a different problem, and nothing complains.

Steady conduction on the unit square, k = 1, left wall T = 1, top/bottom/right
physically insulated. Correct model: only the left condition, so the answer is
uniform T = 1 and the net flux through top+bottom is machine zero. Wrong model:
add T = 0 on top and bottom — the field now spans [0, 1] and those walls carry
a large net flux they must not carry.

Mutation control: T2_MUTATE=1 drops the spurious Dirichlet conditions.
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


def solve(spurious: bool):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 16, 16)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = dolfinx.fem.Constant(msh, 0.0) * v * ufl.dx

    left = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 0.0))
    bcs = [dolfinx.fem.dirichletbc(
        dolfinx.fem.Constant(msh, 1.0),
        dolfinx.fem.locate_dofs_topological(V, fdim, left), V)]
    tb = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    if spurious:
        bcs.append(dolfinx.fem.dirichletbc(
            dolfinx.fem.Constant(msh, 0.0),
            dolfinx.fem.locate_dofs_topological(V, fdim, tb), V))

    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=bcs, petsc_options_prefix="t2_ad_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = prob.solve()
    if isinstance(uh, tuple):
        uh = uh[0]

    tags = dolfinx.mesh.meshtags(msh, fdim, np.sort(tb),
                                np.full(len(tb), 7, dtype=np.int32))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
    n = ufl.FacetNormal(msh)
    flux = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(ufl.dot(ufl.grad(uh), n) * ds(7)))
    return (float(np.min(uh.x.array)), float(np.max(uh.x.array)),
            float(flux))


def main() -> int:
    lo_w, hi_w, flux_w = solve(spurious=not MUTATE)
    lo_c, hi_c, flux_c = solve(spurious=False)
    print(f"natural_only_min={lo_c:.6f} natural_only_max={hi_c:.6f} "
          f"natural_only_flux={flux_c:.3e}")
    print(f"with_dirichlet_zero_min={lo_w:.6f} "
          f"with_dirichlet_zero_max={hi_w:.6f} "
          f"with_dirichlet_zero_flux={flux_w:.3e}")
    uniform = abs(hi_c - 1.0) < 1e-10 and abs(lo_c - 1.0) < 1e-10
    tight = abs(flux_c) < 1e-10
    broken = abs(lo_w) < 1e-12 and abs(flux_w) > 1.0
    print(f"natural_only_is_uniform_one={uniform}")
    print(f"natural_only_flux_is_machine_zero={tight}")
    print(f"dirichlet_zero_pulls_field_to_zero_and_leaks_flux={broken}")
    if uniform and tight and broken:
        print("VERDICT=dirichlet_zero_on_insulated_wall_changes_the_problem")
        return 0
    print("VERDICT=no_difference")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
