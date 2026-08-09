"""Tier-2 for fenics biharmonic#2: interior penalty needs interior-facet (dS)
integrals, every facet is visited from both sides, and the matrix assembly is
much more expensive than the equivalent second-order problem on the same space.
The mixed (u, sigma) formulation avoids dS at the cost of doubling the dof
count. Measured here on a 32x32 unit square, P2, best of five assemblies each,
so the comparison is between three forms on identical meshes and identical
elements.

Wrong assumption: that a C0-IP biharmonic matrix costs about what a Poisson
matrix costs. Right variant, and the mutation: the mixed formulation
(sigma, u) in P2 x P2, whose bilinear form has no dS term at all.

Observed on dolfinx 0.10.0: the C0-IP assembly takes 0.0349 s against Poisson's
0.0028 s on the same 4225-dof P2 space, a factor of 12.5 — above the 5-10x the
claim quotes, so treat 5-10x as a lower bound rather than a range. The mixed
form assembles in 0.0116 s and has exactly 8450 dofs, twice the C0-IP space,
which is the trade the claim describes; normalised per dof it costs 2.1x
Poisson against the C0-IP form's 12.5x, so the gap really is the facet
integrals and not the problem size.

Mutation control: T2_MUTATE=1 makes the mixed dS-free form the primary one, its
cost ratio drops below the threshold and the fixture loses its own expectation.
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

import time  # noqa: E402

import basix.ufl  # noqa: E402

N = 32
DEGREE = 2
REPS = 5


def mesh():
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    return msh


def c0ip_form():
    msh = mesh()
    V = dolfinx.fem.functionspace(msh, ("Lagrange", DEGREE))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    nrm = ufl.FacetNormal(msh)
    h = ufl.CellDiameter(msh)
    al = dolfinx.fem.Constant(msh, 8.0)
    a = (ufl.inner(ufl.div(ufl.grad(u)), ufl.div(ufl.grad(v))) * ufl.dx
         - ufl.inner(ufl.avg(ufl.div(ufl.grad(u))),
                     ufl.jump(ufl.grad(v), nrm)) * ufl.dS
         - ufl.inner(ufl.jump(ufl.grad(u), nrm),
                     ufl.avg(ufl.div(ufl.grad(v)))) * ufl.dS
         + al / ufl.avg(h) * ufl.inner(ufl.jump(ufl.grad(u), nrm),
                                       ufl.jump(ufl.grad(v), nrm)) * ufl.dS)
    ndof = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    return dolfinx.fem.form(a), ndof


def poisson_form():
    msh = mesh()
    V = dolfinx.fem.functionspace(msh, ("Lagrange", DEGREE))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    ndof = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    return dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx), ndof


def mixed_form():
    msh = mesh()
    P = basix.ufl.element("Lagrange", msh.basix_cell(), DEGREE)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P, P]))
    (s, u) = ufl.TrialFunctions(W)
    (t, v) = ufl.TestFunctions(W)
    a = (s * t + ufl.inner(ufl.grad(u), ufl.grad(t))
         + ufl.inner(ufl.grad(s), ufl.grad(v))) * ufl.dx
    ndof = W.dofmap.index_map.size_global * W.dofmap.index_map_bs
    return dolfinx.fem.form(a), ndof


def best_assembly_time(form) -> float:
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        A = dolfinx.fem.petsc.assemble_matrix(form)
        A.assemble()
        times.append(time.perf_counter() - t0)
        A.destroy()
    return float(min(times))


def main() -> int:
    f_ip, n_ip = c0ip_form()
    f_po, n_po = poisson_form()
    f_mx, n_mx = mixed_form()
    t_ip = best_assembly_time(f_ip)
    t_po = best_assembly_time(f_po)
    t_mx = best_assembly_time(f_mx)
    t_primary, n_primary = (t_mx, n_mx) if MUTATE else (t_ip, n_ip)
    ratio = t_primary / t_po
    # Per dof, so the mixed form is not penalised for having twice as many.
    per_dof = (t_primary / n_primary) / (t_po / n_po)
    print(f"poisson_ndofs={n_po} poisson_assemble_s={t_po:.4f}")
    print(f"c0ip_ndofs={n_ip} c0ip_assemble_s={t_ip:.4f} "
          f"c0ip_over_poisson_per_dof="
          f"{(t_ip / n_ip) / (t_po / n_po):.2f}")
    print(f"mixed_ndofs={n_mx} mixed_assemble_s={t_mx:.4f} "
          f"mixed_over_poisson={t_mx / t_po:.2f} "
          f"mixed_over_poisson_per_dof={(t_mx / n_mx) / (t_po / n_po):.2f}")
    print(f"primary_over_poisson_assembly_ratio={ratio:.2f} "
          f"primary_over_poisson_per_dof={per_dof:.2f}")

    expensive = per_dof > 5.0
    dS_beats_nothing = t_ip > t_mx
    doubled = n_mx == 2 * n_ip
    print(f"primary_assembly_costs_more_than_5x_poisson_per_dof={expensive}")
    print(f"interior_penalty_costs_more_than_the_ds_free_mixed_form="
          f"{dS_beats_nothing}")
    print(f"mixed_formulation_doubles_the_dof_count={doubled}")
    if expensive and dS_beats_nothing and doubled:
        print("VERDICT=interior_facet_integrals_dominate_biharmonic_assembly")
        return 0
    print("VERDICT=interior_penalty_assembly_was_not_more_expensive")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
