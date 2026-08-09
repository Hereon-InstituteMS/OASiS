"""Tier-2 for fenics stokes_darcy#7: a large permeability CONTRAST is not by
itself a preconditioner problem -- the geometry of the permeability field is.

Wrong variant: blaming the contrast number and, per the older knowledge text,
expecting block-Jacobi to stall with a residual ratio of about 1 on a
high-contrast Darcy pressure block. Right variant: look at where the
high-permeability region sits, and use hypre boomeramg.

Darcy pressure block -div(K grad p) = 0 on a 32x32 unit square, P1, p = 1 at
x = 0 and p = 0 at x = 1, CG at rtol 1e-8 with six preconditioners (jacobi,
bjacobi+ilu, icc, asm, gamg, hypre boomeramg). K is a DG0 field: the marked region
has K = 1 and the background K = contrast, so the marked region is the
high-permeability one. Three geometries: a single planar jump, a square
high-permeability island that touches no Dirichlet boundary, and a random
cell-by-cell field (seed 0, half the cells).

Observed on dolfinx 0.10.0 / PETSc 3.24.5, serial, taking the contrast from 1 to
1e-9: with the planar jump every preconditioner returns getConvergedReason() = 2
with a true relative residual at or below the requested tolerance and the counts
barely move -- jacobi 41 -> 41, bjacobi/icc/asm 27 -> 32, gamg 12 -> 19,
hypre 5 -> 5. Nothing stalls, and block-Jacobi never shows a residual ratio near 1.
Geometry is what bites: the disconnected island takes jacobi 41 -> 135 and gamg
12 -> 24, the random field takes jacobi 41 -> 181 and gamg 12 -> 17, while hypre
boomeramg stays at 5 to 7 iterations everywhere. The claim's factors are somewhat
larger than this 32x32 mesh gives (jacobi degrades about 4.4x on the random field,
not an order of magnitude, and gamg about 1.4x, not five times), but the ordering
and the conclusion hold exactly.

Mutation control: T2_MUTATE=1 makes both geometry slots under test the planar
jump, so the jacobi-degradation tokens go False.
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
from petsc4py import PETSc  # noqa: E402

DTYPE = dolfinx.default_scalar_type
N = 32
RTOL = 1e-8
PCS = (("jacobi", {}), ("bjacobi", {"sub_pc_type": "ilu"}), ("icc", {}),
       ("asm", {}), ("gamg", {}), ("hypre", {"pc_hypre_type": "boomeramg"}))


def build(geometry: str, contrast: float, seed: int = 0):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N)
    tdim = msh.topology.dim
    DG0 = fem.functionspace(msh, ("DG", 0))
    K = fem.Function(DG0)
    ncell = msh.topology.index_map(tdim).size_local
    msh.topology.create_connectivity(tdim, 0)
    conn = msh.topology.connectivity(tdim, 0)
    mids = np.array([msh.geometry.x[conn.links(c)].mean(axis=0)
                     for c in range(ncell)])
    K.x.array[:] = contrast
    if geometry == "planar":
        sel = mids[:, 1] > 0.5
    elif geometry == "island":
        sel = (np.abs(mids[:, 0] - 0.5) < 0.25) & (np.abs(mids[:, 1] - 0.5) < 0.25)
    elif geometry == "random":
        sel = np.random.default_rng(seed).random(ncell) < 0.5
    else:
        raise ValueError(geometry)
    K.x.array[np.where(sel)[0]] = 1.0

    V = fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = K * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(fem.Constant(msh, DTYPE(0.0)), v) * ufl.dx
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    left = mesh.locate_entities_boundary(msh, fdim,
                                       lambda x: np.isclose(x[0], 0.0))
    right = mesh.locate_entities_boundary(msh, fdim,
                                        lambda x: np.isclose(x[0], 1.0))
    g = fem.Function(V)
    g.interpolate(lambda x: 1.0 - x[0])
    bc = fem.dirichletbc(g, fem.locate_dofs_topological(
        V, fdim, np.concatenate([left, right])))
    af, Lf = fem.form(a), fem.form(L)
    A = dolfinx.fem.petsc.assemble_matrix(af, bcs=[bc])
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(Lf)
    dolfinx.fem.petsc.apply_lifting(b, [af], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, [bc])
    return msh, A, b


def solve(msh, A, b, pc_type: str, opts: dict):
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType(pc_type)
    prefix = f"t2sd7_{pc_type}_"
    ksp.setOptionsPrefix(prefix)
    o = PETSc.Options()
    for k, val in opts.items():
        o.setValue(prefix + k, val)
    ksp.setFromOptions()
    ksp.setTolerances(rtol=RTOL)
    x = A.createVecRight()
    ksp.solve(b, x)
    true_res = float((b - A * x).norm() / b.norm())
    return (int(ksp.getIterationNumber()), int(ksp.getConvergedReason()),
            true_res)


def measure(geometry: str, contrast: float) -> dict:
    msh, A, b = build(geometry, contrast)
    out = {}
    for pc_type, opts in PCS:
        out[pc_type] = solve(msh, A, b, pc_type, opts)
    print(f"geometry={geometry:8s} contrast={contrast:.0e} "
          + " ".join(f"{k}={v[0]}/r{v[1]}/{v[2]:.0e}" for k, v in out.items()))
    return out


def main() -> int:
    geo_b = "planar" if MUTATE else "island"
    geo_c = "planar" if MUTATE else "random"
    if MUTATE:
        print("mutation=both_geometry_slots_under_test_are_the_planar_jump")
    uniform = measure("planar", 1.0)
    planar = measure("planar", 1e-9)
    island = measure(geo_b, 1e-9)
    random = measure(geo_c, 1e-9)

    def flat(ref: dict, got: dict, factor: float) -> bool:
        return all(got[k][1] == 2 and got[k][2] < 1e-6
                   and got[k][0] <= factor * ref[k][0] for k in ref)

    planar_flat = flat(uniform, planar, 2.0)
    bj = [uniform["bjacobi"], planar["bjacobi"], island["bjacobi"],
          random["bjacobi"]]
    bj_fine = all(r[1] == 2 and r[2] < 1e-6 and r[0] <= 2 * bj[0][0] for r in bj)
    jac_random = random["jacobi"][0] / uniform["jacobi"][0]
    jac_island = island["jacobi"][0] / uniform["jacobi"][0]
    hypre_counts = [uniform["hypre"][0], planar["hypre"][0],
                    island["hypre"][0], random["hypre"][0]]
    hypre_flat = max(hypre_counts) <= 12 and \
        max(hypre_counts) <= 2 * min(hypre_counts)
    all_converged = all(r[1] == 2 for d in (uniform, planar, island, random)
                        for r in d.values())
    print(f"jacobi_factor_random_over_uniform={jac_random:.2f} "
          f"jacobi_factor_island_over_uniform={jac_island:.2f} "
          f"hypre_counts={hypre_counts}")
    print(f"every_preconditioner_converged_with_reason_two={all_converged}")
    print(f"planar_contrast_leaves_every_preconditioner_within_two_times="
          f"{planar_flat}")
    print(f"quoted_block_jacobi_stall_does_not_reproduce={bj_fine}")
    print(f"random_geometry_degrades_jacobi_by_over_three_times="
          f"{jac_random > 3.0}")
    print(f"disconnected_island_degrades_jacobi_by_over_two_and_a_half_times="
          f"{jac_island > 2.5}")
    print(f"hypre_boomeramg_stays_flat_and_under_a_dozen_iterations={hypre_flat}")
    if all_converged and planar_flat and bj_fine and hypre_flat \
            and jac_random > 3.0 and jac_island > 2.5:
        print("VERDICT=permeability_geometry_not_contrast_drives_the_counts")
        return 0
    print("VERDICT=contrast_alone_wrecked_the_preconditioners")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
