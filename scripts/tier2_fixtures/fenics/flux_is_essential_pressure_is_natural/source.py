"""Tier-2 for fenics mixed_poisson#6: in the mixed form the ESSENTIAL condition
is the normal flux sigma.n; the pressure is NATURAL and enters through the facet
term -u_D * dot(tau, n) * ds(tag) in the linear form. Putting a dolfinx
dirichletbc on the pressure subspace instead constrains the discontinuous
pressure dofs directly and the answer is nonsense.

Problem: unit square, K = 1, no source, sigma.n = 0 on the top and bottom walls
(essential, on the RT subspace), pressure 1 at x=0 and 0 at x=1. The exact
solution is p = 1 - x, sigma = (1, 0), so the flux leaving through x=1 is
exactly 1 and the global balance of sigma.n over the whole boundary is 0.

Wrong variant (the default run): drop the facet term and instead apply
fem.dirichletbc on W.sub(1) with value 1 in the cells touching x=0 and 0 in the
cells touching x=1.

Observed on dolfinx 0.10.0 (16x16, RT1 x DG0):
  natural pressure term      outflow at x=1 = 1.000000, boundary balance 0
  dirichletbc on the pressure  outflow at x=1 = 0.127551, inflow at x=0 = +54.31
The pressure is pinned to exactly 1.0 and 0.0 in the wall cells, the interior
profile is not 1 - x, and because the Dirichlet rows replace the divergence
equations of those cells, global mass balance is off by O(10). Note also that
locate_dofs_topological on the DG0 pressure subspace with FACET entities returns
zero dofs -- a pressure dirichletbc written the usual way is silently empty, so
the fixture has to look the dofs up by cell to make the pitfall happen at all.

Mutation control: T2_MUTATE=1 puts the natural facet term in the slot under
test, and the outflow is 1.0 with the balance at round-off.
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

from petsc4py import PETSc  # noqa: E402

N = 16


def run(natural: bool):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    msh.topology.create_connectivity(tdim, tdim)
    S = basix.ufl.element("RT", msh.basix_cell(), 1)
    P = basix.ufl.element("DG", msh.basix_cell(), 0)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([S, P]))
    (sig, p) = ufl.TrialFunctions(W)
    (tau, q) = ufl.TestFunctions(W)
    n = ufl.FacetNormal(msh)
    x = ufl.SpatialCoordinate(msh)

    f_left = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda z: np.isclose(z[0], 0.0))
    f_right = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda z: np.isclose(z[0], 1.0))
    f_wall = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda z: np.isclose(z[1], 0.0) | np.isclose(z[1], 1.0))
    idx = np.concatenate([f_left, f_right, f_wall])
    val = np.concatenate([np.full(len(f_left), 1), np.full(len(f_right), 2),
                          np.full(len(f_wall), 3)]).astype(np.int32)
    order = np.argsort(idx)
    tags = dolfinx.mesh.meshtags(msh, fdim, idx[order], val[order])
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)

    a = (ufl.inner(sig, tau) * ufl.dx
         - p * ufl.div(tau) * ufl.dx
         + q * ufl.div(sig) * ufl.dx)

    V0, _ = W.sub(0).collapse()
    Q0, qmap = W.sub(1).collapse()
    zero_flux = dolfinx.fem.Function(V0)
    bcs = [dolfinx.fem.dirichletbc(
        zero_flux,
        dolfinx.fem.locate_dofs_topological((W.sub(0), V0), fdim, f_wall),
        W.sub(0))]
    n_facet_dofs = -1
    if natural:
        # pressure imposed NATURALLY: p_D = 1 on ds(1), p_D = 0 on ds(2)
        L = -1.0 * ufl.dot(tau, n) * ds(1) - 0.0 * ufl.dot(tau, n) * ds(2)
    else:
        L = dolfinx.fem.Constant(msh, 0.0) * ufl.div(tau) * ufl.dx
        h = 1.0 / N
        cl = dolfinx.mesh.locate_entities(msh, tdim, lambda z: z[0] <= 1.01 * h)
        cr = dolfinx.mesh.locate_entities(
            msh, tdim, lambda z: z[0] >= 1.0 - 1.01 * h)
        n_facet_dofs = len(dolfinx.fem.locate_dofs_topological(
            (W.sub(1), Q0), fdim, f_left)[0])
        dl = dolfinx.fem.locate_dofs_topological((W.sub(1), Q0), tdim, cl)
        dr = dolfinx.fem.locate_dofs_topological((W.sub(1), Q0), tdim, cr)
        g_one = dolfinx.fem.Function(Q0)
        g_one.x.array[dl[1]] = 1.0
        g_zero = dolfinx.fem.Function(Q0)
        bcs.append(dolfinx.fem.dirichletbc(g_one, dl, W.sub(1)))
        bcs.append(dolfinx.fem.dirichletbc(g_zero, dr, W.sub(1)))

    a_f, L_f = dolfinx.fem.form(a), dolfinx.fem.form(L)
    A = dolfinx.fem.petsc.assemble_matrix(a_f, bcs=bcs)
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(L_f)
    dolfinx.fem.petsc.apply_lifting(b, [a_f], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs)
    # dense LAPACK LU: PETSc's sparse LU does not pivot and stumbles on the
    # zero pressure block of any saddle-point matrix
    Ad = A.copy().convert("dense")
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(Ad)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.setErrorIfNotConverged(True)
    w = dolfinx.fem.Function(W)
    ksp.solve(b, w.x.petsc_vec)
    w.x.scatter_forward()

    sh, ph = ufl.split(w)
    out = float(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(ufl.dot(sh, n) * ds(2))))
    inn = float(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(ufl.dot(sh, n) * ds(1))))
    perr = float(np.sqrt(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form((ph - (1.0 - x[0])) ** 2 * ufl.dx))))
    parr = w.x.array[np.array(qmap, dtype=np.int32)]
    return out, inn, out + inn, perr, float(parr.min()), float(parr.max()), \
        n_facet_dofs


def main() -> int:
    o_t, i_t, bal_t, err_t, lo_t, hi_t, nfd = run(natural=MUTATE)
    o_r, i_r, bal_r, err_r, _, _, _ = run(natural=True)
    print(f"reference_natural: outflow={o_r:.6f} inflow={i_r:.6f} "
          f"boundary_balance={bal_r:.3e} p_l2_error={err_r:.4e}")
    print(f"under_test: outflow={o_t:.6f} inflow={i_t:.6f} "
          f"boundary_balance={bal_t:.3e} p_l2_error={err_t:.4e} "
          f"p_range=[{lo_t:.4f}, {hi_t:.4f}]")
    if nfd >= 0:
        print(f"dg0_pressure_dofs_found_on_boundary_facets={nfd}")
        print(f"facet_lookup_on_dg0_pressure_is_empty={nfd == 0}")
    print(f"natural_pressure_term_gives_the_analytic_outflow="
          f"{abs(o_r - 1.0) < 1e-9}")
    print(f"natural_pressure_term_conserves_boundary_mass={abs(bal_r) < 1e-9}")
    wrong_flux = abs(o_t - 1.0) > 0.5
    broken_mass = abs(bal_t) > 1.0
    pinned = abs(hi_t - 1.0) < 1e-12 and abs(lo_t) < 1e-12
    worse_p = err_t > 3 * err_r
    print(f"pressure_dirichletbc_outflow_is_wrong={wrong_flux}")
    print(f"pressure_dirichletbc_breaks_boundary_mass_balance={broken_mass}")
    print(f"pressure_dirichletbc_pins_the_wall_cells_exactly={pinned}")
    print(f"pressure_dirichletbc_profile_error_is_over_3x_worse={worse_p}")
    if (abs(o_r - 1.0) < 1e-9 and abs(bal_r) < 1e-9 and wrong_flux
            and broken_mass and pinned and worse_p and nfd == 0):
        print("VERDICT=pressure_must_be_natural_flux_must_be_essential")
        return 0
    print("VERDICT=pressure_dirichletbc_worked")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
