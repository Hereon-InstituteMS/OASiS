"""Tier-2 for fenics mixed_poisson#7: with a heterogeneous permeability K(x) the
flux term of the mixed form must be weighted by K^-1,
ufl.inner(Kinv * sigma, tau) * dx. Omitting the weight solves the homogeneous
K = 1 problem instead, and nothing in the run says so.

Geometry: unit square, two layers PARALLEL to the flow, K = 1 for y < 1/2 and
K = 100 for y > 1/2, pressure 1 at x=0 and 0 at x=1 imposed naturally, sigma.n=0
on the top and bottom walls. The exact solution is p = 1 - x with
sigma = K * (1, 0), so the flux must JUMP by the contrast of 100 across y = 1/2
and the total outflow is 0.5 * 1 + 0.5 * 100 = 50.5.

Observed on dolfinx 0.10.0 (16x16, RT1 x DG0):
  with Kinv     mean sigma_x = 1.000000 (bottom) / 100.000000 (top), outflow 50.5
  without Kinv  mean sigma_x = 1.000000 / 1.000000, outflow 1.0
so the computed flux is continuous across the layer interface instead of
jumping, exactly as the claim says, and the total flow is 50x too small.

Two parts of the claim did NOT reproduce and are pinned here as measured:
(1) the mass-balance residual is NOT O(1) for the unweighted solve -- div(sigma)
    is at round-off in both variants, because H(div) conformity and the second
    equation hold whatever the first equation weighs;
(2) "post-processing sigma = -K grad(p)" cannot even be evaluated on the DG0
    pressure of the stable pair: grad of a cellwise constant is identically
    zero, so the post-processed flux is 0 everywhere.

Mutation control: T2_MUTATE=1 puts the Kinv weight into the slot under test.
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
K_TOP = 100.0


def run(use_kinv: bool):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    S = basix.ufl.element("RT", msh.basix_cell(), 1)
    P = basix.ufl.element("DG", msh.basix_cell(), 0)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([S, P]))
    (sig, p) = ufl.TrialFunctions(W)
    (tau, q) = ufl.TestFunctions(W)
    n = ufl.FacetNormal(msh)
    x = ufl.SpatialCoordinate(msh)

    DG0 = dolfinx.fem.functionspace(msh, ("DG", 0))
    kinv = dolfinx.fem.Function(DG0)
    kinv.x.array[:] = 1.0
    top_cells = dolfinx.mesh.locate_entities(msh, tdim, lambda z: z[1] >= 0.5)
    kinv.x.array[top_cells] = 1.0 / K_TOP

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

    flux_term = (ufl.inner(kinv * sig, tau) if use_kinv
                 else ufl.inner(sig, tau)) * ufl.dx
    a = (flux_term - p * ufl.div(tau) * ufl.dx + q * ufl.div(sig) * ufl.dx)
    L = -1.0 * ufl.dot(tau, n) * ds(1)

    V0, _ = W.sub(0).collapse()
    bcs = [dolfinx.fem.dirichletbc(
        dolfinx.fem.Function(V0),
        dolfinx.fem.locate_dofs_topological((W.sub(0), V0), fdim, f_wall),
        W.sub(0))]

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
    up = ufl.conditional(ufl.gt(x[1], 0.5), 1.0, 0.0)
    dn = ufl.conditional(ufl.lt(x[1], 0.5), 1.0, 0.0)
    sc = lambda e: float(dolfinx.fem.assemble_scalar(dolfinx.fem.form(e)))
    top = sc(sh[0] * up * ufl.dx) / 0.5
    bot = sc(sh[0] * dn * ufl.dx) / 0.5
    out = sc(ufl.dot(sh, n) * ds(2))
    divres = np.sqrt(sc(ufl.div(sh) ** 2 * ufl.dx))
    postproc = np.sqrt(sc(ufl.inner(ufl.grad(ph), ufl.grad(ph))
                          / kinv**2 * ufl.dx))
    return top, bot, out, float(divres), float(postproc)


def main() -> int:
    t_t, b_t, o_t, dr_t, pp_t = run(use_kinv=MUTATE)
    t_r, b_r, o_r, dr_r, _ = run(use_kinv=True)
    ratio_t = t_t / b_t
    ratio_r = t_r / b_r
    print(f"reference_with_kinv: mean_sigma_x_bottom={b_r:.6f} "
          f"top={t_r:.6f} ratio={ratio_r:.6f} outflow={o_r:.6f} "
          f"div_residual={dr_r:.3e}")
    print(f"under_test: mean_sigma_x_bottom={b_t:.6f} top={t_t:.6f} "
          f"ratio={ratio_t:.6f} outflow={o_t:.6f} div_residual={dr_t:.3e}")
    print(f"kinv_flux_ratio_matches_the_contrast="
          f"{abs(ratio_r - K_TOP) < 1e-6 * K_TOP}")
    print(f"kinv_outflow_matches_the_layer_average="
          f"{abs(o_r - 0.5 * (1.0 + K_TOP)) < 1e-6}")
    flat = abs(ratio_t - 1.0) < 1e-9
    homog = abs(o_t - 1.0) < 1e-6
    print(f"no_kinv_flux_is_continuous_across_the_interface={flat}")
    print(f"no_kinv_outflow_is_the_homogeneous_value={homog}")
    print(f"mass_balance_residual_is_at_roundoff_in_both="
          f"{dr_t < 1e-10 and dr_r < 1e-10}")
    print(f"post_processed_k_grad_p_from_dg0_pressure_is_identically_zero="
          f"{pp_t == 0.0}")
    if (flat and homog and abs(ratio_r - K_TOP) < 1e-6 * K_TOP
            and abs(o_r - 0.5 * (1.0 + K_TOP)) < 1e-6):
        print("VERDICT=omitting_kinv_erases_the_permeability_contrast")
        return 0
    print("VERDICT=kinv_made_no_difference")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
