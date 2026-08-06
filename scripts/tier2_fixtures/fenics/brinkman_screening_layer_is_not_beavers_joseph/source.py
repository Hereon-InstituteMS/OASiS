"""Tier-2 for fenics stokes_darcy#6: the single-mesh Brinkman form does NOT impose
the Beavers-Joseph-Saffman interface condition. There is one H1 velocity field over
both regions, so the velocity -- including its tangential component -- is
continuous across the interface by construction and no slip jump and no alpha_BJ
parameter exist anywhere in the form. What the model does produce is a Brinkman
screening layer whose decay length is sqrt(K/mu_eff); size the mesh near the
interface from sqrt(K), not from the geometry.

Wrong variant: choose the mesh from the geometry, here cells about ten times
sqrt(K) tall. Right variant: refine until the cell size reaches sqrt(K).

Pressure-driven channel on [0,1]^2, quadrilaterals, Taylor-Hood Q2/Q1, porous
lower half, mu = mu_eff = 1, unit pressure at x = 0 and zero at x = 1, no-slip on
the two horizontal walls, MUMPS LU. The mesh is deliberately anisotropic (4 cells
across the channel, many along it) because the solution is one-dimensional: the
largest mesh used, 4 x 320, costs about the same as a 32x32 one and the whole
fixture solves in a few seconds.

Observed on dolfinx 0.10.0: the compiled bilinear form carries cell integrals only
-- there is no interface facet integral and no slip parameter to set -- and the
tangential velocity evaluated at the interface point from the cells above and from
the cells below agrees to 1.4e-17, i.e. there is no jump to measure. On a resolved
mesh the fitted decay length equals sqrt(K/mu_eff) to within 0.2% at K = 2.5e-3,
6.25e-4 and 1.5625e-4 (ratios 0.9988, 1.0000, 1.0000). NOTE the claim says the
decay length approaches sqrt(K/mu_eff) FROM ABOVE; on these resolved meshes the
ratio is at or just below one, so it approaches from below, and only the limit is
confirmed here. With cells at ten times sqrt(K) nothing is raised and the KSP
reason is 4 at every resolution, but the interface velocity comes out at 0.6864 of
its resolved value (31% low, "roughly a third") and the total flux at 0.9778 (2.2%
low, "a couple of percent"); refining until h = sqrt(K) brings those to 0.9994 and
1.0000.

Mutation control: T2_MUTATE=1 sizes the mesh under test from sqrt(K) instead of
the geometry, so the under-resolution tokens go False.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import basix.ufl  # noqa: E402
import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402
import dolfinx.geometry  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dolfinx import fem, mesh  # noqa: E402
from petsc4py import PETSc  # noqa: E402

MU, MU_EFF, PIN, NX = 1.0, 1.0, 1.0, 4
TOL = 1e-12
K_MAIN = 1.5625e-4          # sqrt(K) = 0.0125
K_SWEEP = (2.5e-3, 6.25e-4, 1.5625e-4)
NY_SWEEP = (80, 160, 320)   # h = sqrt(K)/1, /1, /4 -- all resolved


def solve(K: float, ny: int):
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(
        comm, [np.array([0.0, 0.0]), np.array([1.0, 1.0])], [NX, ny],
        cell_type=mesh.CellType.quadrilateral)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    porous = mesh.locate_entities(msh, tdim, lambda x: x[1] <= 0.5 + TOL)
    marks = np.full(msh.topology.index_map(tdim).size_local, 1, dtype=np.int32)
    marks[porous] = 2
    ct = mesh.meshtags(msh, tdim, np.arange(len(marks), dtype=np.int32), marks)
    inlet = mesh.locate_entities_boundary(msh, fdim,
                                         lambda x: np.isclose(x[0], 0.0))
    outlet = mesh.locate_entities_boundary(msh, fdim,
                                          lambda x: np.isclose(x[0], 1.0))
    wall = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    fi = np.concatenate([inlet, outlet])
    fv = np.concatenate([np.full(len(inlet), 1, np.int32),
                         np.full(len(outlet), 2, np.int32)])
    o = np.argsort(fi)
    ft = mesh.meshtags(msh, fdim, fi[o], fv[o])
    Ve = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(tdim,))
    Pe = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = fem.functionspace(msh, basix.ufl.mixed_element([Ve, Pe]))
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    dx = ufl.Measure("dx", domain=msh, subdomain_data=ct)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
    nv = ufl.FacetNormal(msh)
    # The whole coupled form: one velocity space, a Darcy drag term on the porous
    # cells, and nothing whatsoever on the interface.
    a = (MU_EFF * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
         + (MU / K) * ufl.inner(u, v) * dx(2)
         - p * ufl.div(v) * dx - q * ufl.div(u) * dx)
    L = -PIN * ufl.dot(v, nv) * ds(1)
    V0, _ = W.sub(0).collapse()
    zero = fem.Function(V0)
    bcs = [fem.dirichletbc(
        zero, fem.locate_dofs_topological((W.sub(0), V0), fdim, wall), W.sub(0))]
    af, Lf = fem.form(a), fem.form(L)
    A = dolfinx.fem.petsc.assemble_matrix(af, bcs=bcs)
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(Lf)
    dolfinx.fem.petsc.apply_lifting(b, [af], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs)
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    wh = fem.Function(W)
    raised = ""
    try:
        ksp.solve(b, wh.x.petsc_vec)
        wh.x.scatter_forward()
    except Exception as exc:  # noqa: BLE001
        raised = f"{type(exc).__name__}: {exc}"
    uh = wh.sub(0).collapse()
    flux = float(fem.assemble_scalar(fem.form(ufl.dot(uh, nv) * ds(2))))
    types = sorted({i.integral_type() for i in a.integrals()})
    n_vel_spaces = len({W.sub(0).element.signature})
    return dict(msh=msh, uh=uh, flux=flux, raised=raised,
                reason=int(ksp.getConvergedReason()), types=types,
                n_velocity_spaces=n_vel_spaces)


def eval_both_sides(msh, uh, point=(0.5, 0.5, 0.0)):
    pt = np.array([list(point)])
    tree = dolfinx.geometry.bb_tree(msh, msh.topology.dim)
    cand = dolfinx.geometry.compute_collisions_points(tree, pt)
    coll = dolfinx.geometry.compute_colliding_cells(msh, cand, pt)
    cells = coll.links(0)
    vals = [float(uh.eval(pt[0], c)[0]) for c in cells]
    return vals


def fit_decay_length(msh, uh, K):
    d = np.sqrt(K * MU_EFF / MU)
    ys = 0.5 - np.linspace(0.4 * d, 3.0 * d, 12)
    pts = np.array([[0.5, y, 0.0] for y in ys])
    tree = dolfinx.geometry.bb_tree(msh, msh.topology.dim)
    cand = dolfinx.geometry.compute_collisions_points(tree, pts)
    coll = dolfinx.geometry.compute_colliding_cells(msh, cand, pts)
    ux = np.array([float(uh.eval(pts[i], coll.links(i)[0])[0])
                   for i in range(pts.shape[0])])
    u_darcy = K / MU * PIN          # dp/dx = -1 over a unit length
    val = ux - u_darcy
    ok = val > 1e-14
    slope = np.polyfit(0.5 - ys[ok], np.log(val[ok]), 1)[0]
    return -1.0 / slope


def main() -> int:
    sq = np.sqrt(K_MAIN * MU_EFF / MU)
    ny_coarse = int(round(1.0 / (10.0 * sq)))     # h = 10*sqrt(K)
    ny_fine = int(round(1.0 / sq))                # h = sqrt(K)
    ny_test = ny_fine if MUTATE else ny_coarse
    if MUTATE:
        print("mutation=mesh_under_test_sized_from_sqrt_K")
    print(f"K={K_MAIN:.4e} sqrt_K_over_mu_eff={sq:.5f} "
          f"ny_from_geometry={ny_coarse} ny_from_sqrt_K={ny_fine}")

    tested = solve(K_MAIN, ny_test)
    fine = solve(K_MAIN, ny_fine)
    ref = solve(K_MAIN, 4 * ny_fine)

    sides = eval_both_sides(tested["msh"], tested["uh"])
    jump = max(sides) - min(sides)
    print(f"cells_meeting_at_the_interface_point={len(sides)} "
          f"tangential_velocity_from_each={['%.9e' % s for s in sides]} "
          f"jump={jump:.3e}")
    print(f"bilinear_form_integral_types={tested['types']} "
          f"velocity_spaces_over_the_two_regions={tested['n_velocity_spaces']}")

    ui_t = eval_both_sides(tested["msh"], tested["uh"])[0]
    ui_f = eval_both_sides(fine["msh"], fine["uh"])[0]
    ui_r = eval_both_sides(ref["msh"], ref["uh"])[0]
    print(f"h_over_sqrtK={1 / ny_test / sq:.2f} interface_velocity={ui_t:.6e} "
          f"flux={tested['flux']:.6e} ksp_reason={tested['reason']} "
          f"raised={tested['raised'][:40]!r}")
    print(f"h_over_sqrtK={1 / ny_fine / sq:.2f} interface_velocity={ui_f:.6e} "
          f"flux={fine['flux']:.6e} ksp_reason={fine['reason']}")
    print(f"h_over_sqrtK={1 / (4 * ny_fine) / sq:.2f} reference_interface_velocity="
          f"{ui_r:.6e} reference_flux={ref['flux']:.6e} ksp_reason={ref['reason']}")
    u_ratio_t, f_ratio_t = ui_t / ui_r, tested["flux"] / ref["flux"]
    u_ratio_f, f_ratio_f = ui_f / ui_r, fine["flux"] / ref["flux"]
    print(f"under_test_interface_velocity_over_reference={u_ratio_t:.4f} "
          f"under_test_flux_over_reference={f_ratio_t:.4f}")
    print(f"sqrt_K_mesh_interface_velocity_over_reference={u_ratio_f:.4f} "
          f"sqrt_K_mesh_flux_over_reference={f_ratio_f:.4f}")

    deltas = []
    for K, ny in zip(K_SWEEP, NY_SWEEP):
        r = solve(K, ny)
        d = fit_decay_length(r["msh"], r["uh"], K)
        deltas.append(d / np.sqrt(K * MU_EFF / MU))
        print(f"K={K:.4e} resolved_ny={ny} fitted_decay_length={d:.5f} "
              f"over_sqrt_K_over_mu_eff={deltas[-1]:.4f}")

    single_field = tested["n_velocity_spaces"] == 1
    no_interface_term = tested["types"] == ["cell"]
    continuous = jump < 1e-12
    quiet = all(r["raised"] == "" and r["reason"] == 4
                for r in (tested, fine, ref))
    third_low = 0.55 <= u_ratio_t <= 0.80
    flux_off = 0.005 <= (1.0 - f_ratio_t) <= 0.05
    refined_ok = u_ratio_f > 0.99 and f_ratio_f > 0.999
    decay_matches = all(abs(x - 1.0) < 0.01 for x in deltas)
    print(f"velocity_is_one_h1_field_over_both_regions={single_field}")
    print(f"form_has_cell_integrals_only_no_interface_coupling={no_interface_term}")
    print(f"tangential_velocity_jump_across_the_interface_is_machine_zero="
          f"{continuous}")
    print(f"no_error_message_at_any_resolution={quiet}")
    print(f"decay_length_equals_sqrt_K_over_mu_eff_on_resolved_meshes="
          f"{decay_matches}")
    print(f"cells_at_ten_times_sqrt_K_put_the_interface_velocity_a_third_low="
          f"{third_low}")
    print(f"cells_at_ten_times_sqrt_K_cost_a_couple_of_percent_of_the_flux="
          f"{flux_off}")
    print(f"refining_to_h_equal_sqrt_K_removes_both_errors={refined_ok}")
    if single_field and no_interface_term and continuous and quiet \
            and decay_matches and third_low and flux_off and refined_ok:
        print("VERDICT=brinkman_has_no_slip_jump_and_needs_a_mesh_sized_from_sqrt_K")
        return 0
    print("VERDICT=brinkman_interface_behaved_differently")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
