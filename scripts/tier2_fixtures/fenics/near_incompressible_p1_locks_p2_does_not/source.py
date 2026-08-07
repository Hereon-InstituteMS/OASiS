"""Tier-2 for fenics hyperelasticity#2: in the near-incompressible regime the
pure-displacement formulation locks, and the fix is a mixed (u, p) space -- but
the severity depends strongly on the polynomial degree, so the ratio has to be
measured rather than assumed.

The measurement is the linear analogue the claim itself points at: a 2D
cantilever, 4 x 1, clamped at x = 0, shear traction on the tip, solved on two
meshes at nu = 0.4999 and at nu = 0.3, with the tip deflection compared against a
P2/P1 Taylor-Hood (u, p) reference on the same mesh.

Observed on dolfinx 0.10.0: at nu = 0.4999 the pure-displacement P1 tip
deflection is 12.6x (8x2 mesh) and 12.5x (16x4 mesh) below the Taylor-Hood
answer, and refining does NOT close the gap -- that is locking. Pure-displacement
P2 lands within 5.2 % and 1.8 % of Taylor-Hood on the same meshes, so the severe
locking is a P1 phenomenon, not a P2 one. At nu = 0.3 the same P1 space is only
1.9x / 1.2x below the reference and improves with refinement, i.e. ordinary
discretisation error rather than locking.

Mutation control: T2_MUTATE=1 puts P2 in the space under test, and the
many-fold stiffness excess disappears.
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

E = 1.0e5
MESHES = ((8, 2), (16, 4))


def tip_deflection(nu, nx, ny, kind, prefix):
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    msh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([4.0, 1.0])],
        [nx, ny])
    d = 2
    msh.topology.create_connectivity(d - 1, d)
    left = dolfinx.mesh.locate_entities_boundary(
        msh, d - 1, lambda x: np.isclose(x[0], 0.0))
    right = dolfinx.mesh.locate_entities_boundary(
        msh, d - 1, lambda x: np.isclose(x[0], 4.0))
    tags = dolfinx.mesh.meshtags(msh, d - 1, np.sort(right),
                                 np.full(len(right), 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
    traction = dolfinx.fem.Constant(msh, (0.0, -100.0))
    eps = lambda w: ufl.sym(ufl.grad(w))  # noqa: E731

    if kind in ("P1", "P2"):
        V = dolfinx.fem.functionspace(
            msh, ("Lagrange", 1 if kind == "P1" else 2, (d,)))
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        a = ufl.inner(2 * mu * eps(u)
                      + lam * ufl.tr(eps(u)) * ufl.Identity(d),
                      eps(v)) * ufl.dx
        L = ufl.inner(traction, v) * ds(1)
        bcs = [dolfinx.fem.dirichletbc(
            np.zeros(d),
            dolfinx.fem.locate_dofs_topological(V, d - 1, left), V)]
        prob = dolfinx.fem.petsc.LinearProblem(
            a, L, bcs=bcs, petsc_options_prefix=prefix,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = prob.solve()
        if isinstance(uh, tuple):
            uh = uh[0]
        return float(np.max(np.abs(uh.x.array.reshape(-1, d)[:, 1])))

    p2 = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(d,))
    p1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([p2, p1]))
    (u, p), (v, q) = ufl.TrialFunctions(W), ufl.TestFunctions(W)
    a = (ufl.inner(2 * mu * eps(u), eps(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx - q * ufl.div(u) * ufl.dx
         - (1.0 / lam) * p * q * ufl.dx)
    L = ufl.inner(traction, v) * ds(1)
    V0, v_map = W.sub(0).collapse()
    dofs = dolfinx.fem.locate_dofs_topological((W.sub(0), V0), d - 1, left)
    bcs = [dolfinx.fem.dirichletbc(dolfinx.fem.Function(V0), dofs, W.sub(0))]
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=bcs, petsc_options_prefix=prefix,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    wh = prob.solve()
    if isinstance(wh, tuple):
        wh = wh[0]
    arr = wh.x.array[np.array(v_map, dtype=np.int32)].reshape(-1, d)
    return float(np.max(np.abs(arr[:, 1])))


def main() -> int:
    tested = "P2" if MUTATE else "P1"
    print(f"pure_displacement_space_under_test={tested}")
    stiff, ok_p2 = {}, {}
    for nu in (0.4999, 0.3):
        for nx, ny in MESHES:
            tag = f"t2_hy2_{tested}_{int(nu * 1e4)}_{nx}_"
            d_test = tip_deflection(nu, nx, ny, tested, tag + "t")
            d_ref = tip_deflection(nu, nx, ny, "TH", tag + "r")
            stiff[(nu, nx)] = d_ref / d_test
            print(f"nu={nu} mesh={nx}x{ny} tip_{tested}={d_test:.6e} "
                  f"tip_taylor_hood={d_ref:.6e} "
                  f"reference_over_test={d_ref / d_test:.3f}")
            if nu == 0.4999:
                d_p2 = tip_deflection(nu, nx, ny, "P2", tag + "p2")
                ok_p2[nx] = abs(d_p2 / d_ref - 1.0)
                print(f"  nu={nu} mesh={nx}x{ny} tip_P2={d_p2:.6e} "
                      f"p2_relative_gap_to_taylor_hood={ok_p2[nx]:.4f}")

    locks = all(stiff[(0.4999, nx)] > 5.0 for nx, _ in MESHES)
    persists = (stiff[(0.4999, MESHES[1][0])]
                > 0.8 * stiff[(0.4999, MESHES[0][0])])
    print(f"under_test_is_many_times_too_stiff_at_nu_4999={locks}")
    print(f"refinement_does_not_cure_it={persists}")

    p2_close = all(v < 0.06 for v in ok_p2.values())
    print(f"p2_is_within_six_percent_of_taylor_hood={p2_close}")

    mild = all(stiff[(0.3, nx)] < 2.0 for nx, _ in MESHES)
    improves = stiff[(0.3, MESHES[1][0])] < stiff[(0.3, MESHES[0][0])]
    print(f"at_nu_030_the_same_space_is_not_locked={mild and improves}")

    if locks and persists and p2_close and mild and improves:
        print("VERDICT=p1_locks_at_nu_4999_p2_is_close_to_taylor_hood")
        return 0
    print("VERDICT=no_locking_measured")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
