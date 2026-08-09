"""Tier-2 for fenics nearly_incompressible_elasticity#1: the inf-sup (LBB)
condition demands a pressure space STRICTLY SMALLER than the displacement space.
The claim's signal is an SVD of the bc-applied saddle-point matrix on the Stokes
analogue: numerical null dimension 1 for P2/P1 Taylor-Hood (the constant
pressure alone) but 8 for equal-order P1/P1, and the extra kernel vectors are
pressure modes.

The fixture assembles the lid-driven Stokes operator on an 8x8 unit square with
Dirichlet velocity on every boundary facet, converts it to dense, and counts the
relative singular values below a tolerance for the tested pair and for
Taylor-Hood. It then inspects the kernel vectors of the tested pair: what
fraction of their norm sits on pressure dofs, and how much of that is NOT the
constant mode.

Mutation control: T2_MUTATE=1 tests the P2/P1 pair, whose kernel is the constant
pressure and nothing else.
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

N = 8
TOL = 1e-10


def saddle_matrix(k_vel: int, k_pre: int):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    gdim, tdim = msh.geometry.dim, msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    V = basix.ufl.element("Lagrange", msh.basix_cell(), k_vel, shape=(gdim,))
    Q = basix.ufl.element("Lagrange", msh.basix_cell(), k_pre)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([V, Q]))
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v))
         - p * ufl.div(v) - q * ufl.div(u)) * ufl.dx
    V0, _ = W.sub(0).collapse()
    lid = dolfinx.fem.Function(V0)
    lid.interpolate(lambda x: np.vstack(
        [np.isclose(x[1], 1.0) * 1.0, np.zeros_like(x[0])]))
    dofs = dolfinx.fem.locate_dofs_topological(
        (W.sub(0), V0), tdim - 1,
        dolfinx.mesh.exterior_facet_indices(msh.topology))
    bcs = [dolfinx.fem.dirichletbc(lid, dofs, W.sub(0))]
    A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a), bcs=bcs)
    A.assemble()
    dense = A.convert("dense").getDenseArray().copy()
    _, pmap = W.sub(1).collapse()
    return dense, np.asarray(pmap, dtype=np.int64)


def nullity(dense: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    u, s, vt = np.linalg.svd(dense)
    rel = s / s[0]
    idx = np.flatnonzero(rel < TOL)
    return len(idx), vt[idx], rel


def main() -> int:
    pair = (2, 1) if MUTATE else (1, 1)
    print(f"tested_pair=P{pair[0]}/P{pair[1]}")
    dense, pmap = saddle_matrix(*pair)
    n_test, kernel, rel = nullity(dense)
    th_dense, _ = saddle_matrix(2, 1)
    n_th, _, _ = nullity(th_dense)
    print(f"tested_matrix_size={dense.shape[0]} pressure_dofs={len(pmap)}")
    print(f"tested_numerical_null_dimension={n_test}")
    print(f"taylor_hood_numerical_null_dimension={n_th}")
    stable = [int((rel < t).sum()) for t in (1e-14, 1e-12, 1e-10, 1e-8, 1e-6)]
    print(f"tested_nullity_over_tolerances_1e14_to_1e6={stable}")

    press_frac, nonconst_frac = [], []
    for k in kernel:
        kp = k[pmap]
        nrm = np.linalg.norm(k)
        press_frac.append(float(np.linalg.norm(kp) / nrm))
        const = float(kp.sum() / np.sqrt(len(kp)))
        nonconst_frac.append(float(
            np.sqrt(max(np.dot(kp, kp) - const ** 2, 0.0)) / nrm))
    print(f"min_pressure_share_of_kernel_vectors={min(press_frac):.6f}")
    print(f"max_non_constant_pressure_share={max(nonconst_frac):.6f}")
    th_is_one = n_th == 1
    extra = n_test > n_th
    robust = len(set(stable)) == 1
    on_pressure = min(press_frac) > 0.99
    nonconst = max(nonconst_frac) > 0.1
    print(f"taylor_hood_null_dimension_is_one={th_is_one}")
    print(f"tested_nullity_exceeds_taylor_hood={extra}")
    print(f"nullity_is_tolerance_robust={robust}")
    print(f"kernel_lives_on_the_pressure_block={on_pressure}")
    print(f"kernel_contains_non_constant_pressure_modes={nonconst}")
    if th_is_one and extra and robust and on_pressure and nonconst:
        print("VERDICT=equal_order_pressure_space_adds_kernel_modes")
        return 0
    print("VERDICT=tested_pair_has_only_the_constant_pressure_kernel")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
