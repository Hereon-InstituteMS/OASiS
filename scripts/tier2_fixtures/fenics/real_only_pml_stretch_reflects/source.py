"""Tier-2 for fenics maxwell#2: a PML needs the COMPLEX coordinate stretch
x -> x (1 + i sigma(x)/omega). A real-only stretch is not a radiating layer.

Wrong variant: the 1D Helmholtz PML -(1/s u')' - k^2 s u = 0 on [0, 1.5], k = 20,
physical region [0, 1], layer [1, 1.5], sigma = 40 ((x-1)/0.5)^2, u(0) = 1,
u(1.5) = 0, P2 on 600 elements, with s = 1 + sigma/k (real) instead of
s = 1 + i sigma/k. Run on the complex dolfinx build (env FENICS_PYTHON).

Observed: with the complex stretch |u| is flat across the physical region - the
standing-wave ratio max|u|/min|u| on [0.1, 0.9] is 1.0000 - and the field decays
by orders of magnitude through the layer. With the real stretch the ratio is
about 1.9e3: the layer does not absorb at all (a real stretch adds no loss, only
an impedance gradient), the wave reflects off it and the physical region fills
with a standing wave whose nodes are near zero.

Mutation control: T2_MUTATE=1 uses the complex stretch, and the standing wave
disappears.
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

K = 20.0
LEN = 1.5
X_PML = 1.0
SIGMA0 = 40.0
NCELL = 600


def solve(stretch: str):
    msh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, NCELL, [0.0, LEN])
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 2))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    xi = ufl.max_value((x[0] - X_PML) / (LEN - X_PML), 0.0)
    sigma = SIGMA0 * xi ** 2
    s = 1.0 + (1j * sigma / K if stretch == "complex" else sigma / K)
    a = ((1.0 / s) * ufl.inner(ufl.grad(u), ufl.grad(v))
         - K ** 2 * s * ufl.inner(u, v)) * ufl.dx
    st = dolfinx.default_scalar_type
    L = ufl.inner(dolfinx.fem.Constant(msh, st(0.0)), v) * ufl.dx
    bcs = [
        dolfinx.fem.dirichletbc(
            dolfinx.fem.Constant(msh, st(1.0)),
            dolfinx.fem.locate_dofs_geometrical(
                V, lambda p: np.isclose(p[0], 0.0)), V),
        dolfinx.fem.dirichletbc(
            dolfinx.fem.Constant(msh, st(0.0)),
            dolfinx.fem.locate_dofs_geometrical(
                V, lambda p: np.isclose(p[0], LEN)), V),
    ]
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=bcs, petsc_options_prefix=f"t2_mw2_{stretch}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = prob.solve()
    assert prob.solver.getConvergedReason() > 0, "solve failed"
    xs = V.tabulate_dof_coordinates()[:, 0]
    mag = np.abs(uh.x.array)
    inner = (xs > 0.1) & (xs < 0.9)
    entry = (xs >= X_PML) & (xs < X_PML + 0.05)
    exit_ = xs > LEN - 0.05
    swr = float(mag[inner].max() / mag[inner].min())
    decay = float(mag[entry].max() / max(mag[exit_].max(), 1e-300))
    return swr, decay


def main() -> int:
    is_complex = np.issubdtype(dolfinx.default_scalar_type, np.complexfloating)
    print(f"scalar_type_is_complex={is_complex}")
    tested = "complex" if MUTATE else "real"
    print(f"tested_stretch={tested}")
    swr, decay = solve(tested)
    ref_swr, ref_decay = solve("complex")
    print(f"tested_standing_wave_ratio={swr:.4f} "
          f"tested_decay_across_the_layer={decay:.4e}")
    print(f"complex_standing_wave_ratio={ref_swr:.4f} "
          f"complex_decay_across_the_layer={ref_decay:.4e}")
    print(f"tested_standing_wave_ratio_above_10={swr > 10.0}")
    print(f"tested_layer_decay_below_10x={decay < 10.0}")
    print(f"complex_stretch_standing_wave_ratio_below_1p05={ref_swr < 1.05}")
    print(f"complex_stretch_decays_by_more_than_100x={ref_decay > 100.0}")
    if (is_complex and swr > 10.0 and decay < 10.0 and ref_swr < 1.05
            and ref_decay > 100.0):
        print("VERDICT=only_the_complex_coordinate_stretch_radiates")
        return 0
    print("VERDICT=real_stretch_behaved_like_a_pml")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
