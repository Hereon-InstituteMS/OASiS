"""Tier-2 for fenics reaction_diffusion#6: species concentrations really do go
negative, and the cause is the step size relative to h^2, not merely "steep
gradients". Backward Euler -- the most stable integrator there is -- undershoots
when dt drops below about h^2/(6D), because the consistent P1 mass matrix is not
an M-matrix.

Pure diffusion (D = 1) of a sharp blob of height 1.0 on a 32x32 unit square, all
boundaries no-flux, 5 backward-Euler steps. h^2/(6D) = 1.628e-04 here.

Observed on dolfinx 0.10.0: min(c) over the five computed states stays
non-negative at dt = 1e-2 and reaches -1.939e-02 at dt = 1e-4, an undershoot of
about two percent of the peak. The total mass is conserved to round-off in BOTH
runs (0.06738281 -> 0.06738281), so a mass check will not catch it. Integrating the time-derivative term with
ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": 1}) lumps the
mass matrix and removes the undershoot at the same dt.

Mutation control: T2_MUTATE=1 lumps the mass matrix in the small-dt run.
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

N, D, NSTEP = 32, 1.0, 5
DT_BIG, DT_SMALL = 1.0e-2, 1.0e-4


def blob(x):
    return np.where((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 < 0.15 ** 2,
                    1.0, 0.0)


def run(dt: float, lump: bool):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    c_n = dolfinx.fem.Function(V)
    c_n.interpolate(blob)
    c_n.x.scatter_forward()
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dxm = (ufl.dx(metadata={"quadrature_rule": "vertex",
                            "quadrature_degree": 1}) if lump else ufl.dx)
    a = (u / dt) * v * dxm + D * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (c_n / dt) * v * dxm
    c_h = dolfinx.fem.Function(V)
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, u=c_h,
        petsc_options_prefix=f"t2_rd6_{'lump' if lump else 'cons'}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    mass_form = dolfinx.fem.form(c_n * ufl.dx)
    mass0 = float(dolfinx.fem.assemble_scalar(mass_form))
    lo = float("inf")  # minimum over the COMPUTED states, not the blob itself
    for _ in range(NSTEP):
        prob.solve()
        c_h.x.scatter_forward()
        c_n.x.array[:] = c_h.x.array
        lo = min(lo, float(c_n.x.array.min()))
    mass1 = float(dolfinx.fem.assemble_scalar(mass_form))
    return lo, float(c_n.x.array.max()), mass0, mass1


def main() -> int:
    h2_over_6d = (1.0 / N) ** 2 / (6.0 * D)
    lo_big, hi_big, m0_big, m1_big = run(DT_BIG, lump=False)
    lo_small, hi_small, m0_s, m1_s = run(DT_SMALL, lump=MUTATE)
    lo_lump, _, _, _ = run(DT_SMALL, lump=True)

    drift = abs(m1_s - m0_s) / abs(m0_s)
    print(f"h_squared_over_6D={h2_over_6d:.3e}")
    print(f"dt={DT_BIG:.0e} (dt >> h^2/6D): min_c={lo_big:.4e} "
          f"max_c={hi_big:.4f} mass {m0_big:.8f} -> {m1_big:.8f}")
    print(f"dt={DT_SMALL:.0e} (dt < h^2/6D): min_c={lo_small:.4e} "
          f"max_c={hi_small:.4f} mass {m0_s:.8f} -> {m1_s:.8f} "
          f"relative_drift={drift:.3e}")
    print(f"lumped_mass_at_dt={DT_SMALL:.0e}: min_c={lo_lump:.4e}")

    under = lo_small < -1e-3
    print(f"large_dt_stays_non_negative={lo_big >= 0.0}")
    print(f"small_dt_undershoots_below_minus_1e_3={under}")
    print(f"undershoot_is_a_percent_of_the_peak={abs(lo_small) > 0.005}")
    print(f"total_mass_is_conserved_anyway={drift < 1e-12}")
    print(f"vertex_lumping_removes_the_undershoot={lo_lump >= 0.0}")

    if (lo_big >= 0.0 and under and drift < 1e-12 and lo_lump >= 0.0
            and abs(lo_small) > 0.005):
        print("VERDICT=small_dt_undershoots_and_mass_does_not_notice")
        return 0
    print("VERDICT=no_undershoot_observed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
