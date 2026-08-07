"""Tier-2 for fenics maxwell#2: a PML needs a COMPLEX coordinate stretch. If the
stretch coefficient is built with a real sigma, the layer is just a different
real material: it does not absorb, it reflects, and the physical region fills
with a standing wave.

Setup: 1D Helmholtz on (0, 1.25) with the physical region (0, 1) and a PML in
(1, 1.25). The stretched equation is
((1/s) u')' + k^2 s u = 0 with s = 1 + i sigma(x)/omega inside the layer, k = 20,
omega = k, a quadratic sigma profile, u(0) = 1 and a natural condition at the
outer end. s is a DG0 fem.Function, so the difference between the two variants is
exactly the dtype and value the claim talks about. With a perfect PML the
physical region carries the pure outgoing wave, |u| = 1 flat, and |u| collapses
inside the layer.

Observed on dolfinx 0.10.0 with a complex PETSc build: the complex stretch gives
a flat |u| in the physical region (swing 0.000000) and |u| down by a factor
7.5e-06 across the layer; the real-only stretch (same magnitude, no imaginary
part) gives an |u| that swings by 1.017 across the physical region -- the
reflected wave -- and does not decay at all in the layer (|u| at the outer end is
4.6 times the value at the layer entry).

Mutation control: T2_MUTATE=1 builds the stretch with numpy.complex128 values,
i.e. the correct PML; the standing wave disappears.
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
X_PML = 1.0
X_END = 1.25
SIGMA0 = 150.0
NCELLS = 300


def solve(complex_stretch: bool):
    msh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, NCELLS,
                                       [0.0, X_END])
    tdim = msh.topology.dim
    ncells = msh.topology.index_map(tdim).size_local
    mid = dolfinx.mesh.compute_midpoints(
        msh, tdim, np.arange(ncells, dtype=np.int32)).T[0]

    DG0 = dolfinx.fem.functionspace(msh, ("DG", 0))
    s = dolfinx.fem.Function(DG0)
    s.x.array[:] = 1.0
    inside = mid > X_PML
    prof = SIGMA0 * ((mid[inside] - X_PML) / (X_END - X_PML)) ** 2 / K
    s.x.array[inside] = (1.0 + 1j * prof) if complex_stretch else (1.0 + prof)
    print(f"stretch_dtype={s.x.array.dtype.name} "
          f"max_imag_part={float(np.abs(s.x.array.imag).max()):.4f} "
          f"max_real_part={float(s.x.array.real.max()):.4f}")

    V = dolfinx.fem.functionspace(msh, ("Lagrange", 2))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ((1.0 / s) * ufl.inner(ufl.grad(u), ufl.grad(v))
         - K ** 2 * s * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(dolfinx.fem.Constant(msh, dolfinx.default_scalar_type(0.0)),
                  v) * ufl.dx
    left = dolfinx.fem.locate_dofs_geometrical(
        V, lambda x: np.isclose(x[0], 0.0))
    bc = dolfinx.fem.dirichletbc(
        dolfinx.default_scalar_type(1.0), left, V)
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix=f"t2_mx2_{'c' if complex_stretch else 'r'}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})
    uh = prob.solve()
    if isinstance(uh, tuple):
        uh = uh[0]
    assert prob.solver.getConvergedReason() > 0

    coords = V.tabulate_dof_coordinates()[:, 0]
    mag = np.abs(uh.x.array)
    phys = (coords > 0.1) & (coords < 0.9)
    swing = float(mag[phys].max() - mag[phys].min())
    at_interface = float(mag[np.argmin(np.abs(coords - X_PML))])
    at_end = float(mag[np.argmin(np.abs(coords - X_END))])
    return swing, at_interface, at_end


def main() -> int:
    print(f"scalar_type={np.dtype(dolfinx.default_scalar_type).name}")
    is_complex = bool(np.issubdtype(dolfinx.default_scalar_type,
                                    np.complexfloating))
    print(f"complex_build={is_complex}")
    if not is_complex:
        print("VERDICT=needs_the_complex_build")
        return 1

    tag = "complex" if MUTATE else "real_only"
    print(f"stretch_under_test={tag}")
    swing, interface, end = solve(MUTATE)
    print(f"under_test standing_wave_swing_in_physical_region={swing:.6f} "
          f"abs_u_at_pml_entry={interface:.6e} abs_u_at_outer_end={end:.6e}")
    decay = end / max(interface, 1e-300)
    print(f"under_test_decay_through_the_layer={decay:.3e}")

    cswing, cif, cend = solve(True)
    cdecay = cend / max(cif, 1e-300)
    print(f"complex_pml standing_wave_swing={cswing:.6f} "
          f"decay_through_the_layer={cdecay:.3e}")

    print(f"complex_pml_physical_region_is_flat={cswing < 0.05}")
    print(f"complex_pml_decays_by_orders_of_magnitude={cdecay < 1e-2}")
    print(f"under_test_shows_standing_wave_reflection={swing > 0.2}")
    print(f"under_test_fails_to_absorb={decay > 1e-2}")

    if cswing < 0.05 and cdecay < 1e-2 and swing > 0.2 and decay > 1e-2:
        print("VERDICT=real_only_stretch_reflects_complex_stretch_absorbs")
        return 0
    print("VERDICT=stretch_under_test_absorbed_like_a_pml")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
