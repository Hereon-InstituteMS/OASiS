"""Tier-2 for fenics magnetostatics#5: `ufl.curl` applied to a 2D SCALAR
returns the 2-vector (dAz/dy, -dAz/dx) - the plane restriction of the 3D
curl((0, 0, Az)) - so the components never have to be written by hand. The
reason that matters: if you do write them by hand and flip the sign, the field
MAGNITUDE is unchanged, so every |B| check still passes while the winding
direction is reversed.

The fixture builds a hand-written B against `ufl.curl(Az)` for
Az = x*y + 0.3*x - 0.2*y, and measures the L2 norms (equal) and the alignment
integral B_hand . B_curl / (|B_hand| |B_curl|) (which is -1, not +1). It also
pins the two shape facts that make a rank mix-up silent: curl of a 2D scalar has
shape (2,), curl of a 2D vector has shape (), both compile through fem.form, and
inner(curl(s), curl(s))*dx assembles bit-identically to inner(grad(s),
grad(s))*dx for a scalar s.

Mutation control: T2_MUTATE=1 writes the hand-made components with the correct
sign; the alignment becomes +1 and the direction check passes.
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


def norm(expr) -> float:
    return float(np.sqrt(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(ufl.inner(expr, expr) * ufl.dx))))


def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    Az = dolfinx.fem.Function(V)
    Az.interpolate(lambda x: x[0] * x[1] + 0.3 * x[0] - 0.2 * x[1])
    Vv = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (2,)))
    A2 = dolfinx.fem.Function(Vv)
    A2.interpolate(lambda x: np.vstack([x[1], -x[0]]))

    b_curl = ufl.curl(Az)
    if MUTATE:
        b_hand = ufl.as_vector((Az.dx(1), -Az.dx(0)))
        print("hand_written_spelling=(dAz/dy,-dAz/dx)")
    else:
        b_hand = ufl.as_vector((-Az.dx(1), Az.dx(0)))
        print("hand_written_spelling=(-dAz/dy,+dAz/dx)")

    n_curl, n_hand = norm(b_curl), norm(b_hand)
    align = float(dolfinx.fem.assemble_scalar(dolfinx.fem.form(
        ufl.inner(b_hand, b_curl) * ufl.dx))) / (n_curl * n_hand)
    print(f"L2_norm_of_ufl_curl={n_curl:.12f}")
    print(f"L2_norm_of_hand_written={n_hand:.12f}")
    print(f"alignment_integral={align:.12f}")
    same_mag = abs(n_curl - n_hand) < 1e-12 * n_curl
    same_dir = abs(align - 1.0) < 1e-10
    reversed_dir = abs(align + 1.0) < 1e-10
    print(f"magnitude_agrees_with_curl_reference={same_mag}")
    print(f"direction_agrees_with_curl_reference={same_dir}")
    print(f"alignment_is_minus_one={reversed_dir}")

    # The pointwise API facts the claim makes, at one cell.
    W = dolfinx.fem.functionspace(msh, ("DG", 0, (2,)))

    def cell0(expr, sp):
        f = dolfinx.fem.Function(sp)
        f.interpolate(dolfinx.fem.Expression(
            expr, sp.element.interpolation_points))
        return np.round(f.x.array.reshape(-1, sp.value_shape[0])[0], 12)

    lin = dolfinx.fem.Function(V)
    lin.interpolate(lambda x: x[1])
    v_curl = cell0(ufl.curl(lin), W)
    v_hand = cell0(ufl.as_vector((lin.dx(1), -lin.dx(0))), W)
    m3 = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    V3 = dolfinx.fem.functionspace(m3, ("Lagrange", 1))
    lin3 = dolfinx.fem.Function(V3)
    lin3.interpolate(lambda x: x[1])
    zero = dolfinx.fem.Constant(m3, 0.0)
    W3 = dolfinx.fem.functionspace(m3, ("DG", 0, (3,)))
    v3 = cell0(ufl.curl(ufl.as_vector((zero, zero, lin3))), W3)
    print(f"for_Az_equal_y_ufl_curl={list(v_curl)}")
    print(f"for_Az_equal_y_hand_written={list(v_hand)}")
    print(f"for_Az_equal_y_3d_curl={list(v3)}")
    agree3 = bool(np.allclose(v_curl, v3[:2]) and np.allclose(v_curl, v_hand)
                  and abs(v3[2]) < 1e-14)
    print(f"three_spellings_agree_for_Az_equal_y={agree3}")

    print(f"curl_of_2d_scalar_shape={ufl.curl(Az).ufl_shape}")
    print(f"curl_of_2d_vector_shape={ufl.curl(A2).ufl_shape}")
    e_curl = dolfinx.fem.assemble_scalar(dolfinx.fem.form(
        ufl.inner(ufl.curl(Az), ufl.curl(Az)) * ufl.dx))
    e_grad = dolfinx.fem.assemble_scalar(dolfinx.fem.form(
        ufl.inner(ufl.grad(Az), ufl.grad(Az)) * ufl.dx))
    dolfinx.fem.form(ufl.inner(ufl.curl(A2), ufl.curl(A2)) * ufl.dx)
    print("both_ranks_compile_through_fem_form=True")
    print(f"scalar_curl_energy_equals_grad_energy={e_curl == e_grad}")

    if same_mag and reversed_dir and not same_dir and agree3:
        print("VERDICT=sign_flip_keeps_the_magnitude_and_reverses_the_field")
        return 0
    print("VERDICT=hand_written_components_match_ufl_curl")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
