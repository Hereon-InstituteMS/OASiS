"""Tier-2 for fenics magnetostatics#5: `ufl.curl` applied to a 2D SCALAR
returns the 2-vector (dAz/dy, -dAz/dx) -- exactly the plane restriction of the
3D curl((0, 0, Az)) -- so there is nothing to write by hand and no sign
convention to get wrong. The trap is the OTHER direction: `ufl.curl(<2D
scalar>)` has shape (2,) while `ufl.curl(<2D vector>)` has shape (), BOTH
compile through fem.form, and for a scalar s the curl-curl energy
inner(curl(s), curl(s))*dx assembles bit-for-bit the same number as the
Laplacian energy inner(grad(s), grad(s))*dx -- so a scalar/vector mix-up in a
curl-curl form is completely silent.

Wrong variant: build the curl-curl energy on a SCALAR Lagrange space (a vector
potential component) as if it were the vector unknown.

Mutation control: T2_MUTATE=1 puts the intended 2D VECTOR field in the same
form. curl then has shape (), the curl energy stops agreeing with the gradient
energy, and the silent-equality signal disappears.
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


def interp_vector(msh, expr, dim: int) -> np.ndarray:
    W = dolfinx.fem.functionspace(msh, ("DG", 0, (dim,)))
    f = dolfinx.fem.Function(W)
    f.interpolate(dolfinx.fem.Expression(expr, W.element.interpolation_points))
    return np.round(f.x.array.reshape(-1, dim)[0], 12)


def main() -> int:
    msh2 = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    msh3 = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)

    # --- the sign convention: all three spellings of curl agree for Az = y ---
    V2 = dolfinx.fem.functionspace(msh2, ("Lagrange", 1))
    az = dolfinx.fem.Function(V2)
    az.interpolate(lambda x: x[1])
    a_curl = interp_vector(msh2, ufl.curl(az), 2)
    a_hand = interp_vector(msh2, ufl.as_vector((az.dx(1), -az.dx(0))), 2)
    V3 = dolfinx.fem.functionspace(msh3, ("Lagrange", 1, (3,)))
    az3 = dolfinx.fem.Function(V3)
    az3.interpolate(lambda x: np.vstack((np.zeros_like(x[1]),
                                        np.zeros_like(x[1]), x[1])))
    a_3d = interp_vector(msh3, ufl.curl(az3), 3)
    print(f"curl_2d_scalar={a_curl.tolist()}")
    print(f"hand_written_as_vector={a_hand.tolist()}")
    print(f"curl_3d_of_0_0_Az={a_3d.tolist()}")
    agree = (np.allclose(a_curl, a_hand)
             and np.allclose(a_3d, np.array([a_curl[0], a_curl[1], 0.0])))
    print(f"all_three_spellings_agree={agree}")

    # --- the silent mix-up: shapes differ, both compile, values coincide ---
    Vvec = dolfinx.fem.functionspace(msh2, ("Lagrange", 1, (2,)))
    print(f"curl_of_2d_scalar_shape={ufl.curl(ufl.TrialFunction(V2)).ufl_shape}")
    print("curl_of_2d_vector_shape="
          f"{ufl.curl(ufl.TrialFunction(Vvec)).ufl_shape}")

    space = Vvec if MUTATE else V2
    print(f"space_in_curl_curl_form={'vector_P1' if MUTATE else 'scalar_P1'}")
    s = dolfinx.fem.Function(space)
    if MUTATE:
        s.interpolate(lambda x: np.vstack((np.sin(3.0 * x[1]),
                                           np.cos(2.0 * x[0]))))
    else:
        s.interpolate(lambda x: np.sin(3.0 * x[1]) + np.cos(2.0 * x[0]))

    compiled = []
    vals = {}
    for name, expr in (("curl_curl", ufl.inner(ufl.curl(s), ufl.curl(s))),
                       ("grad_grad", ufl.inner(ufl.grad(s), ufl.grad(s)))):
        try:
            frm = dolfinx.fem.form(expr * ufl.dx)
            vals[name] = float(dolfinx.fem.assemble_scalar(frm))
            compiled.append(True)
            print(f"{name}_compiles=True value={vals[name]:.12e}")
        except Exception as exc:  # pragma: no cover - would be the finding
            compiled.append(False)
            print(f"{name}_compiles=False {type(exc).__name__}: {exc}")

    print(f"both_forms_compile={all(compiled)}")
    same = (len(vals) == 2
            and abs(vals["curl_curl"] - vals["grad_grad"])
            <= 1e-12 * abs(vals["grad_grad"]))
    print(f"curl_curl_energy_equals_grad_grad_energy={same}")
    if agree and all(compiled) and same:
        print("VERDICT=curl_of_a_2d_scalar_is_the_plane_curl_and_the_"
              "scalar_vector_mixup_is_silent")
        return 0
    print("VERDICT=mixup_was_not_silent")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
