"""Tier-2 for fenics biharmonic#3: writing the naive single fourth-order form
inner(div(grad(u)), div(grad(v)))*dx on a C0 Lagrange space raises NOTHING.
dolfinx does not raise a "H2 conformity required" error (no such error exists)
and it does not silently substitute the interior-penalty form. It compiles,
assembles, and hands back a singular or inconsistent operator, so the failure
is numerical rather than diagnostic and you must write the dS interior-penalty
terms yourself.

Wrong variant: the naive form on plain C0 Lagrange. Right variant: the same
volume term plus the three C0-IP interior-facet terms.

Observed on an 8x8 unit square, dolfinx 0.10.0: on P2 the naive form compiles
and assembles 3073 stored nonzeros with a largest entry of 6144.0; on P1 it
assembles 497 stored nonzeros whose largest absolute entry is exactly 0.0,
because div(grad(.)) of a P1 function vanishes cell-wise — an identically zero
matrix, produced without a single warning. The C0-IP form on the same P1 space
is not zero: 849 stored nonzeros, largest entry 6992.31.

Mutation control: T2_MUTATE=1 assembles the C0-IP form in place of the naive
one, the P1 matrix is no longer identically zero and the fixture loses its own
expectation.
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

N = 8


def assemble(degree: int, interior_penalty: bool):
    """Return (error text, nz_used, max |entry|) for the requested form."""
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", degree))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.div(ufl.grad(u)), ufl.div(ufl.grad(v))) * ufl.dx
    if interior_penalty:
        nrm = ufl.FacetNormal(msh)
        h = ufl.CellDiameter(msh)
        al = dolfinx.fem.Constant(msh, 8.0)
        a += (-ufl.inner(ufl.avg(ufl.div(ufl.grad(u))),
                         ufl.jump(ufl.grad(v), nrm)) * ufl.dS
              - ufl.inner(ufl.jump(ufl.grad(u), nrm),
                          ufl.avg(ufl.div(ufl.grad(v)))) * ufl.dS
              + al / ufl.avg(h) * ufl.inner(ufl.jump(ufl.grad(u), nrm),
                                            ufl.jump(ufl.grad(v), nrm))
              * ufl.dS)
    try:
        form = dolfinx.fem.form(a)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}", -1, float("nan")
    A = dolfinx.fem.petsc.assemble_matrix(form)
    A.assemble()
    vals = A.getValuesCSR()[2]
    nz = int(A.getInfo()["nz_used"])
    return "", nz, float(np.max(np.abs(vals))) if vals.size else 0.0


def main() -> int:
    use_ip = MUTATE
    err1, nz1, mx1 = assemble(1, use_ip)
    err2, nz2, mx2 = assemble(2, use_ip)
    _, nz_ip, mx_ip = assemble(1, True)
    print(f"P1_primary_error={err1!r} P1_nz_used={nz1} P1_max_abs_entry={mx1}")
    print(f"P2_primary_error={err2!r} P2_nz_used={nz2} P2_max_abs_entry={mx2}")
    print(f"P1_c0ip_nz_used={nz_ip} P1_c0ip_max_abs_entry={mx_ip:.2f}")

    no_error = not err1 and not err2
    p1_zero = nz1 > 0 and mx1 == 0.0
    p2_nonzero = nz2 > 0 and mx2 > 0.0
    ip_nonzero = mx_ip > 0.0
    print(f"naive_fourth_order_form_raised_nothing={no_error}")
    print(f"P1_matrix_is_identically_zero={p1_zero}")
    print(f"P2_matrix_has_nonzero_entries={p2_nonzero}")
    print(f"interior_penalty_form_on_P1_is_not_zero={ip_nonzero}")
    if no_error and p1_zero and p2_nonzero and ip_nonzero:
        print("VERDICT=naive_h2_form_compiles_and_yields_a_silently_wrong_operator")
        return 0
    print("VERDICT=naive_h2_form_was_rejected_or_corrected")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
