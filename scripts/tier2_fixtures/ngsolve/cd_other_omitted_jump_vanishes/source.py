"""Tier-2: drop the .Other() and the jump term is identically zero -- the form
assembles, the matrix loses every cross-element entry, and nothing says so.

Claim: ngsolve convection_diffusion#1 -- "u.Other() accesses neighboring element
value.  Signal: forgetting .Other() in a skeleton form gives `u - u = 0`
integrands everywhere -- the jump term vanishes and the DG scheme reduces to a
discontinuous polynomial fit per element with no inter-element coupling."

Wrong variant: (u - u)*(v - v)*dx(skeleton=True) in place of the jump.

Setup: L2 order 1 with dgjumps=True on unit_square maxh 0.3, the interior-penalty
jump term written both ways.  Both forms are assembled and the matrices compared
entry by entry.  nze counts ALLOCATED slots, which dgjumps fixes at space
construction and which is therefore identical for both, so the comparison is over
entries that actually received a value.

What this fixture pins, all re-measured on this run:
  * the integrand really is identically zero without .Other(): evaluated over the
    domain it has zero L2 norm, so "u - u = 0" is arithmetic here;
  * the mistake is silent -- the form assembles with no exception and no output,
    captured at the file-descriptor level;
  * the resulting matrix has NO entries at all: every value is below roundoff,
    so the scheme really has "no inter-element coupling";
  * the correct form writes cross-element entries, counted;
  * a full DG solve built on the broken form is element-local -- adding a
    per-element mass term makes it invertible, and its solution is exactly the
    elementwise L2 projection, which the fixture verifies against an independent
    per-element projection.

Mutation control: T2_MUTATE=1 applies the documented fix at both pathology sites
-- .Other() is restored, so the broken form's integrand becomes the real jump
(u - u.Other())*(v - v.Other()) -- and the matrix then couples elements. Re-run
with `T2_MUTATE=1 python source.py`.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy
from netgen.geom2d import unit_square
from ngsolve import (
    BilinearForm,
    GridFunction,
    Integrate,
    L2,
    LinearForm,
    Mesh,
    dx,
    sqrt,
    x,
    y,
)

MUTATE = os.environ.get("T2_MUTATE") == "1"


def _capture_fds(fn):
    o, e = sys.stdout.fileno(), sys.stderr.fileno()
    sys.stdout.flush(); sys.stderr.flush()
    so, se = os.dup(o), os.dup(e)
    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        os.dup2(tmp.fileno(), o); os.dup2(tmp.fileno(), e)
        try:
            r, exc = fn(), None
        except Exception as ex:                                # noqa: BLE001
            r, exc = None, f"{type(ex).__name__}: {ex}"
        sys.stdout.flush(); sys.stderr.flush()
    finally:
        os.dup2(so, o); os.dup2(se, e); os.close(so); os.close(se)
    tmp.seek(0); t = tmp.read().decode("utf-8", "replace"); tmp.close()
    return r, exc, t


def written(form, fes, mesh):
    rows, cols, vals = form.mat.COO()
    r = numpy.asarray(rows)
    c = numpy.asarray(cols)
    v = numpy.abs(numpy.asarray(vals))
    live = v > 1e-14
    blk = fes.ndof // mesh.ne
    same = (r[live] // blk) == (c[live] // blk)
    return int(live.sum()), int((~same).sum())


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    fes = L2(mesh, order=1, dgjumps=True)
    u, v = fes.TnT()

    # The integrand without .Other() is identically zero.
    gf = GridFunction(fes)
    gf.Set(x * x - y)
    zero_cf = gf - gf
    zero_norm = float(sqrt(Integrate(zero_cf * zero_cf, mesh)))
    field_norm = float(sqrt(Integrate(gf * gf, mesh)))
    print(f"field_norm={field_norm:.6f}")
    print(f"u_minus_u_norm={zero_norm:.3e}")
    print(f"u_minus_u_is_identically_zero={zero_norm == 0.0}")

    def _broken():
        a = BilinearForm(fes)
        # The pathology: the .Other() is missing on both factors.
        # T2_MUTATE=1 restores it, turning this into the real jump.
        if MUTATE:
            a += (u - u.Other()) * (v - v.Other()) * dx(skeleton=True)
        else:
            a += (u - u) * (v - v) * dx(skeleton=True)
        a.Assemble()
        return a

    a_bad, exc, out = _capture_fds(_broken)
    print(f"broken_form_raised={exc!r}")
    print(f"broken_form_printed={out.strip()!r}")
    print(f"broken_form_assembles_silently="
          f"{exc is None and out.strip() == ''}")

    n_bad, off_bad = written(a_bad, fes, mesh)
    print(f"broken_entries_written={n_bad} cross_element={off_bad}")
    print(f"broken_matrix_has_no_entries_at_all={n_bad == 0}")
    print(f"broken_form_has_no_inter_element_coupling={off_bad == 0}")

    a_ok = BilinearForm(fes)
    a_ok += (u - u.Other()) * (v - v.Other()) * dx(skeleton=True)
    a_ok.Assemble()
    n_ok, off_ok = written(a_ok, fes, mesh)
    print(f"correct_entries_written={n_ok} cross_element={off_ok}")
    print(f"correct_form_couples_elements={off_ok > 0}")

    # A DG solve on the broken form is the elementwise L2 projection.
    a_solve = BilinearForm(fes)
    a_solve += u * v * dx
    # Same pathology, same fix under T2_MUTATE=1.
    if MUTATE:
        a_solve += (u - u.Other()) * (v - v.Other()) * dx(skeleton=True)
    else:
        a_solve += (u - u) * (v - v) * dx(skeleton=True)
    a_solve.Assemble()
    f = LinearForm(fes)
    f += (x * x - y) * v * dx
    f.Assemble()
    g_broken = GridFunction(fes)
    g_broken.vec.data = a_solve.mat.Inverse(inverse="umfpack") * f.vec

    a_proj = BilinearForm(fes)
    a_proj += u * v * dx
    a_proj.Assemble()
    g_proj = GridFunction(fes)
    g_proj.vec.data = a_proj.mat.Inverse(inverse="umfpack") * f.vec

    diff = float(numpy.abs(g_broken.vec.FV().NumPy()
                           - g_proj.vec.FV().NumPy()).max())
    print(f"broken_solve_vs_elementwise_projection={diff:.3e}")
    print(f"broken_scheme_is_just_a_per_element_fit={diff < 1e-12}")

    ok = (
        zero_norm == 0.0
        and exc is None and out.strip() == ""
        and n_bad == 0 and off_bad == 0
        and off_ok > 0
        and diff < 1e-12
    )
    if ok:
        return 0
    print("FAIL: missing-Other() invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
