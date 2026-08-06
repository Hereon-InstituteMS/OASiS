"""Tier-2: at Pe = 100 the upwind flux does remove the Galerkin undershoot, but
dropping the SIP diffusion penalty does not leave "small-amplitude ringing" --
it destroys the solution.

Claim: ngsolve dg_methods#5 -- "For convection-dominated DG (Pe >> 1): DG is
naturally stable due to upwind flux; SIP diffusion STILL needs the alpha/h *
jump penalty.  Signal: switching from Galerkin-CG to DG on Pe=100 removes the
gross oscillations but the diffusion-dominated regions still show
small-amplitude ringing if the SIP penalty is omitted (alpha=0); always add the
diffusion penalty even when advection dominates the bulk."

Wrong variant: the same DG discretisation with the SIP penalty coefficient set
to zero.

CORRECTION this fixture records.  The direction of the advice is right and the
CG-versus-DG half is confirmed, but "small-amplitude ringing" badly understates
what alpha=0 does.  With the penalty removed the DG solution does not become a
slightly noisy version of the right answer; it undershoots by several times the
largest value the correct solution takes anywhere.  An agent told to look for a
small ripple can accept a field that is entirely wrong, or blame a different
bug.

Setup: steady advection-diffusion on the unit square, b = (1, 0),
eps = 1/Pe = 0.01, unit source, homogeneous Dirichlet on the whole boundary.
A non-negative source with zero boundary data has a non-negative solution, so
ANY negative value in the discrete field is a discretisation artefact and
nothing else -- that is the yardstick used here, and it needs no reference
solution.  All three discretisations run on the SAME mesh.

What this fixture pins, all re-measured on this run:
  * the cell Peclet number really is above 1, i.e. the Galerkin problem is in
    the regime where it is expected to oscillate;
  * H1 Galerkin does go negative on that mesh;
  * DG with the SIP penalty does not -- the claim's "removes the gross
    oscillations", confirmed;
  * DG with alpha = 0 goes negative by several times the field's own positive
    peak, which is the correction: not a ripple.
"""
from __future__ import annotations

import sys

import numpy
from netgen.geom2d import SplineGeometry
from ngsolve import (
    BilinearForm,
    CoefficientFunction,
    GridFunction,
    H1,
    IfPos,
    L2,
    LinearForm,
    Mesh,
    ds,
    dx,
    grad,
    specialcf,
)

PE = 100.0
EPS = 1.0 / PE
BVEC = (1.0, 0.0)
ORDER = 1
MAXH = 0.15


def rect(maxh):
    g = SplineGeometry()
    g.AddRectangle((0, 0), (1, 1), bcs=["bot", "right", "top", "left"])
    return Mesh(g.GenerateMesh(maxh=maxh))


def sample(gfu, mesh, npts=41):
    ts = numpy.linspace(0.002, 0.998, npts)
    vals = []
    for yy in (0.25, 0.5, 0.75):
        vals += [float(gfu(mesh(float(t), yy))) for t in ts]
    return min(vals), max(vals)


def galerkin(mesh):
    fes = H1(mesh, order=ORDER, dirichlet=".*")
    u, v = fes.TnT()
    b = CoefficientFunction(BVEC)
    a = BilinearForm(fes)
    a += EPS * grad(u) * grad(v) * dx + (b * grad(u)) * v * dx
    f = LinearForm(fes)
    f += 1 * v * dx
    a.Assemble()
    f.Assemble()
    gfu = GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="umfpack") * f.vec
    return fes.ndof, gfu


def dg(mesh, penalty_scale):
    fes = L2(mesh, order=ORDER, dgjumps=True)
    u, v = fes.TnT()
    b = CoefficientFunction(BVEC)
    n = specialcf.normal(2)
    h = specialcf.mesh_size
    ju, jv = u - u.Other(), v - v.Other()
    mdu = 0.5 * (grad(u) + grad(u.Other()))
    mdv = 0.5 * (grad(v) + grad(v.Other()))
    alpha = penalty_scale * 4 * (ORDER + 1) ** 2

    a = BilinearForm(fes)
    a += EPS * grad(u) * grad(v) * dx
    a += EPS * alpha / h * ju * jv * dx(skeleton=True)
    a += EPS * (-mdu * n * jv - mdv * n * ju) * dx(skeleton=True)
    a += EPS * alpha / h * u * v * ds(skeleton=True)
    a += EPS * (-grad(u) * n * v - grad(v) * n * u) * ds(skeleton=True)
    a += -u * (b * grad(v)) * dx
    a += IfPos(b * n, u, u.Other()) * (b * n) * jv * dx(skeleton=True)
    a += IfPos(b * n, u, 0) * (b * n) * v * ds(skeleton=True)

    f = LinearForm(fes)
    f += 1 * v * dx
    a.Assemble()
    f.Assemble()
    gfu = GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(inverse="umfpack") * f.vec
    return fes.ndof, gfu


def main() -> int:
    cell_pe = 1.0 * MAXH / (2 * EPS)
    print(f"global_peclet={PE:g}")
    print(f"cell_peclet={cell_pe:.2f}")
    print(f"cell_peclet_above_one={cell_pe > 1.0}")

    mesh = rect(MAXH)

    nd_cg, g_cg = galerkin(mesh)
    cg_lo, cg_hi = sample(g_cg, mesh)
    print(f"galerkin_ndof={nd_cg} min={cg_lo:.6f} max={cg_hi:.6f}")
    print(f"galerkin_goes_negative={cg_lo < -1e-3}")

    nd_p, g_p = dg(mesh, 1.0)
    p_lo, p_hi = sample(g_p, mesh)
    print(f"dg_penalised_ndof={nd_p} min={p_lo:.6f} max={p_hi:.6f}")
    print(f"dg_penalised_stays_nonnegative={p_lo > -1e-3}")
    print(f"upwind_dg_removes_the_galerkin_undershoot="
          f"{cg_lo < -1e-3 and p_lo > -1e-3}")

    nd_0, g_0 = dg(mesh, 0.0)
    z_lo, z_hi = sample(g_0, mesh)
    print(f"dg_alpha0_ndof={nd_0} min={z_lo:.6f} max={z_hi:.6f}")
    print(f"dg_alpha0_goes_negative={z_lo < -1e-3}")

    # How bad is "bad"?  Measure the undershoot against the field's own positive
    # peak, so the number is dimensionless and needs no reference solution.
    ratio = (-z_lo) / max(p_hi, 1e-30)
    print(f"alpha0_undershoot_over_penalised_peak={ratio:.4f}")
    print(f"alpha0_undershoot_exceeds_the_whole_field={ratio > 1.0}")
    print(f"alpha0_is_not_small_amplitude_ringing={ratio > 1.0}")

    ok = (
        cell_pe > 1.0
        and cg_lo < -1e-3
        and p_lo > -1e-3
        and z_lo < -1e-3
        and ratio > 1.0
    )
    if ok:
        return 0
    print("FAIL: SIP-penalty-at-high-Pe invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
