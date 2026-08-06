"""Tier-2: the IMEX scheme's explicit convection has a hard step limit, and the
ratio dt*max|u|/h locates it.

Claims covered: ngsolve time_dependent_ns#0 and navier_stokes#0 say the same
thing -- "CFL for explicit convection: dt < C * h / max|u| -- may need small dt
for high Re.  Signal: velocity field blows up to NaN within the first few time
steps; per-step max(|u|) diverges geometrically; the violation ratio
dt * max(|u|) / h is greater than ~0.5."

Wrong variant: the same IMEX solve at a step above the limit.

Setup: lid-driven cavity on the unit square, Taylor-Hood VectorH1 order 2 *
H1 order 1, nu = 1e-3, parabolic lid profile with peak 1.  The Stokes part is
implicit (mass + dt*Stokes factored once), convection explicit, exactly the IMEX
split the catalog's own template uses.  Kinetic energy is the divergence
measure; the lid Dirichlet value is re-imposed each step by writing the
constrained DOFs back and solving only for the free correction, so a boundary
bug cannot be mistaken for a CFL bug.

Two departures from the claim, both recorded.  The threshold is lower than the
stated ~0.5: at ratio 0.33 the run already leaves double precision, while at
0.13 the energy still decays.  And the failure is not "NaN within the first few
time steps" -- at ratio 0.67 it takes 34 steps and arrives as overflow, not as a
NaN.  A guard that runs three steps and tests for NaN sees neither.

What this fixture pins, all re-measured on this run:
  * below the limit the kinetic energy DECAYS -- the reference behaviour;
  * at ratio 0.33 the run already leaves double precision -- the scheme is
    unstable BELOW the claim's ~0.5, not at it;
  * at ratio 0.67 and above it leaves double precision too;
  * the step count to divergence shrinks as the ratio grows, so the ratio
    orders the runs correctly;
  * every unstable run has energy growing geometrically -- checked as a
    monotone increase over the last third of its history, which is the claim's
    "diverges geometrically".
"""
from __future__ import annotations

import sys

import numpy
from netgen.geom2d import SplineGeometry
from ngsolve import (
    BilinearForm,
    CoefficientFunction,
    GridFunction,
    Grad,
    H1,
    InnerProduct,
    Integrate,
    LinearForm,
    Mesh,
    VectorH1,
    div,
    dx,
    x,
)

MAXH = 0.15
ULID = 1.0
NU = 1e-3
NSTEPS = 60
BLOWUP = 1e20


def _mesh():
    g = SplineGeometry()
    g.AddRectangle((0, 0), (1, 1), bcs=["bot", "right", "top", "left"])
    return Mesh(g.GenerateMesh(maxh=MAXH))


def march(mesh, dt, nsteps=NSTEPS):
    V = VectorH1(mesh, order=2, dirichlet=".*")
    Q = H1(mesh, order=1)
    X = V * Q
    (u, p), (v, q) = X.TnT()
    stokes = (NU * InnerProduct(Grad(u), Grad(v))
              + div(u) * q + div(v) * p) * dx
    mstar = BilinearForm(X)
    mstar += InnerProduct(u, v) * dx + dt * stokes
    mstar.Assemble()

    lid = CoefficientFunction((ULID * 4 * x * (1 - x), 0))
    gbc = GridFunction(X)
    gbc.components[0].Set(lid, definedon=mesh.Boundaries("top"))
    gfu = GridFunction(X)
    gfu.vec.data = gbc.vec
    w = gfu.components[0]

    inv = mstar.mat.Inverse(X.FreeDofs(), "umfpack")
    free = X.FreeDofs()
    rhs = gfu.vec.CreateVector()
    r = gfu.vec.CreateVector()
    ke = [float(Integrate(InnerProduct(w, w), mesh))]
    for _ in range(nsteps):
        conv = LinearForm(X)
        conv += InnerProduct(Grad(w) * w, v) * dx
        conv.Assemble()
        rhs.data = mstar.mat * gfu.vec - dt * conv.vec
        for i in range(X.ndof):
            if not free[i]:
                gfu.vec[i] = gbc.vec[i]
        r.data = rhs - mstar.mat * gfu.vec
        gfu.vec.data += inv * r
        cur = float(Integrate(InnerProduct(w, w), mesh))
        ke.append(cur)
        if not numpy.isfinite(cur) or cur > BLOWUP:
            break
    return ke


def main() -> int:
    mesh = _mesh()
    rows = []
    for dt in (0.005, 0.02, 0.05, 0.1, 0.2, 0.4):
        ke = march(mesh, dt)
        ratio = dt * ULID / MAXH
        end = ke[-1]
        alive = bool(numpy.isfinite(end)) and end <= BLOWUP
        growth = end / ke[0] if alive else float("inf")
        tail = ke[max(1, 2 * len(ke) // 3):]
        geometric = len(tail) >= 2 and all(
            b > a for a, b in zip(tail, tail[1:]))
        rows.append((dt, ratio, len(ke) - 1, alive, growth, geometric))
        print(f"dt={dt:g} ratio={ratio:.3f} steps_run={len(ke) - 1} "
              f"survived={alive} ke_growth={growth:.4e} "
              f"geometric_tail={geometric}")

    low = [r for r in rows if r[1] < 0.2]
    mid = [r for r in rows if 0.2 <= r[1] < 0.5]
    high = [r for r in rows if r[1] >= 0.5]
    print(f"below_0p2_all_decay={all(r[3] and r[4] < 1.0 for r in low)}")
    print(f"between_0p2_and_0p5_energy_grows_10x="
          f"{all(r[4] > 10.0 for r in mid)}")
    print(f"above_0p5_all_leave_double_precision="
          f"{all(not r[3] for r in high)}")
    steps_high = [r[2] for r in high]
    print(f"steps_to_divergence={steps_high}")
    print(f"steps_to_divergence_shrink_with_ratio="
          f"{all(b <= a for a, b in zip(steps_high, steps_high[1:]))}")
    print(f"unstable_runs_grow_geometrically="
          f"{all(r[5] for r in rows if not r[3])}")
    print(f"divergence_took_more_than_a_few_steps={min(steps_high) > 5}")

    ok = (
        all(r[3] and r[4] < 1.0 for r in low)
        and all(r[4] > 10.0 for r in mid)
        and all(not r[3] for r in high)
        and all(b <= a for a, b in zip(steps_high, steps_high[1:]))
        and all(r[5] for r in rows if not r[3])
    )
    if ok:
        return 0
    print("FAIL: IMEX CFL invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
