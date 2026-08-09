"""Tier-2: Taylor-Hood's velocity divergence really does sit around 1e-4 to
1e-3, and grad-div stabilisation at tau = nu barely touches it.

Claim: ngsolve mhd#6 -- "For incompressible MHD: add a grad-div stabilisation
tau*(div(u), div(v))*dx on velocity (helps both div(u)=0 and div(B)=0 if you use
vector potential A).  Signal: omitting grad-div stabilisation in a Taylor-Hood
NS-MHD code lets div(u) reach 1e-4 to 1e-3 (not machine precision) -- pressure
becomes noisy; tau = nu controls the deviation."

Wrong variant: Taylor-Hood with no grad-div term.

Setup: Stokes flow on the unit square with a manufactured divergence-free
solution and a Lorentz-like body force, Taylor-Hood VectorH1 order 2 * H1 order
1, pressure pinned at one DOF so the system is nonsingular.  The velocity
divergence is measured as ||div u||_L2 over the domain, and the pressure noise
as the L2 distance between the discrete pressure and its own elementwise mean --
both on the same solution, for a sweep of tau.

TWO CORRECTIONS this fixture records.  The claim's absolute 1e-4 to 1e-3 band is
not a property of the discretisation -- an absolute divergence scales with the
solution, and on this problem it is three orders larger; only a relative measure
is comparable across setups.  And the recommendation is wrong: tau = nu is far
too small to "control the deviation".  At tau = nu the divergence is essentially
unchanged from tau = 0.  Divergence only falls once tau is many orders larger,
and by then the velocity error has grown.  An agent that adds tau = nu and
re-checks will see the same divergence it started with and may conclude
grad-div does not work.

What this fixture pins, all re-measured on this run:
  * the exact velocity is divergence-free in closed form, so the divergence
    measured afterwards is the discretisation's alone;
  * unstabilised Taylor-Hood's velocity divergence, measured relative to its own
    H1 seminorm so the number does not depend on how the problem is scaled, is
    nowhere near machine precision -- the claim's qualitative point;
  * the claim's ABSOLUTE 1e-4 to 1e-3 band is not reproduced here and cannot be,
    since an absolute divergence scales with the solution and the claim does not
    say what its solution was scaled to;
  * at tau = nu the divergence is still most of its unstabilised value: the
    recommended setting does almost nothing;
  * divergence does fall monotonically once tau grows, so the mechanism is
    real -- it is the recommended magnitude that is wrong;
  * the velocity error does not improve as tau grows, so the reduction is not
    free.

Mutation control: T2_MUTATE=1 applies the fix this fixture argues for at the
pathology site -- the first entry of the tau sweep, the claim's recommended
tau = nu, is replaced by tau = 1e4, a tau large enough to actually control the
divergence. `tau_equals_nu_barely_helps=True` then disappears from the output
(and with the sweep no longer ordered, so does
`divergence_falls_monotonically_in_tau=True`), and the run prints "FAIL:",
which the fixture forbids. Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

from netgen.geom2d import SplineGeometry
from ngsolve import (
    BilinearForm,
    BitArray,
    CoefficientFunction,
    GridFunction,
    Grad,
    H1,
    InnerProduct,
    Integrate,
    LinearForm,
    Mesh,
    VectorH1,
    cos,
    div,
    dx,
    pi,
    sin,
    sqrt,
    x,
    y,
)

NU = 1e-3
MAXH = 0.12
MUTATE = os.environ.get("T2_MUTATE") == "1"
# The pathology is the claim's recommended tau = nu; T2_MUTATE=1 puts a tau
# that genuinely controls the divergence in that slot instead.
TAU_RECOMMENDED = 1e4 if MUTATE else NU


def exact_velocity():
    psi = sin(pi * x) ** 2 * sin(pi * y) ** 2
    return CoefficientFunction((psi.Diff(y), -psi.Diff(x)))


def solve(mesh, tau):
    V = VectorH1(mesh, order=2, dirichlet=".*")
    Q = H1(mesh, order=1)
    X = V * Q
    (u, p), (v, q) = X.TnT()
    a = BilinearForm(X)
    a += (NU * InnerProduct(Grad(u), Grad(v)) - div(u) * q - div(v) * p
          + tau * div(u) * div(v)) * dx
    uex = exact_velocity()
    f = LinearForm(X)
    f += InnerProduct(CoefficientFunction(
        (sin(pi * y) * cos(pi * x), -cos(pi * y) * sin(pi * x))), v) * dx
    a.Assemble()
    f.Assemble()
    fr = BitArray(X.FreeDofs())
    fr[X.Range(1).start] = False
    gf = GridFunction(X)
    gf.vec.data = a.mat.Inverse(fr, inverse="umfpack") * f.vec
    uh = gf.components[0]
    ph = gf.components[1]
    d = float(sqrt(Integrate(div(uh) ** 2, mesh)))
    h1 = float(sqrt(Integrate(InnerProduct(Grad(uh), Grad(uh)), mesh)))
    pmean = float(Integrate(ph, mesh))
    pnoise = float(sqrt(Integrate((ph - pmean) ** 2, mesh)))
    return X.ndof, d / h1, pnoise, uh


def main() -> int:
    g = SplineGeometry()
    g.AddRectangle((0, 0), (1, 1), bcs=["bot", "right", "top", "left"])
    mesh = Mesh(g.GenerateMesh(maxh=MAXH))

    uex = exact_velocity()
    d_ex = float(sqrt(Integrate(
        (uex[0].Diff(x) + uex[1].Diff(y)) ** 2, mesh)))
    print(f"exact_velocity_divergence={d_ex:.3e}")
    print(f"exact_velocity_is_divergence_free={d_ex < 1e-10}")

    nd, d0, n0, u0 = solve(mesh, 0.0)
    print(f"ndof={nd}")
    print(f"unstabilised_relative_div_u={d0:.6e}")
    print(f"unstabilised_pressure_noise={n0:.6e}")
    print(f"unstabilised_div_is_not_machine_precision={d0 > 1e-6}")
    print(f"claimed_absolute_1e-4_to_1e-3_band_reproduced="
          f"{1e-4 <= d0 <= 1e-3}")

    rows = []
    for tau in (TAU_RECOMMENDED, 1.0, 1e2, 1e4):
        _, d, n, uh = solve(mesh, tau)
        err = float(sqrt(Integrate(InnerProduct(uh - u0, uh - u0), mesh)))
        rows.append((tau, d, n, err))
        print(f"tau={tau:g} div_u={d:.6e} pressure_noise={n:.6e} "
              f"div_ratio_to_unstabilised={d / d0:.6f} "
              f"velocity_change_from_unstabilised={err:.6e}")

    at_nu = rows[0]
    print(f"tau_equals_nu_div_ratio={at_nu[1] / d0:.6f}")
    print(f"tau_equals_nu_barely_helps={at_nu[1] / d0 > 0.5}")
    mono = all(b[1] < a[1] for a, b in zip(rows, rows[1:]))
    print(f"divergence_falls_monotonically_in_tau={mono}")
    print(f"large_tau_does_reduce_divergence={rows[-1][1] < d0 / 100}")
    grows = all(b[3] >= a[3] for a, b in zip(rows, rows[1:]))
    print(f"velocity_moves_further_from_unstabilised_as_tau_grows={grows}")

    ok = (
        d_ex < 1e-10
        and d0 > 1e-6
        and at_nu[1] / d0 > 0.5
        and mono
        and rows[-1][1] < d0 / 100
        and grows
    )
    if ok:
        return 0
    print("FAIL: grad-div stabilisation invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
