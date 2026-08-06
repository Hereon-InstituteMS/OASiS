"""Tier-2: with only the normal constraint the contacting body keeps an exact
zero-energy sliding mode, and the Coulomb limit is what decides stick from slip.

Claim: ngsolve contact#4 -- "For frictional contact: add a tangential-direction
penalty with Coulomb's law |f_t| <= mu * |f_n| as a complementarity constraint.
Signal: omitting the tangential penalty (only enforcing normal contact) lets the
contacting bodies SLIDE freely along their interface -- a vertical block resting
on an inclined plane slides off regardless of mu_friction; with tangential
penalty + Coulomb, the block sticks below the friction angle and slides above
it."

Wrong variant: normal constraint only, no tangential term.

Setup: a unit-square elastic block, E = 1000, nu = 0.3, resting on a plane
inclined by theta.  The normal constraint is imposed exactly -- u_y is Dirichlet
on the bottom edge -- so the normal direction is not what is being tested.  The
tangential direction is left to the tangential term.  Gravity is resolved in the
inclined frame: body force (g sin(theta), -g cos(theta)), so the tangential
demand is g sin(theta) and the normal pressure g cos(theta), and Coulomb's
condition tan(theta) <= mu is exactly the friction angle.

"Slides freely" is measured structurally, not by watching a displacement grow:
with no tangential term the free-DOF stiffness has an exact zero eigenvalue --
the rigid tangential translation costs no energy, so no equilibrium exists for
any nonzero tangential load, at any mu.  Adding the tangential penalty makes the
smallest eigenvalue strictly positive.  Both are eigenvalues of the same
assembled matrix, so the test holds on any host.

What this fixture pins, all re-measured on this run:
  * without the tangential term the smallest eigenvalue of the free-DOF
    stiffness is zero to roundoff, and it is the ONLY such mode;
  * that null mode is the tangential rigid translation -- checked by taking its
    eigenvector and comparing its x-part against a constant;
  * the applied tangential load has a nonzero component along it, so the
    singular system genuinely has no solution -- "slides freely";
  * adding the tangential penalty makes the smallest eigenvalue strictly
    positive with the same free-DOF count;
  * below the friction angle the trial tangential traction is under mu times the
    normal pressure (stick admissible), and above it the same computation
    exceeds the Coulomb limit (slip) -- the claim's stick/slip transition,
    measured on both sides of arctan(mu).

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology site
-- build() adds the tangential penalty GAMMA_T*ux*vx*ds("bot") even in the
"no tangential term" branch.  The free-DOF stiffness is then nonsingular, so
'tangential_slide_mode_costs_no_energy=True' and
'null_mode_is_tangential_translation=True' are absent from the output and the
fixture goes red.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy
import scipy.sparse
from netgen.geom2d import SplineGeometry
from ngsolve import (
    BilinearForm,
    CoefficientFunction,
    FESpace,
    GridFunction,
    H1,
    Id,
    InnerProduct,
    LinearForm,
    Mesh,
    Trace,
    ds,
    dx,
    grad,
)

E, NU = 1000.0, 0.3
MUE = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
GRAV = 100.0
MU_FRICTION = 0.3
GAMMA_T = 1e5
MAXH = 0.25
MUTATE = os.environ.get("T2_MUTATE") == "1"


def _mesh():
    g = SplineGeometry()
    g.AddRectangle((0, 0), (1, 1), bcs=["bot", "right", "top", "left"])
    return Mesh(g.GenerateMesh(maxh=MAXH))


def _sym(G):
    return 0.5 * (G + G.trans)


def _stress(G):
    return 2 * MUE * _sym(G) + LAM * Trace(_sym(G)) * Id(2)


def build(mesh, theta, tangential):
    fx = H1(mesh, order=1)                       # tangential: nothing pinned
    fy = H1(mesh, order=1, dirichlet="bot")      # normal contact imposed exactly
    fes = FESpace([fx, fy])
    (ux, uy), (vx, vy) = fes.TnT()
    gu = CoefficientFunction((grad(ux), grad(uy)), dims=(2, 2))
    gv = CoefficientFunction((grad(vx), grad(vy)), dims=(2, 2))
    a = BilinearForm(fes, symmetric=True)
    a += InnerProduct(_stress(gu), _sym(gv)) * dx
    # The pathology: no tangential term at all.  Under mutation the documented
    # fix is applied everywhere -- the tangential penalty is always present.
    if tangential or MUTATE:
        a += GAMMA_T * ux * vx * ds("bot")
    f = LinearForm(fes)
    f += (GRAV * math.sin(theta) * vx - GRAV * math.cos(theta) * vy) * dx
    a.Assemble()
    f.Assemble()
    return fes, a, f


def free_block(fes, a):
    rows, cols, vals = a.mat.COO()
    A = scipy.sparse.coo_matrix(
        (numpy.asarray(vals), (numpy.asarray(rows), numpy.asarray(cols))),
        shape=(fes.ndof, fes.ndof)).toarray()
    fd = fes.FreeDofs()
    idx = [i for i in range(fes.ndof) if fd[i]]
    S = A[numpy.ix_(idx, idx)]
    return 0.5 * (S + S.T), idx


def main() -> int:
    mesh = _mesh()
    theta_f = math.atan(MU_FRICTION)
    print(f"mu_friction={MU_FRICTION}")
    print(f"friction_angle_deg={math.degrees(theta_f):.4f}")

    # --- no tangential term: an exact zero-energy sliding mode ----------
    fes0, a0, f0 = build(mesh, 0.5 * theta_f, tangential=False)
    S0, idx0 = free_block(fes0, a0)
    w0, V0 = numpy.linalg.eigh(S0)
    scale = float(numpy.abs(S0).max())
    print(f"no_tangential_free_dofs={len(idx0)}")
    print(f"no_tangential_lambda_min={w0[0]:+.6e}")
    print(f"no_tangential_lambda_second={w0[1]:+.6e}")
    zero_mode = abs(w0[0]) < 1e-8 * scale
    only_one = abs(w0[1]) > 1e-6 * scale
    print(f"tangential_slide_mode_costs_no_energy={zero_mode}")
    print(f"exactly_one_such_mode={only_one}")

    # Is that null mode the tangential rigid translation?  Its x-part must be
    # (nearly) constant and carry (nearly) all of its norm.
    vec = V0[:, 0]
    nx = fes0.Range(0).stop                      # x-DOFs come first
    xpart = numpy.array([vec[k] for k, i in enumerate(idx0) if i < nx])
    ypart = numpy.array([vec[k] for k, i in enumerate(idx0) if i >= nx])
    x_energy = float(xpart @ xpart)
    y_energy = float(ypart @ ypart)
    spread = float(xpart.std() / max(abs(xpart.mean()), 1e-30))
    print(f"null_mode_x_fraction={x_energy / (x_energy + y_energy):.6f}")
    print(f"null_mode_x_relative_spread={spread:.3e}")
    print(f"null_mode_is_tangential_translation="
          f"{x_energy > 0.99 * (x_energy + y_energy) and spread < 1e-6}")

    # ...and the load pushes along it, so there is no equilibrium at all.
    rhs0 = numpy.array([f0.vec[i] for i in idx0])
    overlap = abs(float(vec @ rhs0)) / max(1e-30, float(numpy.abs(rhs0).max()))
    print(f"load_component_on_null_mode={overlap:.6e}")
    print(f"load_excites_the_slide_mode={overlap > 1e-6}")

    # --- with the tangential penalty -----------------------------------
    fes1, a1, f1 = build(mesh, 0.5 * theta_f, tangential=True)
    S1, idx1 = free_block(fes1, a1)
    w1 = numpy.linalg.eigvalsh(S1)
    print(f"with_tangential_free_dofs={len(idx1)}")
    print(f"with_tangential_lambda_min={w1[0]:+.6e}")
    print(f"same_free_dof_count={len(idx0) == len(idx1)}")
    print(f"tangential_penalty_removes_the_null_mode="
          f"{w1[0] > 1e-6 * float(numpy.abs(S1).max())}")

    # --- stick below the friction angle, slip above --------------------
    verdicts = {}
    for name, theta in (("below", 0.5 * theta_f), ("above", 1.6 * theta_f)):
        fes, a, f = build(mesh, theta, tangential=True)
        gfu = GridFunction(fes)
        gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="umfpack") * f.vec
        uxv = [float(gfu.components[0](mesh(float(t), 0.0)))
               for t in numpy.linspace(0.005, 0.995, 40)]
        traction = GAMMA_T * abs(float(numpy.mean(uxv)))
        limit = MU_FRICTION * GRAV * math.cos(theta)
        verdicts[name] = traction <= limit
        print(f"{name}_theta_deg={math.degrees(theta):.4f} "
              f"trial_traction={traction:.4f} coulomb_limit={limit:.4f} "
              f"stick_admissible={verdicts[name]}")
    print(f"sticks_below_the_friction_angle={verdicts['below']}")
    print(f"slips_above_the_friction_angle={not verdicts['above']}")

    ok = (
        zero_mode and only_one
        and x_energy > 0.99 * (x_energy + y_energy) and spread < 1e-6
        and overlap > 1e-6
        and len(idx0) == len(idx1)
        and w1[0] > 1e-6 * float(numpy.abs(S1).max())
        and verdicts["below"] and not verdicts["above"]
    )
    if ok:
        return 0
    print("FAIL: frictional-contact invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
