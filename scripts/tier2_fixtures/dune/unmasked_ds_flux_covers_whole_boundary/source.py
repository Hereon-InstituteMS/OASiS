"""Tier-2: an unmasked ds integral applies the flux to the WHOLE
boundary, and a wrong BC indicator degrades to no BC at all.

  poisson#17                             -Laplace(u) = 0 with u = 0 on
                                         x=0 and du/dn = 1 on x=1 has
                                         the answer u = x, so max(u) is
                                         1. Writing the flux as g*v*ds
                                         instead of g*mask*v*ds returns
                                         a converged solution with
                                         max(u_h) about twice that. The
                                         masked version is exact.
  _general natural_bc_measured.
  Signal_unmasked_ds                     the same measurement.
The sibling claim poisson#8 (an indicator whose threshold selects no
facet) is NOT re-tested here: the 'never' variant of
dirichletbc_not_in_scheme_list_silent already solves that exact singular
system, and repeating it cost this fixture a 764 s run that had to be
killed.

Verified by execution against dune-fem 2.12.0.2.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from dune.grid import structuredGrid                            # noqa: E402
from dune.fem.space import lagrange                             # noqa: E402
from dune.fem.scheme import galerkin                            # noqa: E402
from dune.ufl import DirichletBC                                # noqa: E402
import dune.fem as dfem                                         # noqa: E402
from ufl import (TrialFunction, TestFunction, SpatialCoordinate, # noqa: E402
                 dot, grad, dx, ds, conditional)


def main() -> int:
    fail: list[str] = []
    tol = 1e-8

    # ── poisson#17: the mask ────────────────────────────────────────
    gridView = structuredGrid([0, 0], [1, 1], [8, 8])
    space = lagrange(gridView, order=1)
    u, v = TrialFunction(space), TestFunction(space)
    x = SpatialCoordinate(space)
    a = dot(grad(u), grad(v)) * dx
    clamp = DirichletBC(space, 0, conditional(x[0] < tol, 1, 0))
    mask = conditional(x[0] > 1 - tol, 1.0, 0.0)

    PARAMS = {"linear.tolerance": 1e-14}
    scheme_ok = galerkin([a == 1.0 * mask * v * ds, clamp],
                         solver="cg", parameters=PARAMS)
    uh_ok = space.interpolate(0, name="uh_ok")
    info_ok = scheme_ok.solve(target=uh_ok)
    vals_ok = np.array(uh_ok.as_numpy)
    err = float(np.sqrt(dfem.integrate((uh_ok - x[0]) ** 2,
                                       gridView=gridView, order=4)))
    print(f"masked_converged={bool(info_ok['converged'])}")
    print(f"masked_max={vals_ok.max():.6f}")
    print(f"masked_l2_error_against_x={err:.3e}")
    print(f"masked_is_exact={err < 1e-12}")
    # 1e-12 is the linear-solver floor, not a discretisation
    # floor: u = x is in the P1 space exactly, so the only
    # error left is how tightly CG was asked to converge.
    if not info_ok["converged"] or err >= 1e-12:
        fail.append(f"the correctly masked Neumann problem did not "
                    f"reproduce u = x (L2 error {err:.3e}); without "
                    f"that control the unmasked number means nothing")

    scheme_bad = galerkin([a == 1.0 * v * ds, clamp],
                          solver="cg", parameters=PARAMS)
    uh_bad = space.interpolate(0, name="uh_bad")
    info_bad = scheme_bad.solve(target=uh_bad)
    vals_bad = np.array(uh_bad.as_numpy)
    ratio = float(vals_bad.max()) / float(vals_ok.max())
    print(f"unmasked_converged={bool(info_bad['converged'])}")
    print(f"unmasked_max={vals_bad.max():.6f}")
    print(f"unmasked_over_masked_ratio={ratio:.4f}")
    print(f"unmasked_raised_nothing=True")
    print(f"unmasked_is_a_small_multiple_of_the_answer="
          f"{1.5 < ratio < 3.0}")
    if not info_bad["converged"]:
        fail.append("the unmasked run did NOT converge; the claim is "
                    "that it converges and is silently wrong")
    if not (1.5 < ratio < 3.0):
        fail.append(f"the unmasked flux gave {ratio:.4f}x the correct "
                    f"maximum; the claim is a small integer multiple, "
                    f"roughly double")

    if not fail:
        print("dune_unmasked_ds_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
