"""FEniCSx <-> scikit-fem exchanging a VECTOR across the interface.

WHAT THIS ESTABLISHES, and why it is not covered by any existing fixture.
Every coupling fixture in this tree before it exchanges ONE SCALAR
(a temperature and a heat flux). The primitive every multiphysics coupling is
built on — TSI, FSI, contact — is a VECTOR exchange: a displacement in and a
traction out, both with two components, in both directions, across two
interface meshes that do not match. A scalar fixture cannot establish it, and
several of the ways it goes wrong are invisible to one:

  * a mapping that interleaves the components instead of interpolating each on
    its own returns an array of the right length and converges;
  * a participant that exports the raw traction (sigma . n) instead of the
    shipped convention -(sigma . n) flips the sign the Neumann side applies and
    converges to a different answer;
  * a component that is dropped or held at zero is invisible in any norm the
    other component dominates.

So the assertions here are componentwise throughout: continuity of BOTH
displacement components across the interface, equilibrium of BOTH traction
components, conservation of each component separately, and agreement with two
independent references — the closed form of the split problem and an un-split
monolithic solve of the same problem in one code.

THE PROBLEM is a plane-strain bimaterial strip split by a straight interface,
with a prescribed displacement on the whole outer boundary taken from a field
that is piecewise linear in (x, y). It is a patch test, so P1 reproduces it
EXACTLY and the only error a converged coupling can carry is the coupling's
own. Both components are alive at the interface and both VARY along it — with a
constant interface profile any mapping reproduces it, including a wrong one,
and the non-matching-mesh claim would be vacuous.

Nothing here is pinned. The tolerances are what the physics has to beat and the
errors are printed, so the run reports its own numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The shared library lives in a sibling `_lib/` DIRECTORY, not as a bare file,
# because scripts/mutate_tier2_fixtures.py stages a fixture into a scratch tree
# and copies only sibling directories whose name starts with `_`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                     # noqa: E402


def body() -> None:
    L.require_available("fenics", "skfem")
    p = L.VECTOR_DEFAULT
    # theta = 1/(1 + max_c rho_c). The max is not caution: the driver applies
    # ONE theta to the whole interface state, its per-component amplification
    # is sqrt((1-theta)^2 + rho_c theta^2), and that is below one only while
    # theta < 2/(1+rho_c) — so the LARGEST rho is the binding one. See
    # coupling/vector_relaxation_needs_the_worst_component.
    L.vector_arrangement("fenics_D_skfem_N", "fenics", "skfem", "left")
    L.vector_arrangement("skfem_D_fenics_N", "skfem", "fenics", "left")
    # …and with the roles swapped, so each code is proven on BOTH sides of the
    # interface against this partner rather than only in one arrangement. The
    # mirrored problem keeps the softer block on the Dirichlet side; what makes
    # the stiff-Dirichlet direction slow is the relaxation, not the backends,
    # and that is measured on its own rather than paid for here.
    L.vector_arrangement("fenics_N_skfem_D", "fenics", "skfem", "right",
                         problem=L.VECTOR_MIRRORED)
    print(f"rho_x={p.rho_x:.6f}")
    print(f"rho_y={p.rho_y:.6f}")
    print(f"theta_dirichlet_left={p.theta_opt('left'):.6f}")
    print("vector_arrangements_run=3")


L.main(body)
