"""Tier-2: gfu.components is a tuple of ComponentGridFunction views, not a list
and not a vector slice -- and the views are live.

Claim: ngsolve linear_elasticity#2 -- "gfu.components on a VectorH1
GridFunction returns a tuple of ComponentGridFunction views (NOT a list, NOT
direct vector slices).  Each component i is callable as
gfu.components[i](mesh(x,y)) for pointwise evaluation of the i-th displacement
component."

Wrong variant: treating .components as a mutable list, or as a copy.

What this fixture pins, all re-measured on this run:
  * the container's runtime type is exactly tuple and each element's class name
    is ComponentGridFunction;
  * because it is a tuple, item assignment raises TypeError -- the concrete
    consequence of "NOT a list", exercised rather than asserted;
  * the components are callable at a mesh point and their values match the
    field they were set from;
  * they are VIEWS, not copies: writing through a component changes the parent
    GridFunction's own vector, and writing through the parent changes what the
    component evaluates to;
  * the component's vector length is the scalar DOF count, and the two together
    account for the whole VectorH1 vector.

Mutation control (INVERTED direction): this fixture executes no pathology -- it
only demonstrates the correct behaviour -- so there is nothing to remove.
``T2_MUTATE=1 python source.py`` instead COMMITS the documented mistake at the
site the pitfall is about: the component's vector is copied
(``comps[0].vec.CreateVector()``) before the +7 is written into it, i.e. the
component is treated as a copy rather than a live view.  The parent
GridFunction's vector then does not move, so ``component_is_a_view_not_a_copy=
True`` disappears (and with it ``component_sees_parent_write=True``, because the
compensating -7 write on the parent is no longer cancelled) and the fixture goes
red.
"""
from __future__ import annotations

import os
import sys

import numpy
from netgen.geom2d import unit_square
from ngsolve import CoefficientFunction, GridFunction, H1, Mesh, VectorH1, x, y

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.25))
    fes = VectorH1(mesh, order=2)
    gf = GridFunction(fes)
    gf.Set(CoefficientFunction((x * x, -3 * y)))

    comps = gf.components
    print(f"container_type={type(comps).__name__}")
    print(f"container_is_tuple={type(comps).__name__ == 'tuple'}")
    names = [type(c).__name__ for c in comps]
    print(f"element_types={names}")
    print(f"elements_are_componentgridfunction="
          f"{all(n == 'ComponentGridFunction' for n in names)}")

    assign_err = ""
    try:
        comps[0] = comps[1]
    except TypeError as exc:
        assign_err = str(exc)
    print(f"item_assignment_raises_typeerror={bool(assign_err)}")

    pt = mesh(0.4, 0.6)
    v0 = float(comps[0](pt))
    v1 = float(comps[1](pt))
    print(f"component0_at_point={v0:.8f} expected={0.4 * 0.4:.8f}")
    print(f"component1_at_point={v1:.8f} expected={-3 * 0.6:.8f}")
    print(f"components_are_callable_and_correct="
          f"{abs(v0 - 0.16) < 1e-8 and abs(v1 + 1.8) < 1e-8}")

    scalar_ndof = H1(mesh, order=2).ndof
    print(f"component_vector_size={comps[0].vec.size} scalar_ndof={scalar_ndof}")
    print(f"component_size_is_scalar_ndof={comps[0].vec.size == scalar_ndof}")
    print(f"components_account_for_the_whole_vector="
          f"{comps[0].vec.size + comps[1].vec.size == fes.ndof}")

    # Views, not copies: write through the component.
    before = gf.vec.FV().NumPy().copy()
    write_vec = comps[0].vec
    if MUTATE:
        # the documented MISTAKE: treat the component as a copy, not a view
        write_vec = comps[0].vec.CreateVector()
        write_vec.data = comps[0].vec
    write_vec[0] += 7.0
    after = gf.vec.FV().NumPy().copy()
    moved = float(numpy.abs(after - before).max())
    print(f"parent_changed_by_component_write={moved:.6f}")
    print(f"component_is_a_view_not_a_copy={abs(moved - 7.0) < 1e-12}")

    # ...and the other way round.
    gf.vec[0] -= 7.0
    back = float(comps[0](pt))
    print(f"component_sees_parent_write={abs(back - v0) < 1e-12}")

    ok = (
        type(comps).__name__ == "tuple"
        and all(n == "ComponentGridFunction" for n in names)
        and bool(assign_err)
        and abs(v0 - 0.16) < 1e-8 and abs(v1 + 1.8) < 1e-8
        and comps[0].vec.size == scalar_ndof
        and comps[0].vec.size + comps[1].vec.size == fes.ndof
        and abs(moved - 7.0) < 1e-12
        and abs(back - v0) < 1e-12
    )
    if ok:
        return 0
    print("FAIL: components-view invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
