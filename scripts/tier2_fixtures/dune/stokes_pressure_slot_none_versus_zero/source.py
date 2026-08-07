"""Tier-2: a 0 where a None belongs in a DirichletBC value list pins
the pressure and over-constrains the Stokes system.

The claim (dune.stokes #5) says the pressure entry of the value list
must be None, not 0, and gives the mechanism: dune.ufl.DirichletBC
stores the raw list in ``.value`` and a None-to-0 substitution in
``.ufl_value``, and it is ``.value`` that builds the component mask.
Both are checked — the mechanism at the object level, the consequence
by COUNTING constrained degrees of freedom.

Counting is what makes this decisive. Looking at the pressure field
alone is not enough: with the 0 in place the pressure maximum only
drops from 2.0 to 1.5, which reads as "changed the answer somehow"
rather than "pinned the pressure", and an earlier attempt on this
claim concluded from exactly that number that the claim did not
reproduce. It does. The scheme's own dirichletBlocks show the same
264 velocity degrees of freedom constrained either way and 0 versus 68
pressure ones, and the pressure on the constrained boundary is exactly
zero in the second case. The refinement level is run twice so it is
clear the over-constraint is not a coarse-grid artefact.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from dune.grid import structuredGrid                            # noqa: E402
from dune.fem.space import lagrange, composite                  # noqa: E402
from dune.fem.scheme import galerkin                            # noqa: E402
from dune.ufl import Constant, DirichletBC                      # noqa: E402
from ufl import (TrialFunction, TestFunction, SpatialCoordinate,  # noqa: E402
                 as_vector, div, grad, inner, dx, lt, gt, Or)

TOL = 1e-8
PARAMS = {"linear.tolerance": 1e-12, "linear.maxiterations": 200000}
LEVELS = (8, 16)


def stokes(n):
    gridView = structuredGrid([0, 0], [1, 1], [n, n])
    velocity = lagrange(gridView, dimRange=2, order=2)
    pressure = lagrange(gridView, order=1)
    W = composite(velocity, pressure)
    t, s = TrialFunction(W), TestFunction(W)
    u, p = as_vector([t[0], t[1]]), t[2]
    v, q = as_vector([s[0], s[1]]), s[2]
    x = SpatialCoordinate(W)
    a = (inner(grad(u), grad(v)) - p * div(v) - q * div(u)) * dx
    L = Constant(0.0, name="zero") * q * dx
    return gridView, W, a, L, x, velocity.size


def solve(n, pressure_slot, tag):
    gridView, W, a, L, x, nv = stokes(n)
    inlet = lt(x[0], TOL)
    walls = Or(lt(x[1], TOL), gt(x[1], 1 - TOL))
    profile = x[1] * (1 - x[1])
    bcs = [DirichletBC(W, [profile, 0, pressure_slot], inlet),
           DirichletBC(W, [0, 0, pressure_slot], walls)]
    scheme = galerkin([a == L] + bcs, solver=("suitesparse", "umfpack"),
                      parameters=PARAMS)
    wh = W.interpolate([0, 0, 0], name=f"{tag}{n}")
    info = scheme.solve(target=wh)
    blocks = np.array(scheme.dirichletBlocks).reshape(-1)
    values = np.array(wh.as_numpy)
    p_values = values[nv:]
    # Coordinates of the PRESSURE dofs: interpolate x and y INTO the
    # pressure leg. Interpolating them into the velocity leg and then
    # slicing the pressure one gives a block of zeros, which silently
    # marks every pressure dof as sitting on x = 0.
    px = np.array(W.interpolate([0, 0, x[0]],
                                name=f"{tag}{n}cx").as_numpy)[nv:]
    py = np.array(W.interpolate([0, 0, x[1]],
                                name=f"{tag}{n}cy").as_numpy)[nv:]
    on_boundary = (px < TOL) | (py < TOL) | (py > 1 - TOL)
    return {
        "converged": bool(info["converged"]),
        "velocity_constrained": int(blocks[:nv].sum()),
        "pressure_constrained": int(blocks[nv:].sum()),
        "p_on_boundary_max": float(np.abs(p_values[on_boundary]).max()),
        "p_max": float(np.abs(p_values).max()),
        "boundary_p_dofs": int(on_boundary.sum()),
        "values": values,
        "bcs": bcs,
    }


def main() -> int:
    fail: list[str] = []

    # 1. The mechanism, at the object level: the None survives in
    #    .value and is replaced by 0 in .ufl_value.
    _gv, W, _a, _L, x, _nv = stokes(LEVELS[0])
    none_bc = DirichletBC(W, [x[1] * (1 - x[1]), 0, None], lt(x[0], TOL))
    zero_bc = DirichletBC(W, [x[1] * (1 - x[1]), 0, 0], lt(x[0], TOL))
    none_kept = none_bc.value[2] is None
    zero_kept = zero_bc.value[2] == 0
    substituted = str(none_bc.ufl_value) == str(zero_bc.ufl_value)
    print(f"none_survives_in_bc_value={none_kept}")
    print(f"zero_is_stored_in_bc_value={zero_kept}")
    print(f"ufl_value_is_identical_for_none_and_zero={substituted}")
    if not (none_kept and zero_kept and substituted):
        fail.append(f"DirichletBC did not keep the None in .value while "
                    f"mapping it to 0 in .ufl_value (value[2]="
                    f"{none_bc.value[2]!r}, ufl_value match "
                    f"{substituted}); that asymmetry is the claim's "
                    f"stated mechanism")

    # 2. The consequence, counted.
    runs = {}
    for n in LEVELS:
        runs[(n, "none")] = solve(n, None, "none")
        runs[(n, "zero")] = solve(n, 0, "zero")
        for tag in ("none", "zero"):
            r = runs[(n, tag)]
            print(f"n{n}_{tag}_converged={r['converged']}")
            print(f"n{n}_{tag}_velocity_dofs_constrained="
                  f"{r['velocity_constrained']}")
            print(f"n{n}_{tag}_pressure_dofs_constrained="
                  f"{r['pressure_constrained']}")
            print(f"n{n}_{tag}_boundary_pressure_max="
                  f"{r['p_on_boundary_max']:.6e}")
            print(f"n{n}_{tag}_pressure_max={r['p_max']:.6f}")

    same_velocity = all(
        runs[(n, "none")]["velocity_constrained"]
        == runs[(n, "zero")]["velocity_constrained"] for n in LEVELS)
    none_frees_pressure = all(
        runs[(n, "none")]["pressure_constrained"] == 0 for n in LEVELS)
    zero_pins_pressure = all(
        runs[(n, "zero")]["pressure_constrained"] > 0 for n in LEVELS)
    print(f"velocity_constraints_are_identical={same_velocity}")
    print(f"none_slot_constrains_no_pressure_dofs={none_frees_pressure}")
    print(f"zero_slot_constrains_pressure_dofs={zero_pins_pressure}")
    if not (same_velocity and none_frees_pressure and zero_pins_pressure):
        fail.append("the 0 in the pressure slot did not add pressure "
                    "constraints on top of an unchanged velocity "
                    "constraint set; that difference IS the "
                    "over-constraint the claim describes")

    # 3. And the pinned value is exactly zero on that boundary, while
    #    the None form leaves it free.
    pinned = all(runs[(n, "zero")]["p_on_boundary_max"] == 0.0
                 for n in LEVELS)
    free = all(runs[(n, "none")]["p_on_boundary_max"] > 1e-6
               for n in LEVELS)
    print(f"zero_slot_pins_boundary_pressure_to_exactly_zero={pinned}")
    print(f"none_slot_leaves_boundary_pressure_free={free}")
    if not (pinned and free):
        fail.append("the boundary pressure was not driven to exactly "
                    "zero by the 0 entry while staying free under None")

    # 4. Refinement does not remove it — the extra constraints grow
    #    with the mesh rather than washing out.
    coarse = runs[(LEVELS[0], "zero")]["pressure_constrained"]
    fine = runs[(LEVELS[-1], "zero")]["pressure_constrained"]
    print(f"zero_slot_pressure_constraints_coarse={coarse}")
    print(f"zero_slot_pressure_constraints_fine={fine}")
    grows = fine > coarse
    print(f"over_constraint_grows_with_refinement={grows}")
    if not grows:
        fail.append(f"the number of pinned pressure dofs did not grow "
                    f"from {coarse} to more under refinement, so the "
                    f"claim's 'no refinement removes it' cannot be read "
                    f"off this measurement")

    # 5. It changes the answer, and by a lot.
    for n in LEVELS:
        gap = float(np.abs(runs[(n, "none")]["values"]
                           - runs[(n, "zero")]["values"]).max())
        print(f"n{n}_solution_difference={gap:.6e}")
    big = all(float(np.abs(runs[(n, "none")]["values"]
                           - runs[(n, "zero")]["values"]).max()) > 0.1
              for n in LEVELS)
    print(f"zero_slot_changes_the_solution={big}")
    if not big:
        fail.append("the two boundary-condition forms gave the same "
                    "answer")

    print(f"pressure_slot_must_be_none_not_zero="
          f"{none_frees_pressure and zero_pins_pressure and pinned}")

    if not fail:
        print("dune_stokes_pressure_slot_gate=OK")
        return 0
    for reason in fail:
        print(f"FAIL: {reason}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
