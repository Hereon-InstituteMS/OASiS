"""Tier-2: dimRange makes a space vector-valued, and its dofs come back
INTERLEAVED.

  linear_elasticity#0   lagrange(gridView, order=k) is SCALAR
                        (dimRange=1); without dimRange=2 the elasticity
                        form dies in UFL at the first sym(grad(u))
                        product, and space.interpolate([0, 0]) on a
                        scalar space raises before that.
  linear_elasticity#5   as_numpy on a dimRange=2 space is ONE flat
                        interleaved array [u0_x, u0_y, u1_x, u1_y, ...],
                        not two blocks — 578 entries for a 16x16 Q1
                        vector space, i.e. 289 nodes x 2. Reading the
                        first half as 'u_x' silently gives the
                        x-displacement of half the nodes.

The interleaving is proven by CONSTRUCTION, not by inspection: the
interpolant of (x, 2y) has a known value at every node, so the
even/odd de-interleaving is checked against those values, and the
half/half reading is shown to disagree with them.

Verified by execution against dune-fem 2.12.0.2.

MUTATION CONTROL. T2_MUTATE=1 re-orders the flat dof array into the
BLOCKED layout [all u_x | all u_y] before the layout tests run — the
pathology (interleaving) removed. reshape(-1, 2) then no longer
recovers (x, 2y) and the entries above 1 are no longer at odd indices,
so 'interleaved_reshape_is_correct=True',
'entries_above_one_are_all_at_odd_indices=True' and
'half_split_disagrees=True' are no longer printed and a FAIL: line
appears. Pure numpy; nothing extra compiles.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dune.grid import structuredGrid                           # noqa: E402
from dune.fem.space import lagrange                            # noqa: E402
from ufl import (TrialFunction, TestFunction, SpatialCoordinate, # noqa: E402
                 as_vector, sym, grad, inner, dx)


def main() -> int:
    fail: list[str] = []
    gridView = structuredGrid([0, 0], [1, 1], [16, 16])

    # ── linear_elasticity#0: the default is scalar ───────────────────
    scalar = lagrange(gridView, order=1)
    print(f"default_dimRange={scalar.dimRange}")
    if scalar.dimRange != 1:
        fail.append(f"lagrange(gridView, order=1) has dimRange "
                    f"{scalar.dimRange}; the claim is that the scalar "
                    f"space is the default")

    # interpolating a 2-vector into it raises
    try:
        scalar.interpolate([0, 0], name="bad")
        print("scalar_interpolate_vector_rejected=False")
        fail.append("space.interpolate([0, 0]) on a SCALAR space was "
                    "accepted; the claim is that it raises")
    except Exception as exc:                                 # noqa: BLE001
        print(f"scalar_interpolate_vector_rejected="
              f"{type(exc).__name__}")

    # the elasticity form dies in UFL on a scalar space
    us, vs = TrialFunction(scalar), TestFunction(scalar)
    try:
        inner(sym(grad(us)), sym(grad(vs))) * dx
        print("scalar_elasticity_form_rejected=False")
        fail.append("sym(grad(u)) on a SCALAR space built a form; the "
                    "claim is that the elasticity form dies in UFL "
                    "without dimRange")
    except Exception as exc:                                 # noqa: BLE001
        msg = " ".join(str(exc).split())
        print(f"scalar_elasticity_form_rejected={type(exc).__name__}")
        print(f"scalar_elasticity_message={msg[:160]}")

    # ── the vector space, and what dimRange buys ────────────────────
    vec = lagrange(gridView, order=1, dimRange=2)
    uv, vv = TrialFunction(vec), TestFunction(vec)
    form = inner(sym(grad(uv)), sym(grad(vv))) * dx
    print(f"vector_dimRange={vec.dimRange}")
    print(f"vector_elasticity_form_builds={form is not None}")
    print(f"vector_form_arguments={len(form.arguments())}")
    if len(form.arguments()) != 2:
        fail.append("the vector elasticity form does not carry two "
                    "arguments")

    # ── linear_elasticity#5: layout ─────────────────────────────────
    x = SpatialCoordinate(vec)
    uh = vec.interpolate(as_vector([x[0], 2 * x[1]]), name="uh")
    vals = np.array(uh.as_numpy)
    if MUTATE:
        # The pathology removed: hand the layout tests a BLOCKED array
        # [all u_x | all u_y] instead of the interleaved one.
        print("mutation=the_dof_array_is_re_ordered_into_a_blocked_"
              "layout")
        vals = np.concatenate([vals[0::2], vals[1::2]])
    n_nodes = 17 * 17
    print(f"as_numpy_len={len(vals)}")
    print(f"as_numpy_is_one_flat_array={vals.ndim == 1}")
    print(f"expected_nodes_times_2={n_nodes * 2}")
    if vals.ndim != 1 or len(vals) != n_nodes * 2:
        fail.append(f"as_numpy has shape {vals.shape}; the claim is a "
                    f"single flat array of {n_nodes * 2} entries for a "
                    f"16x16 Q1 vector space")

    # de-interleave the documented way and check against the known
    # interpolant: u_x = x in [0,1], u_y = 2y in [0,2]
    pair = vals.reshape(-1, 2)
    ux, uy = pair[:, 0], pair[:, 1]
    ok_interleaved = (
        abs(ux.min()) < 1e-12 and abs(ux.max() - 1.0) < 1e-12
        and abs(uy.min()) < 1e-12 and abs(uy.max() - 2.0) < 1e-12)
    print(f"interleaved_reshape_ux_range={ux.min():.6f},{ux.max():.6f}")
    print(f"interleaved_reshape_uy_range={uy.min():.6f},{uy.max():.6f}")
    print(f"interleaved_reshape_is_correct={ok_interleaved}")
    if not ok_interleaved:
        fail.append(f"reshape(-1, 2) did not recover u=(x, 2y): ux in "
                    f"[{ux.min()},{ux.max()}], uy in "
                    f"[{uy.min()},{uy.max()}]")

    # Decisive test of the LAYOUT rather than of the reshape: u_x never
    # exceeds 1, u_y reaches 2, so every entry above 1 must be a u_y
    # dof. Interleaved means those all sit at ODD flat indices; a
    # blocked [all u_x | all u_y] layout would put them in the second
    # half instead.
    big = np.where(vals > 1.0 + 1e-12)[0]
    all_odd = bool(len(big)) and bool(np.all(big % 2 == 1))
    in_second_half = bool(len(big)) and bool(
        np.all(big >= len(vals) // 2))
    print(f"entries_above_one={len(big)}")
    print(f"entries_above_one_are_all_at_odd_indices={all_odd}")
    # Informational only: with u_y = 2y the dofs above 1 are also the
    # upper half of the DOMAIN, which for this node ordering is the
    # second half of the array — so this alone proves nothing either
    # way. The odd-index test above, and the reshape test before it,
    # are what discriminate: under a blocked layout reshape(-1, 2)
    # would mix u_x and u_y and BOTH columns would reach 2.0.
    print(f"entries_above_one_are_all_in_second_half={in_second_half}")
    if not all_odd:
        fail.append(f"the u_y dofs (the only ones that can exceed 1) do "
                    f"not all sit at odd flat indices; the interleaved "
                    f"layout claim does not hold. Indices: "
                    f"{big[:20].tolist()}")
    if abs(ux.max() - 1.0) > 1e-12 or abs(uy.max() - 2.0) > 1e-12:
        fail.append("the two reshape columns do not separate cleanly "
                    "into u_x (max 1) and u_y (max 2), which is what a "
                    "blocked layout would look like")

    # …and the half/half reading disagrees with the correct one, which
    # is what makes it a silent-wrong rather than a crash.
    half = vals[:len(vals) // 2]
    print(f"half_split_first_half_range="
          f"{half.min():.6f},{half.max():.6f}")
    disagreement = float(np.abs(half - ux[:len(half)]).max())
    print(f"half_split_vs_interleaved_maxdiff={disagreement:.6f}")
    print(f"half_split_disagrees={disagreement > 1e-9}")
    if disagreement <= 1e-9:
        fail.append("reading the first half of the flat array as u_x "
                    "agrees with the interleaved reading, so this "
                    "fixture cannot demonstrate the trap")

    if not fail:
        print("dune_vector_space_layout_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
