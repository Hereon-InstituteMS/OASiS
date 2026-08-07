"""Tier-2: dune-fem differentiates the form itself, so the form must
carry a TRIAL function.

  nonlinear#0        DUNE-fem linearises and applies Newton internally;
                     scheme.solve() returns a dict carrying 'iterations'
                     (Newton) alongside 'linear_iterations'.
  nonlinear#1        writing the nonlinearity in the DISCRETE solution
                     (a = (1+uh**2)*dot(grad(uh),grad(v))*dx) leaves the
                     form with ONE argument and is rejected with
                     ValueError 'Integrands model requires form with at
                     least two arguments.' Both working spellings —
                     a == b and F == 0 — converge to the same solution.
  nonlinear#2        the returned object is a plain dict with keys
                     ['converged', 'iterations', 'linear_iterations',
                     'timing']; that dict, not a message string, is the
                     detector. (The catalog's retracted 'Newton did not
                     converge' string is asserted absent.)
  navier_stokes#0    dot(b, grad(u))*v raises ValueError 'Invalid ranks
                     1 and 1 in product.' — the convective term is
                     dot(grad(u), u), contracted with inner.
  navier_stokes#1    a residual built from a discrete function is
                     rejected the same way as nonlinear#1.

One nonlinear problem, -div((1+u^2) grad u) = 1 on a small grid,
supports all five.

Verified by execution against dune-fem 2.12.0.2.

MUTATION CONTROL. T2_MUTATE=1 writes the "discrete function" form with
the TRIAL function instead — the pathology removed, because the form
then carries two arguments. Nothing is rejected, so
'discrete_form_argument_count=1' and
'discrete_function_form_rejected=ValueError' are no longer printed and
a FAIL: line appears. The corrected form is the one the fixture already
compiles for the a == b spelling.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dune.grid import structuredGrid                            # noqa: E402
from dune.fem.space import lagrange                             # noqa: E402
from dune.fem.scheme import galerkin                            # noqa: E402
from dune.ufl import DirichletBC                                # noqa: E402
from ufl import (TrialFunction, TestFunction, dot, grad, dx,     # noqa: E402
                 inner, as_vector)


def main() -> int:
    fail: list[str] = []
    gridView = structuredGrid([0, 0], [1, 1], [8, 8])
    space = lagrange(gridView, order=1)
    u, v = TrialFunction(space), TestFunction(space)
    dbc = DirichletBC(space, 0)

    # ── nonlinear#1 / navier_stokes#1: the rejected spelling ────────
    uh_bad = space.interpolate(0, name="uh_bad")
    if MUTATE:
        print("mutation=the_rejected_form_is_written_in_the_trial_"
              "function_so_it_carries_two_arguments")
        bad = (1 + u ** 2) * dot(grad(u), grad(v)) * dx
    else:
        bad = (1 + uh_bad ** 2) * dot(grad(uh_bad), grad(v)) * dx
    print(f"discrete_form_argument_count={len(bad.arguments())}")
    try:
        galerkin([bad == 0, dbc], solver="cg")
        print("discrete_function_form_rejected=False")
        fail.append("a form written in the discrete solution was "
                    "ACCEPTED; the claim is that dune-fem needs two "
                    "arguments and rejects it")
    except ValueError as exc:
        msg = " ".join(str(exc).split())
        print(f"discrete_function_form_rejected={type(exc).__name__}")
        print(f"discrete_function_form_message={msg[:180]}")
        if "at least two arguments" not in msg:
            fail.append(f"the rejection no longer says 'at least two "
                        f"arguments': {msg[:180]}")
    if len(bad.arguments()) != 1:
        fail.append(f"the discrete-function form carries "
                    f"{len(bad.arguments())} arguments; the claim rests "
                    f"on it having exactly one")

    # ── the two working spellings ──────────────────────────────────
    a = (1 + u ** 2) * dot(grad(u), grad(v)) * dx
    b = 1.0 * v * dx
    scheme_ab = galerkin([a == b, dbc], solver="cg")
    uh1 = space.interpolate(0, name="uh1")
    info1 = scheme_ab.solve(target=uh1)

    F = ((1 + u ** 2) * dot(grad(u), grad(v)) - 1.0 * v) * dx
    scheme_F = galerkin([F == 0, dbc], solver="cg")
    uh2 = space.interpolate(0, name="uh2")
    info2 = scheme_F.solve(target=uh2)

    m1 = float(np.array(uh1.as_numpy).max())
    m2 = float(np.array(uh2.as_numpy).max())
    print(f"a_equals_b_converged={bool(info1['converged'])}")
    print(f"F_equals_zero_converged={bool(info2['converged'])}")
    print(f"a_equals_b_max={m1:.8f}")
    print(f"F_equals_zero_max={m2:.8f}")
    print(f"both_spellings_agree={abs(m1 - m2) < 1e-10}")
    if not (info1["converged"] and info2["converged"]):
        fail.append("one of the two working spellings did not converge")
    if abs(m1 - m2) >= 1e-10:
        fail.append(f"the two spellings gave different answers "
                    f"({m1} vs {m2}); the claim is that they are "
                    f"identical")

    # ── nonlinear#0 / #2: what solve() returns ─────────────────────
    print(f"info_type={type(info1).__name__}")
    keys = sorted(info1)
    print(f"info_keys={keys}")
    expected = ["converged", "iterations", "linear_iterations", "timing"]
    if type(info1).__name__ != "dict" or keys != expected:
        fail.append(f"solve() returned a {type(info1).__name__} with "
                    f"keys {keys}; the claim is a plain dict with "
                    f"{expected}")
    n_newton = int(info1["iterations"])
    n_linear = int(info1["linear_iterations"])
    print(f"newton_iterations={n_newton}")
    print(f"linear_iterations={n_linear}")
    print(f"newton_ran_without_a_manual_loop={n_newton >= 1}")
    print(f"linear_iterations_exceed_newton={n_linear > n_newton}")
    if n_newton < 1:
        fail.append(f"the internal Newton reported {n_newton} "
                    f"iterations on a genuinely nonlinear problem")
    if n_linear <= n_newton:
        fail.append(f"linear_iterations ({n_linear}) did not exceed the "
                    f"Newton count ({n_newton}); the two numbers are "
                    f"supposed to be distinct quantities")

    # the retracted string is nowhere in what solve() gives you
    blob = repr(info1)
    print(f"info_repr_contains_newton_did_not_converge="
          f"{'Newton did not converge' in blob}")
    if "Newton did not converge" in blob:
        fail.append("solve() now returns the string the catalog "
                    "retracted; the retraction would be wrong")

    # ── navier_stokes#0: the convective term ───────────────────────
    # It has to be checked on a VECTOR velocity: with a scalar u,
    # dot(b, grad(u)) is a scalar and dot(b, grad(u))*v is perfectly
    # legal — measured, it builds. The claim is about the momentum
    # equation, where both factors are rank 1.
    uvec = lagrange(gridView, order=1, dimRange=2)
    uu, vv = TrialFunction(uvec), TestFunction(uvec)
    bvec = as_vector([1.0, 0.0])
    print(f"scalar_dot_b_grad_u_times_v_is_legal="
          f"{(dot(bvec, grad(u)) * v) is not None}")
    try:
        dot(bvec, grad(uu)) * vv
        print("vector_times_vector_rejected=False")
        fail.append("dot(b, grad(u))*v on a VECTOR velocity built a "
                    "product; the claim is that UFL rejects a vector "
                    "times a vector")
    except ValueError as exc:
        msg = " ".join(str(exc).split())
        print(f"vector_times_vector_rejected={type(exc).__name__}")
        print(f"vector_times_vector_message={msg[:140]}")
        if "Invalid ranks" not in msg:
            fail.append(f"the rejection no longer says 'Invalid ranks': "
                        f"{msg[:140]}")

    # and the spelling that works
    conv = inner(dot(grad(uu), uu), vv) * dx
    print(f"convective_term_builds={conv is not None}")
    print(f"convective_term_arguments={len(conv.arguments())}")
    if len(conv.arguments()) != 2:
        fail.append("inner(dot(grad(u), u), v)*dx does not carry two "
                    "arguments")

    if not fail:
        print("dune_nonlinear_form_requirements_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
