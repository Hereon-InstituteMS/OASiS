"""Tier-2: the scheme parameter dictionary rewrites some keys, warns
about some, and silently swallows the rest.

  poisson#10        'newton.*' is deprecated in favour of 'nonlinear.*'
                    and dune-fem REWRITES it, with a UserWarning.
                    Passing both spellings is accepted and the newton.*
                    value is dropped; only the nested
                    'newton.linear.*' / 'nonlinear.linear.*' collision
                    raises.
  nonlinear#3       Same prefix rule, plus: the Newton cap is
                    'maxiterations', not 'maxiter', and an unrecognised
                    key is accepted with no exception and no warning.
  poisson_mms#1     'newton.linear.*' -> 'linear.*' warning text, and
                    the old keys still WORK (warning only).

One scheme's worth of compiled code answers all three, because every
assertion is about the parameter dictionary, not about the form.

Verified by execution against dune-fem 2.12.0.2.

MUTATION CONTROL. T2_MUTATE=1 passes the CURRENT spelling
'nonlinear.tolerance' where the base run passes the deprecated
'newton.tolerance' — the pathology removed. Nothing is rewritten and
nothing warns, so 'newton_prefix_warned=True' is no longer printed and
a FAIL: line appears. Same form, so no new module is compiled.
"""
from __future__ import annotations

import os
import sys
import warnings

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dune.grid import structuredGrid                           # noqa: E402
from dune.fem.space import lagrange                            # noqa: E402
from dune.fem.scheme import galerkin                           # noqa: E402
from dune.ufl import DirichletBC                               # noqa: E402
from ufl import (TrialFunction, TestFunction,                   # noqa: E402
                 dot, grad, dx)


def _build(parameters):
    """Build a scheme, returning (scheme, captured warning texts)."""
    gridView = structuredGrid([0, 0], [1, 1], [4, 4])
    space = lagrange(gridView, order=1)
    u, v = TrialFunction(space), TestFunction(space)
    eqn = [dot(grad(u), grad(v)) * dx == 1.0 * v * dx,
           DirichletBC(space, 0)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scheme = galerkin(eqn, solver="cg", parameters=parameters)
        texts = [" ".join(str(w.message).split()) for w in caught]
    return space, scheme, texts


def main() -> int:
    fail: list[str] = []

    # 1. newton.* is rewritten to nonlinear.*, with a warning.
    tolerance_key = "nonlinear.tolerance" if MUTATE else "newton.tolerance"
    if MUTATE:
        print("mutation=the_caller_uses_the_current_nonlinear_prefix")
    space, scheme, texts = _build({tolerance_key: 1e-10})
    blob = " || ".join(texts)
    print(f"newton_prefix_warned="
          f"{any('newton' in t and 'deprecated' in t for t in texts)}")
    print(f"newton_prefix_warning_text={blob[:220]}")
    params = dict(scheme.parameters)
    print(f"rewritten_parameters={sorted(params.items())}")
    if "nonlinear.tolerance" not in params:
        fail.append(f"newton.tolerance was NOT rewritten to "
                    f"nonlinear.tolerance; scheme.parameters = {params}")
    if "newton.tolerance" in params:
        fail.append("the deprecated key survived the rewrite")
    if not any("deprecated" in t for t in texts):
        fail.append(f"no deprecation warning was emitted for "
                    f"newton.tolerance; warnings were {texts}")

    # …and it still WORKS: warning only, not an error.
    uh = space.interpolate(0, name="uh")
    info = scheme.solve(target=uh)
    print(f"deprecated_key_still_solves={bool(info['converged'])}")
    if not info["converged"]:
        fail.append("a scheme built with the deprecated key did not "
                    "solve; the claim is that the old keys still work")

    # 2. The nested newton.linear.* spelling has its own warning.
    _, _, texts_nl = _build({"newton.linear.tolerance": 1e-12})
    blob_nl = " || ".join(texts_nl)
    print(f"newton_linear_warning_text={blob_nl[:220]}")
    if not any("newton.linear" in t for t in texts_nl):
        fail.append(f"no 'newton.linear' deprecation warning: "
                    f"{texts_nl}")

    # 3. Both spellings at once: accepted, newton.* silently dropped.
    _, scheme_both, texts_both = _build({
        "newton.tolerance": 1e-4, "nonlinear.tolerance": 1e-12})
    got = dict(scheme_both.parameters).get("nonlinear.tolerance")
    print(f"both_spellings_accepted=True kept={got}")
    if got != 1e-12:
        fail.append(f"with both spellings passed, the surviving "
                    f"tolerance is {got!r}; the claim is that the "
                    f"nonlinear.* value wins and newton.* is dropped "
                    f"with no error")

    # 4. The nested COLLISION is the one thing that raises.
    try:
        _build({"newton.linear.tolerance": 1e-4,
                "nonlinear.linear.tolerance": 1e-12})
        print("nested_collision_raises=False")
        fail.append("newton.linear.* together with nonlinear.linear.* "
                    "was accepted; the claim is a KeyError")
    except KeyError as exc:
        print(f"nested_collision_raises={type(exc).__name__}")

    # 5. A typo'd or wholly bogus key: accepted, no warning, default
    #    silently left in place. This is the trap.
    _, scheme_typo, texts_typo = _build({"nonlinear.maxiter": 3,
                                         "totally.bogus.key": 42})
    print(f"bogus_keys_warnings={texts_typo}")
    print(f"bogus_keys_accepted_silently={len(texts_typo) == 0}")
    if texts_typo:
        fail.append(f"unrecognised keys now warn ({texts_typo}); the "
                    f"claim is that they are accepted in total silence")
    uh2 = space.interpolate(0, name="uh2")
    info2 = scheme_typo.solve(target=uh2)
    print(f"bogus_keys_run_to_normal_answer={bool(info2['converged'])}")
    if not info2["converged"]:
        fail.append("the run with a typo'd Newton cap did not converge; "
                    "the claim is that the typo leaves the DEFAULT in "
                    "place, so the answer is the normal one")

    # …and the correctly spelled cap is honoured, which is what makes
    # the silence dangerous rather than harmless.
    _, scheme_cap, _ = _build({"nonlinear.maxiterations": 1})
    print("correct_cap_key_is_maxiterations="
          + str("nonlinear.maxiterations"
                in dict(scheme_cap.parameters)))
    if "nonlinear.maxiterations" not in dict(scheme_cap.parameters):
        fail.append("nonlinear.maxiterations did not survive into "
                    "scheme.parameters, so the fixture cannot show "
                    "that 'maxiter' is the typo and 'maxiterations' "
                    "the real key")

    if not fail:
        print("dune_parameter_key_handling_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
