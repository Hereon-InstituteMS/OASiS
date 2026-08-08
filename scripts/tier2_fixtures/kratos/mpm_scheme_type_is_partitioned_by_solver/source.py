"""Tier-2 (kratos.mpm::8): scheme_type is partitioned by solver, and a static
solver has no scheme_type key at all.

    implicit dynamic   newmark, bossak                     -- and nothing else
    explicit dynamic   forward_euler, central_difference   -- and nothing else
    static             no scheme_type key exists

Crossing the partition is not caught when the solver is created. CreateSolver
accepts every combination below; the rejection happens later, when the scheme is
built:

    implicit + central_difference
        Exception: The requested scheme type "central_difference" is not
        available! Available options are: "newmark", "bossak"
    explicit + newmark
        Exception: The requested scheme type "newmark" is not available!
        Available options are: "forward_euler", "central_difference"
    static + scheme_type
        RuntimeError: The item with name "scheme_type" is present in this
        Parameters but NOT in the default values

The last one is a different mechanism from the first two -- flat parameter
validation rather than a scheme factory -- which is why all three are probed.

MUTATION CONTROL (T2_MUTATE=1): the two crossed combinations are UNCROSSED --
implicit is given newmark and explicit is given central_difference, and the
static deck drops its scheme_type. The three cases that must be rejected are
then all legal, every probe still really calls into MPMApplication, and the
fixture reports the disagreement and exits 1.
"""
from __future__ import annotations

import json
import os
import sys

import KratosMultiphysics as KM
import KratosMultiphysics.MPMApplication  # noqa: F401
from KratosMultiphysics.MPMApplication import python_solvers_wrapper_mpm as wrapper

MUTATE = os.environ.get("T2_MUTATE") == "1"


def _build(solver_type, extra):
    """CreateSolver then build the scheme; return (phase, kind, message).

    phase is 'create' if CreateSolver itself objected, 'scheme' if the scheme
    factory did, and 'ok' if neither did.
    """
    model = KM.Model()
    model.CreateModelPart("Background_Grid")
    settings = dict({
        "solver_type": solver_type, "domain_size": 2,
        "model_import_settings": {"input_type": "mdpa", "input_filename": "body"},
        "grid_model_import_settings": {"input_type": "mdpa",
                                       "input_filename": "grid"},
        "material_import_settings": {"materials_filename": "m.json"}}, **extra)
    params = KM.Parameters(json.dumps({"problem_data": {"parallel_type": "OpenMP"},
                                       "solver_settings": settings}))
    try:
        solver = wrapper.CreateSolver(model, params)
    except Exception as exc:                 # noqa: BLE001 - classifying
        return "create", type(exc).__name__, str(exc).replace("\n", " ")
    try:
        # The scheme factory reads grid_model_part, which ImportModelPart would
        # normally set; give it the empty grid so no mdpa is needed.
        solver.grid_model_part = model.GetModelPart("Background_Grid")
        solver._CreateSolutionScheme()
    except Exception as exc:                 # noqa: BLE001 - classifying
        return "scheme", type(exc).__name__, str(exc).replace("\n", " ")
    return "ok", "", ""


def main() -> int:
    implicit_scheme = "newmark" if MUTATE else "central_difference"
    explicit_scheme = "central_difference" if MUTATE else "newmark"
    if MUTATE:
        print("mutation=crossed_scheme_types_uncrossed_and_static_scheme_dropped")

    imp = _build("dynamic", {"time_integration_method": "implicit",
                             "scheme_type": implicit_scheme})
    exp = _build("dynamic", {"time_integration_method": "explicit",
                             "scheme_type": explicit_scheme})
    sta = _build("static", {} if MUTATE else {"scheme_type": "newmark"})

    ok_imp = _build("dynamic", {"time_integration_method": "implicit",
                                "scheme_type": "bossak"})
    ok_exp = _build("dynamic", {"time_integration_method": "explicit",
                                "scheme_type": "forward_euler"})

    checks = [
        ("implicit_rejects_the_explicit_scheme", imp[0], "scheme"),
        ("implicit_lists_newmark_and_bossak",
         'Available options are: "newmark", "bossak"' in imp[2], True),
        ("explicit_rejects_the_implicit_scheme", exp[0], "scheme"),
        ("explicit_lists_forward_euler_and_central_difference",
         'Available options are: "forward_euler", "central_difference"'
         in exp[2], True),
        ("static_has_no_scheme_type_key_at_all", sta[0], "create"),
        ("static_rejection_is_flat_validation",
         "NOT in the default values" in sta[2], True),
        ("implicit_bossak_is_accepted", ok_imp[0], "ok"),
        ("explicit_forward_euler_is_accepted", ok_exp[0], "ok"),
    ]
    return _report(checks, [("implicit_message", imp[2][:90]),
                            ("explicit_message", exp[2][:90])],
                   "mpm_scheme_partition_check",
                   "the scheme_type-partitioning claim")


def _report(checks, extras, ok_line, what):
    mismatches = 0
    for label, got, must in checks:
        if got != must:
            mismatches += 1
        print("probe[%s]=%s_expected=%s" % (label, got, must))
    for k, v in extras:
        print("%s=%s" % (k, v))
    print("probe_mismatches=%d" % mismatches)
    if mismatches:
        print("FIXTURE_FAILED: %s does not hold on this build" % what,
              file=sys.stderr)
        return 1
    print("%s=ok" % ok_line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
