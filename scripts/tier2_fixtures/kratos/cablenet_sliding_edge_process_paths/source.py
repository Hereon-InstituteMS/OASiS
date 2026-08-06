"""Tier-2: every documented way into SlidingEdgeProcess, walked.

The claim (kratos.cable_net #13) says the process is broken through
every user path and that no input configuration runs end-to-end. Three
of its four statements reproduce; the fourth does not, and the fixture
pins what actually happens rather than what was predicted.

  python wrapper, partial input   -> RuntimeError, and NOT the one the
                                     claim names. The wrapper calls
                                     ``default_settings.ValidateAndAssign
                                     Defaults(settings)`` with receiver
                                     and argument the wrong way round, so
                                     the DEFAULTS are validated against
                                     the USER block and the first default
                                     key is reported as an unknown extra.
  python wrapper, complete input  -> NameError on `model_part_name`,
                                     exactly as claimed: the wrapper
                                     reads a name that was never bound.
  C++ binding, the 8 declared keys-> constructs, ExecuteInitialize
                                     succeeds, and it dies at the FIRST
                                     SOLUTION STEP looking up a key that
                                     is not in the defaults. The claim's
                                     mechanism is right; the key it leads
                                     with is not the first one hit.
  C++ binding, keys + the 4 that
  are read but undeclared         -> RUNS. It constructs, initialises,
                                     and builds master-slave constraints.
                                     The claim says no valid input exists
                                     and that adding the missing keys is
                                     rejected. Both are false here.

So the honest reading is narrower than the claim: the PYTHON path is
unusable, and the C++ path is usable only if the caller supplies keys
the defaults block never mentions.
"""
from __future__ import annotations

import os
import sys

sys.excepthook = sys.__excepthook__
os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA    # noqa: F401
import KratosMultiphysics.CableNetApplication as CN
import KratosMultiphysics.CableNetApplication.sliding_edge_process as SEP

# The eight keys the wrapper's own defaults block declares.
DECLARED = """{
  "constraint_name": "LinearMasterSlaveConstraint",
  "master_sub_model_part_name": "master_connect",
  "slave_sub_model_part_name": "slave_connect",
  "variable_names": ["DISPLACEMENT_Y", "DISPLACEMENT_Z"],
  "reform_every_step": true,
  "debug_info": false,
  "angled_initial_line": false,
  "follow_line": false }"""

# The same, plus the four keys the implementation reads but never
# declares.
DECLARED_PLUS_UNDECLARED = """{
  "constraint_name": "LinearMasterSlaveConstraint",
  "constraint_set_name": "LinearMasterSlaveConstraint",
  "master_sub_model_part_name": "master_connect",
  "slave_sub_model_part_name": "slave_connect",
  "variable_names": ["DISPLACEMENT_Y", "DISPLACEMENT_Z"],
  "reform_every_step": true,
  "debug_info": false,
  "angled_initial_line": false,
  "follow_line": false,
  "bucket_size": 10,
  "neighbor_search_radius": 5.0,
  "must_find_neighbor": true }"""

# What the wrapper's defaults block contains, verbatim — the wrapper
# only gets past its inverted validation call if the user hands it
# every one of these.
WRAPPER_COMPLETE = """{ "Parameters": {
  "constraint_name": "LinearMasterSlaveConstraint",
  "master_sub_model_part_name": "master_connect",
  "slave_sub_model_part_name": "slave_connect",
  "computing_model_part": "Structure",
  "variable_names": ["DISPLACEMENT_Y", "DISPLACEMENT_Z"],
  "reform_every_step": true,
  "debug_info": true,
  "angled_initial_line": false,
  "follow_line": false } }"""


def structure():
    """A model part with a three-node master rail and one slave node."""
    model = KM.Model()
    mp = model.CreateModelPart("Structure", 2)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
    for var in (KM.DISPLACEMENT, KM.REACTION):
        mp.AddNodalSolutionStepVariable(var)
    master = mp.CreateSubModelPart("master_connect")
    slave = mp.CreateSubModelPart("slave_connect")
    for i, (x, y, z) in enumerate([(0, 0, 0), (1, 0, 0), (2, 0, 0)], 1):
        mp.CreateNewNode(i, float(x), float(y), float(z))
    mp.CreateNewNode(10, 0.5, 0.1, 0.0)
    for node in mp.Nodes:
        node.AddDof(KM.DISPLACEMENT_X, KM.REACTION_X)
        node.AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y)
        node.AddDof(KM.DISPLACEMENT_Z, KM.REACTION_Z)
    master.AddNodes([1, 2, 3])
    slave.AddNodes([10])
    return model, mp


def attempt(fn):
    """(exception type name or '', one-line message)."""
    try:
        fn()
    except Exception as exc:                                  # noqa: BLE001
        return type(exc).__name__, " ".join(str(exc).split())
    return "", ""


def main() -> int:
    fail: list[str] = []

    # ── the python wrapper, partial input ──────────────────────────
    def wrapper_partial():
        model, _mp = structure()
        SEP.Factory(KM.Parameters('{"Parameters": {}}'), model)

    kind, msg = attempt(wrapper_partial)
    inverted = (kind == "RuntimeError"
                and "NOT in the default values" in msg)
    print(f"wrapper_partial_input_error={kind}")
    print(f"wrapper_validates_defaults_against_user_input={inverted}")
    print(f"wrapper_partial_message={msg[:140]}")
    if not inverted:
        fail.append(f"a partial parameter block did not trip the "
                    f"wrapper's inverted ValidateAndAssignDefaults; got "
                    f"{kind or 'no exception'}: {msg[:120]}")

    # ── the python wrapper, the complete nine-key block ────────────
    def wrapper_complete():
        model, _mp = structure()
        SEP.Factory(KM.Parameters(WRAPPER_COMPLETE), model)

    kind, msg = attempt(wrapper_complete)
    name_error = (kind == "NameError" and "model_part_name" in msg)
    print(f"wrapper_complete_input_error={kind}")
    print(f"wrapper_reads_an_unbound_name={name_error}")
    print(f"wrapper_complete_message={msg[:140]}")
    if not name_error:
        fail.append(f"the wrapper accepted its own complete defaults "
                    f"block without raising NameError on model_part_name; "
                    f"got {kind or 'no exception'}: {msg[:120]}")

    # ── the C++ binding with only the declared keys ────────────────
    model, mp = structure()
    kind, msg = attempt(lambda: CN.SlidingEdgeProcess(mp,
                                                      KM.Parameters(DECLARED)))
    print(f"cpp_declared_keys_construct_error={kind or 'NONE'}")
    constructed = (kind == "")
    if not constructed:
        fail.append(f"the C++ binding refused the eight declared keys at "
                    f"construction: {kind}: {msg[:120]}. The claim and "
                    f"this fixture both say construction succeeds and the "
                    f"failure is deferred.")
        proc = None
    else:
        proc = CN.SlidingEdgeProcess(mp, KM.Parameters(DECLARED))
    print(f"cpp_declared_keys_constructed={constructed}")

    deferred = False
    if proc is not None:
        kind, msg = attempt(proc.ExecuteInitialize)
        print(f"cpp_declared_keys_initialize_error={kind or 'NONE'}")
        kind, msg = attempt(proc.ExecuteInitializeSolutionStep)
        deferred = (kind == "RuntimeError"
                    and "Getting a value that does not exist" in msg)
        print(f"cpp_declared_keys_solution_step_error={kind or 'NONE'}")
        print(f"cpp_declared_keys_fail_on_undeclared_lookup={deferred}")
        print(f"cpp_declared_keys_message={msg[:140]}")
    if not deferred:
        fail.append("the eight declared keys did not fail at the first "
                    "solution step on a key that is read but not "
                    "declared; that lookup is the whole schema/code "
                    "mismatch this claim is about")

    # ── the C++ binding with the undeclared keys supplied ──────────
    model, mp = structure()
    runs, constraints, kind, msg = False, -1, "", ""
    try:
        proc = CN.SlidingEdgeProcess(mp,
                                     KM.Parameters(DECLARED_PLUS_UNDECLARED))
        proc.ExecuteInitialize()
        proc.ExecuteInitializeSolutionStep()
        constraints = mp.NumberOfMasterSlaveConstraints()
        proc.ExecuteFinalizeSolutionStep()
        runs = True
    except Exception as exc:                                  # noqa: BLE001
        kind, msg = type(exc).__name__, " ".join(str(exc).split())
    print(f"cpp_with_undeclared_keys_runs={runs}")
    print(f"cpp_with_undeclared_keys_constraints_built={constraints}")
    print(f"cpp_with_undeclared_keys_error={kind or 'NONE'}")
    if not (runs and constraints > 0):
        fail.append(f"supplying the four read-but-undeclared keys did NOT "
                    f"produce a working process ({kind}: {msg[:110]}); "
                    f"this fixture records that it does, against the "
                    f"claim's 'there is NO valid input configuration'")

    # The claim's two halves, stated as one line each so a mutation
    # that repairs either flips exactly one.
    print(f"python_wrapper_is_unusable_by_both_paths="
          f"{inverted and name_error}")
    print(f"claim_no_valid_input_configuration_is_false={runs}")

    if not fail:
        print("cablenet_sliding_edge_process_paths_verified=True")
        return 0
    for reason in fail:
        print(f"FAIL: {reason}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
