"""Tier-2: KratosFSIApplication ships its OWN convergence
accelerator classes — distinct from the CoSim file stems.

Catalog under audit (src/backends/kratos/generators/fsi.py
field fsi_application_accelerators, added 2026-06-01):

    AitkenConvergenceAccelerator
    ConstantRelaxationConvergenceAccelerator
    IBQNMVQNConvergenceAccelerator
    IBQNMVQNRandomizedSVDConvergenceAccelerator
    MVQNFullJacobianConvergenceAccelerator
    MVQNRandomizedSVDConvergenceAccelerator
    MVQNRecursiveJacobianConvergenceAccelerator

These are Python attributes on
KratosMultiphysics.FSIApplication — NOT the CoSim
file-stem names (block_ibqnls, iqnils, etc.). The prior
catalog conflated the two surfaces and used "ibqn" which
is neither.

Also locks the install-gap diagnostic: importing
KratosMultiphysics.FSIApplication on a fresh .venv raises
ModuleNotFoundError unless pip-installed.
"""
from __future__ import annotations

import importlib
import sys


REQUIRED_ACCEL_CLASSES = {
    "AitkenConvergenceAccelerator",
    "ConstantRelaxationConvergenceAccelerator",
    "IBQNMVQNConvergenceAccelerator",
    "IBQNMVQNRandomizedSVDConvergenceAccelerator",
    "MVQNFullJacobianConvergenceAccelerator",
    "MVQNRandomizedSVDConvergenceAccelerator",
    "MVQNRecursiveJacobianConvergenceAccelerator",
}
REQUIRED_PARTITIONED_UTILS = {
    "FSIUtils",
    "SharedPointsMapper",
    "PartitionedFSIUtilitiesArray2D",
    "PartitionedFSIUtilitiesArray3D",
    "PartitionedFSIUtilitiesDouble2D",
    "PartitionedFSIUtilitiesDouble3D",
}


def main() -> int:
    try:
        fsi = importlib.import_module(
            "KratosMultiphysics.FSIApplication")
    except ImportError as exc:
        print(f"install_gap_detected={exc!r}")
        # Fixture FAILS if the install gap fires — campaign
        # already pip-installed KratosFSIApplication; if a
        # future env regresses we want to see it loudly.
        return 2

    attrs = set(dir(fsi))
    accel_present = REQUIRED_ACCEL_CLASSES & attrs
    accel_missing = REQUIRED_ACCEL_CLASSES - attrs
    util_present = REQUIRED_PARTITIONED_UTILS & attrs
    util_missing = REQUIRED_PARTITIONED_UTILS - attrs
    print(f"accel_classes_present={sorted(accel_present)}")
    print(f"accel_classes_missing={sorted(accel_missing)}")
    print(f"partitioned_utils_present={sorted(util_present)}")
    print(f"partitioned_utils_missing={sorted(util_missing)}")

    # Sanity: 'ibqn' (bare) is NOT a class — only the
    # FULL name 'IBQNMVQNConvergenceAccelerator' is.
    bare_ibqn_present = "ibqn" in attrs
    bare_ibqn_lower_present = any(
        a.lower() == "ibqn" for a in attrs)
    print(f"bare_ibqn_attr_present={bare_ibqn_present}")
    print(
        f"any_lowercase_ibqn_attr_present="
        f"{bare_ibqn_lower_present}")

    # Cross-surface check: CoSim file stems must NOT
    # appear as FSI Application attrs (proves they really
    # are different surfaces, not the same surface under
    # different names).
    cosim_stems = {
        "block_ibqnls", "iqnils", "mvqn", "block_mvqn",
        "aitken", "anderson", "constant_relaxation",
    }
    cosim_stem_in_fsi = cosim_stems & attrs
    print(f"cosim_file_stems_in_fsi_attrs="
          f"{sorted(cosim_stem_in_fsi)}")

    ok = (
        not accel_missing
        and not util_missing
        and not bare_ibqn_present
        and not bare_ibqn_lower_present
        and not cosim_stem_in_fsi
    )
    if ok:
        return 0
    print("FAIL: FSI Application accelerator-class "
          "catalog invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
