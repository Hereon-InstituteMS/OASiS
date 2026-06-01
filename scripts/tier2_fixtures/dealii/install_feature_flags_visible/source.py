"""Tier-2: deal.II install feature-flag detection.

The catalog (data/dealii_knowledge.py + multiple
generators/*.py) makes parallelism claims that depend on
compile-time features:

  DEAL_II_WITH_MPI       — parallel::distributed::Triangulation
  DEAL_II_WITH_P4EST     — distributed mesh refinement
  DEAL_II_WITH_PETSC     — PETScWrappers::*, PETSc MUMPS solver
  DEAL_II_WITH_TRILINOS  — TrilinosWrappers::*, Amesos solvers
  DEAL_II_WITH_SLEPC     — Eigenvalue solvers via SLEPc

The conda-forge dealii-9.1.1 default install has ALL of
the above UNDEFINED (verified 2026-06-01). This fixture:

  * Locates the install's config.h (via $CONDA_PREFIX or
    the known miniconda env path).
  * Parses the '/* #undef DEAL_II_WITH_X */' lines.
  * Reports which features are ON vs OFF.

Pass condition (deliberately tolerant): the fixture
PASSES if it can detect a deal.II install at all and emit
a feature-flag report. It FAILS if it cannot find any
config.h (install completely missing).

The point is to expose the install state in the fixture
runner output so the MCP catalog ↔ install alignment
discrepancy stays visible, and to lock in the install-
detection (task #30 deal.II rebuild will, when complete,
flip the feature flags from OFF to ON and the same
fixture will record that).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path


CONFIG_CANDIDATES = [
    Path.home() / "miniconda3" / "envs" / "ofa-dealii"
    / "include" / "deal.II" / "base" / "config.h",
    Path("/usr/include/deal.II/base/config.h"),
    Path("/usr/local/include/deal.II/base/config.h"),
]

FEATURES_OF_INTEREST = [
    "DEAL_II_WITH_MPI",
    "DEAL_II_WITH_P4EST",
    "DEAL_II_WITH_PETSC",
    "DEAL_II_WITH_TRILINOS",
    "DEAL_II_WITH_SLEPC",
    "DEAL_II_WITH_LAPACK",
    "DEAL_II_WITH_HDF5",
    "DEAL_II_WITH_METIS",
    "DEAL_II_WITH_MUPARSER",
]


def find_config() -> Path | None:
    cp = os.environ.get("CONDA_PREFIX")
    if cp:
        p = Path(cp) / "include" / "deal.II" / "base" / "config.h"
        if p.is_file():
            return p
    for p in CONFIG_CANDIDATES:
        if p.is_file():
            return p
    return None


def main() -> int:
    cfg = find_config()
    if cfg is None:
        print("FAIL: no deal.II config.h found at any "
              "known path", file=sys.stderr)
        return 2
    print(f"config_h_path={cfg}")

    text = cfg.read_text()

    # Version
    version_match = re.search(
        r"DEAL_II_VERSION_MAJOR\s+(\d+).*?"
        r"DEAL_II_VERSION_MINOR\s+(\d+).*?"
        r"DEAL_II_VERSION_SUBMINOR\s+(\d+)",
        text, re.DOTALL)
    if version_match:
        major, minor, sub = version_match.groups()
        print(f"dealii_version={major}.{minor}.{sub}")
    else:
        print("dealii_version=UNKNOWN")

    # Parse features
    on = []
    off = []
    for feat in FEATURES_OF_INTEREST:
        undef_pattern = (r"/\*\s*#undef\s+" +
                         re.escape(feat) + r"\s*\*/")
        define_pattern = (r"#define\s+" + re.escape(feat))
        if re.search(undef_pattern, text):
            off.append(feat)
        elif re.search(define_pattern, text):
            on.append(feat)
        else:
            print(f"feat_unknown={feat}")
    print(f"features_on={sorted(on)}")
    print(f"features_off={sorted(off)}")

    # Diagnostic: list of broken catalog claims for THIS install.
    catalog_broken = []
    if "DEAL_II_WITH_P4EST" in off:
        catalog_broken.append(
            "p4est-distributed AMR claim (data/dealii_"
            "knowledge.py adaptive_refinement.parallel)")
    if "DEAL_II_WITH_MPI" in off:
        catalog_broken.append(
            "MPI-parallel Poisson generator (parallel.py)")
    if "DEAL_II_WITH_PETSC" in off:
        catalog_broken.append(
            "PETSc MUMPS / PETScWrappers solver claims")
    if "DEAL_II_WITH_TRILINOS" in off:
        catalog_broken.append(
            "Trilinos Amesos solver claims")
    if "DEAL_II_WITH_SLEPC" in off:
        catalog_broken.append(
            "SLEPc eigenvalue solver claims")
    print(f"catalog_claims_install_doesnt_support="
          f"{catalog_broken}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
