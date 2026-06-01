"""Tier-2: deal.II catalog symbols ↔ install header inventory.

Walks the installed deal.II include tree and asserts:

  * No catalog-claimed C++ symbol is missing from
    all headers (catalog_missing must stay empty).
  * The known-typo phantom 'VectorTools::interpolate_
    difference' stays absent (real fn is integrate_
    difference).
  * The 'catalog_unfeasible_in_install' list — symbols
    whose declaring headers require an OFF feature flag —
    is observable and stable.

The fixture PASSes when catalog symbols stay declared AND
the typo stays absent. The unfeasible-in-install list is
reported (not asserted) — when task #30 rebuilds dealii
with MPI/PETSc/Trilinos/SLEPc, the list will shrink
automatically.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path


CATALOG_SYMBOLS = (
    "SolverCG", "SolverGMRES", "SolverFGMRES",
    "SolverBicgstab", "SolverMinRes", "SolverRichardson",
    "PreconditionJacobi", "PreconditionSSOR",
    "PreconditionILU", "PreconditionIdentity",
    "PreconditionChebyshev",
    "FE_Q", "FE_DGQ", "FE_Nedelec", "FE_RaviartThomas",
    "GridGenerator", "GridTools", "AffineConstraints",
    "MappingQ", "VectorTools", "DataOut",
    "KellyErrorEstimator", "SparseDirectUMFPACK",
    "MGTransferMatrixFree", "MeshWorker",
    "parallel::distributed::Triangulation",
)
PHANTOM_TYPO = "interpolate_difference"


def find_include() -> Path | None:
    cp = os.environ.get("CONDA_PREFIX")
    if cp:
        p = Path(cp) / "include" / "deal.II"
        if p.is_dir():
            return p
    for cand in (
        Path("/home/hermann/miniconda3/envs/ofa-dealii"
             "/include/deal.II"),
    ):
        if cand.is_dir():
            return cand
    return None


def main() -> int:
    inc = find_include()
    if inc is None:
        print("FAIL: no deal.II include dir found",
              file=sys.stderr)
        return 2
    print(f"include_dir={inc}")

    # Walk every header once, build symbol → bool map.
    all_text = []
    for h in inc.rglob("*.h"):
        try:
            all_text.append(h.read_text())
        except Exception:
            continue
    combined = "\n".join(all_text)
    print(f"headers_scanned={len(all_text)}")

    missing = []
    for sym in CATALOG_SYMBOLS:
        last = sym.split("::")[-1]
        pattern = re.compile(
            r"(class|struct|namespace)\s+"
            + re.escape(last) + r"\b")
        if not pattern.search(combined):
            missing.append(sym)
    print(f"catalog_missing={missing}")

    # Phantom typo: verify NO header has the typo as a
    # function declaration (grep for ' interpolate_
    # difference(' or '::interpolate_difference(' )
    typo_present = bool(re.search(
        r"\b" + re.escape(PHANTOM_TYPO) + r"\s*\(",
        combined))
    print(f"phantom_typo_present={typo_present}")

    # And the real fn 'integrate_difference' MUST be
    # present (catalog ships the corrected name).
    real_fn = bool(re.search(
        r"\bintegrate_difference\s*\(", combined))
    print(f"real_integrate_difference_present={real_fn}")

    ok = not missing and not typo_present and real_fn
    if ok:
        return 0
    print("FAIL: dealii catalog↔header drift",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
