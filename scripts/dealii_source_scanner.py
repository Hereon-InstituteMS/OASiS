"""Layer A — deal.II header scanner.

deal.II is C++ so the scanner walks the installed headers
under $CONDA_PREFIX/include/deal.II (or system fallback)
rather than introspecting at runtime.

For each catalog-claimed C++ symbol the scanner reports:
  (a) whether the class/function is declared in any
      shipped header — name-presence check.
  (b) whether the declaring header REQUIRES an external
      compile-time feature (PETSc, Trilinos, SLEPc, p4est,
      MPI) that the current install has switched OFF.

  → catalog claims with (a)=present AND (b)=requires-OFF-
    feature are 'declared-but-link-unfeasible' — runs of
    catalog-generated code will fail at link/runtime.

Output: tabular report + JSON dump under
scripts/scan_results/dealii_source_scan.json.
"""
from __future__ import annotations

import os
import re
import json
import sys
from pathlib import Path
from collections import defaultdict


# Catalog C++ symbols extracted via grep of
# src/backends/dealii/generators/*.py + data/dealii_*.py.
CATALOG_SYMBOLS = (
    # Solvers
    "SolverCG", "SolverGMRES", "SolverFGMRES",
    "SolverBicgstab", "SolverMinRes", "SolverQMRS",
    "SolverRichardson",
    # Preconditioners
    "PreconditionJacobi", "PreconditionSSOR",
    "PreconditionILU", "PreconditionIdentity",
    "PreconditionChebyshev",
    "PreconditionAMG", "PreconditionBoomerAMG",
    "PreconditionICC",
    # Finite elements
    "FE_Q", "FE_DGQ", "FE_Nedelec", "FE_RaviartThomas",
    "FE_BDM", "FE_Q_DG0",
    # Mesh + grid utils
    "GridGenerator", "GridTools", "GridIn", "GridOut",
    # DoF/finite element
    "DoFTools", "DoFHandler", "DoFRenumbering",
    "AffineConstraints",
    # Quadrature/Mapping
    "MappingQ", "QGauss", "QGaussLobatto",
    # Numerics
    "VectorTools", "MatrixCreator", "DataOut",
    "KellyErrorEstimator", "SolutionTransfer",
    # Linear algebra
    "SparseDirectUMFPACK", "FullMatrix", "SparseMatrix",
    "Vector", "BlockVector", "BlockSparseMatrix",
    # Parallelism (external libs)
    "TrilinosWrappers", "PETScWrappers", "SLEPcWrappers",
    # Multigrid + matrix-free
    "MGTransferMatrixFree", "MGTools", "MatrixFree",
    "MeshWorker",
    # Parallel distributed
    "parallel::distributed::Triangulation",
)
# False-name catalog typos to assert STAY absent
PHANTOM_SYMBOLS = (
    "VectorTools::interpolate_difference",  # typo for integrate_difference
)
# Feature-flag dependencies (header path → required flag)
HEADER_TO_FLAG = {
    "lac/trilinos_": "DEAL_II_WITH_TRILINOS",
    "lac/petsc_": "DEAL_II_WITH_PETSC",
    "lac/slepc_": "DEAL_II_WITH_SLEPC",
    "distributed/tria": "DEAL_II_WITH_P4EST",
}


def find_include_dir() -> Path | None:
    cp = os.environ.get("CONDA_PREFIX")
    if cp:
        p = Path(cp) / "include" / "deal.II"
        if p.is_dir():
            return p
    for cand in (
        Path("/home/hermann/miniconda3/envs/ofa-dealii"
             "/include/deal.II"),
        Path("/usr/include/deal.II"),
        Path("/usr/local/include/deal.II"),
    ):
        if cand.is_dir():
            return cand
    return None


def parse_feature_flags(config_h: Path) -> dict[str, bool]:
    """Return {flag: True if ON, False if OFF}."""
    out = {}
    text = config_h.read_text()
    for flag in (
        "DEAL_II_WITH_MPI", "DEAL_II_WITH_P4EST",
        "DEAL_II_WITH_PETSC", "DEAL_II_WITH_TRILINOS",
        "DEAL_II_WITH_SLEPC", "DEAL_II_WITH_LAPACK",
    ):
        undef = re.search(
            r"/\*\s*#undef\s+" + re.escape(flag) + r"\s*\*/",
            text)
        define = re.search(
            r"#define\s+" + re.escape(flag) + r"(?:\s|$)",
            text)
        if undef:
            out[flag] = False
        elif define:
            out[flag] = True
    return out


def required_flag_for(header: Path) -> str | None:
    rel = str(header).split("include/deal.II/")[-1]
    for prefix, flag in HEADER_TO_FLAG.items():
        if rel.startswith(prefix):
            return flag
    return None


def find_symbol(include_dir: Path, sym: str
                ) -> list[tuple[Path, str | None]]:
    """Return [(header, required_flag)] for every header
    declaring this symbol."""
    # Symbol may include :: for nested namespaces — split
    # and search for the last component as 'class X' or
    # 'namespace X' or '::X('.
    last = sym.split("::")[-1]
    pattern = re.compile(
        r"(class|struct|namespace)\s+" + re.escape(last)
        + r"\b")
    hits = []
    for h in include_dir.rglob("*.h"):
        try:
            text = h.read_text()
        except Exception:
            continue
        if pattern.search(text):
            hits.append((h, required_flag_for(h)))
    return hits


def main() -> int:
    inc = find_include_dir()
    if inc is None:
        print("FAIL: no deal.II include dir found",
              file=sys.stderr)
        return 2
    print(f"include_dir={inc}")

    config_h = inc / "base" / "config.h"
    flags = parse_feature_flags(config_h)
    print(f"feature_flags={flags}")

    catalog_present_but_unfeasible = []
    catalog_declared_in = {}
    catalog_missing = []
    for sym in CATALOG_SYMBOLS:
        hits = find_symbol(inc, sym)
        if not hits:
            catalog_missing.append(sym)
            continue
        feasible_hits = [
            (h, fl) for (h, fl) in hits
            if fl is None or flags.get(fl, False)]
        unfeasible_hits = [
            (h, fl) for (h, fl) in hits
            if fl is not None and not flags.get(fl, False)]
        catalog_declared_in[sym] = {
            "feasible_in_install": [str(h.relative_to(inc))
                                    for h, _ in feasible_hits],
            "unfeasible_in_install": [
                {"header": str(h.relative_to(inc)),
                 "needs_flag": fl}
                for h, fl in unfeasible_hits],
        }
        if not feasible_hits and unfeasible_hits:
            catalog_present_but_unfeasible.append(sym)
    print(f"catalog_missing={catalog_missing}")
    print(f"catalog_unfeasible_in_install="
          f"{catalog_present_but_unfeasible}")

    # Phantom symbols MUST stay absent
    phantom_status = {}
    for ps in PHANTOM_SYMBOLS:
        # Phantom is a function not a class — grep for
        # the function name in numerics/vector_tools*.h
        fn_name = ps.split("::")[-1]
        found = False
        for h in (inc / "numerics").glob("vector_tools*.h"):
            if fn_name in h.read_text():
                found = True
                break
        phantom_status[ps] = found
    print(f"phantom_symbols_present={phantom_status}")

    out_dir = Path(__file__).resolve().parent / "scan_results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "dealii_source_scan.json"
    out.write_text(json.dumps({
        "include_dir": str(inc),
        "feature_flags": flags,
        "catalog_missing": catalog_missing,
        "catalog_unfeasible_in_install":
            catalog_present_but_unfeasible,
        "catalog_declared_in": catalog_declared_in,
        "phantom_symbols_present": phantom_status,
    }, indent=2))
    print(f"wrote={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
