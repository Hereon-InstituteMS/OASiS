"""Layer A — NGSolve source scanner.

Introspects the installed ngsolve + netgen Python packages
and compares against catalog claims in
src/backends/ngsolve/generators/*.py.

Reports:
  * Top-level attribute presence (H1, HCurl, FESpaces, ...)
  * Submodule API presence (solvers.Newton, krylovspace.*,
    fem.NewtonCF, ...)
  * pybind11 callable signatures (pml.Radial, solvers.Newton)
  * Star-import-only names (pml is exported by 'from
    ngsolve import *' but NOT as an attribute under
    importlib.import_module('ngsolve.pml'))
"""
from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path


CATALOG_TOPLEVEL = (
    "H1", "HCurl", "HDiv", "L2", "VectorH1", "VectorL2",
    "HCurlDiv", "HDivDiv", "HCurlCurl",
    "FacetFESpace", "TangentialFacetFESpace",
    "NormalFacetFESpace", "NumberSpace", "NodalFESpace",
    "SurfaceL2", "Compress", "CompressCompound", "Periodic",
    "FESpace", "BilinearForm", "LinearForm", "GridFunction",
    "CoefficientFunction", "Mesh", "Draw", "Integrate",
    "InnerProduct", "Grad", "grad", "div", "curl",
    "IfPos", "Trace", "specialcf", "x", "y", "z",
    "TaskManager", "SetNumThreads", "CGSolver",
)
CATALOG_PHANTOM_TOPLEVEL = (
    # claimed-or-implied but actually NOT at top level
    "NewtonCF",       # in ngsolve.fem
    "MinimizationCF", # in ngsolve.fem
    "GMResSolver",    # in ngsolve.krylovspace
    "MinResSolver",   # in ngsolve.krylovspace
    "Curl",           # only ngs.curl (lowercase)
    "Div",            # only ngs.div  (lowercase)
)
CATALOG_SUBMODULE_APIS = (
    "ngsolve.fem.NewtonCF",
    "ngsolve.fem.MinimizationCF",
    "ngsolve.fem.IfPos",
    "ngsolve.solvers.Newton",
    "ngsolve.solvers.CG",
    "ngsolve.solvers.BVP",
    "ngsolve.solvers.PreconditionedRichardson",
    "ngsolve.krylovspace.CGSolver",
    "ngsolve.krylovspace.GMResSolver",
    "ngsolve.krylovspace.MinResSolver",
    "ngsolve.krylovspace.BramblePasciakCG",
)


def resolve(dotted: str) -> bool:
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            mod = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        obj = mod
        ok = True
        for attr in parts[cut:]:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok:
            return True
    return False


def main() -> int:
    import ngsolve as ngs
    print(f"ngsolve_version={ngs.__version__}")

    top = {n: hasattr(ngs, n) for n in CATALOG_TOPLEVEL}
    phantom = {n: hasattr(ngs, n)
               for n in CATALOG_PHANTOM_TOPLEVEL}
    sub = {n: resolve(n) for n in CATALOG_SUBMODULE_APIS}
    print(f"toplevel_missing="
          f"{sorted([k for k, v in top.items() if not v])}")
    print(f"phantom_at_toplevel_present="
          f"{sorted([k for k, v in phantom.items() if v])}")
    print(f"submodule_missing="
          f"{sorted([k for k, v in sub.items() if not v])}")

    # pml.Radial signature — pybind11 builtin, must check via
    # docstring scraping.
    from ngsolve import pml
    radial_doc = pml.Radial.__doc__ or ""
    radial_requires_origin = "origin" in radial_doc
    print(f"pml_radial_doc_has_origin="
          f"{radial_requires_origin}")
    # Empirical TypeError sentinel
    radial_typeerror_without_origin = False
    try:
        pml.Radial(rad=0.7, alpha=2j)
    except TypeError as e:
        radial_typeerror_without_origin = (
            "incompatible function arguments" in str(e)
            or "origin" in str(e))
    print(f"pml_radial_typeerror_without_origin="
          f"{radial_typeerror_without_origin}")
    # Right call works
    radial_with_origin_ok = False
    try:
        pml.Radial(origin=(0, 0), rad=0.7, alpha=2j)
        radial_with_origin_ok = True
    except Exception:
        pass
    print(f"pml_radial_with_origin_ok="
          f"{radial_with_origin_ok}")

    # Newton kwargs
    import ngsolve.solvers as solv
    sig = inspect.signature(solv.Newton)
    print(f"newton_params={list(sig.parameters)}")

    out_dir = Path(__file__).resolve().parent / "scan_results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "ngsolve_source_scan.json"
    out.write_text(json.dumps({
        "version": ngs.__version__,
        "toplevel": top,
        "phantom_at_toplevel_present": {
            k: v for k, v in phantom.items() if v},
        "submodule": sub,
        "pml_radial_requires_origin": radial_requires_origin,
        "newton_params": list(sig.parameters),
    }, indent=2))
    print(f"wrote={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
