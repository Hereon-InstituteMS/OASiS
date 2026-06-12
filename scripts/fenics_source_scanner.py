"""Layer A — fenics dolfinx source scanner.

Introspects the installed dolfinx + ufl + basix modules and
compares against the catalog's claimed APIs in:

  src/backends/fenics/generators/*.py
  data/fenics_knowledge.py

For each catalog-claimed attribute path (e.g. 'ufl.FiniteElement',
'fem.petsc.LinearProblem'), this scanner reports whether the
attribute resolves at runtime in the current env.

Layer A in the catalog-validation framework: walk the source/
binary directly and emit a 'claimed-but-absent' diff. Tier-2
fixtures then lock specific findings as regression gates.

Invoke from the fenics conda env:

  conda activate ofa-fenicsx
  python scripts/fenics_source_scanner.py

Output is line-delimited 'attr=present|absent' suitable for
piping into a diff against catalog claims.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


# Catalog-claimed attributes, derived from grep of
# src/backends/fenics/generators/*.py + a few well-known
# dolfinx 0.10 surfaces that catalog text references.
CLAIMED_API: dict[str, str] = {
    # dolfinx.fem core (top-level attrs)
    "dolfinx.fem.functionspace": "function-space factory (lowercase)",
    "dolfinx.fem.FunctionSpace": "class (callable via factory in 0.10)",
    "dolfinx.fem.Function": "Function class",
    "dolfinx.fem.Constant": "Constant class",
    "dolfinx.fem.Expression": "Expression class",
    "dolfinx.fem.form": "form compiler entry point",
    "dolfinx.fem.dirichletbc": "Dirichlet BC factory",
    "dolfinx.fem.locate_dofs_geometrical": "DOF locator (geometric)",
    "dolfinx.fem.locate_dofs_topological": "DOF locator (topological)",
    "dolfinx.fem.assemble_matrix": "non-PETSc matrix assembly",
    "dolfinx.fem.assemble_vector": "non-PETSc vector assembly",
    # dolfinx.fem.petsc (submodule)
    "dolfinx.fem.petsc.LinearProblem": "PETSc linear problem",
    "dolfinx.fem.petsc.NonlinearProblem": "PETSc nonlinear problem",
    "dolfinx.fem.petsc.assemble_matrix": "PETSc matrix assembly",
    "dolfinx.fem.petsc.assemble_vector": "PETSc vector assembly",
    "dolfinx.fem.petsc.apply_lifting": "Lifting BC application",
    # dolfinx.nls.petsc
    "dolfinx.nls.petsc.NewtonSolver": "Newton solver wrapper",
    # dolfinx.mesh
    "dolfinx.mesh.create_box": "box mesh constructor",
    "dolfinx.mesh.create_rectangle": "2D rectangle",
    "dolfinx.mesh.create_unit_cube": "unit-cube convenience",
    "dolfinx.mesh.create_unit_square": "unit-square convenience",
    "dolfinx.mesh.CellType": "cell-type enum",
    "dolfinx.mesh.GhostMode": "ghost-mode enum",
    "dolfinx.mesh.locate_entities_boundary": "boundary entity locator",
    "dolfinx.mesh.exterior_facet_indices": "exterior-facet locator",
    "dolfinx.mesh.meshtags": "MeshTags factory",
    # dolfinx.io
    "dolfinx.io.XDMFFile": "XDMF I/O",
    "dolfinx.io.VTXWriter": "VTX I/O",
    "dolfinx.io.gmshio.model_to_mesh": "Gmsh model importer (submodule)",
    # ufl
    "ufl.TrialFunction": "trial function",
    "ufl.TestFunction": "test function",
    "ufl.dx": "volume integration measure",
    "ufl.ds": "exterior-facet measure",
    "ufl.dS": "interior-facet measure",
    "ufl.MixedFunctionSpace": "mixed function space helper",
    # ufl REMOVED in 2024+ — catalog uses must trigger
    "ufl.FiniteElement": "REMOVED — moved to basix.ufl.element",
    "ufl.VectorElement": "REMOVED — basix.ufl.element(shape=(d,))",
    "ufl.MixedElement": "REMOVED — basix.ufl.mixed_element",
    "ufl.TensorElement": "REMOVED — basix.ufl.element(shape=(d,d))",
    # basix.ufl
    "basix.ufl.element": "factory replacing ufl.FiniteElement",
    "basix.ufl.mixed_element": "factory replacing ufl.MixedElement",
    "basix.ufl.blocked_element": "blocked element factory",
}


def resolve_attr(dotted: str) -> bool:
    """Resolve 'pkg.sub.attr' by importing module then
    getattr-walking. Submodules are imported on demand."""
    parts = dotted.split(".")
    # Walk modules: try longest prefix as importable,
    # remaining as attrs.
    for cut in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:cut])
        try:
            mod = importlib.import_module(mod_name)
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
    results = {dotted: resolve_attr(dotted)
               for dotted in CLAIMED_API}
    for k in sorted(results):
        present = results[k]
        print(f"{k}={'present' if present else 'absent'}")

    # Versions for provenance
    versions = {}
    for pkg in ("dolfinx", "ufl", "basix"):
        try:
            m = importlib.import_module(pkg)
            versions[pkg] = getattr(m, "__version__", "?")
        except ImportError:
            versions[pkg] = "MISSING"
    print(f"versions={json.dumps(versions)}")

    absent_claimed = {k: CLAIMED_API[k]
                      for k, v in results.items() if not v}
    print(f"absent_count={len(absent_claimed)}")

    # Write structured output
    out_dir = Path(__file__).resolve().parent / "scan_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "fenics_source_scan.json"
    out_path.write_text(json.dumps({
        "versions": versions,
        "results": results,
        "absent_count": len(absent_claimed),
    }, indent=2))
    print(f"wrote={out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
