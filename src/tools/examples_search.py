"""
MCP tools for searching and retrieving example input files across all backends.

For 4C: Searches ~2,872 test input files + tutorials
For FEniCS: Returns parametrized Python script templates
For deal.II: Returns C++ source templates based on tutorial steps
For FEBio: Returns XML templates
"""

import json
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from core.backend import detect_template_language
from core.registry import get_backend, available_backends

# Auto-detect 4C paths
FOURC_ROOT = Path(os.environ.get("FOURC_ROOT", ""))
FOURC_TESTS = FOURC_ROOT / "tests" if FOURC_ROOT else None
FOURC_TUTORIALS = FOURC_TESTS / "tutorials" if FOURC_TESTS else None


def _search_4c_input_files(keyword: str, max_results: int = 5) -> list[dict]:
    """Search 4C test input files by keyword in filename."""
    if not FOURC_TESTS or not FOURC_TESTS.is_dir():
        return []

    results = []
    keyword_lower = keyword.lower()
    for yaml_file in FOURC_TESTS.rglob("*.4C.yaml"):
        if keyword_lower in yaml_file.name.lower() or keyword_lower in str(yaml_file.parent.name).lower():
            rel = yaml_file.relative_to(FOURC_TESTS)
            results.append({
                "file": str(rel),
                "name": yaml_file.stem.replace(".4C", ""),
                "dir": yaml_file.parent.name,
                "size": yaml_file.stat().st_size,
            })
            if len(results) >= max_results:
                break
    return results


def _get_4c_tutorial_files() -> list[dict]:
    """List all 4C tutorial input files."""
    if not FOURC_TUTORIALS or not FOURC_TUTORIALS.is_dir():
        return []

    results = []
    for tut_dir in sorted(FOURC_TUTORIALS.iterdir()):
        if not tut_dir.is_dir():
            continue
        for yaml_file in tut_dir.glob("*.4C.yaml"):
            results.append({
                "tutorial": tut_dir.name,
                "file": yaml_file.name,
                "path": str(yaml_file),
            })
    return results


def register_example_tools(mcp: FastMCP):

    @mcp.tool()
    def get_example_inputs(module: str, solver: str = "fourc", max_results: int = 3) -> str:
        """Get working example input files for a given physics module.

        Returns the FULL content of real, tested input files from the solver's
        own test suite. These are the best reference for correct setup.

        IMPORTANT: Always call this before generating a new input file so you
        have a correct, validated template to work from.

        Args:
            module: Physics keyword (e.g. 'peridynamic', 'fsi', 'beam', 'contact',
                    'poisson', 'heat', 'fluid', 'elasticity')
            solver: Backend name (default: 'fourc')
            max_results: Maximum number of example files to return (default 3)
        """
        results = []

        # 4C: return full content of real test files
        if solver.lower() in ("fourc", "4c"):
            if FOURC_TESTS and FOURC_TESTS.is_dir():
                matches = []
                keyword_lower = module.lower()
                for f in sorted(FOURC_TESTS.rglob("*.4C.yaml")):
                    if keyword_lower in f.name.lower():
                        matches.append(f)
                        if len(matches) >= max_results:
                            break

                for f in matches:
                    try:
                        content = f.read_text()
                        rel_path = f.relative_to(FOURC_TESTS.parent)
                        results.append(
                            f"### Real test file: `{rel_path}`\n\n```yaml\n{content[:20000]}```\n"
                        )
                    except Exception:
                        pass

        # deal.II: return step tutorial content
        elif solver.lower() in ("dealii", "deal.ii"):
            dealii_dir = Path("/usr/share/doc/libdeal.ii-doc/examples")
            if dealii_dir.is_dir():
                keyword_lower = module.lower()
                for step_dir in sorted(dealii_dir.iterdir()):
                    if keyword_lower in step_dir.name.lower():
                        cc_file = step_dir / f"{step_dir.name}.cc"
                        if cc_file.is_file():
                            content = cc_file.read_text()
                            results.append(
                                f"### deal.II `{step_dir.name}`\n\n```cpp\n{content[:8000]}```\n"
                            )
                            if len(results) >= max_results:
                                break

        # FEniCS: return demo scripts from conda env
        elif solver.lower() in ("fenics", "fenicsx"):
            fenics_demo = Path.home() / "miniconda3" / "envs" / "fenics" / "share" / "dolfinx" / "demo"
            if not fenics_demo.is_dir():
                for p in Path.home().glob("miniconda3/envs/fenics/**/demo"):
                    if p.is_dir():
                        fenics_demo = p
                        break
            if fenics_demo.is_dir():
                keyword_lower = module.lower()
                for f in sorted(fenics_demo.rglob("*.py")):
                    if keyword_lower in f.name.lower() or keyword_lower in f.parent.name.lower():
                        content = f.read_text()
                        results.append(
                            f"### FEniCS demo: `{f.name}`\n\n```python\n{content[:8000]}```\n"
                        )
                        if len(results) >= max_results:
                            break

        # NGSolve, scikit-fem, Kratos, DUNE: generate from our templates as reference
        else:
            backend = get_backend(solver)
            if backend:
                keyword_lower = module.lower()
                # Normalize spaces/underscores/hyphens for fuzzy matching
                keyword_normalized = keyword_lower.replace(" ", "_").replace("-", "_")
                # Split into individual words for OR matching
                # "eigenvalue maxwell" matches physics containing "eigenvalue" OR "maxwell"
                keywords = [w.replace("-", "_") for w in keyword_lower.split()]
                for p in backend.supported_physics():
                    name_norm = p.name.lower().replace("-", "_")
                    desc_norm = p.description.lower()
                    match = (keyword_lower in name_norm or keyword_normalized in name_norm
                             or keyword_lower in desc_norm or keyword_normalized in desc_norm)
                    if not match and len(keywords) > 1:
                        # OR matching: any individual keyword matches
                        match = any(kw in name_norm or kw in desc_norm for kw in keywords)
                    if match:
                        for v in p.template_variants[:max_results]:
                            try:
                                content = backend.generate_input(p.name, v, {})
                                fmt = detect_template_language(content, backend.input_format().value)
                                results.append(
                                    f"### {backend.display_name()} template: `{p.name}/{v}`\n\n```{fmt}\n{content[:8000]}```\n"
                                )
                            except Exception:
                                pass
                        break

        if not results:
            return (
                f"No example files found for '{module}' in {solver}. "
                f"Try different keywords or use generate_input() for a template."
            )

        header = f"## {len(results)} example(s) for '{module}' from {solver}\n\n"
        return header + "\n---\n".join(results)

    @mcp.tool()
    def search_examples(keyword: str, solver: str = "", max_results: int = 5) -> str:
        """Search for example input files across all backends.

        For 4C: Searches ~2,872 test files + tutorials by keyword
        For FEniCS/deal.II/FEBio: Lists available template variants

        Args:
            keyword: Search term (e.g. 'poisson', 'fsi', 'beam', 'contact')
            solver: Limit to specific solver (empty = search all)
            max_results: Maximum results per backend (default 5)
        """
        results = {}

        # 4C: search actual test files AND include content preview
        if not solver or solver.lower() in ("fourc", "4c"):
            fourc_results = _search_4c_input_files(keyword, max_results)
            if fourc_results:
                # Include actual file content for the first match so the agent
                # can see real parameter values used in validated test cases
                for r in fourc_results[:2]:
                    fpath = FOURC_TESTS / r["file"] if FOURC_TESTS else None
                    if fpath and fpath.is_file():
                        try:
                            content = fpath.read_text()
                            # Include first 3000 chars of content
                            r["content_preview"] = content[:3000]
                        except Exception:
                            pass
                results["4C"] = {
                    "source": f"tests/ directory ({FOURC_TESTS})" if FOURC_TESTS else "not available",
                    "matches": fourc_results,
                }

        # All backends: search template variants
        backends_to_search = []
        if solver:
            b = get_backend(solver)
            if b:
                backends_to_search = [b]
        else:
            backends_to_search = available_backends()

        keyword_lower = keyword.lower()
        for b in backends_to_search:
            matches = []
            for p in b.supported_physics():
                if keyword_lower in p.name.lower() or keyword_lower in p.description.lower():
                    matches.append({
                        "physics": p.name,
                        "description": p.description,
                        "variants": p.template_variants,
                        "dims": p.spatial_dims,
                    })
            if matches:
                results[b.display_name()] = {"templates": matches[:max_results]}

        if not results:
            return f"No examples found for '{keyword}'. Try broader terms like 'elasticity', 'flow', 'heat'."

        return json.dumps(results, indent=2)

    @mcp.tool()
    def get_example_input(solver: str, physics: str, variant: str = "",
                          source: str = "template") -> str:
        """Retrieve a complete, runnable example input file.

        For 4C: Can return tutorial files or generated templates
        For FEniCS: Returns complete Python script
        For deal.II: Returns complete C++ source + CMakeLists
        For FEBio: Returns complete XML

        Args:
            solver: Backend name ('fenics', 'fourc', 'dealii', 'febio')
            physics: Physics type (e.g. 'poisson', 'linear_elasticity')
            variant: Template variant (e.g. '2d', '3d', 'poisson_2d')
            source: 'template' for generated, 'tutorial' for 4C tutorial files
        """
        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        # For 4C tutorials
        if source == "tutorial" and solver.lower() in ("fourc", "4c"):
            tutorials = _get_4c_tutorial_files()
            for t in tutorials:
                if physics.lower() in t["tutorial"].lower() or physics.lower() in t["file"].lower():
                    try:
                        content = Path(t["path"]).read_text()
                        return f"# 4C Tutorial: {t['tutorial']}/{t['file']}\n\n```yaml\n{content}\n```"
                    except Exception as e:
                        return f"Error reading {t['path']}: {e}"
            return f"No 4C tutorial found for '{physics}'. Available: {[t['tutorial'] for t in tutorials]}"

        # Generate from template
        if not variant:
            # Auto-select first variant
            for p in backend.supported_physics():
                if p.name == physics.lower() and p.template_variants:
                    variant = p.template_variants[0]
                    break

        if not variant:
            return f"No variant specified and no default found for {physics} in {solver}"

        try:
            content = backend.generate_input(physics, variant, {})
            fmt = detect_template_language(content, backend.input_format().value)
            return f"```{fmt}\n{content}\n```"
        except ValueError as e:
            return str(e)

    @mcp.tool()
    def list_tutorials(solver: str = "") -> str:
        """List available tutorial examples for a solver backend.

        For 4C: Lists complete tutorials with mesh files from the 4C test suite.
        For FEniCS/deal.II: Lists template variants available for generation.
        For all: Shows what ready-to-run examples exist.

        Args:
            solver: Backend name (empty = list all available)
        """
        lines = []

        # 4C tutorials (file-based)
        if not solver or solver.lower() in ("fourc", "4c"):
            tutorials = _get_4c_tutorial_files()
            if tutorials:
                lines.append("# 4C Tutorials\n")
                current_tut = ""
                for t in tutorials:
                    if t["tutorial"] != current_tut:
                        current_tut = t["tutorial"]
                        lines.append(f"\n## {current_tut}")
                    lines.append(f"- `{t['file']}`")
                lines.append("")

        # Other backends: list template variants
        from core.registry import available_backends, get_backend
        backends_to_show = []
        if solver and solver.lower() not in ("fourc", "4c"):
            b = get_backend(solver)
            if b:
                backends_to_show = [b]
        elif not solver:
            backends_to_show = available_backends()

        for b in backends_to_show:
            if b.name() == "fourc":
                continue  # already handled above
            b_lines = []
            for p in b.supported_physics():
                if p.template_variants:
                    b_lines.append(f"  - **{p.name}**: {', '.join(p.template_variants)} — {p.description}")
            if b_lines:
                lines.append(f"# {b.display_name()} Templates\n")
                lines.extend(b_lines)
                lines.append("")

        if not lines:
            return "No tutorials found. Check solver installation."
        return "\n".join(lines)

    @mcp.tool()
    def browse_solver_tests(solver: str, keyword: str = "", max_results: int = 10) -> str:
        """Browse real test/example files from a solver's own test suite.

        This is the PRIMARY source for understanding how each solver is actually
        used. These are real, validated input files from the solver's own test
        suite — not simplified templates.

        Args:
            solver: Backend name (fourc, fenics, dealii, ngsolve, kratos, dune, skfem)
            keyword: Filter by keyword in filename (empty = list all)
            max_results: Maximum number of results
        """
        results = []

        # Define where each solver's real tests/examples live
        test_dirs = {
            "fourc": FOURC_ROOT / "tests" / "input_files" if FOURC_ROOT else None,
            "4c": FOURC_ROOT / "tests" / "input_files" if FOURC_ROOT else None,
            "dealii": Path("/usr/share/doc/libdeal.ii-doc/examples"),
            "fenics": None,  # detected below
            "ngsolve": None,  # detected below
        }

        # FEniCS demos
        fenics_demo = Path.home() / "miniconda3" / "envs" / "fenics" / "share" / "dolfinx" / "demo"
        if not fenics_demo.is_dir():
            for p in Path.home().glob("miniconda3/envs/fenics/**/demo"):
                if p.is_dir():
                    fenics_demo = p
                    break
        if fenics_demo.is_dir():
            test_dirs["fenics"] = fenics_demo

        solver_key = solver.lower()
        test_dir = test_dirs.get(solver_key)

        if test_dir is None or not test_dir.is_dir():
            return (
                f"No test directory found for '{solver}'. "
                f"The agent can still use generate_input() for templates, "
                f"or read files directly if the path is known."
            )

        # Search
        extensions = {
            "fourc": "*.4C.yaml", "4c": "*.4C.yaml",
            "dealii": "*.cc",
            "fenics": "*.py",
            "ngsolve": "*.py",
        }
        ext = extensions.get(solver_key, "*")
        keyword_lower = keyword.lower()

        found_files = []
        for f in sorted(test_dir.rglob(ext)):
            if keyword and keyword_lower not in f.name.lower() and keyword_lower not in str(f.parent.name).lower():
                continue
            found_files.append(f)
            if len(found_files) >= max_results:
                break

        if not found_files:
            return f"No files matching '{keyword}' in {test_dir}"

        header = f"## {solver} test files in `{test_dir}`\n\n"
        header += f"Found {len(found_files)} matches" + (f" for '{keyword}'" if keyword else "") + ":\n\n"

        parts = [header]
        for f in found_files:
            rel = f.relative_to(test_dir)
            parts.append(f"### `{rel}` ({f.stat().st_size} bytes)\n")
            # Include content preview for ALL solvers
            try:
                content = f.read_text()
                preview = content[:3000]
                if len(content) > 3000:
                    preview += "\n... (truncated)"
                parts.append(f"```\n{preview}\n```\n")
            except Exception:
                parts.append("(could not read file)\n")

        return "\n".join(parts)

    @mcp.tool()
    def read_solver_test_file(solver: str, filepath: str) -> str:
        """Read the content of a specific test/example file from a solver.

        Use browse_solver_tests() first to find files, then this to read them.
        This gives the agent access to real, validated input files from each
        solver's own test suite — the best reference for correct usage.

        Args:
            solver: Backend name
            filepath: Relative path within the test directory (from browse_solver_tests output)
        """
        test_dirs = {
            "fourc": FOURC_ROOT / "tests" / "input_files" if FOURC_ROOT else None,
            "4c": FOURC_ROOT / "tests" / "input_files" if FOURC_ROOT else None,
            "dealii": Path("/usr/share/doc/libdeal.ii-doc/examples"),
        }

        solver_key = solver.lower()
        test_dir = test_dirs.get(solver_key)

        if test_dir is None or not test_dir.is_dir():
            return f"No test directory for '{solver}'"

        full_path = test_dir / filepath
        if not full_path.is_file():
            return f"File not found: {full_path}"

        try:
            content = full_path.read_text()
            if len(content) > 50000:
                content = content[:50000] + "\n\n... (truncated, file is very large)"
            return f"# {filepath}\n\n```\n{content}\n```"
        except Exception as e:
            return f"Error reading {full_path}: {e}"

    @mcp.tool()
    def get_input_file_guide(solver: str = "fourc") -> str:
        """Get a comprehensive guide for writing input files for a solver.

        Covers file structure, required sections, common patterns, and
        the most frequent mistakes.

        Args:
            solver: Backend name (default: 'fourc')
        """
        if solver.lower() in ("fourc", "4c"):
            return _4C_INPUT_GUIDE
        elif solver.lower() in ("fenics", "fenicsx"):
            return _FENICS_INPUT_GUIDE
        elif solver.lower() in ("dealii", "deal.ii"):
            return _DEALII_INPUT_GUIDE
        elif solver.lower() == "febio":
            return _FEBIO_INPUT_GUIDE
        else:
            return f"Unknown solver: {solver}"


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT FILE GUIDES
# ═══════════════════════════════════════════════════════════════════════════════

_4C_INPUT_GUIDE = """\
# 4C Input File Guide (.4C.yaml)

## File Structure
4C uses YAML input files with ALL-CAPS section names:
```yaml
TITLE:
  - "Description of the simulation"
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"  # or Scalar_Transport, Fluid, etc.
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  # ... time integration parameters
SOLVER 1:
  SOLVER: "UMFPACK"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 210000
      NUE: 0.3
      DENS: 7.85e-9
STRUCTURE GEOMETRY:
  ELEMENT_BLOCKS:
    - ID: 1
      SOLID:
        HEX8:
          MAT: 1
          KINEM: linear
  FILE: mesh.e
```

## Key Rules
1. Section names are ALL CAPS: STRUCTURAL DYNAMIC, not Structural Dynamic
2. PROBLEMTYPE determines which *_DYNAMIC section is read
3. Each physics needs its own GEOMETRY section (STRUCTURE, FLUID, TRANSPORT)
4. Materials are listed under MATERIALS with MAT: <id>
5. Solvers are SOLVER 1, SOLVER 2, etc.
6. Boundary conditions reference node/surface sets by ID (ENTITY_TYPE: node_set_id)

## Common Mistakes
- Using wrong section name (SCATRA DYNAMIC vs SCALAR TRANSPORT DYNAMIC)
- Missing VELOCITYFIELD: zero for pure diffusion
- Using NUMDOF inconsistently with the physics
- Forgetting to set KINEM: nonlinear for large-deformation problems
- Missing DENS in dynamics (zero mass matrix = singular)
"""

_FENICS_INPUT_GUIDE = """\
# FEniCSx (dolfinx) Script Guide

## Script Structure
```python
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem  # or NonlinearProblem
import ufl
import numpy as np

# 1. Create mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32, mesh.CellType.triangle)

# 2. Define function space
V = fem.functionspace(domain, ("Lagrange", 1))

# 3. Apply boundary conditions
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)

# 4. Define weak form
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = fem.Constant(domain, default_scalar_type(1.0)) * v * ufl.dx

# 5. Solve
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# 6. Output (XDMF for dolfinx, convert to VTU for visualization)
from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)
```

## Key Rules (dolfinx 0.10.0)
1. Use `fem.functionspace()` (not `FunctionSpace()`)
2. Use `basix.ufl.element()` and `mixed_element()` for mixed spaces
3. `NonlinearProblem` requires `petsc_options_prefix` kwarg
4. `NonlinearProblem.solve()` directly — no separate NewtonSolver
5. P2 functions can't write to XDMF — interpolate to P1 first
6. BCs on sub-spaces: use `fem.Function` (not constant array)
7. Always use `default_scalar_type` for PETSc compatibility
"""

_DEALII_INPUT_GUIDE = """\
# deal.II C++ Source Guide

## Source Structure
```cpp
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
// ... other includes

using namespace dealii;

int main()
{
    // 1. Grid
    Triangulation<2> triangulation;
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(5);

    // 2. FE space
    FE_Q<2> fe(1);
    DoFHandler<2> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);

    // 3. Sparsity + matrices
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    SparsityPattern sp; sp.copy_from(dsp);
    SparseMatrix<double> system_matrix; system_matrix.reinit(sp);

    // 4. Assembly (loop over cells)
    // 5. Boundary conditions
    // 6. Solve (CG + preconditioner)
    // 7. Output (DataOut → VTU)
}
```

## Build (CMakeLists.txt)
```cmake
cmake_minimum_required(VERSION 3.1)
find_package(deal.II 9.0 REQUIRED)
deal_ii_initialize_cached_variables()
project(my_problem)
add_executable(my_problem main.cpp)
deal_ii_setup_target(my_problem)
```

## Key Rules
1. Always refine BEFORE distributing DOFs
2. Use DynamicSparsityPattern → copy_from → SparsityPattern
3. Vector FE: FESystem<dim>(FE_Q<dim>(1), dim)
4. Boundary IDs depend on GridGenerator (hyper_cube: all=0, hyper_rectangle: 0-3)
5. DataOut for VTU output
"""

_DUNE_INPUT_GUIDE = """\
# DUNE-fem Input Guide (Python scripts using UFL)

## Core Imports
```python
from dune.grid import structuredGrid          # structured mesh
from dune.alugrid import aluConformGrid       # unstructured mesh
from dune.fem.space import lagrange, dglagrange  # FE spaces
from dune.fem.scheme import galerkin          # solver
from dune.fem.function import gridFunction    # post-processing
from dune.ufl import DirichletBC             # boundary conditions
from ufl import *                            # UFL form language
```

## Grid Creation
```python
# Structured grid on [0,1]^2
gridView = structuredGrid([0, 0], [1, 1], [N, N])

# Unstructured from Gmsh
from dune.alugrid import aluConformGrid
gridView = aluConformGrid("mesh.msh", dimgrid=2)
```

## Scalar vs Vector Spaces
```python
# Scalar (1 DOF per node)
space = lagrange(gridView, order=1)           # P1
space = lagrange(gridView, order=2)           # P2

# Vector-valued (dim DOFs per node, e.g. elasticity)
space = lagrange(gridView, dimRange=2, order=1)  # 2D vector P1

# DG space
space = dglagrange(gridView, order=1)         # DG-P1
```

## Weak Forms (same as FEniCS UFL)
```python
u = TrialFunction(space)
v = TestFunction(space)

# Bilinear + linear form
a = inner(grad(u), grad(v)) * dx
L = f * v * dx
scheme = galerkin([a == L, DirichletBC(space, 0)], solver="cg")
u_h = space.interpolate(0, name="solution")
scheme.solve(target=u_h)

# Nonlinear: use galerkin with residual form
# galerkin([F == 0]) triggers Newton's method automatically
# F must be written with u_h (the solution function), NOT TrialFunction
F = (inner(grad(u_h), grad(v)) - f * v) * dx
scheme = galerkin([F == 0, DirichletBC(space, 0)])
scheme.solve(target=u_h)
```

## Time Stepping
```python
# Semi-implicit: diffusion implicit, reaction explicit
u_n = space.interpolate(u0_expr, name="u")
u = TrialFunction(space)
v = TestFunction(space)

a = (u * v / dt + D * inner(grad(u), grad(v))) * dx
L = (u_n * v / dt + reaction(u_n) * v) * dx
scheme = galerkin([a == L], solver="cg")

for step in range(n_steps):
    scheme.solve(target=u_n)  # u_n updated in-place
```

## Coupled Systems (Two Scalar Fields)
```python
# For multi-species: use two separate scalar spaces + Gauss-Seidel
space = lagrange(gridView, order=1)
u_n = space.interpolate(u0, name="u")
v_n = space.interpolate(v0, name="v")

# Scheme for u (can reference v_n as a coefficient)
scheme_u = galerkin([a_u == L_u], solver="cg")
# Scheme for v (reference u_n — updated value for Gauss-Seidel)
scheme_v = galerkin([a_v == L_v], solver="cg")

for step in range(n_steps):
    scheme_u.solve(target=u_n)  # solve u first
    scheme_v.solve(target=v_n)  # then v using new u_n
```

## Coefficient Functions & Reassembly
```python
# DUNE-fem evaluates coefficients lazily at solve time.
# If u_n appears in a form, its CURRENT values are used each solve().
# No manual reassembly needed — just update the discrete function.
```

## Output
```python
# VTK output
gridView.writeVTK("filename", pointdata={"u": u_h, "v": v_h})

# Access DOF values as numpy array
vals = u_h.as_numpy  # returns a numpy view (read/write)
```

## Key Pitfalls
1. First run is slow (60-120s) due to JIT C++ compilation. Subsequent runs are fast.
2. For coupled systems: dimRange=2 with Newton is possible but less documented.
   Safer approach: two scalar spaces with Gauss-Seidel coupling.
3. `galerkin([F == 0])` triggers Newton automatically. `galerkin([a == L])` solves linear.
4. DOF ordering for dimRange>1: components are interleaved (u0,v0,u1,v1,...).
5. No built-in time integrator — manual time loop required.
6. Set timeout >= 600s for first run to allow JIT compilation.
"""

_FEBIO_INPUT_GUIDE = """\
# FEBio Input File Guide (.feb XML)

## Structure (v4.0)
```xml
<?xml version="1.0"?>
<febio_spec version="4.0">
  <Module type="solid"/>  <!-- solid, biphasic, heat, etc. -->
  <Control>...</Control>
  <Globals>...</Globals>
  <Material>...</Material>
  <Mesh>
    <Nodes>...</Nodes>
    <Elements>...</Elements>
    <NodeSet>...</NodeSet>
  </Mesh>
  <MeshDomains>...</MeshDomains>
  <Boundary>...</Boundary>
  <LoadData>...</LoadData>
  <Output>...</Output>
</febio_spec>
```

## Key Rules
1. Poisson's ratio: lowercase 'v' (not 'nu')
2. All indices are 1-based
3. MeshDomains links elements to materials (required in v4.0)
4. LoadData with load_controller for time-varying BCs
5. Module type determines available materials and BCs
"""
