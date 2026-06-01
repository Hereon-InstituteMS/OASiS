"""
MCP tools for accessing physics knowledge and input generation.
"""

import json
from mcp.server.fastmcp import FastMCP
from core.registry import get_backend, available_backends


def discover_test_dirs() -> dict:
    """Return a {solver_key: Path} mapping of locally-present test/
    demo directories.

    The same lookup is needed by prepare_simulation (via
    _find_reference_test_files) and the `examples` MCP tool. Before
    2026-06-01 these were two separate hardcoded fourc+dealii-only
    dicts — meaning the examples tool returned 0 results for fenics
    / ngsolve / kratos / dune / febio even though demo trees existed
    locally. Centralising here keeps the two surfaces in sync.

    Probes (each gated on directory existence):
      fourc / 4c   -> $FOURC_ROOT/tests/input_files
      dealii       -> /usr/share/doc/libdeal.ii-doc/examples
      fenics(x)    -> any *fenics* conda env's
                      share/dolfinx/demo OR
                      etc/conda/test-files/fenics-dolfinx/0/python/demo
      ngsolve      -> .venv .../ngsolve/demos OR
                      conda *fenics* envs ../ngsolve/demos
      kratos       -> .venv .../KratosMultiphysics
    """
    import os
    from pathlib import Path

    test_dirs = {
        "fourc": Path(os.environ.get("FOURC_ROOT", "")) / "tests" / "input_files",
        "4c": Path(os.environ.get("FOURC_ROOT", "")) / "tests" / "input_files",
        "dealii": Path("/usr/share/doc/libdeal.ii-doc/examples"),
    }

    # FEniCS demos — probe several known conda-forge layouts.
    # Without this, prepare_simulation('fenics', 'poisson')
    # silently emits no reference test file because the
    # hardcoded share/dolfinx/demo path does not exist on
    # current ofa-fenicsx installs (conda-forge moved demos
    # under etc/conda/test-files/fenics-dolfinx/0/python/demo).
    candidates = [
        # Legacy path (older conda-forge layout)
        Path.home() / "miniconda3" / "envs" / "fenics"
        / "share" / "dolfinx" / "demo",
        # Current conda-forge ofa-fenicsx layout (probed 2026-06-01)
        Path.home() / "miniconda3" / "envs" / "ofa-fenicsx"
        / "etc" / "conda" / "test-files" / "fenics-dolfinx"
        / "0" / "python" / "demo",
    ]
    # Also probe any *fenics* conda env present locally
    conda_envs = Path.home() / "miniconda3" / "envs"
    if conda_envs.is_dir():
        for env in conda_envs.iterdir():
            if "fenics" in env.name.lower():
                candidates.extend([
                    env / "share" / "dolfinx" / "demo",
                    env / "etc" / "conda" / "test-files"
                    / "fenics-dolfinx" / "0" / "python" / "demo",
                ])
    for fenics_demo in candidates:
        if fenics_demo.is_dir():
            test_dirs["fenics"] = fenics_demo
            test_dirs["fenicsx"] = fenics_demo
            break

    # NGSolve ships demos inside the installed wheel:
    # site-packages/ngsolve/demos/{intro,howto,mpi,
    # TensorProduct,...}. Probe the active .venv plus
    # any conda env that includes ngsolve.
    ngsolve_candidates = [
        Path(__file__).resolve().parents[2] / ".venv" / "lib"
        / "python3.12" / "site-packages" / "ngsolve" / "demos",
    ]
    for envname in ("ofa-fenicsx",):  # other envs may ship ngsolve too
        ngsolve_candidates.append(
            Path.home() / "miniconda3" / "envs" / envname / "lib"
            / "python3.12" / "site-packages" / "ngsolve" / "demos")
    for ng_demo in ngsolve_candidates:
        if ng_demo.is_dir():
            test_dirs["ngsolve"] = ng_demo
            break

    # Kratos ships Python tests inside each Application:
    # site-packages/KratosMultiphysics/<App>/tests/*.py. The
    # tree is broad (many Applications) so we point at the
    # top KratosMultiphysics dir and let the rglob walk find
    # *.py matching the keyword.
    kratos_candidates = [
        Path(__file__).resolve().parents[2] / ".venv" / "lib"
        / "python3.12" / "site-packages" / "KratosMultiphysics",
    ]
    for kr_dir in kratos_candidates:
        if kr_dir.is_dir():
            test_dirs["kratos"] = kr_dir
            break

    return test_dirs


def _find_reference_test_files(solver: str, physics: str) -> str:
    """Find real test files from a solver's test suite as reference.

    Returns a note with file paths and content previews so the agent
    can see how validated simulations are actually configured. Empty
    string when no local demos are available for the (solver, physics)
    pair.
    """
    from pathlib import Path
    test_dirs = discover_test_dirs()

    # Map physics to search keywords for ALL physics types
    search_terms = {
        "particle_pd": "pdbody",
        "particle_sph": "sph",
        "fsi": "fsi",
        "tsi": "tsi",
        "ssi": "ssi",
        "ssti": "ssti",
        "sti": "sti",
        "fluid": "fluid",
        "contact": "contact",
        "beams": "beam",
        "poisson": "scatra",
        "heat": "thermo",
        "linear_elasticity": "solid",
        "structural_dynamics": "genalpha",
        "ale": "ale",
        "electrochemistry": "elch",
        "level_set": "level_set",
        "low_mach": "loma",
        "lubrication": "lubrication",
        "cardiac_monodomain": "cardiac",
        "arterial_network": "art_",
        "ehl": "ehl",
        "fpsi": "fpsi",
        "fbi": "fbi",
        "pasi": "pasi",
        "beam_interaction": "beam_contact",
        "multiscale": "multi_scale",
        "reduced_airways": "red_airway",
        # deal.II step tutorials
        "stokes": "step-22",
        "helmholtz": "step-29",
        "eigenvalue": "step-36",
        "wave": "step-23",
        "hyperelasticity": "step-44",
        "nonlinear": "step-15",
        "convection_diffusion": "step-9",
        "hp_adaptive": "step-27",
        "dg_transport": "step-12",
        "parallel": "step-40",
        # FEniCS demos
        "navier_stokes": "navier",
        "mixed_poisson": "mixed",
        "biharmonic": "biharmonic",
        "reaction_diffusion": "reaction",
    }

    solver_key = solver.lower()
    test_dir = test_dirs.get(solver_key)

    if not test_dir or not test_dir.is_dir():
        return ""

    ext = "*.4C.yaml" if solver_key in ("fourc", "4c") else "*.cc" if solver_key == "dealii" else "*.py"

    # The search_terms map is biased toward 4C / deal.II
    # filename conventions (e.g. poisson -> scatra,
    # heat -> thermo). Those keys do NOT match the FEniCS
    # demo names (demo_poisson.py, demo_helmholtz.py, ...).
    # Audit 2026-06-01: prepare_simulation('fenics','poisson')
    # surfaced zero reference test files because the keyword
    # was 'scatra' which doesn't appear in any dolfinx demo.
    # Try BOTH the mapped keyword AND the raw physics name —
    # also try with hyphens (dolfinx demos use hyphens for
    # compound names: demo_navier-stokes.py, demo_mixed-
    # poisson.py, demo_cahn-hilliard.py).
    # NGSolve demo filenames don't match the catalog or 4C /
    # deal.II keys. Demos: poisson.py / navierstokes.py /
    # elasticity.py / cmagnet.py (magnetostatics) / pml.py
    # (helmholtz / PML) / hhj.py (Hellan-Herrmann-Johnson
    # biharmonic) / hybrid_dg.py (DG methods) / nonlin.py
    # (nonlinear elasticity) / mixed.py (mixed_poisson) /
    # timeDG.py (time-dependent DG) / tdnns.py.
    ngsolve_aliases = {
        "navier_stokes": "navierstokes",
        "maxwell": "cmagnet",
        "magnetostatics": "cmagnet",
        "helmholtz": "pml",
        "biharmonic": "hhj",
        "hdivdiv": "hhj",
        "dg_methods": "hybrid_dg",
        "hyperelasticity": "nonlin",
        "nonlinear_elasticity": "nonlin",
        "mixed_poisson": "mixed",
        "time_dependent_heat": "timeDG",
        "time_dependent_ns": "timeDG",
    }

    keywords = [physics]
    if physics in search_terms:
        keywords.insert(0, search_terms[physics])
    if solver_key == "ngsolve" and physics in ngsolve_aliases:
        keywords.insert(0, ngsolve_aliases[physics])
    if "_" in physics:
        keywords.append(physics.replace("_", "-"))
    # Common substring trims so the FEniCS demo naming
    # convention (demo_elasticity.py, no "linear_" prefix)
    # is reachable from the catalog name (linear_elasticity).
    # Audit 2026-06-01.
    for prefix in ("linear_", "nonlinear_", "time_dependent_"):
        if physics.startswith(prefix):
            keywords.append(physics[len(prefix):])

    matches = []
    for kw in keywords:
        for f in sorted(test_dir.rglob(ext)):
            if kw.lower() in f.name.lower() and f not in matches:
                matches.append(f)
                if len(matches) >= 2:
                    break
        if len(matches) >= 2:
            break

    if not matches:
        return ""

    parts = ["## Reference: Real test files from the solver's own test suite\n"]
    for f in matches:
        rel = f.relative_to(test_dir)
        parts.append(f"### `{rel}`")
        try:
            content = f.read_text()[:2000]
            parts.append(f"```\n{content}\n```\n")
        except Exception:
            pass

    return "\n".join(parts)


def register_knowledge_tools(mcp: FastMCP):

    @mcp.tool()
    def get_physics_knowledge(solver: str, physics: str) -> str:
        """Get domain knowledge for a physics module from a specific solver backend.

        Returns materials, solver recommendations, pitfalls, and best practices.

        Args:
            solver: Backend name (e.g. 'fenics', 'fourc', 'dealii', 'febio')
            physics: Physics type (e.g. 'poisson', 'linear_elasticity', 'heat')
        """
        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        knowledge = backend.get_knowledge(physics)
        if not knowledge:
            return f"No knowledge available for '{physics}' in {backend.display_name()}"

        result = json.dumps(knowledge, indent=2, default=str)

        # Automatically append real test file examples for ALL solvers
        ref = _find_reference_test_files(solver, physics)
        if ref:
            result += f"\n\n{ref}"

        return result

    @mcp.tool()
    def generate_input(solver: str, physics: str, variant: str = "2d",
                       params: str = "{}") -> str:
        """Generate a complete, runnable input for a solver backend.

        The generated input is solver-specific:
        - 4C: YAML input file (.4C.yaml)
        - FEniCS: Python script using dolfinx
        - deal.II: C++ source code
        - FEBio: XML input file (.feb)

        Args:
            solver: Backend name (e.g. 'fenics', 'fourc', 'dealii', 'febio')
            physics: Physics type (e.g. 'poisson', 'linear_elasticity')
            variant: Template variant (e.g. '2d', '3d', '2d_steady')
            params: JSON string of parameters to override defaults,
                    e.g. '{"kappa": 2.5, "nx": 64}'
        """
        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        import json as _json
        try:
            param_dict = _json.loads(params)
        except _json.JSONDecodeError as e:
            return f"Invalid params JSON: {e}"

        try:
            content = backend.generate_input(physics, variant, param_dict)
            format_name = backend.input_format().value
            result = f"```{format_name}\n{content}\n```"

            # Include real test file references so the agent can see
            # validated parameter values from the solver's own test suite
            ref_note = _find_reference_test_files(solver, physics)
            if ref_note:
                result += f"\n\n{ref_note}"

            return result
        except ValueError as e:
            return str(e)

    @mcp.tool()
    def validate_input(solver: str, content: str) -> str:
        """Validate solver-specific input content before running.

        Args:
            solver: Backend name
            content: The input content (YAML / Python / C++ / XML)
        """
        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        errors = backend.validate_input(content)
        if not errors:
            return "Input is valid."
        return "Validation errors:\n" + "\n".join(f"- {e}" for e in errors)

    @mcp.tool()
    def get_coupling_knowledge() -> str:
        """Get complete knowledge for cross-solver domain decomposition coupling.

        Returns theory, implementation patterns, pitfalls, and best practices
        for Dirichlet-Neumann domain decomposition across independent FEM codes.
        This is essential reading before using coupled_solve or transfer_field.
        """
        return '''\
# Cross-Solver Coupling via Dirichlet-Neumann Domain Decomposition

## Theory

Dirichlet-Neumann (DN) domain decomposition splits a PDE domain at an interface.
Two independent solvers handle the subdomains, exchanging boundary data iteratively:

```
Initialize: guess T_interface (e.g. linear interpolation of BCs)

for iteration in range(max_iter):
    1. Solve subdomain A (Dirichlet at interface): u_A = solve(BC: T_interface)
    2. Extract flux: q = -k * du_A/dn at interface
    3. Solve subdomain B (Neumann at interface): u_B = solve(BC: flux = q)
    4. Extract temperature: T_new = u_B at interface
    5. Update: T_interface = θ * T_new + (1-θ) * T_interface
    6. Check convergence: |T_new - T_old| / |T_new| < tolerance
```

## Relaxation Parameter θ

- **θ = 1.0**: No relaxation. Works for linear problems WITHOUT source terms
  (e.g. steady heat conduction T=100→0). Converges in 1 iteration.
- **θ = 0.5**: Required for problems WITH source terms (e.g. Poisson -Δu=f).
  Without relaxation, DN oscillates and never converges!
- **θ = 0.7**: Good default for nonlinear problems (elasticity, coupled physics).
- **Rule of thumb**: If DN oscillates (residual doesn't decrease), reduce θ.

## Neumann BC Sign Convention

At the interface between subdomains A and B:
- Domain A solves with Dirichlet BC at interface
- The flux from A is: q = -k * ∂u_A/∂n_A (outward normal from A)
- Domain B receives this flux as its Neumann BC
- In 4C: DESIGN LINE NEUMANN VAL receives the flux value directly
  (4C convention: the Neumann value is k * ∂u/∂n on the boundary)
- The outward normal at B's interface boundary points AWAY from B (toward A)
- Therefore: q_4C = q_FEniCS (same sign — the physical flux is continuous)

## Solver-Specific Details

### FEniCS (Dirichlet subdomain — typically Domain A)
- Use `mesh.create_rectangle()` for subdomain mesh
- Apply interface Dirichlet via per-DOF interpolation from coupled values
- Compute flux via finite difference from neighboring interior nodes
- Write interface data to `interface_data.json` for transfer

### 4C (Neumann subdomain — typically Domain B)
- Use TRANSP QUAD4 elements for scalar transport (heat/Poisson)
- DESIGN LINE NEUMANN CONDITIONS for interface flux
- Node coordinates must be offset to match subdomain position
- No IO/RUNTIME VTK OUTPUT section for scatra — use post_vtu conversion
- Field name in VTU output is `phi_1` (not `temperature`)
- Always use the LAST VTU file (scatra-00001-0.vtu), not the initial condition

### Field Transfer Between Solvers
- 4C uses duplicate nodes per element (QUAD4 = 4 nodes per cell → more points)
- FEniCS uses shared nodes → fewer points
- Use `extract_interface_from_vtu()` to get interface values from either
- Use `interpolate_to_points()` for non-matching mesh interpolation
- Sort interface nodes by tangential coordinate for consistent ordering

## Verified Results (All Solver Combinations)

### Heat DD (T=100 left, T=0 right, no source)
- Exact solution: T(x) = 100*(1-x), linear

| Solver A | Solver B | Iterations | Final Residual | T(0.5) Error |
|----------|----------|------------|----------------|--------------|
| FEniCS   | 4C       | 1          | 3.4e-16        | 0.0          |
| FEniCS   | FEniCS   | 1          | 4.3e-15        | 4.3e-15      |

### Poisson DD (-Δu=1, u=0 on boundary, θ=0.5)
- Reference: single-domain FEniCS solve

| Solver A | Solver B | Iterations | Final Residual |
|----------|----------|------------|----------------|
| FEniCS   | 4C       | 2          | 1.2e-16        |
| FEniCS   | FEniCS   | 7          | 9.7e-05        |

- Without relaxation (θ=1.0): oscillates indefinitely!

### Supported Backend Combinations
- FEniCS (Dirichlet) ↔ 4C (Neumann): fully tested, production ready
- FEniCS (Dirichlet) ↔ FEniCS (Neumann): fully tested, proves solver-agnosticism
- Any combination works if `_generate_domain_b_input()` supports the backend

## Common Pitfalls

1. **Missing relaxation**: DN with source oscillates without θ<1. Always test.
2. **Wrong VTU timestep**: 4C writes initial condition + solution. Use LAST file.
3. **Field name mismatch**: 4C=phi_1, FEniCS=temperature. Handle in extraction.
4. **Node duplication**: 4C VTU has 4× more nodes than expected. Still works with
   extract_interface but interpolation target must match.
5. **IO/RUNTIME VTK OUTPUT/SCATRA**: May crash 4C. Omit — use post_vtu instead.
6. **Neumann sign**: Easy to get wrong. Test with linear solution first where
   exact answer is known.

## Extending to New Problems

The same DN pattern works for:
- **Elasticity**: Replace temperature with displacement, flux with traction
- **Coupled thermal-structural**: One-way coupling (heat→stress)
- **Different physics per subdomain**: e.g. fluid (FEniCS) + structure (4C)

Key changes needed for new physics:
1. New subdomain script generator (analogous to `_fenics_heat_subdomain_script`)
2. New 4C input generator (analogous to `_fourc_heat_subdomain_input`)
3. Appropriate relaxation parameter
4. Correct field names and transfer format

## Available Coupling Problem Types

| Problem | Description | Solvers |
|---------|-------------|---------|
| `heat_dd` | Heat conduction, DN domain decomposition | FEniCS↔4C, FEniCS↔FEniCS |
| `poisson_dd` | Poisson with source, DN decomposition | FEniCS↔4C, FEniCS↔FEniCS |
| `one_way` | FEniCS thermal → 4C structural (TSI) | FEniCS + 4C |
| `tsi_dd` | Two-way iterative TSI coupling | FEniCS + 4C |
| `poisson_dd_study` | Relaxation parameter comparison | Any backend pair |
| `l_bracket_tsi` | L-bracket thermal stress concentration | FEniCS + 4C |
| `heat_dd_precice` | Our DN vs preCICE comparison | Any + preCICE config |

## Relaxation Parameter Selection Guide

| Scenario | Recommended θ | Reason |
|----------|--------------|--------|
| Linear, no source | 1.0 | Exact in 1 iteration |
| Linear, with source | 0.5 | DN oscillates without relaxation |
| Nonlinear | 0.7 | Good starting point |
| Unknown / complex | Aitken | Automatic acceleration |
| Failing to converge | Reduce θ | Try 0.3, then 0.1 |
'''

    @mcp.tool()
    def get_tsi_knowledge() -> str:
        """Get complete knowledge for thermo-structural interaction (TSI) coupling.

        Returns 4C TSI patterns, material types, CLONING MAP, coupling algorithms,
        and cross-solver TSI workflow. Essential for thermal-structural simulations.
        """
        return '''\
# Thermo-Structural Interaction (TSI) Knowledge

## 4C Native TSI

4C has built-in thermo-structural coupling via `PROBLEMTYPE: "Thermo_Structure_Interaction"`.

### Required Components

1. **Element type:** `SOLIDSCATRA HEX8` (3D) — TSI needs the
   combined eletype, NOT plain `SOLID HEX8` (structure-only) or
   the legacy `WALL` 2D eletype.
   - SOLIDSCATRA combines structural + scalar transport capabilities
   - Must be 3D (no 2D TSI elements in 4C)

2. **Material:** `MAT_Struct_ThermoStVenantK`
   ```yaml
   MATERIALS:
     - MAT: 1
       MAT_Struct_ThermoStVenantK:
         YOUNGNUM: 1
         YOUNG: [200000]      # Young's modulus (Pa or MPa)
         NUE: 0.3             # Poisson's ratio
         DENS: 1.0            # Density
         THEXPANS: 1.2e-5     # Thermal expansion coefficient (1/K)
         INITTEMP: 0.0        # Reference temperature
         THERMOMAT: 2         # Links to thermal material ID
     - MAT: 2
       MAT_Fourier:
         CAPA: 1.0            # Heat capacity
         CONDUCT:
           constant: [1.0]    # Thermal conductivity
   ```

3. **Cloning material map** (required for multi-field coupling):
   ```yaml
   CLONING MATERIAL MAP:
     - SRC_FIELD: "structure"
       SRC_MAT: 1
       TAR_FIELD: "thermo"
       TAR_MAT: 2
   ```

4. **Three dynamics sections:**
   - `STRUCTURAL DYNAMIC`: structural solver parameters
   - `THERMAL DYNAMIC`: thermal solver parameters + INITIALFIELD
   - `TSI DYNAMIC`: coupling algorithm control

### TSI Coupling Algorithms

| Algorithm | COUPALGO value | Use case |
|-----------|---------------|----------|
| One-way | `tsi_oneway` | Thermal → structural (no feedback) |
| Iterative staggered | `tsi_iterstagg` | Two-way, sequential |
| Aitken staggered | `tsi_iterstaggaitken` | Two-way with Aitken acceleration |
| Monolithic | (use `TSI DYNAMIC/MONOLITHIC`) | Simultaneous, tight coupling |

### Thermal Boundary Conditions

- `DESIGN SURF THERMO DIRICH CONDITIONS`: prescribed temperature on surfaces
- `DESIGN SURF THERMO NEUMANN CONDITIONS`: prescribed heat flux on surfaces
- `DESIGN VOL THERMO DIRICH CONDITIONS`: prescribed temperature on volumes
- **Note:** use "THERMO" not "THERMAL" in the section name

### Initial Temperature Field

```yaml
THERMAL DYNAMIC:
  INITIALFIELD: "field_by_function"
  INITFUNCNO: 1
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "100.0 * (1.0 - x)"
```

### Cross-Solver TSI Workflow

1. FEniCS solves heat equation → temperature field
2. 4C TSI receives same thermal BCs → solves coupled problem
3. Compare displacements → cross-validation
4. Verify against analytical: ΔL = α · T_avg · L (for 1D, ν=0)

### Pitfalls

1. **Must use SOLIDSCATRA elements** — standard SOLID or WALL elements cannot couple
2. **CLONING MAP is mandatory** — without it, 4C crashes at initialization
3. **Two LINEAR_SOLVERs needed** — one for thermal, one for structural
4. **THEXPANS units** — must be consistent with temperature units (1/K or 1/°C)
5. **INITTEMP** — the reference temperature for zero thermal strain
6. **3D only** — no 2D TSI elements available in 4C
'''

    @mcp.tool()
    def get_precice_knowledge() -> str:
        """Get knowledge about preCICE coupling and comparison with MCP approach.

        Returns preCICE XML config patterns, adapter ecosystem status, and
        comparison between preCICE and our MCP-orchestrated coupling.
        """
        return '''\
# preCICE Coupling Knowledge

## What is preCICE?

preCICE is an open-source coupling library for partitioned multi-physics.
It provides mesh mapping, data communication, and coupling schemes between
independent solvers via adapters.

## preCICE vs MCP-Orchestrated Coupling

| Aspect | preCICE | MCP Agent (ours) |
|--------|---------|------------------|
| Architecture | Library linked into each solver | External orchestrator (no code changes) |
| Config | XML + adapter code per solver | Python (auto-generated) |
| Mesh mapping | Built-in (RBF, nearest-neighbor) | scipy.griddata + numpy.interp |
| Coupling schemes | Parallel/serial implicit, explicit | DN iteration (MCP-controlled) |
| Performance | Optimized C++ | Python loop (sufficient for demos) |
| 4C support | No adapter exists | Built-in (YAML generation) |
| FEniCS support | fenicsprecice adapter | Built-in (script generation) |
| deal.II support | No official adapter | Built-in (template generation) |
| Setup complexity | Install C++ lib + adapters | pip install (pure Python) |
| Novelty | Established (2016+) | First MCP-based coupling |

## preCICE XML Configuration Pattern

```xml
<precice-configuration>
  <data:scalar name="Temperature" />
  <data:scalar name="Heat-Flux" />

  <mesh name="Mesh-A" dimensions="2">
    <use-data name="Temperature" />
    <use-data name="Heat-Flux" />
  </mesh>

  <participant name="Dirichlet">
    <provide-mesh name="Mesh-A" />
    <write-data name="Temperature" mesh="Mesh-A" />
    <read-data name="Heat-Flux" mesh="Mesh-A" />
  </participant>

  <coupling-scheme:serial-implicit>
    <acceleration:aitken>
      <initial-relaxation value="0.5" />
    </acceleration:aitken>
  </coupling-scheme:serial-implicit>
</precice-configuration>
```

## Adapter Ecosystem

| Solver | preCICE Adapter | Status |
|--------|----------------|--------|
| FEniCS | fenicsprecice | Official, maintained |
| OpenFOAM | openfoam-adapter | Official, widely used |
| deal.II | No official | Community experiments only |
| 4C | None | No adapter available |
| CalculiX | calculix-adapter | Official |
| SU2 | su2-adapter | Official |

## Key Advantage of MCP Approach

Our coupling does NOT require solver-specific adapters. Any solver that
produces VTU output can be coupled. The MCP agent handles:
1. Input generation for each solver
2. Running solvers independently
3. Extracting results from VTU
4. Transferring data between non-matching meshes
5. Controlling the iteration loop
6. Checking convergence

This is fundamentally different from preCICE: we treat solvers as black
boxes orchestrated by an intelligent agent, rather than requiring
library-level integration.

## Installation (if needed)

```bash
# Requires C++ library first
sudo apt install libprecice-dev  # Ubuntu
# Then Python bindings
pip install pyprecice
# FEniCS adapter
pip install fenicsprecice
```
'''

    @mcp.tool()
    def list_physics(solver: str = "") -> str:
        """List all physics problems solvable by available backends.

        Args:
            solver: Optional — filter by backend name. If empty, shows all.
        """
        if solver:
            backend = get_backend(solver)
            if not backend:
                return f"Unknown solver: {solver}"
            backends = [backend]
        else:
            backends = available_backends()

        if not backends:
            return "No backends available."

        lines = []
        for b in backends:
            lines.append(f"## {b.display_name()}")
            for p in b.supported_physics():
                lines.append(f"- **{p.name}**: {p.description}")
                lines.append(f"  Dims: {p.spatial_dims}, Variants: {', '.join(p.template_variants)}")
            lines.append("")

        return "\n".join(lines)
