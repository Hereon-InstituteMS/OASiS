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


def resolve_search_keywords(solver: str, physics: str) -> list[str]:
    """Return a prioritised list of filename-substring keywords to
    probe when looking for upstream test/demo files for the given
    (solver, physics) pair.

    Audit 2026-06-01: this logic was previously baked into
    _find_reference_test_files, so the MCP `examples` tool (which
    has its own file walk) couldn't reach NGSolve demos via aliases
    — examples('hyperelasticity', solver='ngsolve') walked for the
    literal substring 'hyperelasticity' and missed nonlin.py.
    Factoring this out keeps the two LLM-facing surfaces in sync.
    """
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

    return keywords


def _find_reference_test_files(solver: str, physics: str) -> str:
    """Find real test files from a solver's test suite as reference.

    Returns a formatted block with file paths and content previews,
    or empty string when no local demos are available.
    """
    # Empty / whitespace-only physics would match every filename
    # via the substring-of-everything pattern that bit
    # prepare_simulation, examples('search'), and discover(
    # 'recommend'). Callers in this module already guard before
    # reaching here, but the helper is publicly importable
    # from src/tools/knowledge.py — guard it here too so a
    # future caller can't reintroduce the bug. (Audit
    # 2026-06-01.)
    if not physics or not physics.strip():
        return ""

    test_dirs = discover_test_dirs()
    solver_key = solver.lower()
    test_dir = test_dirs.get(solver_key)
    if not test_dir or not test_dir.is_dir():
        return ""

    ext = ("*.4C.yaml" if solver_key in ("fourc", "4c")
           else "*.cc" if solver_key == "dealii"
           else "*.py")

    keywords = resolve_search_keywords(solver, physics)
    # Drop any empty / whitespace-only keyword the resolver
    # produced (a defensive de-dup against the same class of
    # bug in resolve_search_keywords' alias map).
    keywords = [k for k in keywords if k and k.strip()]
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
    def get_coupling_knowledge(solver: str = "") -> str:
        """Complete knowledge for partitioned multi-code coupling via `couple`.

        With no solver: the participant contract, the InterfaceData shapes, how
        the driver iterates and relaxes, the interface-flux sign convention,
        which side each backend can take, and the failure modes.
        With a solver name: a COMPLETE runnable participant script for that
        backend plus the traps specific to it.
        """
        from tools.coupling_knowledge import coupling_knowledge
        return coupling_knowledge(solver)

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

## Participant Adapter Pattern (generic — every coupled code follows this)

Each participant is a small script wrapping ONE solver that exchanges the coupling
fields every time window. Fill in the `<SOLVER>` specifics yourself:

```python
import precice, numpy as np
p = precice.Participant("<PARTICIPANT_NAME>", "precice-config.xml", 0, 1)
# coordinates of YOUR coupling-interface points (the surface/edge shared with the partner):
coords = np.array([[x0, y0], ...])
vid = p.set_mesh_vertices("<MESH_NAME>", coords)
if p.requires_initial_data():
    p.write_data("<MESH_NAME>", "<WRITE_FIELD>", vid, initial_outgoing)
p.initialize()
while p.is_coupling_ongoing():
    # --- IMPLICIT coupling: save state at the start of a window so the window can be redone ---
    if p.requires_writing_checkpoint():
        saved_state = your_solver.save_state()        # deep-copy solver state (fields, time)
    dt = p.get_max_time_step_size()
    incoming = p.read_data("<MESH_NAME>", "<READ_FIELD>", vid, dt)   # field FROM the partner
    # --- advance YOUR solver by dt, using `incoming` as a boundary condition ---
    outgoing = ...                                                   # field YOUR solver produces
    p.write_data("<MESH_NAME>", "<WRITE_FIELD>", vid, outgoing)
    p.advance(dt)
    # --- IMPLICIT coupling: if not yet converged, REDO the window from the checkpoint ---
    if p.requires_reading_checkpoint():
        your_solver.restore_state(saved_state)
    # else: window converged -> commit and move to the next time window
p.finalize()
```

CHECKPOINTS ARE MANDATORY FOR IMPLICIT SCHEMES. With `serial-implicit`/`parallel-implicit`
(needed when explicit Dirichlet–Neumann diverges — values blow up to infinity), preCICE
SUB-ITERATES each time window to convergence, so each participant MUST handle the checkpoint
calls above: `requires_writing_checkpoint()` (save state) and `requires_reading_checkpoint()`
(restore state and redo the window). If you ignore these, an implicit run never advances and
preCICE reports the window unconverged. For `*-explicit` schemes these calls always return
False, so the same loop works unchanged. Add convergence acceleration
(`<acceleration:aitken>` or IQN) under the coupling-scheme to make implicit converge fast.

CRITICAL LAUNCH: both participant scripts must run AT THE SAME TIME — they handshake through
preCICE and each blocks at `initialize()` until the other connects. Start each as its own
concurrent process (NOT one after the other, and NOT a single `mpirun` over both files), or
let the `couple_precice` tool launch every participant for you. Set `LD_LIBRARY_PATH` to the
preCICE lib and match the `pyprecice` version to `libprecice`.

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
