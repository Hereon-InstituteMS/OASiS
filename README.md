<p align="center">
  <img src="logo/logo_w_text.png" alt="OASiS — open-source multi-physics and multi-code framework for verified computer simulations" width="640"/>
</p>

# OASiS

**O**pen-source **A**gent for **Si**mulation across multiple FEM **S**olvers — an open-source multi-physics and multi-code framework for verified computer simulations.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20388035.svg)](https://doi.org/10.5281/zenodo.20388035)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

OASiS is a **Model Context Protocol (MCP) server** that connects AI coding agents to **eight independent finite element codes** (FEniCSx, deal.II, 4C Multiphysics, NGSolve, scikit-fem, Kratos Multiphysics, DUNE-fem, FEBio). Any MCP-compatible AI tool (Claude Code, Cursor, Windsurf, GitHub Copilot) can **operate** solvers, **couple** them across codes, **develop** new solver capabilities, and **verify** simulation correctness — all through one protocol.

## Key Numbers

| Metric | Value |
|--------|-------|
| FEM backends | **8 working** (FEniCSx, deal.II, 4C, NGSolve, scikit-fem, Kratos, DUNE-fem, FEBio) |
| MCP tools | **13** consolidated tools |
| Physics types | **180+** across all backends |
| Coupling modes | **7** (heat DD, Poisson DD, one-way TSI, two-way TSI, relaxation study, L-bracket, preCICE) |
| Supported solver pairs | **20** for domain decomposition (any Python solver + any backend) |
| Tests | **400+ passing** |
| E2E stress tests | **24 completed** (24 pass) |
| Cross-backend collation topics | **26** topics, **45+** pitfalls (the actual differentiator) |
| Signal: retrieval-anchor coverage | **100%** across all 8 backends |

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Hereon-InstituteMS/OASiS.git && cd OASiS
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 2. Install solver backends (pick what you need)

```bash
# pip-installable (any combination)
pip install ngsolve scikit-fem dune-fem
pip install KratosMultiphysics KratosStructuralMechanicsApplication

# FEniCSx (requires conda)
conda create -n fenics -c conda-forge fenics-dolfinx
# or: pip install fenics-dolfinx (if available for your platform)

# deal.II (system package)
sudo apt install libdeal.ii-dev   # Ubuntu/Debian

# 4C Multiphysics (build from source — see 4C documentation)

# FEBio (binary download)
# https://febio.org/downloads/  — set FEBIO_BINARY env var or place in PATH
```

### 3. Connect to your AI tool

The MCP server runs as a stdio process. Each AI tool has its own way to register it.

**Claude Code** (recommended — supports sub-agent spawning for critic workflow):

```bash
# From the project root:
claude mcp add oasis \
  .venv/bin/python -- -m server \
  -e PYTHONPATH=src \
  -e PYVISTA_OFF_SCREEN=true
```

Or manually create `.claude/settings.json` in the project directory:

```json
{
  "mcpServers": {
    "oasis": {
      "type": "stdio",
      "command": ".venv/bin/python",
      "args": ["-m", "server"],
      "cwd": "src",
      "env": {
        "PYTHONPATH": "src",
        "PYVISTA_OFF_SCREEN": "true"
      }
    }
  }
}
```

**Cursor**:

Go to Settings > MCP Servers > Add Server, then enter:
- Name: `oasis`
- Command: `/path/to/OASiS/.venv/bin/python`
- Args: `-m server`
- Working directory: `/path/to/OASiS/src`
- Environment: `PYTHONPATH=src`, `PYVISTA_OFF_SCREEN=true`

**Windsurf**:

Add to your Windsurf MCP configuration (Settings > MCP):
```json
{
  "oasis": {
    "command": "/path/to/OASiS/.venv/bin/python",
    "args": ["-m", "server"],
    "cwd": "/path/to/OASiS/src",
    "env": { "PYTHONPATH": "src", "PYVISTA_OFF_SCREEN": "true" }
  }
}
```

**OpenAI Codex / any MCP client**:

The server speaks standard MCP over stdio. Point your client at:
```
command: .venv/bin/python -m server
cwd: src/
env: PYTHONPATH=src
```

### 4. Verify

```bash
source .venv/bin/activate
cd src && python -m server  # should start without errors (Ctrl+C to stop)

# Run tests
cd .. && pytest tests/ -v
```

### 5. Optional: configure solver source access

To enable developer mode (source browsing, modification, rebuild), add solver paths to your MCP environment. See `.claude/settings.json.example` for the template.

## Environment Variables

Solver backends are auto-detected from pip/conda installations. For compiled solvers or source-level development access, set these in your MCP settings:

| Variable | Purpose | Example |
|----------|---------|---------|
| `FOURC_ROOT` | 4C source tree | `/home/user/4C` |
| `FOURC_BINARY` | 4C binary path | `/home/user/4C/build/4C` |
| `DEALII_ROOT` | deal.II source | `/opt/dealii` |
| `FENICS_ROOT` | FEniCSx source | `/home/user/dolfinx` |
| `NGSOLVE_ROOT` | NGSolve source | `/home/user/ngsolve` |
| `KRATOS_ROOT` | Kratos source | `/home/user/Kratos` |
| `DUNE_ROOT` | DUNE-fem source | `/home/user/dune-fem` |
| `SKFEM_ROOT` | scikit-fem source | `/home/user/scikit-fem` |
| `FEBIO_BINARY` | FEBio binary | `/home/user/febio4/bin/febio4` |

When a `*_ROOT` variable is set, the agent can browse, modify, and rebuild the solver source code (developer mode).

## Architecture

```
User --> AI Agent (any MCP client) --> MCP Protocol --> OASiS
                                                          |
   +-------------+---------+------+--------+--------+--------+--------+-------+
   |             |         |      |        |        |        |        |       |
FEniCSx     deal.II      4C   NGSolve   skfem   Kratos    DUNE    FEBio
(Python)     (C++)    (YAML) (Python) (Python) (JSON)   (Python)  (XML)
```

## Three Modes of Operation

### 1. Operate: Run Simulations
The agent generates solver-specific input, runs the simulation, and validates results.

### 2. Couple: Multi-Solver Workflows
Domain decomposition, field transfer, and multi-physics coupling across different FEM codes.

### 3. Develop: Extend Solver Capabilities
When a solver lacks a needed feature, the agent can read source code, implement the missing piece, rebuild, and test.

## MCP Tools

| Tool | Purpose |
|------|---------|
| `prepare_simulation` | Knowledge + examples + template in ONE call |
| `run_simulation` | Execute Python-based solvers (FEniCS, NGSolve, scikit-fem, DUNE) |
| `run_with_generator` | Generate input + run compiled solvers (4C, deal.II, Kratos) |
| `knowledge` | Physics knowledge, pitfalls, materials, coupling docs, cross-backend collation |
| `discover` | List solvers, check availability, capabilities matrix |
| `examples` | Real test files and templates from solver test suites |
| `coupled_solve` | Cross-solver domain decomposition (20 solver pair combinations) |
| `transfer_field` | Extract and transfer fields between solver outputs |
| `visualize` | Field statistics, plots, automated validation |
| `generate_mesh` | Gmsh mesh generation (L-domain, plate with hole, channel) |
| `developer` | Source architecture, file browsing, extension points |
| `session_insights` | Review session patterns, contribute knowledge back |
| `reload_catalog` | Hot-reload backend catalogs without restarting the MCP server |

## The Cross-Backend Differentiator

OASiS ships **26 cross-backend collation topics** with **45+ pitfalls** that describe gotchas at the *delta* between two or more solvers — content that has no equivalent in any individual solver's docs:

- units, element node ordering, linear-elastic semantics, Dirichlet BC enforcement
- restart/checkpoint compatibility, MPI launch idioms, element-type naming
- time-integration defaults, solver-tolerance defaults, contact formulation defaults
- output-format conventions, integration-order defaults, boundary-tag semantics
- plasticity return-mapping, turbulence model defaults, material orientation
- frequency-unit conventions, mesh-quality thresholds, stress-measure conventions
- periodic-BC implementation, damping-convention defaults, timestamp output
- initial-condition interpolation, frame-of-reference BC
- **linear-solver defaults** (direct vs iterative dispatch, GMRES restart, tolerance scaling)
- **nonlinear convergence criteria** (relative vs absolute, energy-norm, line-search, failure-signal paths)

Reach via `knowledge(topic='cross_backend')` or filter by physics: `knowledge(topic='cross_backend', physics='units')`.

## Tested Benchmarks

These benchmarks have been run as end-to-end stress tests with a fresh AI agent. Each prompt was given verbatim to the agent with no additional guidance.

### Single-Solver Benchmarks (9/9 pass)

| # | Prompt | Solver | Result |
|---|--------|--------|--------|
| 1 | `Solve the lid-driven cavity problem at Re=400 using FEniCS and visualize the vortex structure in ParaView.` | FEniCS | PASS |
| 2 | `Run a 3D magnetostatics problem in NGSolve: a permanent magnet inside a steel housing. Show the B-field distribution.` | NGSolve | PASS |
| 3 | `Simulate a cantilever beam with Neo-Hookean hyperelastic material under large deformation using deal.II. Apply 30% compression.` | deal.II | PASS |
| 4 | `Run an eigenvalue analysis on an L-shaped membrane in scikit-fem. Find the first 10 eigenfrequencies and compare against known values.` | scikit-fem | PASS |
| 5 | `Simulate fluid-structure interaction of a flexible flag behind a cylinder in 4C.` | 4C | PASS |
| 6 | `Solve a transient reaction-diffusion system (Turing patterns) on a unit square using DUNE-fem. Show the pattern evolution.` | DUNE-fem | PASS |
| 7 | `Simulate a 3D cantilever beam subjected to a sudden tip load using Kratos Multiphysics. Track the tip displacement over time and compare the oscillation frequency against the analytical first natural frequency.` | Kratos | PASS |
| 8 | `Simulate 2D flow past a circular cylinder at Re=100 using FEniCS. Run long enough to capture periodic vortex shedding and measure the Strouhal number. Compare against the accepted value St~0.164.` | FEniCS | PASS |
| 9 | `Solve the Poisson equation with a known analytical solution on a 3D unit cube using NGSolve. Run an h-convergence study with 4 mesh refinement levels and verify optimal L2 convergence rate for P1 and P2 elements.` | NGSolve | PASS |

### Cross-Solver Validation (2/2 pass)

| # | Prompt | Solvers | Result |
|---|--------|---------|--------|
| 10 | `Solve Stokes flow in a backward-facing step on FEniCS, NGSolve, and scikit-fem. Compare the reattachment length.` | 3 solvers | PASS |
| 11 | `Run linear elasticity on a plate with a circular hole under uniaxial tension. Compare stress concentration factor across deal.II, FEniCS, and 4C.` | 3 solvers | PASS |

### Multi-Solver Coupling (4/4 pass)

| # | Prompt | Solvers | Result |
|---|--------|---------|--------|
| 12 | `Solve heat conduction on an L-domain with FEniCS, transfer the temperature field to NGSolve, and solve thermoelasticity there. Show the thermal stress distribution.` | FEniCS + NGSolve | PASS |
| 13 | `Run a Poisson problem with domain decomposition: left half on DUNE-fem, right half on scikit-fem. Iterate until convergence.` | DUNE + scikit-fem | PASS |
| 14 | `Simulate a heated steel beam in 4C (TSI one-way) and independently verify the thermal expansion using FEniCS. Compare displacements.` | 4C + FEniCS | PASS |
| 15 | `Model electromagnetic wave scattering in NGSolve around an obstacle, then use the Joule heating field as a thermal load in a Kratos structural analysis.` | NGSolve + Kratos | PASS |

### Advanced (7/7 pass)

| # | Prompt | Solver | Result |
|---|--------|--------|--------|
| 16 | `Run a poroelasticity consolidation problem in 4C (Terzaghi's problem) and verify against the analytical solution.` | 4C | PASS (re-run: settlement 0.1%, pressure < 0.4% error, agent derived correct BULKMODULUS→alpha mapping) |
| 17 | `Simulate crack propagation in a double-cantilever beam using 4C peridynamics and compare the energy release rate against LEFM predictions.` | 4C | PASS (G_eff = G_Ic exact, CMOD 8.5% of LEFM) |
| 18 | `Set up a fluid-beam interaction problem in 4C: flow around a slender elastic beam. Monitor the beam tip displacement over time.` | 4C | PASS (4.36mm tip deflection, monotonic growth) |
| 19 | `Simulate gravity-driven packing of 500 spherical particles into a cylindrical container using Kratos DEM. Measure the final packing fraction and compare against the random close packing limit (~0.64).` | Kratos | PASS |
| 20 | `Compute the first 6 electromagnetic resonant frequencies of a 3D rectangular cavity using NGSolve Nédélec elements. Compare against the analytical TM/TE mode frequencies.` | NGSolve | PASS (all 6 modes match to <10⁻⁶ relative error) |
| 21 | `Generate a 3D thick-walled cylinder mesh with Gmsh, then solve internal pressure loading with Neo-Hookean material in FEniCS. Compare the radial displacement against the analytical Lamé solution at small strain.` | FEniCS | PASS (0.42% L2 error vs Lamé, P2 curved elements) |

### Solver Selection & Agent Intelligence (1/1 pass)

| # | Prompt | Solver Chosen | Result |
|---|--------|---------------|--------|
| 22 | `Solve the heat equation on a unit square with T=1 on the left, T=0 on the right, and zero-flux top/bottom. Pick the best solver and verify against the analytical solution.` | FEniCS (auto-selected) | PASS (L2 error = 7e-15, machine precision) |
| 23 | `Simulate two elastic blocks being pressed together with contact. Use whichever solver is most appropriate.` | 4C (auto-selected) | PASS (mortar penalty contact, 10 load steps, stress concentration 1.66x) |
| 24 | `Set up a simple fluid-structure interaction problem in Kratos using the CoSimulation application: flow in a channel with a flexible wall segment. Monitor the wall deflection.` | Kratos | PASS (14 Aitken iterations, 13.99mm deflection, 1.3% vs beam theory) |

## Contributing

We welcome contributions that improve the **general-purpose** capabilities of OASiS. The key principle:

**Every improvement must benefit ALL simulations, not be fine-tuned for specific examples.**

### How to contribute

1. **Report agent behavior** — Run any of the benchmark prompts (or your own) with your AI tool and solver setup. Report:
   - Which AI tool you used (Claude Code, Cursor, Windsurf, etc.)
   - The exact prompt you gave
   - What worked and what didn't
   - The agent's retrospective (ask the debrief questions below)

2. **Improve solver knowledge** — Add pitfalls, element catalogs, or API documentation that would help any agent set up simulations correctly. Focus on things the agent had to discover by trial-and-error.

3. **Add solver backends** — Implement the `SolverBackend` interface for a new FEM code.

4. **Expand coupling** — Add script generators for new solver pairs in `src/tools/coupling.py`.

### What NOT to contribute

- Benchmark-specific parameter databases (the agent should research these per-problem)
- Model-specific templates (e.g., "Turek-Hron FSI2 template") — these are fine-tuning
- Hardcoded paths or machine-specific configurations

### Debrief questions (ask after every stress test)

```
- "What went wrong and what workarounds did you have to use?"
- "Which MCP tools were useful and which were missing or unhelpful?"
- "What information did you have to look up online that should have been available through the MCP?"
- "What parameters did you struggle with and why?"
- "If you had to do this again, what would you do differently?"
```

Feed the answers back as general-purpose improvements to the MCP knowledge and tools.

## AI Tool Compatibility

The MCP server works with any MCP-compatible AI tool. Agent instructions are provided in multiple formats:

| File | AI Tool |
|------|---------|
| `CLAUDE.md` | Claude Code |
| `AGENTS.md` | Cross-tool standard |
| `.cursorrules` | Cursor |
| `.windsurfrules` | Windsurf |
| `.github/copilot-instructions.md` | GitHub Copilot |

## License

MIT

## Citation

If you use OASiS in your research, please cite:

```bibtex
@article{oasis2026,
  title={OASiS: an open-source multi-physics and multi-code framework for verified computer simulations},
  author={Hermann, Alexander and Shojaei, Arman and Scheider, Ingo and Cyron, Christian},
  year={2026},
}
```
