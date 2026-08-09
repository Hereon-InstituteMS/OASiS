<p align="center">
  <img src="logo/logo_w_text.png" alt="OASiS — open-source multi-physics and multi-code framework for verified computer simulations" width="640"/>
</p>

# OASiS

**O**pen-source **A**gent for **Si**mulation across multiple FEM **S**olvers

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20543501.svg)](https://doi.org/10.5281/zenodo.20543501)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> [!NOTE]
> Development happens on the experimental fork <https://github.com/alhermann/OASiS>; stable releases live here.

## What it is

OASiS is an **agentic simulation system**: it lets any AI language model operate **eight professional finite-element and multiphysics codes** — FEniCSx, deal.II, 4C Multiphysics, NGSolve, scikit-fem, Kratos Multiphysics, DUNE-fem, and FEBio — through one common interface. You describe the physics problem in plain language; the AI agent picks a solver, writes correct input for it, runs the simulation, checks the result against analytical solutions or published benchmarks, and shows you the outcome. Under the hood, OASiS gives the model curated solver knowledge (pitfalls, working examples, element catalogs) and verified execution, so the agent does not have to rediscover each code's quirks by trial and error.

## What it can do

- **Classic PDEs** — Poisson and other elliptic problems, heat conduction, diffusion
- **Solid mechanics** — linear elasticity, hyperelasticity, plasticity, contact, eigenfrequency analysis
- **Nonlinear problems** — large deformation, nonlinear material laws, Newton-solver setup with sensible defaults
- **Flow** — Stokes and Navier-Stokes (lid-driven cavity, vortex shedding, channel flow)
- **Transport** — transient heat, reaction-diffusion (e.g. Turing patterns), convection
- **Electromagnetics** — Maxwell, cavity resonances, magnetostatics (NGSolve)
- **Particle methods** — SPH, DEM, peridynamics (4C, Kratos)
- **Cross-code coupling** — split a problem across two different solvers and iterate to convergence: thermo-mechanics, fluid-structure interaction, domain decomposition, and even multi-paradigm couplings such as **FEM ↔ DSMC** (continuum solid + rarefied-gas particle code via the experimental SPARTA backend), plus a bridge to the preCICE coupling library
- **Mesh generation** — Gmsh-based geometries (L-domain, plate with hole, channel, custom)
- **Visualization & checking** — field statistics, plots, automated sanity checks
- **Convergence studies** — h-refinement studies with error norms against analytical solutions
- **Solver development** — when a code lacks a feature, the agent can browse its source, implement the change, rebuild, and test

## How to use it — with any model

OASiS speaks the **Model Context Protocol (MCP)** — the standard way to plug tools into AI assistants. That gives you two ways in.

### Option A: connect to an MCP-capable app (Claude Desktop, Claude Code, Cursor, ...)

If you already use an AI coding app with a subscription, this is the zero-extra-cost path: the app brings the model, OASiS brings the solvers.

**Claude Code** (from the project root):

```bash
claude mcp add oasis \
  .venv/bin/python -- -m server \
  -e PYTHONPATH=src \
  -e PYVISTA_OFF_SCREEN=true
```

**Claude Desktop** — add to `claude_desktop_config.json` (Settings > Developer > Edit Config):

```json
{
  "mcpServers": {
    "oasis": {
      "command": "/path/to/OASiS/.venv/bin/python",
      "args": ["-m", "server"],
      "cwd": "/path/to/OASiS/src",
      "env": { "PYTHONPATH": "/path/to/OASiS/src", "PYVISTA_OFF_SCREEN": "true" }
    }
  }
}
```

**Cursor / Windsurf / any MCP client** — same idea: command `/path/to/OASiS/.venv/bin/python`, args `-m server`, working directory `/path/to/OASiS/src`, env `PYTHONPATH=/path/to/OASiS/src` and `PYVISTA_OFF_SCREEN=true`. The server speaks standard MCP over stdio.

Then just ask, e.g.: *"Solve the Poisson equation on a unit square with a known analytical solution and verify the convergence rate."*

### Option B: drive it from your own code with ANY API model (LangGraph)

If you prefer an API key over an app subscription — OpenAI, OpenRouter, Anthropic, a local vLLM or Ollama server — this repository ships a working **LangGraph agent harness** in [`langgraph_eval/`](langgraph_eval/). It attaches every OASiS tool to a LangGraph agent via `langchain-mcp-adapters` and works with any OpenAI-compatible endpoint. The shipped scaffold (`langgraph_eval/agent.py`) builds the complete agent — model client, host-side tools, OASiS MCP tools — as a reusable function you call from your own driver script; pointing it at any provider is a one-line change to the model client:

```python
import os
from langchain_openai import ChatOpenAI

# OpenRouter (any hosted model)
llm = ChatOpenAI(base_url="https://openrouter.ai/api/v1",
                 api_key=os.environ["OPENROUTER_API_KEY"],
                 model="anthropic/claude-sonnet-4.5")

# OpenAI:  base_url default,             api_key=os.environ["OPENAI_API_KEY"]
# local vLLM:  base_url="http://localhost:8000/v1", api_key="not-needed"
# Ollama:      base_url="http://localhost:11434/v1", api_key="not-needed"
```

The harness spawns the OASiS server as an MCP subprocess (see `_load_oasis_mcp_tools()` in `langgraph_eval/agent.py` for the exact launch configuration), gives the model file/shell/web tools alongside the solver tools, and even lets it spawn a sub-agent to criticize its own setup before running. Install the extra dependencies with `pip install -r langgraph_eval/requirements-langgraph.txt` (a separate virtualenv, e.g. `.venv-lg`, is recommended).

### Subscription or API key?

- **Subscription** (Claude Pro/Max, Cursor, ...): use Option A. No API key, no per-token billing; the app you already pay for becomes a simulation front-end.
- **API key**: use Option B. Full control over which model runs the agent loop — including free/cheap models via OpenRouter or fully local open-weight models via vLLM/Ollama. Costs scale with usage.

## Install

### Prerequisites by backend tier

| Tier | Backends | Install effort |
|------|----------|----------------|
| pip-installable | NGSolve, scikit-fem, Kratos | `pip install ...` — seconds |
| conda-installable | FEniCSx, DUNE-fem | `conda create -c conda-forge ...` — minutes |
| system package | deal.II | `sudo apt install libdeal.ii-dev` |
| compiled from source | 4C Multiphysics | CMake build, see 4C docs |
| binary download | FEBio | <https://febio.org/downloads/>, set `FEBIO_BINARY` |

You do **not** need all eight — install what you need; `discover(query='list')` reports what is available and how to get the rest.

### Quickstart: five lines to a first Poisson solve

```bash
git clone https://github.com/Hereon-InstituteMS/OASiS.git && cd OASiS
python3 -m venv .venv && source .venv/bin/activate
pip install -e . && pip install scikit-fem
claude mcp add oasis .venv/bin/python -- -m server -e PYTHONPATH=src -e PYVISTA_OFF_SCREEN=true
claude "Solve the Poisson equation -Δu = 1 on a unit square with u=0 on the boundary using scikit-fem, and report the max value."
```

### Backend examples

```bash
# pip-installable (any combination)
pip install ngsolve scikit-fem
pip install KratosMultiphysics KratosStructuralMechanicsApplication

# FEniCSx and DUNE-fem (conda-forge is the supported path)
conda create -n fenics -c conda-forge fenics-dolfinx
conda create -n ofa-dune -c conda-forge dune-fem

# deal.II (Ubuntu/Debian)
sudo apt install libdeal.ii-dev
```

On macOS, install deal.II from the official `deal.II.app` bundle and point
`DEAL_II_DIR` at its `Contents/Resources/Libraries`. If a deal.II build then
fails inside `<complex>`/`<cmath>` (an Xcode SDK header clash baked into the
app bundle, not an OASiS issue), set the SDK sysroot consistently:

```bash
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
```

Conda envs are auto-detected (envs whose name contains `fenics`/`dolfinx`
or `dune` are preferred). For non-standard layouts point OASiS at the
interpreter explicitly — both discovery **and** execution use it:

```bash
export FENICS_PYTHON=/path/to/env/bin/python   # or FENICS_CONDA_PREFIX=/path/to/env
export DUNE_PYTHON=/path/to/env/bin/python     # or DUNE_CONDA_PREFIX=/path/to/env
```

### Verify

```bash
source .venv/bin/activate
cd src && python -m server   # should start without errors (Ctrl+C to stop)
cd .. && pytest tests/ -v    # run the test suite
```

### Optional: solver source access (developer mode)

Set `*_ROOT` environment variables (`FOURC_ROOT`, `DEALII_ROOT`, `FENICS_ROOT`, `NGSOLVE_ROOT`, `KRATOS_ROOT`, `DUNE_ROOT`, `SKFEM_ROOT`) in your MCP settings to let the agent browse, modify, and rebuild solver source code. For compiled binaries set `FOURC_BINARY` / `FEBIO_BINARY`. See `.claude/settings.json.example`.

## Architecture

```
You --> AI model (any app or API) --> MCP --> OASiS (src/server.py)
                                                |
   +-------------+---------+------+--------+--------+--------+--------+-------+
   |             |         |      |        |        |        |        |       |
FEniCSx     deal.II      4C   NGSolve   skfem   Kratos    DUNE    FEBio
(Python)     (C++)    (YAML) (Python) (Python) (JSON)   (Python)  (XML)
```

- **Server** — `src/server.py`, a stdio MCP server; backends load as plugins via `src/core/registry.py`.
- **Backends** — one package per code under `src/backends/`, each with a physics catalog and input **generators** for the compiled codes (plus an experimental SPARTA DSMC backend for rarefied-gas problems).
- **Curated knowledge** — per-backend pitfalls, element catalogs, installed-version API references, and a cross-backend collation layer, served through `knowledge` and `prepare_simulation`.
- **Coupling orchestrator** — `src/core/coupling_driver.py`: each participant is a black box that reads `imports.json` / writes `exports.json` under an explicit contract; OASiS validates the contract, runs the fixed-point iteration with Aitken relaxation, and reports convergence-or-failure. A non-converged run is never presented as a result.
- **preCICE bridge** — `src/core/precice_config.py` + the `couple_precice` tool: generates a valid `precice-config.xml` for standard scenarios and verifies the coupling end-to-end.

### Tools the agent gets

| Tool | Purpose |
|------|---------|
| `discover` | List solvers, availability, capabilities matrix |
| `prepare_simulation` | Knowledge + real examples + template in one call — always the first step |
| `run_simulation` | Execute Python-based solvers (FEniCSx, NGSolve, scikit-fem, DUNE-fem) |
| `run_with_generator` | Generate input + run compiled solvers (4C, deal.II, Kratos) |
| `verify_mesh_independence` | Heuristic mesh-refinement study for problems without an exact solution: re-runs at refined resolutions, compares global norms + probe values, converged/not-converged verdict |
| `knowledge` | Physics knowledge, pitfalls, materials, coupling docs, cross-backend collation |
| `examples` | Real test files from the solvers' own test suites |
| `couple` | General partitioned coupling for any physics (contract + Aitken relaxation) |
| `couple_precice` | preCICE-based coupling: config generation and end-to-end run |
| `coupled_solve` | Legacy fixed-geometry domain decomposition (deprecated — prefer `couple`) |
| `transfer_field` | Extract and transfer fields between solver outputs |
| `generate_mesh` | Gmsh mesh generation |
| `visualize` | Field statistics, plots, automated validation |
| `developer` | Solver source architecture, file browsing, extension points |
| `session_insights` | Review session patterns, contribute reusable knowledge back |
| `setup_backend` / `rediscover_backends` / `reload_catalog` | Install hints, re-scan for new solvers, hot-reload catalogs |

## Methodology

Three principles shape everything in this repository:

- **General knowledge, never problem constants.** Curated entries describe a *class* of failure and its general fix. Templates use placeholders, not the dimensions of any particular benchmark — anything that would anchor the agent to one geometry or one paper's parameters is rejected. The agent researches problem-specific values per task; the knowledge layer only removes tooling friction.
- **Verification is not optional.** Every workflow should end with a check against an analytical solution, a published benchmark, or an independent solver. The coupling orchestrator enforces this in code: contract violations and non-converged iterations are failures, never results. The server also asks the agent to have an independent critic review each setup before running.
- **Operate and develop.** The agent both runs the solvers and, when a feature is missing, extends them: developer mode exposes the source tree, and the agent reads, modifies, rebuilds, and re-tests the code itself. Fixes flow back into the knowledge layer in general form.

## Contributing

Contributions are welcome. One rule above all: **every improvement must benefit all simulations**, not be fine-tuned for a specific example. Solver pitfalls, element catalogs, new backends, and new coupling generators are great contributions; benchmark-specific parameter databases and model-specific templates are not. New capabilities are typically developed and stress-tested on the experimental fork (<https://github.com/alhermann/OASiS>) before they land here.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use OASiS in your research, please cite the archived release:

```bibtex
@software{oasis2026,
  title  = {OASiS: an open-source multi-physics and multi-code framework for verified computer simulations},
  author = {Hermann, Alexander and Shojaei, Arman and Scheider, Ingo and Cyron, Christian},
  year   = {2026},
  doi    = {10.5281/zenodo.20543501},
  url    = {https://github.com/Hereon-InstituteMS/OASiS}
}
```
