# OASiS — Master Plan

**Updated:** 2026-03-16 (comprehensive knowledge overhaul)

## Vision

A **general-purpose, solver-agnostic MCP server** that makes an AI agent as knowledgeable as an expert user of EVERY supported FEM code. Not a wrapper. Not a tutorial replayer. A genuinely knowledgeable multi-solver interface that:

1. **Knows everything each code can do** — every element type, every physics, every solver option, every material model, every boundary condition, every coupling mechanism
2. **Can operate any code** — generate correct input, run simulations, extract and compare results across solvers
3. **Can develop and extend codes** — read source, modify templates, add new capabilities, recompile
4. **Can couple codes intelligently** — orchestrate multi-solver workflows, transfer fields, validate across solvers

---

## Current State (2026-03-16)

### Infrastructure
| Metric | Value |
|--------|-------|
| Backends | **8** (7 available: FEniCSx, deal.II, 4C, NGSolve, scikit-fem, Kratos, DUNE-fem; FEBio needs binary) |
| Tests | **80 passed**, 0 skipped, 0 failures |
| MCP tools | **43+** (discovery, simulation, knowledge, visualization, benchmark, coupling, developer) |
| Coupling modes | **7** (heat_dd, poisson_dd, one_way TSI, tsi_dd, poisson_dd_study, l_bracket_tsi, heat_dd_precice) |
| preCICE | **Built from source** (v3.1.2 at /opt/precice, pyprecice installed) |
| Python source | **24,559 lines** across 51+ files |

### Cross-Solver Poisson Benchmark (7 solvers agree to 99.9%)
| Solver | max(u) | Element |
|--------|--------|---------|
| FEniCSx | 0.07361 | P1 tri |
| deal.II | 0.07373 | Q1 quad |
| 4C | 0.07373 | Q1 quad |
| NGSolve | 0.07364 | P1 tri |
| scikit-fem | 0.07373 | Q1 quad |
| Kratos | 0.07361 | P1 tri |
| DUNE-fem | 0.07373 | Q1 quad |

---

## What Must Be Done (Priority Order)

### Phase A: Deep Knowledge Encoding (CURRENT FOCUS)

Each backend Python file must be a **comprehensive reference** for that code. Not scattered knowledge in separate files — the backend itself must contain or directly serve ALL knowledge. Each backend needs:

1. **Complete physics catalogue** — every PDE type the code can solve, with working template
2. **Complete element catalogue** — every element type with description and use case
3. **Complete solver catalogue** — every solver option, preconditioner, and configuration
4. **Complete material catalogue** — every constitutive law with parameters
5. **Complete BC catalogue** — every boundary condition type
6. **Coupling capabilities** — what can couple with what, and how
7. **Pitfalls and best practices** — real expert knowledge, not generic tips

**Per-backend targets:**

| Backend | Target Physics | Target Knowledge Depth |
|---------|---------------|----------------------|
| **FEniCSx** | 15+ (all from dolfinx demos + jsdokken tutorials) | 14 Basix families, PETSc/SLEPc solvers, UFL reference |
| **deal.II** | 20+ (from 97 step tutorials) | All FE types, all grid generators, AMR, hp, matrix-free, parallel |
| **4C** | 40 problem types | 120+ materials, 130+ conditions, all coupling algorithms |
| **NGSolve** | 15+ (from i-tutorials) | All spaces (H1/HCurl/HDiv/L2/Facet), DG/HDG, GPU, parallel |
| **scikit-fem** | 12+ (from 50 examples) | 50+ element types, assembly patterns, mixed methods |
| **Kratos** | 10+ applications | Structural, fluid, FSI, contact, DEM, MPM, CoSimulation |
| **DUNE-fem** | 10+ (tutorials + dune-fem-dg) | UFL interop with FEniCS, DG methods, VEM, adaptivity |

### Phase B: Cross-Solver Capabilities

After deep knowledge is encoded, demonstrate:
1. **N-solver benchmark** — same problem on ALL backends, compare results
2. **Solver recommendation** — given a problem description, rank backends by suitability
3. **Cross-solver coupling** — orchestrate multi-solver workflows (already have 7 modes)
4. **Field transfer** — extract from one solver's VTU, inject into another's input

### Phase C: Developer Mode

Each backend must support source code modification:
1. **Template extension** — add new physics templates to any backend
2. **Source modification** — for C++ backends (4C, deal.II), read/modify/recompile
3. **Knowledge update** — when a new capability is added, update knowledge automatically

### Phase D: Deployment & Paper

1. **README** with installation, quick start, architecture diagram
2. **pyproject.toml** with all dependencies
3. **Paper figures** — architecture, benchmark table, coupling workflow, relaxation study
4. **Reproducible benchmarks** — scripts that regenerate all paper results

---

## Architecture

```
User → LLM (Claude, GPT, etc.) → MCP Protocol → OASiS
                                                    │
                    ┌───────────────────────────────┤
                    │         │         │         │         │         │         │
                FEniCSx    deal.II     4C      NGSolve  scikit-fem  Kratos   DUNE-fem
               (dolfinx)    (C++)   (YAML+MPI) (Netgen)  (scipy)   (JSON)    (UFL)
                Python     CMake      mpirun    Python   Python    Python    Python

Each backend provides:
├── generate_input(physics, variant, params) → solver-specific input
├── run(input, work_dir) → JobHandle with VTU output
├── get_knowledge(physics) → deep domain knowledge
├── supported_physics() → complete capability list
├── validate_input(content) → error checking
└── get_result_files(job) → VTU/VTK paths for PyVista
```

---

## Competitive Position

| Feature | MCP-SIM | FeaGPT | MooseAgent | **OASiS** |
|---------|---------|--------|------------|-------------------|
| Real FEM solvers | No | Yes (1) | Yes (1) | **Yes (7+)** |
| Multi-solver | No | No | No | **Yes (7 backends)** |
| Cross-validation | No | No | No | **Yes (7-solver Poisson)** |
| Standard MCP | No | No | No | **Yes** |
| Open source | Yes | No | Yes | **Yes** |
| Deep knowledge | Limited | Unknown | Vector DB | **54 physics, 84 templates, 51 knowledge entries** |
| Coupling | No | No | No | **7 modes + preCICE** |
| Developer mode | No | No | No | **Yes (read/modify/extend source)** |
