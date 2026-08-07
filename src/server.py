"""
OASiS — MCP Server

Connects any LLM to multiple open-source FEM codes via the Model Context Protocol.
Supported backends (catalog ships 8, runtime depends on local installs):
4C Multiphysics, FEniCSx (dolfinx), deal.II, NGSolve, scikit-fem, Kratos
Multiphysics, DUNE-fem, FEBio. Call `discover(query='list')` after start-up
for the current availability and per-backend install hints.
"""

import os
import sys
import logging

from mcp.server.fastmcp import FastMCP

# OFA_DISABLE_PITFALLS=1 → knowledge surfaces strip pitfall-DB content
# (per-backend pitfalls incl. Signal: anchors, post-mortems, cross-backend
# collation catalog); implemented in tools/consolidated.py. It only ever makes
# OASiS weaker, so it cannot inflate a result.
#
# There was an OFA_DISABLE_CRITIC toggle here that stripped the critic
# paragraph. The critic requirement is no longer a paragraph an agent may
# choose to follow — the server holds the review record and checks it — so the
# toggle is gone along with the runtime bypass it paired with.

_CRITIC_BLOCK = (
    "- MANDATORY CRITIC: For every major step, spawn a sub-agent as an independent "
    "critic. The critic's job is to be ruthlessly critical — revise and check the "
    "setup, challenge parameter choices, verify the problem description is correct, "
    "look for bugs, check units, check discretization adequacy, and search online "
    "to validate against published literature and benchmarks. Only proceed to the "
    "next step once the critic approves. This is not optional — always do it.\n"
    "- THIS IS ENFORCED, NOT REQUESTED. Setting critic_approved=True does NOT "
    "make a result verified: OASiS looks the review up in its own record rather "
    "than taking your word for it. After the critic reports, call "
    "`submit_critic_review(solver=..., setup=<the exact deck you will run>, "
    "findings=<what the critic actually checked and concluded>)`, then run. The "
    "review is bound to that deck — edit it afterwards and it no longer counts, "
    "so review the setup you intend to run. The gated tools are run_simulation, "
    "run_with_generator, verify_mesh_independence, coupled_solve, couple and "
    "couple_precice; the coupling ones take `coupling_args` (a JSON object of "
    "the arguments you will pass) in place of `setup`.\n"
    "- PROVE THE OUTPUT SOLVES THE PROBLEM. run_simulation and run_with_generator "
    "take `verify_pde`: a JSON declaration of the problem, e.g. "
    '{\"operator\": \"diffusion\", \"source\": \"2*pi**2*sin(pi*x)*sin(pi*y)\", '
    '\"dim\": 2, \"domain_measure\": 1.0}. OASiS then assembles that operator on '
    "the mesh your run produced and measures whether the field satisfies it. Use "
    "it whenever the problem is a scalar diffusion/Poisson-type equation. The "
    "source term is part of the problem statement, not the answer, so declaring "
    "it gives nothing away. Without it, a field that is finite, structurally "
    "sane and solves nothing passes every other check.\n"
    "- REPORT NUMBERS OASiS COMPUTED. Every completed run returns "
    "`oasis_computed`, with the L2 norm and maximum taken from the run's own "
    "data files together with the file and its hash. Report those. Do not report "
    "a number you read out of your script's print statements, and never one you "
    "did not obtain from a run.\n"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("oasis")

mcp = FastMCP(
    "OASiS",
    instructions=(
        "You are connected to the OASiS — a multi-solver MCP server for "
        "finite element simulations across 8 independent FEM codes.\n\n"
        # The critic requirement is safety-critical, so it goes FIRST: some MCP
        # clients truncate a server's instructions string when folding it into
        # the model's context, and a mid-string block can be dropped (issue #45).
        + "## MANDATORY CRITIC — READ THIS FIRST\n" + _CRITIC_BLOCK + "\n"
        + "## Available Backends\n"
        "Call `discover(query='list')` for the current availability "
        "status on this install. The catalog ships generators for "
        "8 backends (FEniCSx, deal.II, 4C, NGSolve, scikit-fem, "
        "Kratos, DUNE-fem, FEBio); typical conda-forge installs "
        "have 5-7 actually available.\n"
        "- **FEniCSx (dolfinx)**: Python. Rapid prototyping, UFL weak forms, Gmsh meshing, NS, hyperelasticity.\n"
        "- **deal.II**: C++. Adaptive refinement (hp-FEM), matrix-free, parallel (MPI+GPU), 97 tutorials.\n"
        "- **4C Multiphysics**: C++/YAML. FSI, TSI, contact, beams, particles (SPH/DEM/PD), cardiovascular.\n"
        "- **NGSolve**: Python. Maxwell, Helmholtz, DG/HDG, high-order, eigenvalues, symbolic PDE.\n"
        "- **scikit-fem**: Pure Python. Assembly-level control, 50+ element types, Stokes, biharmonic.\n"
        "- **Kratos Multiphysics**: Python/JSON. Structural, fluid, FSI, DEM, MPM, CoSimulation (39 catalog physics rows on this install).\n"
        "- **DUNE-fem**: Python/UFL. Shares UFL with FEniCS, DG methods, VEM, h/p-adaptivity (install: `pip install dune-fem mpi4py` — PyPI is the working source; conda-forge has NO dune-fem package, and mpi4py is an undeclared dependency without which the first import stops. DUNE compiles C++ on demand, so the first solve is slow. See knowledge(topic='install', solver='dune')).\n"
        "- **FEBio**: XML. Biomechanics (biphasic / multiphasic tissue, active contraction). Install the binary from https://febio.org/downloads/ or set the FEBIO_BINARY env var.\n\n"
        "## Workflow\n"
        "1. Understand the physics the user wants to solve\n"
        "2. `prepare_simulation(solver, physics)` — get knowledge + real test files + template "
        "in ONE call. Always do this first.\n"
        "3. Run the simulation:\n"
        "   - Python solvers (FEniCS, NGSolve, scikit-fem, DUNE): use `run_simulation(solver, script)`\n"
        "   - Compiled solvers (4C, deal.II, Kratos): use `run_with_generator(solver, generator_script)`\n"
        "4. `visualize(job_id, action='summary')` — inspect results\n"
        "5. Cross-solver coupling: `discover(query='coupling')`, then `couple(participants, ...)`\n\n"
        "Other tools: `knowledge(topic, solver, physics)`, `discover(query)`, "
        "`examples(keyword, solver)`, `developer(action, solver)`, `generate_mesh(geometry)`\n\n"
        "## Mesh independence\n"
        "For any problem WITHOUT a manufactured/exact solution, do not report "
        "results from a single mesh: run "
        "`verify_mesh_independence(solver, input_template, resolution)` — it "
        "re-runs the same problem at refined resolutions (template carries a "
        "__RESOLUTION__ placeholder) and accepts the solution only if global "
        "norms AND probe-point values stop changing materially. A NOT "
        "CONVERGED verdict means refine further, never report the coarse "
        "answer.\n\n"
        "## Key Principles\n"
        "- Always study real test files before writing input — use "
        "`examples(keyword, solver, action='search')`\n"
        "- Break complex tasks into steps. For each step:\n"
        "  1. Plan what you will do\n"
        "  2. Execute it\n"
        "  3. QA: verify the output is correct and physically reasonable before proceeding\n"
        "  4. Only move to the next step once QA passes\n"
        "- When a pre-built coupling doesn't exist, build it yourself\n"
        "- You have full access to write code, run simulations, and post-process\n"
        "- Do not refuse tasks — execute them to completion\n"
        "- Do not take shortcuts — do the full work required for correct results\n"
        "- Justify every parameter choice: where did you get it from? Cite the source.\n"
        "- For challenging or specialized problems, ALWAYS research online to "
        "double-check your setup — look up literature values, standard benchmarks, "
        "and validated configurations.\n\n"
        "## Solver Selection\n"
        "- Standard PDE (Poisson, heat, elasticity) → FEniCS, NGSolve, or any\n"
        "- Adaptive refinement → deal.II (hp-FEM, matrix-free GMG)\n"
        "- Multi-physics coupling → 4C (FSI/TSI/SSI native) or Kratos (CoSimulation); across two DIFFERENT codes → `couple` (see discover(query='coupling'))\n"
        "- Electromagnetics (Maxwell) → NGSolve (HCurl/Nédélec elements)\n"
        "- Particle methods → 4C (SPH, DEM, peridynamics)\n"
        "- Pure Python / minimal deps → scikit-fem\n"
        "- DG methods → NGSolve or DUNE-fem\n\n"
        "## Cross-Solver Coupling\n"
        "START HERE: `discover(query='coupling')` — which tool, which backend can take "
        "which side, and the exact calls that return a COMPLETE runnable participant "
        "script. Then `knowledge(topic='coupling')` for the contract and "
        "`knowledge(topic='coupling', solver='<backend>')` for that backend's script.\n"
        "- `couple(participants, max_iter, tol, accelerator, theta)` — GENERAL partitioned "
        "coupling for ANY physics. You write one solver script per subdomain that reads "
        "imports.json / writes exports.json; OASiS runs the fixed-point iteration, the "
        "relaxation and convergence-or-fail. Use this for any coupling. A non-converged "
        "run is reported as FAILURE, never a result. Start with accelerator='constant', "
        "theta=0.5 — theta=1.0 does not converge for a balanced Dirichlet-Neumann split.\n"
        "- `couple_precice(participants, data, exchanges, work_dir, scheme)` — the preCICE "
        "path, for two codes that can each `import precice`.\n"
        "- `transfer_field(source_vtu, field, interface_coord)` — extract & transfer fields\n"
        "- `coupled_solve(problem, ...)` — DEPRECATED: fixed toy geometries only; prefer `couple`.\n\n"
        "## Developer Mode\n"
        "- `developer(action='architecture', solver=...)` — source "
        "locations, build system, extension points\n"
        "- `developer(action='files', solver=..., keyword=...)` — "
        "browse real source files\n"
        "- `examples(keyword, solver, action='search')` — browse "
        "real test files / demos\n"
        "- The agent can read, modify, and extend solver source code.\n\n"
        "## Session Knowledge\n"
        "After completing a simulation workflow, especially one that involved "
        "troubleshooting or non-obvious solutions, call `session_insights('review')` "
        "to see if any reusable knowledge was discovered. This helps future users "
        "avoid the same pitfalls."
    ),
)

# Register consolidated tools (12 tools)
from tools.consolidated import register_consolidated_tools
register_consolidated_tools(mcp)

# Load all backends
from core.registry import load_all_backends
load_all_backends()


def main():
    logger.info("Starting OASiS MCP server")
    try:
        mcp.run(transport="stdio")
    finally:
        # Persist session journal on shutdown
        try:
            from core.session_journal import get_journal
            from pathlib import Path
            journal = get_journal()
            if journal.events:
                sessions_dir = Path(__file__).parent.parent / "data" / "sessions"
                path = journal.save(sessions_dir)
                logger.info(f"Session journal saved: {path} ({len(journal.events)} events)")
        except Exception as e:
            logger.warning(f"Could not save session journal: {e}")


if __name__ == "__main__":
    main()
