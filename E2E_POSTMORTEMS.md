# E2E Stress Test Post-Mortems

Collected from 24 end-to-end stress tests run with fresh Claude Code agents using the OASiS MCP server. Each test used the exact prompt from STRESS_TESTS.md with no additional guidance.

---

## Test #1: FEniCS — Lid-Driven Cavity Re=400
**Result: PASS** | ψ_min matches Ghia et al. to 3.6%

Clean first-attempt pass. No significant issues reported.

---

## Test #2: NGSolve — 3D Magnetostatics
**Result: PASS** | B-field physically reasonable

Clean pass. No significant issues.

---

## Test #3: deal.II — Hyperelasticity 30% Compression
**Result: PASS (re-run)** | First attempt: agent bailed to FEniCS due to timeout

**What went wrong:**
1. 600s timeout forced 90-cell mesh (useless results) — agent switched to FEniCS
2. On re-run (no timeout): 5 iterations needed for correct C++
3. `cell->n_faces()` doesn't exist in deal.II 9.1.1 — needed `GeometryInfo<dim>::faces_per_cell`
4. `ComponentMask(dim, false)` with `.set()` — wrong API for 9.1.1
5. Second-order Gmsh elements (Tri6) — deal.II can't read them
6. Triangles not supported in 2D — needed RecombineAll=1
7. Boundary ID collision: `subdivided_hyper_rectangle` defaults all faces to boundary_id=0

**MCP improvements applied:**
- Removed all timeouts from simulation tools
- Added colorize=true pitfall for deal.II
- Added incremental constraint approach for displacement-controlled loading
- Added FEValuesExtractors::Vector requirement for FESystem
- Added Gmsh element order ≠ FE degree warning
- Added UMFPACK recommendation for < 50k DOFs

---

## Test #4: scikit-fem — Eigenvalue Analysis
**Result: PASS** | All 10 eigenvalues match literature

Clean pass. No significant issues.

---

## Test #5: 4C — FSI Turek-Hron
**Result: PASS (re-run, ramp phase)** | First attempt: FUNCT syntax bug, 30x too low velocity

**What went wrong (first run):**
1. FUNCT COMPONENT:0 missing — variable resolution failed silently, velocities 30x too low
2. MPI_ABORT swallowed error messages (no stdbuf)
3. ExodusII block ID 0-indexed vs 1-indexed
4. Shared-node NUMDOF conflict at FSI interface
5. DESIGN FLUID LINE LIFT&DRAG doesn't exist in 2D
6. IO section EVERY_ITERATION not valid
7. CG+SSOR couldn't converge — switched to UMFPACK

**What went wrong (re-run):**
1. WALL TRI3 doesn't exist in 4C
2. ExodusII block ID offset (same issue, now documented with netCDF4 fix)
3. Node duplication needed at FSI interface for monolithic coupling
4. Dirichlet on FSI slave side conflicts
5. IO/RUNTIME VTK OUTPUT/STRUCTURE incompatible with FSI
6. 2D VTK pressure shows NaN (artifact)

**MCP improvements applied:**
- stdbuf -oL for 4C binary
- FUNCT COMPONENT:0 requirement documented
- ExodusII block ID fix with actual netCDF4 patch code
- Shared-node NUMDOF pitfall
- WALL TRI3 removed from element list (doesn't exist)
- FSI separate nodes requirement
- ALE boundary condition rules
- All valid COUPALGO values listed
- FSI slave/master Dirichlet constraint
- IO/RUNTIME VTK OUTPUT/ALE invalid
- 2D VTK NaN artifact for fluid AND porofluid
- Inflow ramp rate stability warning
- THICK parameter semantics (out-of-plane depth = 1.0)

---

## Test #6: DUNE-fem — Turing Patterns
**Result: PASS** | Pattern matches theory, converged to steady state

**What went wrong:**
1. Timeout on first run (N=64, dt=0.005) — DUNE JIT compilation + 10k steps
2. Operator splitting stability — Jacobi splitting unconditionally unstable for f_u > 0
3. No DUNE-fem examples for coupled systems

**Workarounds:**
- Reduced to N=40, dt=0.01
- Gauss-Seidel coupling (solve u first, then v with updated u)

**MCP improvements applied:**
- Added DUNE-fem input guide (API for spaces, forms, coupling, Newton)
- Fixed space/underscore fuzzy matching in examples search

---

## Test #7: Kratos — Cantilever Dynamics
**Result: PASS** | f1=41.82 Hz (analytical 41.78 Hz, 0.12% error)

**What went wrong:**
1. Shear locking with linear hex8 — 18% frequency error
2. Kratos hex20 node ordering is CGNS, not VTK — element size errors
3. assign_vector_by_direction_process crashes for POINT_LOAD
4. Missing echo_level in problem_data
5. Generator tried subprocess.run() — wrong Python env

**MCP improvements applied:**
- Shear locking warning for linear elements (all solvers)
- POINT_LOAD process guidance
- echo_level requirement
- .vtk file support in visualize tool
- run_with_generator file discovery documentation

---

## Test #8: FEniCS — Vortex Shedding Re=100
**Result: PASS** | St=0.1667 (reference 0.164, 1.6% error)

**What went wrong:**
1. dolfinx 0.10 API: gmshio → gmsh, model_to_mesh returns NamedTuple
2. XDMF can't write P2 functions — interpolated to P1
3. No way to test-compile without full simulation run

**MCP improvements applied:**
- Added XDMF support in visualize tool

---

## Test #9: NGSolve — 3D Poisson Convergence Study
**Result: PASS** | P1 rate=2.06, P2 rate=3.02 (textbook perfect)

**What went wrong:**
- Pre-asymptotic regime on coarse meshes (h=0.5 too coarse)

**MCP improvements:** None needed.

---

## Test #10: Stokes BFS — 3-Solver Comparison
**Result: PASS** | FEniCS 0.3503, NGSolve 0.3378, scikit-fem 0.3492

**What went wrong:**
1. FEniCS: dolfinx 0.10 API changes (3 failed runs)
2. NGSolve: umfpack not available — needed solver fallback
3. scikit-fem: Nbfun returns per-element count (not global!) — silent corruption
4. scikit-fem: ElementVector DOF ordering ambiguous — abandoned for scalar bases
5. scikit-fem: asm() quadrature mismatch with mixed bases

**MCP improvements applied:**
- scikit-fem Nbfun vs N pitfall
- scikit-fem ElementVector ordering warning
- scikit-fem intorder matching requirement
- NGSolve solver fallback pattern (pardiso → mumps → umfpack)
- Pressure sign convention warning (FEniCS vs NGSolve)

---

## Test #11: Plate with Hole — 3-Solver SCF Comparison
**Result: PASS** | deal.II 3.038, FEniCS 3.087, 4C 3.129 (all near Kirsch 3.0)

**What went wrong:**
1. deal.II compilation failures (cell->n_faces(), ComponentMask API, Tri6 elements)
2. MCP server disconnected mid-run
3. Dead placeholder code in C++

**MCP improvements applied:**
- deal.II Gmsh element order pitfall (always use first-order geometry)
- deal.II 2D: quads only (RecombineAll=1)
- Plane stress λ* formula as code snippet

---

## Test #12: FEniCS → NGSolve Thermal-Structural Coupling
**Result: PASS** | Max displacement 1.07mm, von Mises peak 942 MPa

**What went wrong:**
1. dolfinx 0.10 API changes (2 failed runs)
2. generate_mesh failed with import error
3. transfer_field only works at interface planes, not whole-domain

**MCP improvements:** None critical (transfer_field limitation noted).

---

## Test #13: DUNE-fem ↔ scikit-fem Poisson DD
**Result: PASS** | 6 iterations, residual 3e-7, 0.74% vs analytical

**What went wrong:**
1. coupled_solve hardcodes FEniCS for domain A — fails with solver_a='dune'
2. MCP client timeout killed connection (DUNE JIT + DD iterations)

**MCP improvements applied:**
- coupled_solve now supports 20 solver-pair combinations (was 2)
- Progress keepalive (ctx.report_progress) on all simulation tools
- coupled_solve description documents supported combinations

---

## Test #14: 4C TSI vs FEniCS Cross-Validation
**Result: PASS** | Machine-precision agreement (rel error 2.3e-15)

**What went wrong:**
1. Missing TYPE Undefined in SOLIDSCATRA element definition
2. Monolithic TSI requires Belos, not UMFPACK
3. Default COUPVARIABLE is Displacement (backwards for heating)
4. Examples truncated — critical info at bottom cut off

**MCP improvements applied:**
- TYPE Undefined requirement documented
- COUPVARIABLE: Temperature for one-way thermal→structural
- Monolithic TSI requires Belos warning
- Volume thermal Dirichlet documented
- Examples truncation limit raised from 8K to 20K chars

---

## Test #15: NGSolve Maxwell → Kratos Thermal-Structural
**Result: PASS** | G_eff matches, temperature rise 0.055 mK, stress 54.2 Pa

**What went wrong:**
1. run_with_generator vs run_simulation confusion for Kratos
2. Kratos "solver" is manual scipy assembly, not real Kratos binary
3. Critic found plane-strain bugs ((1+ν) factor, σ_zz term)

**MCP improvements applied:**
- run_with_generator documentation clarified
- Kratos DEM generator rewritten to use real DEMApplication

---

## Test #16: 4C Terzaghi Consolidation
**Result: PASS (re-run)** | Settlement 0.1%, pressure < 0.4% error

**What went wrong (first run):**
1. WALLQUAD4PORO doesn't exist — needed WALLQ4PORO
2. THICK=0.1 caused 10x force error (should be 1.0)
3. VTK pressure NaN in 2D poro
4. Dynamic waves corrupted transient results
5. α≈0 due to wrong BULKMODULUS

**What the agent did on re-run:**
- Derived that BULKMODULUS >> K_skeleton gives α → 1 (classical Biot)
- Set BULKMODULUS=2e8 (vs K_skeleton=16667) → α=0.9999
- Settlement and pore pressure both match analytical solution

**MCP improvements applied:**
- porous_media registered as PhysicsCapability
- Poro element catalog (WALLQ4PORO, SOLIDH8PORO, etc.)
- THICK pitfall (out-of-plane depth for plane strain)
- 2D VTK NaN expanded to porofluid
- Poro dynamic formulation warning (slow ramp needed)
- Synonym mapping: poroelasticity/poro/consolidation → porous_media

---

## Test #17: 4C Peridynamic Fracture (DCB)
**Result: PASS** | G_eff = G_Ic exact, CMOD 8.5% of LEFM

**What went wrong:**
1. Boundary particles are repulsive-only — cannot apply tensile loads
2. VTK output sections crash particle problems
3. Initial velocity too low (1000 mm/s) — crack didn't propagate
4. meshio can't read particle-only VTU (no cells)

**MCP improvements applied:**
- Boundary particle repulsive-only limitation documented
- PDFIXED per-particle flag documented
- INITIAL_VELOCITY_FIELD documented
- IO/RUNTIME VTK OUTPUT incompatible with particles
- Critical stretch formulas (plane stress vs plane strain)

---

## Test #18: 4C Fluid-Beam Interaction
**Result: PASS** | 4.36mm tip deflection, monotonic growth

**What went wrong:**
1. IO/RUNTIME VTK OUTPUT/ALE crashes 4C
2. Dirichlet on FSI slave interface conflicts
3. Newton divergence from aggressive inflow ramp

**MCP improvements applied:**
- FSI slave interface Dirichlet constraint
- IO/RUNTIME VTK OUTPUT/ALE invalid
- All valid COUPALGO values listed
- Inflow ramp rate stability warning

---

## Test #19: Kratos DEM Particle Packing
**Result: PASS** | Packing fraction consistent

Agent used standalone scipy script (not real Kratos DEM binary). MCP DEM generator was subsequently rewritten to use real DEMApplication.

---

## Test #20: NGSolve Maxwell Cavity Eigenvalues
**Result: PASS** | All 6 modes match to < 10⁻⁶ relative error

**What went wrong:**
1. nograds=True degraded accuracy 1-3% (wrong for eigenvalue problems)
2. PINVIT + Poisson projection gave completely wrong results
3. ArnoldiSolver shift=1 too far from eigenvalue range

**MCP improvements applied:**
- nograds pitfall: WRONG for eigenvalue problems (only for source problems)
- ArnoldiSolver shift guidance (set near expected eigenvalue range)
- Examples search OR-matching for multi-keyword queries

---

## Test #21: FEniCS 3D Thick-Walled Cylinder (Lamé)
**Result: PASS** | 0.42% L2 error vs corrected Lamé solution

**What went wrong:**
1. Wrong analytical formula from research (common online error in Lamé)
2. dolfinx 0.10 API changes (2 failed runs)

**MCP improvements:** None needed (dolfinx API already documented).

---

## Test #22: Open-Ended "Solve the Heat Equation"
**Result: PASS** | Agent auto-selected FEniCS, L2 error = 7e-15

**What went wrong:**
- visualize returned empty (VTXWriter produces .bp files, not .vtu)

**MCP improvements applied:**
- Added .bp (ADIOS2/VTX) file support in visualize

---

## Test #23: Open-Ended "Contact Mechanics"
**Result: PASS** | Agent auto-selected 4C, mortar penalty contact

Clean pass using 4C tutorial mesh. No workarounds needed.

---

## Test #24: Kratos FSI CoSimulation
**Result: PASS** | 14 Aitken iterations, 13.99mm deflection (1.3% vs beam theory)

**What went wrong:**
1. Penalty Stokes diverged the FSI coupling (p = -λ·div(u) amplifies mesh deformation noise)
2. Workaround: switched to mixed P1/P1 + PSPG formulation
3. Initial Aitken ω=0.5 too aggressive — reduced to 0.1 with delayed activation

Agent used manual scipy implementation mirroring CoSimulation concepts (not actual Kratos CoSimulationApplication).

---

## Summary of All MCP Improvements Applied

### Infrastructure
- Removed all simulation timeouts
- Progress keepalive (ctx.report_progress) on run_simulation, run_with_generator, visualize
- post_vtu fire-and-forget (no longer blocks MCP server)
- stdbuf -oL for 4C binary (prevents lost error messages)
- NaN-safe JSON in visualize summary
- .vtk, .xdmf, .bp file support in visualize
- Skip .pvtu files in visualize
- Visualize groups multi-field outputs (structure/fluid/ale)
- Examples OR-matching for multi-keyword queries
- Examples truncation limit 8K → 20K chars
- Fuzzy keyword matching (spaces/underscores/hyphens)
- Source build detection (get_env_with_source_root)
- All vendor agent instruction files (CLAUDE.md, AGENTS.md, .cursorrules, etc.)

### 4C Knowledge
- FUNCT COMPONENT:0 requirement
- ExodusII block ID fix with netCDF4 code
- WALL TRI3 doesn't exist
- FSI separate nodes requirement
- ALE BC rules
- All COUPALGO values
- FSI slave/master Dirichlet constraint
- IO/RUNTIME VTK OUTPUT/ALE invalid
- 2D VTK NaN for fluid and porofluid
- THICK semantics (out-of-plane depth = 1.0)
- Inflow ramp rate stability
- PD loading mechanisms (PDFIXED, INITIAL_VELOCITY_FIELD, repulsive-only boundary)
- VTK incompatible with particle problems
- Critical stretch plane stress vs plane strain
- TSI: TYPE Undefined, COUPVARIABLE, monolithic requires Belos
- Large inline YAML: use ExodusII for >200 nodes
- Poro element catalog, dynamic formulation warning

### deal.II Knowledge
- colorize=true for subdivided_hyper_rectangle
- Incremental constraint approach
- FEValuesExtractors::Vector for FESystem
- Gmsh element order ≠ FE degree
- UMFPACK for < 50k DOFs
- 2D quads only (RecombineAll=1)
- Plane stress λ* formula

### Kratos Knowledge
- Shear locking warning
- POINT_LOAD process (assign_vector_variable_process)
- echo_level requirement
- damp_factor_m naming
- DEM generator rewritten to use real DEMApplication

### NGSolve Knowledge
- nograds pitfall for eigenvalue problems
- ArnoldiSolver shift guidance
- Solver fallback (pardiso → mumps → umfpack)
- Pressure sign convention

### scikit-fem Knowledge
- Nbfun vs N (per-element vs global DOF count)
- ElementVector ordering warning
- intorder matching for mixed bases
- Pressure sign convention

### DUNE-fem
- Input guide (API for spaces, forms, coupling, Newton)

### Coupling
- coupled_solve: 20 solver-pair combinations (was 2)
- NGSolve, scikit-fem, DUNE-fem generators for domain A and B
