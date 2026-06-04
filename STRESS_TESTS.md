# OASiS — Stress Tests

These tests verify the agent's ability to set up, run, and validate simulations across all backends. Each test exposes different capabilities and potential gaps. Findings from each test should be used to improve the **general-purpose** infrastructure — not to fine-tune for specific examples.

---

## Single-Solver Benchmarks

### 1. FEniCS — Lid-Driven Cavity (Navier-Stokes)
```
Solve the lid-driven cavity problem at Re=400 using FEniCS and visualize the vortex structure in ParaView.
```

### 2. NGSolve — 3D Magnetostatics
```
Run a 3D magnetostatics problem in NGSolve: a permanent magnet inside a steel housing. Show the B-field distribution.
```

### 3. deal.II — Large-Deformation Hyperelasticity
```
Simulate a cantilever beam with Neo-Hookean hyperelastic material under large deformation using deal.II. Apply 30% compression.
```

### 4. scikit-fem — Eigenvalue Analysis
```
Run an eigenvalue analysis on an L-shaped membrane in scikit-fem. Find the first 10 eigenfrequencies and compare against known values.
```

### 5. 4C — Fluid-Structure Interaction
```
Simulate fluid-structure interaction of a flexible flag behind a cylinder in 4C.
```

### 6. DUNE-fem — Reaction-Diffusion Turing Patterns
```
Solve a transient reaction-diffusion system (Turing patterns) on a unit square using DUNE-fem. Show the pattern evolution.
```

### 7. Kratos — Structural Dynamics (Cantilever Impact)
```
Simulate a 3D cantilever beam subjected to a sudden tip load using Kratos Multiphysics. Track the tip displacement over time and compare the oscillation frequency against the analytical first natural frequency.
```

### 8. FEniCS — Transient Navier-Stokes (Vortex Shedding)
```
Simulate 2D flow past a circular cylinder at Re=100 using FEniCS. Run long enough to capture periodic vortex shedding and measure the Strouhal number. Compare against the accepted value St≈0.164.
```

### 9. NGSolve — 3D Poisson Convergence Study
```
Solve the Poisson equation with a known analytical solution on a 3D unit cube using NGSolve. Run an h-convergence study with 4 mesh refinement levels and verify optimal L2 convergence rate for P1 and P2 elements.
```

---

## Cross-Solver Validation

### 10. Stokes Flow — 3-Solver Comparison
```
Solve Stokes flow in a backward-facing step on FEniCS, NGSolve, and scikit-fem. Compare the reattachment length.
```

### 11. Stress Concentration — 3-Solver Comparison
```
Run linear elasticity on a plate with a circular hole under uniaxial tension. Compare stress concentration factor across deal.II, FEniCS, and 4C.
```

---

## Multi-Solver Coupling

### 12. FEniCS → NGSolve Thermal-Structural
```
Solve heat conduction on an L-domain with FEniCS, transfer the temperature field to NGSolve, and solve thermoelasticity there. Show the thermal stress distribution.
```

### 13. DUNE-fem ↔ scikit-fem Domain Decomposition
```
Run a Poisson problem with domain decomposition: left half on DUNE-fem, right half on scikit-fem. Iterate until convergence.
```

### 14. 4C TSI vs FEniCS Cross-Validation
```
Simulate a heated steel beam in 4C (TSI one-way) and independently verify the thermal expansion using FEniCS. Compare displacements.
```

### 15. NGSolve Maxwell → Kratos Structural
```
Model electromagnetic wave scattering in NGSolve around an obstacle, then use the Joule heating field as a thermal load in a Kratos structural analysis.
```

---

## Advanced / Research-Level

### 16. 4C — Poroelasticity (Terzaghi)
```
Run a poroelasticity consolidation problem in 4C (Terzaghi's problem) and verify against the analytical solution.
```

### 17. 4C — Peridynamic Fracture Verification
```
Simulate crack propagation in a double-cantilever beam using 4C peridynamics and compare the energy release rate against LEFM predictions.
```

### 18. 4C — Fluid-Beam Interaction
```
Set up a fluid-beam interaction problem in 4C: flow around a slender elastic beam. Monitor the beam tip displacement over time.
```

### 19. Kratos — DEM Particle Packing
```
Simulate gravity-driven packing of 500 spherical particles into a cylindrical container using Kratos DEM. Measure the final packing fraction and compare against the random close packing limit (~0.64).
```

### 20. NGSolve — Maxwell Eigenvalue (Cavity Resonance)
```
Compute the first 6 electromagnetic resonant frequencies of a 3D rectangular cavity using NGSolve Nédélec elements. Compare against the analytical TM/TE mode frequencies.
```

### 21. FEniCS — 3D Hyperelasticity with Gmsh Mesh
```
Generate a 3D thick-walled cylinder mesh with Gmsh, then solve internal pressure loading with Neo-Hookean material in FEniCS. Compare the radial displacement against the analytical Lamé solution at small strain.
```

---

## Solver Selection & Agent Intelligence

### 22. Open-Ended: "Solve the Heat Equation"
```
Solve the heat equation on a unit square with T=1 on the left, T=0 on the right, and zero-flux top/bottom. Pick the best solver and verify against the analytical solution.
```

### 23. Open-Ended: "I Need Contact Mechanics"
```
Simulate two elastic blocks being pressed together with contact. Use whichever solver is most appropriate.
```

### 24. Kratos — FSI Co-Simulation
```
Set up a simple fluid-structure interaction problem in Kratos using the CoSimulation application: flow in a channel with a flexible wall segment. Monitor the wall deflection.
```

---

## How to Use These Tests

1. Run each prompt in a **fresh interactive Claude terminal** (`cd /path/to/oasis && claude`)
2. Let the agent work through the sub-agent workflow (research → setup → critic → execution)
3. Note what works and what fails
4. After the test completes, ask the agent these debrief questions:
   - "What went wrong and what workarounds did you have to use?"
   - "Which MCP tools were useful and which were missing or unhelpful?"
   - "What information did you have to look up online that should have been available through the MCP?"
   - "What parameters did you struggle with and why?"
   - "If you had to do this again, what would you do differently?"
5. Feed the debrief answers back to the development agent for general-purpose improvements
6. Never fine-tune for the specific test case — fix the underlying infrastructure issue
7. After fixing, re-run the same test to verify the fix works
8. Purge conversation history between tests: `./clear_history.sh`
