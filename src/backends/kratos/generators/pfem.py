"""Kratos PFEM (Particle Finite Element Method) generators and knowledge.

Covers free-surface flows, sloshing, fluid-structure with topology changes.
Applications: PfemFluidDynamicsApplication, PfemApplication, PFEM2Application.
"""


# NOTE (2026-06-26 honesty audit): the three PFEM generators
# (_pfem_fluid_2d, _pfem_solid_2d, _pfem2_2d) were availability-probe stubs
# (import-check + {"note": ...}, no solver run) — the KNOWLEDGE pitfalls
# below even self-document them as such. The PFEM applications
# (PfemFluidDynamicsApplication, PfemSolidMechanicsApplication,
# PFEM2Application) are not published on PyPI and are NOT importable in the
# installed Kratos stack, so 'pfem_fluid', 'pfem_solid' and 'pfem2' have been
# removed from the generator registry and from
# KratosBackend.supported_physics(). KNOWLEDGE retained for reference.


KNOWLEDGE = {
    "pfem_fluid": {
        "description": "Particle FEM for free-surface flows (dam break, sloshing, wave impact)",
        "application": (
            "PfemFluidDynamicsApplication — NOT on PyPI as of "
            "Kratos 10.4.2; pip install KratosPfemFluidDynamics"
            "Application yields 'No matching distribution'. "
            "Build from source with -DPFEM_FLUID_DYNAMICS_"
            "APPLICATION=ON + -DDELAUNAY_MESHING_APPLICATION=ON."
        ),
        "elements": {
            "2D": ["TwoStepUpdatedLagrangianVPImplicitNodallyIntegratedElement2D3N",
                   "TwoStepUpdatedLagrangianVPImplicitFluidElement2D3N"],
            "3D": ["TwoStepUpdatedLagrangianVPImplicitNodallyIntegratedElement3D4N"],
        },
        "capabilities": ["free_surface_tracking", "remeshing", "alpha_shape_boundary_detection",
                         "fluid_structure_with_topology_changes"],
        "solver_types": ["two_step_v_p_solver (velocity-pressure split)"],
        "pitfalls": [
                        '[Integration] PFEM applications '
                        '(PfemFluidDynamicsApplication, '
                        'DelaunayMeshingApplication, '
                        'PfemSolidMechanicsApplication, '
                        'PFEM2Application) are NOT published on '
                        'PyPI as of Kratos 10.4.2. The pip-install '
                        'hint in some legacy templates fails with '
                        '"ERROR: No matching distribution found for '
                        'KratosPfemFluidDynamicsApplication". Build '
                        'Kratos from source with '
                        '-DPFEM_FLUID_DYNAMICS_APPLICATION=ON + '
                        '-DDELAUNAY_MESHING_APPLICATION=ON to enable. '
                        "Signal: pip install of any KratosPfem*"
                        "Application package returns 'No matching "
                        "distribution found' from the index. "
                        "(Verified empirically 2026-06-01.)",
                        '[Numerical] Requires DelaunayMeshingApplication for remeshing '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Physics] Alpha-shape parameter controls free-surface detection (default ~1.25) '
                        'Signal: the post-processed VtkOutput .post.bin shows the integrated_flux / max_displacement / PRESSURE channels disagreeing with analytic / textbook reference by 10-100%.',
                        '[Numerical] Time step must be small enough for remeshing stability '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Output: particles move, so mesh changes every step '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                    ],
    },
    "pfem_solid": {
        "description": "PFEM for large-deformation solid mechanics with remeshing",
        "application": "PfemSolidMechanicsApplication",
        "capabilities": ["large_deformation_solids", "cutting", "forming", "erosion"],
        "pitfalls": [
            "[Integration] Catalog template is an availability-"
            "probe STUB, not a solver. It imports "
            "KratosMultiphysics + the relevant Application "
            "module, prints whether the import succeeded, and "
            "writes a 1-line results_summary.json. No "
            "ModelPart / AnalysisStage / SolverWrapper is "
            "scaffolded — run_simulation on this template "
            "reports 'Available' or 'not installed' but does "
            "NOT solve anything. Signal: the emitted script is "
            "< 30 lines, contains no Model.CreateModelPart() "
            "and no AnalysisStage subclass; results_summary.json "
            "has a single 'note' key set to 'Available' or 'not "
            "installed'. For an actual solve, scaffold a full "
            "ProjectParameters.json + MDPA mesh + AnalysisStage. "
            "(Verified empirically 2026-06-01 — catalog audit.)",
        ],
    },
    "pfem2": {
        "description": "PFEM2 (streamline integration) for two-phase flows",
        "application": "PFEM2Application",
        "capabilities": ["two_phase_flow", "interface_tracking", "bubble_dynamics"],
        "pitfalls": [
            "[Integration] Catalog template is an availability-"
            "probe STUB, not a solver — same pattern as "
            "pfem_solid: imports Kratos + the Application "
            "module, prints availability, writes a 1-line "
            "summary. No actual PFEM2 streamline integration "
            "or two-phase interface tracking is set up. "
            "Signal: emitted script < 30 lines, "
            "results_summary.json has only a 'note' key. "
            "(Verified empirically 2026-06-01.)",
        ],
    },
}

# Empty: no PFEM application is installable in this Kratos stack; the prior
# generators (pfem_fluid_2d, pfem_solid_2d, pfem2_2d) were no-solve probe
# stubs (removed — see honesty-audit note at top of file).
GENERATORS = {}
