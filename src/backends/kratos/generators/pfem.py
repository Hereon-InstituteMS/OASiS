"""Kratos PFEM (Particle Finite Element Method) generators and knowledge.

Covers free-surface flows, sloshing, fluid-structure with topology changes.
Applications: PfemFluidDynamicsApplication, PfemApplication, PFEM2Application.
"""


def _pfem_fluid_2d(params: dict) -> str:
    """FORMAT TEMPLATE — PFEM free-surface fluid simulation."""
    return '''\
"""PFEM free-surface flow — Kratos PfemFluidDynamicsApplication"""
import json
try:
    import KratosMultiphysics as KM
    import KratosMultiphysics.PfemFluidDynamicsApplication
    print("PfemFluidDynamicsApplication available")
    summary = {"note": "PfemFluidDynamicsApplication available",
               "capabilities": ["free_surface", "sloshing", "dam_break", "wave_breaking",
                                "fluid_structure_topology_change"]}
except ImportError:
    print("PfemFluidDynamicsApplication not installed")
    print("NOTE: PFEM apps are NOT published on PyPI as of "
          "Kratos 10.4.2 — pip install does not work. "
          "Build Kratos from source with "
          "-DPFEM_FLUID_DYNAMICS_APPLICATION=ON "
          "and -DDELAUNAY_MESHING_APPLICATION=ON.")
    summary = {"note": "not installed"}
with open("results_summary.json", "w") as f: json.dump(summary, f, indent=2)
'''


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
                        'Signal: post-processed quantity (max displacement, integrated flux, pressure) disagrees with analytic / textbook reference by 10-100%.',
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
    },
    "pfem2": {
        "description": "PFEM2 (streamline integration) for two-phase flows",
        "application": "PFEM2Application",
        "capabilities": ["two_phase_flow", "interface_tracking", "bubble_dynamics"],
    },
}

def _pfem_solid_2d(params: dict) -> str:
    """PFEM for large-deformation solids."""
    return '''\
"""PFEM Solid — Kratos PfemSolidMechanicsApplication"""
import json
try:
    import KratosMultiphysics as KM
    import KratosMultiphysics.PfemSolidMechanicsApplication
    print("PfemSolidMechanicsApplication available")
    summary = {"note": "Available", "capabilities": ["large_deformation", "cutting", "forming"]}
except ImportError:
    print("PfemSolidMechanicsApplication not installed")
    summary = {"note": "not installed"}
with open("results_summary.json", "w") as f: json.dump(summary, f, indent=2)
'''

def _pfem2_2d(params: dict) -> str:
    """PFEM2 two-phase flow."""
    return '''\
"""PFEM2 Two-Phase — Kratos PFEM2Application"""
import json
try:
    import KratosMultiphysics as KM
    import KratosMultiphysics.PFEM2Application
    print("PFEM2Application available")
    summary = {"note": "Available", "capabilities": ["two_phase", "interface_tracking"]}
except ImportError:
    print("PFEM2Application not installed")
    summary = {"note": "not installed"}
with open("results_summary.json", "w") as f: json.dump(summary, f, indent=2)
'''

GENERATORS = {
    "pfem_fluid_2d": _pfem_fluid_2d,
    "pfem_solid_2d": _pfem_solid_2d,
    "pfem2_2d": _pfem2_2d,
}
