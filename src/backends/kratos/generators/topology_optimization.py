"""Kratos topology optimization generators and knowledge.

Application: TopologyOptimizationApplication.
"""


def _topology_opt_2d(params: dict) -> str:
    """FORMAT TEMPLATE — Topology optimization (SIMP/level-set)."""
    return '''\
"""Topology Optimization — Kratos TopologyOptimizationApplication"""
import json
try:
    import KratosMultiphysics as KM
    import KratosMultiphysics.TopologyOptimizationApplication
    print("TopologyOptimizationApplication available")
    summary = {"note": "TopologyOptimizationApplication available",
               "capabilities": ["SIMP", "level_set", "compliance_minimization",
                                "stress_constrained", "multi_material"]}
except ImportError:
    print("TopologyOptimizationApplication not installed")
    summary = {"note": "not installed"}
with open("results_summary.json", "w") as f: json.dump(summary, f, indent=2)
'''


KNOWLEDGE = {
    "topology_optimization": {
        "description": "Topology optimization: SIMP, level-set, compliance/stress objectives",
        "application": "TopologyOptimizationApplication",
        "methods": {
            "SIMP": "Solid Isotropic Material with Penalization (density-based)",
            "level_set": "Level-set topology optimization",
        },
        "objectives": ["compliance_minimization", "stress_minimization",
                       "multi_objective", "frequency_maximization"],
        "constraints": ["volume_fraction", "stress_limit", "displacement_limit"],
        "pitfalls": [
                        '[Integration] Requires StructuralMechanicsApplication as dependency '
                        "Signal: RuntimeError 'KeyError' from JSON parsing OR 'SubModelPart not found' / 'Property ID ... missing' during AnalysisStage.Initialize.",
                        '[Numerical] SIMP penalization factor p=3 is standard '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Filter radius needed to avoid checkerboard patterns '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Mesh-dependent results without proper filtering '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                    ],
    },
}

GENERATORS = {"topology_optimization_2d": _topology_opt_2d}
