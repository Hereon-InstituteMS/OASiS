"""Kratos topology optimization generators and knowledge.

Application: TopologyOptimizationApplication.
"""


# NOTE (2026-06-26 honesty audit): the previous _topology_opt_2d generator
# emitted an availability-probe stub that only import-checked
# TopologyOptimizationApplication and wrote {"note": "not installed"} with
# no solver run. TopologyOptimizationApplication is NOT importable in the
# installed Kratos stack. The stub generator and its
# 'topology_optimization_2d' registry entry have been removed;
# 'topology_optimization' is no longer advertised in
# KratosBackend.supported_physics(). The KNOWLEDGE block below is retained
# as a reference-only entry.


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

# Empty: no runnable topology-optimization generator —
# TopologyOptimizationApplication is not installable in this Kratos stack
# (see honesty-audit note above).
GENERATORS = {}
