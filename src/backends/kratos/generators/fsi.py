"""Kratos FSI generators and knowledge."""


def _fsi_2d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    FSI via Kratos FSIApplication + CoSimulationApplication."""
    return f'''\
"""Fluid-Structure Interaction — Kratos FSI/CoSimulation"""
import json
try:
    import KratosMultiphysics as KM
    import KratosMultiphysics.CoSimulationApplication as CSA
    print("CoSimulationApplication available — enables partitioned FSI")
    summary = {{"note": "Kratos CoSim available — Gauss-Seidel/Jacobi coupling, Aitken/MVQN acceleration"}}
except ImportError:
    try:
        import KratosMultiphysics.FSIApplication as FSI
        print("FSIApplication available — Dirichlet-Neumann partitioned FSI")
        summary = {{"note": "Kratos FSI available"}}
    except ImportError:
        print("Neither FSI nor CoSimulation application installed")
        summary = {{"note": "Install: pip install KratosCoSimulationApplication"}}
with open("results_summary.json", "w") as _f: json.dump(summary, _f, indent=2)
'''


KNOWLEDGE = {
    "fsi": {
        "description": "Fluid-Structure Interaction (FSI/CoSimulation)",
        "applications": ["FSIApplication (partitioned DN)", "CoSimulationApplication (general coupling)"],
        "cosimulation": {
            "coupling_schemes": ["Gauss-Seidel (weak/strong)", "Jacobi (weak/strong)"],
            "convergence_accelerators": ["constant_relaxation", "aitken", "mvqn (Multi-Vector Quasi-Newton)",
                                         "ibqn", "anderson", "iqnils"],
            "data_transfer": ["copy", "kratos_mapping", "empire_mapping"],
            "solver_wrappers": ["kratos (internal)", "external (CoSimIO for coupling with other codes)"],
        },
        "pitfalls": [
                        '[Numerical] FSI needs: FluidDynamics + StructuralMechanics + MeshMoving + Mapping applications '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] CoSimIO enables coupling Kratos with any external solver (including our MCP agents) '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Aitken relaxation recommended for initial runs; MVQN for production '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                    ],
    },
}

GENERATORS = {
    "fsi_2d": _fsi_2d_kratos,
}
