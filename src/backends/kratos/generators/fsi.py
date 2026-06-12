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
            # See cosimulation physics catalog for the cross-checked
            # set of CoSim file-stem accelerator names.
            "convergence_accelerators_via_cosim": [
                "constant_relaxation", "aitken", "mvqn",
                "block_mvqn", "block_ibqnls", "iqnils", "anderson",
            ],
            "data_transfer_via_cosim": [
                "nearest_neighbor", "nearest_element", "barycentric",
                "coupling_geometry", "radial_basis_function",
            ],
            "solver_wrappers": ["kratos (internal)", "external (CoSimIO for coupling with other codes)"],
        },
        # FSIApplication ships its OWN convergence-accelerator
        # classes (Python attributes on the FSI module, distinct
        # from the CoSim file stems above). Verified by walking
        # dir(KratosMultiphysics.FSIApplication) 2026-06-01.
        "fsi_application_accelerators": [
            "AitkenConvergenceAccelerator",
            "ConstantRelaxationConvergenceAccelerator",
            "IBQNMVQNConvergenceAccelerator",
            "IBQNMVQNRandomizedSVDConvergenceAccelerator",
            "MVQNFullJacobianConvergenceAccelerator",
            "MVQNRandomizedSVDConvergenceAccelerator",
            "MVQNRecursiveJacobianConvergenceAccelerator",
        ],
        "fsi_application_partitioned_utils": [
            "FSIUtils", "SharedPointsMapper",
            "PartitionedFSIUtilitiesArray2D",
            "PartitionedFSIUtilitiesArray3D",
            "PartitionedFSIUtilitiesDouble2D",
            "PartitionedFSIUtilitiesDouble3D",
        ],
        "pitfalls": [
                        '[Integration] KratosFSIApplication is a '
                        'SEPARATE pip package (pip install '
                        'KratosFSIApplication) — it is NOT pulled '
                        'in by KratosMultiphysics core. The FSI '
                        'catalog previously inherited the cosim '
                        'accelerator list verbatim, conflating '
                        'TWO distinct surfaces: '
                        '(a) CoSimulationApplication accelerator '
                        'FILE stems (block_ibqnls, iqnils, mvqn, '
                        'block_mvqn, aitken, anderson) under '
                        'convergence_accelerators/, vs '
                        '(b) FSIApplication CLASS names '
                        '(IBQNMVQNConvergenceAccelerator, '
                        'MVQNFullJacobianConvergenceAccelerator, '
                        'AitkenConvergenceAccelerator, ...). '
                        '"ibqn" (bare) is neither — neither a '
                        'CoSim file nor an FSI class. Signal: '
                        '"import KratosMultiphysics.FSIApplication" '
                        'on a fresh .venv raises '
                        'ModuleNotFoundError; using "ibqn" as a '
                        'CoSim accelerator "type" raises a factory '
                        'ImportError finding "ibqn.py". (Verified '
                        '2026-06-01 — see also cosimulation '
                        'physics pitfall #0 + Tier-2 fixture '
                        'cosimulation_accelerator_mapper_names.)',
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
