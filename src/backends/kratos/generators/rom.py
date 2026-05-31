"""Kratos Reduced Order Modeling (ROM) generators and knowledge.

Covers POD, HROM, neural network surrogates.
Application: RomApplication.
"""


def _rom_2d(params: dict) -> str:
    """FORMAT TEMPLATE — ROM analysis."""
    return '''\
"""Reduced Order Model — Kratos RomApplication"""
import json
try:
    import KratosMultiphysics as KM
    import KratosMultiphysics.RomApplication
    print("RomApplication available")
    summary = {"note": "RomApplication available",
               "capabilities": ["POD", "HROM", "neural_network_surrogates",
                                "parametric_studies", "real_time_simulation"]}
except ImportError:
    print("RomApplication not installed")
    print("Install: pip install KratosRomApplication")
    summary = {"note": "not installed"}
with open("results_summary.json", "w") as f: json.dump(summary, f, indent=2)
'''


KNOWLEDGE = {
    "rom": {
        "description": "Reduced Order Modeling: POD, HROM, neural network surrogates",
        "application": "RomApplication (pip install KratosRomApplication)",
        "methods": {
            "POD": "Proper Orthogonal Decomposition — project FOM onto reduced basis",
            "HROM": "Hyper-Reduced Order Model — also reduce integration cost",
            "LSPG": "Least-Squares Petrov-Galerkin projection",
            "ANN": "Artificial Neural Network surrogate from training snapshots",
        },
        "workflow": ["1. Run full-order model (FOM) for training parameters",
                     "2. Collect snapshots", "3. Build reduced basis (SVD/POD)",
                     "4. Train ROM/HROM", "5. Evaluate at new parameters in real-time"],
        "pitfalls": [
                        '[Numerical] Training snapshots must cover the parameter space adequately '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] HROM requires empirical cubature method (ECM) for element selection '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] ROM accuracy degrades for strongly nonlinear problems '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Works best with StructuralMechanics and FluidDynamics applications '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                    ],
    },
}

GENERATORS = {
    "rom_2d": _rom_2d,
}
