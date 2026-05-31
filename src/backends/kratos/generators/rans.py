"""Kratos RANS turbulence modeling generators and knowledge.

Covers k-epsilon, k-omega SST, and other RANS turbulence models.
Application: RANSApplication.
"""


def _rans_2d(params: dict) -> str:
    """FORMAT TEMPLATE — RANS turbulent flow simulation."""
    return '''\
"""RANS turbulent flow — Kratos RANSApplication"""
import json
try:
    import KratosMultiphysics as KM
    import KratosMultiphysics.RANSApplication
    print("RANSApplication available")
    summary = {"note": "RANSApplication available",
               "capabilities": ["k_epsilon", "k_omega", "k_omega_sst", "spalart_allmaras"]}
except ImportError:
    print("RANSApplication not installed")
    print("Install: pip install KratosRANSApplication")
    summary = {"note": "not installed"}
with open("results_summary.json", "w") as f: json.dump(summary, f, indent=2)
'''


KNOWLEDGE = {
    "rans": {
        "description": "RANS turbulence modeling for incompressible flow",
        "application": "RANSApplication (pip install KratosRANSApplication)",
        "models": {
            "k_epsilon": "Standard k-epsilon with wall functions",
            "k_omega": "Wilcox k-omega model",
            "k_omega_sst": "Menter SST model (recommended for general use)",
            "spalart_allmaras": "One-equation SA model",
        },
        "elements": ["RansKEpsilonKElement2D3N", "RansKEpsilonEpsilonElement2D3N",
                     "RansKOmegaSSTKElement2D3N", "RansKOmegaSSTOmegaElement2D3N"],
        "wall_treatment": ["wall_functions (log law)", "low_Re (resolve boundary layer)"],
        "pitfalls": [
                        '[Integration] Requires FluidDynamicsApplication as dependency '
                        "Signal: RuntimeError 'KeyError' from JSON parsing OR 'SubModelPart not found' / 'Property ID ... missing' during AnalysisStage.Initialize.",
                        '[Numerical] Wall distance computation needed for SST model '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Physics] Inlet turbulence: specify k and epsilon/omega from turbulence intensity '
                        'Signal: post-processed quantity (max displacement, integrated flux, pressure) disagrees with analytic / textbook reference by 10-100%.',
                        '[Numerical] y+ must be appropriate for chosen wall treatment (30-300 for wall functions) '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                    ],
    },
}

GENERATORS = {
    "rans_2d": _rans_2d,
}
