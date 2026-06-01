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
        "elements": [
            # Real RANSApplication names: drop the 'Element'
            # suffix and add an AFC / CWD / RFC stabilization tag
            # in the middle (verified empirically 2026-06-01 via
            # kratos_eletype_scanner.py). The catalog previously
            # listed RansKEpsilonKElement2D3N etc. — none
            # registered.
            "RansKEpsilonK{AFC,CWD,RFC}2D3N",
            "RansKEpsilonK{AFC,CWD,RFC}3D4N",
            "RansKEpsilonEpsilon{AFC,CWD,RFC}2D3N",
            "RansKEpsilonEpsilon{AFC,CWD,RFC}3D4N",
            "RansKOmegaK{AFC,CWD,RFC}{2D3N,3D4N}",
            "RansKOmegaOmega{AFC,CWD,RFC}{2D3N,3D4N}",
            "RansKOmegaSSTK{AFC,CWD,RFC}{2D3N,3D4N}",
            "RansKOmegaSSTOmega{AFC,CWD,RFC}{2D3N,3D4N}",
            # Example expansions actually verified:
            "RansKEpsilonKAFC2D3N",
            "RansKEpsilonEpsilonCWD2D3N",
            "RansKOmegaSSTKRFC2D3N",
        ],
        "wall_treatment": ["wall_functions (log law)", "low_Re (resolve boundary layer)"],
        "pitfalls": [
                        '[API] RANSApplication element names DROP the '
                        '"Element" suffix and ADD an AFC / CWD / RFC '
                        'stabilization-scheme tag in the middle. '
                        'Examples: RansKEpsilonKAFC2D3N, '
                        'RansKEpsilonEpsilonCWD2D3N, '
                        'RansKOmegaSSTKRFC2D3N. The catalog '
                        'previously listed RansKEpsilonKElement2D3N '
                        '/ RansKOmegaSSTKElement2D3N — none of '
                        'those names are registered. '
                        "Signal: model_part.CreateNewElement("
                        "\"RansKEpsilonKElement2D3N\", ...) raises "
                        "'is not registered' from kratos/python/"
                        "add_model_part_to_python.cpp:173; dropping "
                        "'Element' and inserting AFC / CWD / RFC "
                        "lets the call succeed. (Verified "
                        "empirically 2026-06-01 — Tier-2 fixture "
                        "rans_shallowwater_element_naming in "
                        "scripts/tier2_fixtures/kratos/.)",
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
