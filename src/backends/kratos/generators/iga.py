"""Kratos Isogeometric Analysis (IGA) generators and knowledge.

Application: IgaApplication.
"""


def _iga_2d(params: dict) -> str:
    """FORMAT TEMPLATE — IGA shell/membrane analysis."""
    return '''\
"""Isogeometric Analysis — Kratos IgaApplication"""
import json
try:
    import KratosMultiphysics as KM
    import KratosMultiphysics.IgaApplication
    print("IgaApplication available")
    summary = {"note": "IgaApplication available",
               "capabilities": ["NURBS_shells", "NURBS_membranes", "trimmed_surfaces",
                                "multi_patch", "form_finding"]}
except ImportError:
    print("IgaApplication not installed")
    summary = {"note": "not installed"}
with open("results_summary.json", "w") as f: json.dump(summary, f, indent=2)
'''


KNOWLEDGE = {
    "iga": {
        "description": "Isogeometric Analysis: NURBS-based shells, membranes, trimmed surfaces",
        "application": "IgaApplication (pip install KratosIgaApplication)",
        "elements": ["Shell3pElement", "Shell5pElement", "Shell5pHierarchicElement",
                     "SurfaceLoadCondition"],
        "capabilities": ["NURBS_shells", "trimmed_multi_patch", "form_finding",
                         "penalty_coupling", "Nitsche_coupling"],
        "geometry_formats": ["NURBS from CAD (IGES/STEP)", "B-spline patches"],
        "pitfalls": [
                        '[Numerical] Requires NURBS geometry definition (control points, knot vectors, weights) '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Trimmed surfaces need special integration rules '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Multi-patch coupling via penalty or Nitsche method '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Higher continuity (C^p-1) vs C^0 FEM — different error behavior '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                    ],
    },
}

GENERATORS = {"iga_2d": _iga_2d}
