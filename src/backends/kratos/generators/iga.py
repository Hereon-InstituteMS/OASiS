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
        # Real registered IGA Element names (verified empirically
        # 2026-06-01 by walking CreateNewElement against the
        # canonical 'is not registered' list machinery).
        "elements": [
            "Shell3pElement",
            "Shell5pElement",
            "Shell5pHierarchicElement",
            "IgaMembraneElement",
            "TrussElement",
            "TrussEmbeddedEdgeElement",
            "BeamThinElement2D",
            "BeamThickElement2D",
            "SurfaceElement3D3N",
            "SurfaceElement3D4N",
            "SurfaceElement3D6N",
            "SurfaceElement3D8N",
            "SurfaceElement3D9N",
        ],
        # Conditions are a SEPARATE registry; the catalog
        # previously misclassified SurfaceLoadCondition as an
        # element. The bare name without shape suffix is
        # unregistered.
        "conditions": [
            "BrepCurveOnSurfaceCondition",
            "NurbsCurveCondition",
            "SurfaceCondition3D3N",
            "SurfaceCondition3D4N",
            "SurfaceCondition3D6N",
            "SurfaceCondition3D8N",
            "SurfaceCondition3D9N",
            # Surface load conditions live in
            # StructuralMechanicsApplication, NOT IgaApplication:
            "SurfaceLoadCondition3D3N (from StructuralMechanics)",
            "SurfaceLoadCondition3D4N (from StructuralMechanics)",
        ],
        "capabilities": ["NURBS_shells", "trimmed_multi_patch", "form_finding",
                         "penalty_coupling", "Nitsche_coupling"],
        "geometry_formats": ["NURBS from CAD (IGES/STEP)", "B-spline patches"],
        "pitfalls": [
                        '[API] "SurfaceLoadCondition" (bare name) is '
                        'NOT registered as either an Element or a '
                        'Condition in either IgaApplication or '
                        'StructuralMechanicsApplication. The catalog '
                        'previously listed it in the "elements" '
                        'field, which is doubly wrong: (a) it is a '
                        'Condition, not an Element; (b) it needs a '
                        'shape suffix ("SurfaceLoadCondition3D3N" / '
                        '"SurfaceLoadCondition3D4N"), and (c) the '
                        'suffixed form comes from '
                        'StructuralMechanicsApplication, not IGA. '
                        'For IGA-internal surface integration use '
                        'SurfaceCondition3D{3,4,6,8,9}N (no "Load" '
                        'in the name) instead. '
                        "Signal: mp.CreateNewCondition("
                        "\"SurfaceLoadCondition\", ...) raises "
                        "'Error: The Condition X is not registered!' "
                        "from kratos/python/add_model_part_to_python."
                        "cpp:173. Appending the 3D{3,4}N shape "
                        "suffix and loading StructuralMechanics"
                        "Application lets it register. "
                        "(Verified empirically 2026-06-01 — Tier-2 "
                        "fixture iga_surface_condition_naming in "
                        "scripts/tier2_fixtures/kratos/.)",
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
