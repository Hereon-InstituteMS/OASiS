"""Kratos Isogeometric Analysis (IGA) generators and knowledge.

Application: IgaApplication.
"""


# NOTE (2026-06-26 honesty audit): the previous _iga_2d generator emitted an
# availability-probe stub that only import-checked IgaApplication and wrote
# {"note": "not installed"} with no solver run. IgaApplication is NOT
# importable in the installed Kratos stack. The stub generator and its
# 'iga_2d' registry entry have been removed; 'iga' is no longer advertised
# in KratosBackend.supported_physics(). The KNOWLEDGE block below (registered
# element/condition names, pitfalls) is retained as a reference-only entry.


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

# Empty: no runnable IGA generator — IgaApplication is not installable in
# this Kratos stack (see honesty-audit note above).
GENERATORS = {}
