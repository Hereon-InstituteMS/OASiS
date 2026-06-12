"""Kratos contact mechanics generators and knowledge."""


def _contact_2d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Contact mechanics via Kratos ContactStructuralMechanicsApplication."""
    return f'''\
"""Contact mechanics — Kratos ContactStructuralMechanicsApplication"""
import json
try:
    import KratosMultiphysics as KM
    import KratosMultiphysics.ContactStructuralMechanicsApplication as CSMA
    print("ContactStructuralMechanicsApplication available")
    summary = {{"note": "Kratos CSMA available — ALM/penalty mortar contact"}}
except ImportError:
    print("ContactStructuralMechanicsApplication not installed")
    summary = {{"note": "ContactStructuralMechanicsApplication not installed"}}
with open("results_summary.json", "w") as _f: json.dump(summary, _f, indent=2)
'''


KNOWLEDGE = {
    "contact": {
        "description": "Contact mechanics via ContactStructuralMechanicsApplication",
        "application": "ContactStructuralMechanicsApplication",
        "formulations": ["ALM (Augmented Lagrangian Method)", "Penalty method",
                        "Mortar NTN (Node-to-Node)", "Mortar NTS (Node-to-Segment)"],
        "contact_types": ["Frictionless", "Frictional (Coulomb)"],
        "conditions": [
            "ALMFrictionlessMortarContactCondition2D2N",
            "ALMFrictionalMortarContactCondition2D2N",
            "PenaltyFrictionlessMortarContactCondition2D2N",
            "PenaltyFrictionalMortarContactCondition2D2N",
            "ALMFrictionlessMortarContactCondition3D3N",
            "ALMFrictionlessMortarContactCondition3D4N",
            "PenaltyFrictionlessMortarContactCondition3D3N",
            "PenaltyFrictionalMortarContactCondition3D3N",
        ],
        "pitfalls": [
                        '[API] Contact condition names registered by '
                        'KratosContactStructuralMechanicsApplication '
                        'have a "Condition" suffix AND a shape descriptor '
                        '(2D2N, 3D3N, 3D4N, 3D4N3N, etc.). The base names '
                        '"ALMFrictionlessMortarContact" / '
                        '"ALMFrictionalMortarContact" / '
                        '"PenaltyFrictionlessMortarContact" / '
                        '"PenaltyFrictionalMortarContact" — without '
                        'suffixes — are NOT registered and fail '
                        'CreateNewCondition with "is not registered". '
                        'Correct strings: '
                        '"ALMFrictionlessMortarContactCondition2D2N" for '
                        '2D line, "ALMFrictionlessMortarContactCondition3D3N" '
                        'for 3D triangle surface, etc. The "MapperFactory" '
                        'pattern (CreateNewCondition by name) is the only '
                        'public path — there are no Python attributes on '
                        'CSMA for these conditions. Also: '
                        'KratosContactStructuralMechanicsApplication is a '
                        'SEPARATE pip package (not pulled by '
                        'KratosMultiphysics core); '
                        '"pip install KratosContactStructuralMechanicsApplication" '
                        'is required before any contact catalog usage. '
                        "Signal: mp.CreateNewCondition(\"ALMFrictionlessMortarContact\", ...) "
                        "raises 'Error: The Condition X is not registered!' "
                        "from kratos/python/add_model_part_to_python.cpp:173; "
                        "appending 'Condition2D2N' lets the call succeed. "
                        "(Verified empirically 2026-06-01 — Tier-2 fixture "
                        "contact_condition_naming_with_shape_suffix in "
                        "scripts/tier2_fixtures/kratos/.)",
                        '[Integration] Contact surfaces must be defined as SubModelParts containing Conditions of a Mortar contact type — not Elements. Mixing Element / Condition types on a contact SubModelPart triggers obscure assembly errors. '
                        "Signal: AnalysisStage.Initialize raises RuntimeError 'Condition type ... not registered' or 'invalid geometry' when the contact search builds the master-slave pairs.",
                        '[Numerical] Master / slave designation matters for convergence — the slave surface integrates the gap; swapping master and slave with very different mesh densities prevents convergence. '
                        "Signal: ResidualBasedNewtonRaphsonStrategy reports non-converged at max_iteration of convergence_criterion; integrated GAP on the slave Mortar SubModelPart stays O(1); swapping the master/slave designation lets ALMFrictionlessMortarContact reach ResidualCriteria tolerance.",
                        '[Numerical] ALM penalty parameter needs tuning: too small permits inter-penetration; too large makes the tangent stiffness matrix ill-conditioned. Recommended start: penalty = 1e3 * Young modulus / characteristic length. '
                        "Signal: penetration > 1% of characteristic length OR solver reports stiffness condition number > 1e14 / 'matrix is numerically singular' from the linear solver.",
                    ],
    },
}

GENERATORS = {
    "contact_2d": _contact_2d_kratos,
}
