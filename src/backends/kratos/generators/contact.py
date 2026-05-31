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
        "conditions": ["ALMFrictionlessMortarContact", "ALMFrictionalMortarContact",
                      "PenaltyFrictionlessMortarContact", "PenaltyFrictionalMortarContact"],
        "pitfalls": [
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
