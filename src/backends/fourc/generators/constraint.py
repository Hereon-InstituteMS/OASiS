"""Constraint/coupling condition generator for 4C.

Covers multi-point constraints, rigid body constraints, periodic BCs.
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class ConstraintGenerator(BaseGenerator):
    """Generator for constraint problems in 4C."""

    module_key = "constraint"
    display_name = "Constraints and Coupling"
    problem_type = "Structure"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Multi-point constraints, rigid body coupling, periodic boundary "
                "conditions, and general coupling of DOFs across discretizations."
            ),
            "condition_types": {
                "MPC": "Multi-point constraints (linear combinations of DOFs)",
                "Rigid body": "Rigid coupling of a set of nodes to a master node",
                "Periodic": "Periodic boundary conditions for unit cell analysis",
                "Mortar": "Mortar-based surface coupling (non-matching meshes)",
                "Penalty": "Penalty-based constraint enforcement",
            },
            "yaml_sections": [
                "DESIGN LINE COUPLING CONDITIONS",
                "DESIGN SURF COUPLING CONDITIONS",
                "DESIGN LINE MPC NORMAL COMPONENT CONDITIONS",
            ],
            "pitfalls": [
                (
                    "[Numerical] Constraint equations must be "
                    "LINEARLY INDEPENDENT. Signal: linearly-"
                    "dependent rows in the constraint matrix "
                    "C produce a singular saddle-point "
                    "system [K C^T; C 0] — direct LU "
                    "reports 'zero pivot in Schur "
                    "complement'. Verify with rank(C) = "
                    "n_constraints; remove duplicates. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] Penalty parameter affects "
                    "both ACCURACY and CONDITIONING. Signal: "
                    "too small (alpha < 100*K_typical) "
                    "lets the constraint drift; too large "
                    "(alpha > 1e6*K_typical) makes "
                    "cond(K + alpha*C^T*C) > 1e15 and "
                    "iterative solvers stagnate. Sweet "
                    "spot: alpha ~ 1e3 - 1e5 times "
                    "stiffness diagonal. Use Lagrange "
                    "multipliers for exact enforcement. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] Mortar coupling requires "
                    "INTEGRATION on the interface — the "
                    "coupling integrals over master and "
                    "slave segments must be evaluated. "
                    "Signal: missing INTEGRATION on the "
                    "mortar side produces an incorrect "
                    "coupling matrix; the multiplier "
                    "system is rank-deficient and the "
                    "interface jumps stay non-zero. Use "
                    "INTPOINTS_MORTAR appropriate to "
                    "element order. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] Periodic BCs: master and slave "
                    "surfaces must MATCH GEOMETRICALLY "
                    "(same shape, opposite location). "
                    "Signal: a periodic pair where the "
                    "slave is rotated or non-uniformly "
                    "spaced relative to the master gives "
                    "wrong node-to-node mapping — periodic "
                    "BC enforcement fails or creates "
                    "spurious gaps. Verify "
                    "max|x_slave - x_master + L*e_per| < "
                    "tol (where e_per is the period "
                    "direction). (Audit 2026-06-02.)"
                ),
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "constraint_3d", "description": "Multi-point constraint problem"}]

    def get_template(self, variant: str = "constraint_3d") -> str:
        return "# Constraint template — use DESIGN LINE/SURF COUPLING CONDITIONS"

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
