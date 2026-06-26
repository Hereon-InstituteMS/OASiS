"""Membrane element generator for 4C.

Covers thin membrane/shell elements for structural analysis.
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class MembraneGenerator(BaseGenerator):
    """Generator for membrane/thin shell problems in 4C."""

    module_key = "membrane"
    display_name = "Membrane Elements"
    problem_type = "Structure"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Thin membrane elements for inflatable structures, fabric, "
                "biological tissue.  No bending stiffness — pure in-plane stress."
            ),
            "elements": {
                "2D": ["MEMBRANE TRI3", "MEMBRANE TRI6", "MEMBRANE QUAD4", "MEMBRANE QUAD9"],
            },
            "materials": ["MAT_ElastHyper (with membrane kinematics)",
                          "MAT_Struct_StVenantKirchhoff"],
            "pitfalls": [
                (
                    "[Numerical] Membranes have ZERO bending "
                    "stiffness — prone to WRINKLING. Signal: "
                    "an unloaded thin membrane under "
                    "compression produces visible buckling "
                    "with arbitrary wavelength (numerical "
                    "noise dictates) instead of a physical "
                    "wrinkle pattern; the membrane has no "
                    "bending energy to set the wavelength. "
                    "Pre-tensioning or pressure stabilises. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] Use PRESTRESS or pressure "
                    "loading to STABILISE the membrane. "
                    "Signal: a membrane analysis without "
                    "any in-plane tension produces zero-"
                    "eigenvalue modes in the stiffness "
                    "matrix — direct LU reports 'singular', "
                    "iterative solver stalls. Apply "
                    "prestress via PRESTRESS section or "
                    "internal pressure DESIGN SURF NEUMANN. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Input] For wrinkling: enable wrinkling "
                    "model in material definition (e.g. "
                    "MAT_MembraneWrinkling). Signal: a "
                    "standard MAT_ElastHyper on a "
                    "compressive membrane gives negative "
                    "principal stresses (which the "
                    "membrane cannot carry); the wrinkling "
                    "model relaxes compressive states to "
                    "zero stress + wrinkle direction. "
                    "Without it the result is non-"
                    "physical. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] THICK parameter defines "
                    "membrane thickness. Signal: omitting "
                    "THICK uses default 1.0, which "
                    "silently scales all in-plane forces "
                    "linearly — total reaction force at a "
                    "constrained edge is off by exactly "
                    "the THICK factor. Always specify "
                    "THICK to the physical membrane "
                    "thickness. (Audit 2026-06-02.)"
                ),
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "membrane_2d", "description": "Membrane under pressure loading"}]

    def get_template(self, variant: str = "membrane_2d") -> str:
        if variant not in ("membrane_2d", "default"):
            raise ValueError(f"Unknown variant {variant!r}")
        from ..inline_mesh import matched_membrane_2d_input
        return matched_membrane_2d_input()

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
