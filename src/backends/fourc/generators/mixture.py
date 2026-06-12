"""Mixture/multiscale material generator for 4C.

Covers fiber-reinforced composites, biological tissue mixtures.
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class MixtureGenerator(BaseGenerator):
    """Generator for mixture/composite material problems in 4C."""

    module_key = "mixture"
    display_name = "Mixture/Composite Materials"
    problem_type = "Structure"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Mixture theory for fiber-reinforced composites and biological tissues.  "
                "Multiple material constituents with individual constitutive laws "
                "combined via mixture rules."
            ),
            "materials": {
                "MAT_Mixture": "General mixture material with multiple constituents",
                "constituents": [
                    "MAT_ElastHyper (isotropic ground substance)",
                    "MAT_Muscle_Weickenmeier (skeletal muscle)",
                    "MAT_Muscle_Giantesio (active muscle)",
                    "Fiber families with anisotropic response",
                ],
            },
            "applications": ["arterial wall mechanics", "tendon/ligament",
                             "muscle tissue", "fiber-reinforced polymers",
                             "growth and remodeling"],
            "pitfalls": [
                (
                    "[Numerical] Mixture rule: stress = "
                    "sum(volume_fraction_i * stress_i). "
                    "Signal: a mixture of fibre + matrix "
                    "with phi_fibre + phi_matrix != 1 "
                    "produces nonphysical stress scaling "
                    "(e.g. total > sum of constituents); "
                    "the volume fractions must SUM TO 1 "
                    "across all constituents at every "
                    "point. Verify in pre-processing. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Input] Fibre direction must be "
                    "specified per element OR via a "
                    "vector field. Signal: a uniform "
                    "FIBER_VEC across all elements on a "
                    "curved geometry gives fibres "
                    "misaligned with the underlying "
                    "physiology (e.g. arterial wall "
                    "fibres should follow the helical "
                    "structure); the stiffness "
                    "anisotropy ends up aligned wrong. "
                    "Use a per-element FIBER_VEC vector "
                    "field. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] Growth and remodelling "
                    "requires TIME-DEPENDENT mass sources. "
                    "Signal: a steady mixture solve with "
                    "constant mass cannot capture growth; "
                    "the deformation gradient F = F_e * "
                    "F_g remains pure-elastic. Add the "
                    "growth tensor F_g(t) driven by a "
                    "concentration or stress signal, with "
                    "RHO_GROWTH FUNCT to specify time-"
                    "varying mass density. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Numerical] Incompressibility "
                    "constraint handled via PENALTY or "
                    "MIXED (u, p) formulation. Signal: a "
                    "near-incompressible mixture "
                    "(K/mu > 1000) with pure-displacement "
                    "solve LOCKS volumetrically — "
                    "deflection is wrong by 10-1000x. "
                    "Switch to mixed (u, p) or "
                    "augmented-Lagrangian penalty. "
                    "(Audit 2026-06-02.)"
                ),
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "mixture_3d", "description": "Mixture material under loading"}]

    def get_template(self, variant: str = "mixture_3d") -> str:
        return "# Mixture template — use MAT_Mixture with multiple constituents"

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
