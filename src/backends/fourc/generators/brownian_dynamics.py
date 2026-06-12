"""Brownian dynamics generator for 4C.

Covers thermal fluctuations in beam/fiber networks (e.g., biopolymer networks).
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class BrownianDynamicsGenerator(BaseGenerator):
    """Generator for Brownian dynamics of fiber networks in 4C."""

    module_key = "brownian_dynamics"
    display_name = "Brownian Dynamics (Fiber Networks)"
    problem_type = "Structure"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Brownian dynamics for thermal fluctuations in beam/fiber networks.  "
                "Used for modeling biopolymer networks (actin, collagen) at the "
                "mesoscale where thermal forces are significant."
            ),
            "elements": ["BEAM3R LINE2 (Simo-Reissner beam with Brownian forces)"],
            "physics": {
                "thermal_forces": "Random forces from Fluctuation-Dissipation theorem",
                "viscous_drag": "Stokes drag on beam segments",
                "cross_links": "Beam-to-beam coupling via penalty/Lagrange",
            },
            "applications": ["actin network mechanics", "collagen fiber networks",
                             "polymer rheology", "cytoskeleton modeling"],
            "pitfalls": [
                (
                    "[Numerical] Time step must be SMALL "
                    "relative to Brownian relaxation time "
                    "tau_B = ksi / kBT. Signal: dt > tau_B / "
                    "10 gives a discrete random walk that "
                    "looks correlated (consecutive "
                    "fluctuations in same direction) "
                    "rather than diffusive; mean-square "
                    "displacement scales as t instead of "
                    "t^1.0 on the right scale. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] Temperature parameter (kBT) "
                    "controls the FLUCTUATION MAGNITUDE in "
                    "the Langevin noise term. Signal: kBT "
                    "= 0 reduces Brownian dynamics to "
                    "deterministic (no thermal "
                    "fluctuations); too-large kBT "
                    "produces unphysical filament "
                    "stretching beyond bond extension. "
                    "Use kBT = 4.14e-21 J at 300 K for "
                    "physiological systems. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Numerical] Cross-link stiffness "
                    "dramatically affects network "
                    "response. Signal: k_xl too small "
                    "lets the network deform like a "
                    "viscous fluid (no elastic plateau); "
                    "too large makes the network rigid "
                    "and locks Brownian fluctuations to "
                    "zero. Typical actin-network range: "
                    "k_xl in [0.1, 10] * k_bend / L_p. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Input] Periodic boundary conditions "
                    "typically needed for RVE analysis "
                    "of network rheology. Signal: a "
                    "FREE-boundary RVE produces "
                    "non-physical edge effects — "
                    "fibres near the boundary have "
                    "fewer neighbours and the average "
                    "stress is wrong by 5-20%. Use "
                    "DESIGN PERIODIC CONDITIONS for the "
                    "RVE faces. (Audit 2026-06-02.)"
                ),
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "brownian_3d", "description": "Brownian fiber network"}]

    def get_template(self, variant: str = "brownian_3d") -> str:
        return "# Brownian dynamics template — use BEAM3R elements with thermal fluctuation parameters"

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
