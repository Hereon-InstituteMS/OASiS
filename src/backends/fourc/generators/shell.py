"""Shell element generator for 4C.

Covers Kirchhoff-Love and Reissner-Mindlin shell elements.
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class ShellGenerator(BaseGenerator):
    """Generator for shell structure problems in 4C."""

    module_key = "shell"
    display_name = "Shell Elements"
    problem_type = "Structure"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Thin and thick shell elements for plates, curved shells, "
                "and general 3D surface structures.  Includes Kirchhoff-Love "
                "(thin, no transverse shear) and Reissner-Mindlin (thick, "
                "transverse shear) formulations."
            ),
            "elements": {
                "Kirchhoff-Love": ["SHELL KIRCHHOFF TRI3", "SHELL KIRCHHOFF QUAD4",
                                   "SHELL KIRCHHOFF QUAD9"],
                "Reissner-Mindlin": ["SHELL REISSNER TRI3", "SHELL REISSNER QUAD4",
                                     "SHELL REISSNER QUAD9"],
                "solid-shell": ["SOLIDSHELL HEX8 (continuum shell with 3D topology)"],
            },
            "materials": ["MAT_Struct_StVenantKirchhoff", "MAT_ElastHyper",
                          "MAT_Struct_MicroMaterial (multiscale)"],
            "pitfalls": [
                (
                    "[Numerical] Kirchhoff shells need C1 "
                    "continuity — use NURBS or DKT "
                    "(Discrete Kirchhoff Triangle) "
                    "formulation. Signal: declaring a C0 "
                    "SHELL KIRCHHOFF QUAD4 / TRI3 with "
                    "MAT_Struct_StVenantKirchhoff produces "
                    "wrong deflection by ~30-50% on plate "
                    "bending; the shear strain that the C0 "
                    "element cannot suppress contaminates "
                    "the bending energy. Switch to "
                    "SHELL_KL_NURBS or a DKT element. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] Reissner-Mindlin shells can "
                    "LOCK for thin shells — use REDUCED "
                    "integration. Signal: a thin-plate "
                    "test (t/L < 0.01) with full "
                    "integration gives centre deflection "
                    "10-100x smaller than analytic; reduced "
                    "integration (1 Gauss point in shear "
                    "terms) recovers correct deflection. "
                    "Alternative: MITC family or assumed-"
                    "strain methods. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] THICK parameter is the SHELL "
                    "THICKNESS. Signal: omitting THICK uses "
                    "the default (often 1.0) which silently "
                    "scales bending stiffness as t^3 — "
                    "wrong by orders of magnitude for "
                    "typical thin-shell problems. Always "
                    "specify THICK explicitly to the "
                    "physical shell thickness. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] Director vector must be "
                    "SPECIFIED or auto-computed from "
                    "element normal. Signal: a curved-"
                    "shell mesh with auto-computed "
                    "directors aligned only with element "
                    "normals (not the smooth surface "
                    "normal) gives discontinuous director "
                    "field across element edges — visible "
                    "wrinkles in the deformed shape. Use "
                    "smoothed nodal directors for curved "
                    "geometry. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] Shell elements produce "
                    "BOTH in-plane forces (N_xx, N_yy, "
                    "N_xy) AND bending moments (M_xx, M_yy, "
                    "M_xy). Signal: visualising only sigma "
                    "(in-plane stress) misses the bending "
                    "contribution — through-thickness "
                    "stress sigma(z) varies from -6*M/t^2 "
                    "at one face to +6*M/t^2 at the other. "
                    "Output both N and M tensors via "
                    "STRESS_STRAIN and verify both for "
                    "validation. (Audit 2026-06-02.)"
                ),
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "shell_3d", "description": "Shell structure under loading"}]

    def get_template(self, variant: str = "shell_3d") -> str:
        return "# Shell template — use SHELL REISSNER QUAD4 for general purpose"

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
