"""Reduced-dimensional lung model generator for 4C.

Couples reduced airways with alveolar tissue mechanics.
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class ReducedLungGenerator(BaseGenerator):
    """Generator for reduced lung model in 4C."""

    module_key = "reduced_lung"
    display_name = "Reduced Lung Model"
    problem_type = "ReducedLung"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Reduced-dimensional lung model coupling 1-D airway trees with "
                "0-D alveolar tissue compartments.  Used for whole-lung ventilation "
                "simulation."
            ),
            "coupling": "Reduced airways (1D) + alveolar acini (0D) + optional 3D parenchyma",
            "applications": ["ventilation simulation", "mechanical ventilation",
                             "lung disease modeling", "airway pressure distribution"],
            "pitfalls": [
                (
                    "[Input] Airway tree TOPOLOGY must be "
                    "physiologically reasonable — typically "
                    "16-23 generations (Weibel symmetric "
                    "model). Signal: an unbalanced tree "
                    "(e.g. all generations branching into "
                    "only 1 child) gives wrong total airway "
                    "resistance and tidal volume — should "
                    "be ~ 2^23 = 8 million terminal acini "
                    "for normal lung. Verify total alveolar "
                    "surface ~ 130 m^2 against the modelled "
                    "geometry. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] Alveolar compliance varies with "
                    "DISEASE STATE. Healthy lung ~ 0.2 L/"
                    "cmH2O; emphysema is hyper-compliant "
                    "(0.5+); fibrosis is stiff (~ 0.05). "
                    "Signal: using a uniform compliance "
                    "for a diseased-lung model misses the "
                    "spatial heterogeneity that drives the "
                    "clinical pressure-volume curve — total "
                    "lung capacity will be wrong by 20-50% "
                    "vs measured. Vary compliance per "
                    "acinus by region. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Numerical] Coupling between 1D "
                    "airways and 0D acini via flow/pressure "
                    "matching. Signal: unit mismatch at "
                    "the interface (e.g. acinus expects Q "
                    "in L/min but airway delivers L/s) "
                    "gives factor-60 error in tidal "
                    "volume — verify coupling units and "
                    "DOF mapping. The interface must "
                    "match flow Q AND pressure p at each "
                    "terminal node. (Audit 2026-06-02.)"
                ),
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "lung_1d", "description": "Reduced lung ventilation model"}]

    def get_template(self, variant: str = "lung_1d") -> str:
        return (
            "# =====================================================\n"
            "# 4C reduced-lung-tree (variant: lung_1d)\n"
            "# =====================================================\n"
            "# Not a self-contained runnable input. 4C reduced-lung\n"
            "# problems couple a 1-D airway tree (typically derived\n"
            "# from a patient CT) with 0-D alveolar acini and an\n"
            "# optional 3-D parenchyma. The required mesh +\n"
            "# topology depends on:\n"
            "#\n"
            "#   * airway-tree topology (NODE COORDS + ARTERY\n"
            "#     ELEMENTS sections)\n"
            "#   * alveolar compliance distribution per acinus\n"
            "#   * coupling type (1D-0D pressure-flow or 3D-0D\n"
            "#     surface-to-acinus mortar)\n"
            "#\n"
            "# Pitfalls (see knowledge() for the full set):\n"
            "#   * Airway-tree topology must be physiologically\n"
            "#     reasonable (Horsfield / Strahler order)\n"
            "#   * Alveolar compliance varies with disease state\n"
            "#     (emphysema, fibrosis, ARDS)\n"
            "#   * Coupling between 1-D airways and 0-D acini\n"
            "#     happens via flow/pressure matching at outlets\n"
            "# =====================================================\n"
            "TITLE:\n"
            "  - \"4C reduced-lung-tree reference stub\"\n"
            "PROBLEM TYPE:\n"
            "  PROBLEMTYPE: \"ReducedLung\"\n"
            "REDUCED DIMENSIONAL AIRWAYS DYNAMIC:\n"
            "  DYNAMICTYPE: \"OneStepTheta\"\n"
            "  TIMESTEP: 0.01\n"
            "  NUMSTEP: 100\n"
            "# Concrete airway-tree NODE COORDS / ARTERY ELEMENTS\n"
            "# / boundary BCs must be supplied per patient case.\n"
        )

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
