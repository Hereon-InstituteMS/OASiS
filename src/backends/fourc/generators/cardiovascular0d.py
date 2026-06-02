"""0-D cardiovascular / windkessel models generator for 4C.

Covers lumped-parameter heart models, windkessel afterload, closed-loop circulation.
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class Cardiovascular0DGenerator(BaseGenerator):
    """Generator for 0-D cardiovascular models in 4C."""

    module_key = "cardiovascular0d"
    display_name = "0-D Cardiovascular (Windkessel)"
    problem_type = "Structure"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Lumped-parameter (0-D) cardiovascular models: windkessel afterload, "
                "closed-loop circulation, time-varying elastance heart model.  "
                "Coupled to 3-D fluid or structure via surface conditions."
            ),
            "models": {
                "windkessel_3element": "R_p, C, R_d — proximal resistance, compliance, distal resistance",
                "windkessel_4element": "R_p, L, C, R_d — adds inductance",
                "heart_time_varying_elastance": "E(t) model with active contraction",
                "closed_loop": "Full circulation: heart + arterial + venous + pulmonary",
            },
            "coupling": [
                "DESIGN SURF CARDIOVASCULAR0D CONDITIONS",
                "Coupled to 3-D fluid outflow or structural cavity volume",
            ],
            "applications": ["cardiac simulation", "hemodynamics", "afterload modeling",
                             "valve simulation", "ventricular assist device"],
            "pitfalls": [
                (
                    "[Input] Windkessel parameters (R, C) "
                    "must MATCH the vascular impedance. "
                    "Signal: arbitrary R, C give non-"
                    "physiological pressure waveforms — "
                    "too-high R gives strong reflection "
                    "(double-peak central pressure); too-"
                    "low R damps the wave. Tune from "
                    "Z_terminal = rho*c/A and TPVR / "
                    "compliance estimates for the modelled "
                    "vascular bed. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] Time-varying elastance requires "
                    "cardiac cycle TIMING parameters (T_S "
                    "systole, T_D diastole). Signal: a "
                    "constant-elastance 0D model produces "
                    "no pumping action — pressure follows "
                    "volume linearly without the systolic "
                    "spike. For active cardiac models, "
                    "ELASTANCE_FUNCTION over the heart "
                    "cycle is required. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[API] Coupling to 3D: cavity volume "
                    "computed from SURFACE INTEGRAL over "
                    "the closed cavity boundary. Signal: "
                    "an OPEN cavity boundary (mesh hole "
                    "or missing surface) gives wrong "
                    "volume — the surface integral is "
                    "incorrect and the 0D-3D coupling "
                    "drifts. Verify mesh closure with "
                    "Gmsh CheckClosedSurface. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] Initial conditions: set "
                    "initial pressures in the 0D model. "
                    "Signal: default zero pressure with a "
                    "physiological elastance gives a "
                    "transient that takes 5-10 cardiac "
                    "cycles to stabilise; pre-set "
                    "physiological diastolic pressures "
                    "(~ 10 kPa LV diastolic) to skip the "
                    "warm-up. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] Cardiovascular0D is typically "
                    "used with FLUID or FSI, NOT "
                    "standalone. Signal: a standalone "
                    "Cardiovascular0D problem has no field "
                    "to couple to and produces a "
                    "degenerate setup; 4C's "
                    "Cardiovascular0D adapter requires a "
                    "parent field (PROBLEMTYPE: Fluid or "
                    "Structure with this condition "
                    "applied). (Audit 2026-06-02.)"
                ),
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "windkessel_3d", "description": "3-element windkessel coupled to 3D"}]

    def get_template(self, variant: str = "windkessel_3d") -> str:
        return "# Cardiovascular0D template — use DESIGN SURF CARDIOVASCULAR0D CONDITIONS"

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
