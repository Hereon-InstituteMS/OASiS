"""Fluid turbulence (LES/DNS) generator for 4C.

Covers large-eddy simulation and direct numerical simulation capabilities.
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class FluidTurbulenceGenerator(BaseGenerator):
    """Generator for turbulent flow (LES/DNS) in 4C."""

    module_key = "fluid_turbulence"
    display_name = "Fluid Turbulence (LES/DNS)"
    problem_type = "Fluid"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Large-Eddy Simulation (LES) and Direct Numerical Simulation (DNS) "
                "for turbulent incompressible flow.  Uses the fluid module with "
                "additional subgrid-scale modeling."
            ),
            "sgs_models": {
                "Smagorinsky": "Classic constant-coefficient SGS model",
                "DynamicSmagorinsky": "Germano dynamic procedure for C_s",
                "WALE": "Wall-Adapting Local Eddy viscosity",
                "Vreman": "Vreman SGS model",
                "Multifractal": "Multifractal SGS model",
            },
            "stabilization": [
                "Residual-based VMS (variational multiscale) — built into fluid elements",
                "SUPG/PSPG for coarse LES",
            ],
            "applications": ["channel flow DNS/LES", "backward-facing step",
                             "cylinder wake", "jet flow", "mixing layers"],
            "pitfalls": [
                (
                    "[Numerical] DNS requires mesh resolution "
                    "at the KOLMOGOROV scale eta = "
                    "(nu^3/epsilon)^(1/4) — expensive (DOFs "
                    "scale as Re^(9/4)). Signal: a 'DNS' "
                    "at Re_tau = 1000 with only ~10^6 DOFs "
                    "is actually under-resolved; true DNS "
                    "needs ~10^9 DOFs. Verify by checking "
                    "y+ < 1 at first cell. For practical "
                    "Re, use LES or RANS instead. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Numerical] LES mesh should resolve "
                    "~80% of the turbulent kinetic energy. "
                    "Signal: too-coarse LES (resolving < "
                    "50% of TKE) gives unphysical mean "
                    "profiles and under-predicts wall "
                    "shear by 20-50%; compute the "
                    "resolved/total TKE ratio from "
                    "diagnostics and refine if < 0.8. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] LES time step: CFL < 1 for "
                    "explicit, CFL < 5 for implicit with "
                    "fine mesh. Signal: explicit time "
                    "integration at CFL > 1 gives NaN "
                    "within ~10 steps; implicit at CFL > 5 "
                    "is stable but loses time-accuracy "
                    "(transient features smeared). Adjust "
                    "TIMESTEP to keep max(|u|*dt/dx) "
                    "within these bounds. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] Periodic BCs typically needed "
                    "for homogeneous directions (streamwise "
                    "and spanwise in channel flow). Signal: "
                    "using DIRICHLET outlets in a channel-"
                    "flow LES gives blockage and incorrect "
                    "mean flow; the streamwise-periodic "
                    "condition uses DESIGN PERIODIC "
                    "CONDITIONS pairing inflow + outflow "
                    "nodes. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] LES STATISTICS: average "
                    "over MANY flow-through times "
                    "(typically 20-50 T_flowthrough = L / "
                    "U_bulk) for convergence. Signal: in a "
                    "FLUID_DYNAMIC LES with "
                    "TURBULENCE_MODEL: Smagorinsky, stopping "
                    "after 1-2 flow-through times gives the "
                    "running_mean of the VELOCITY second-"
                    "order moments still showing transient "
                    "drift; they converge as 1/sqrt(N_samples). "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Input] Inflow: use recycling/"
                    "rescaling or synthetic turbulence "
                    "generation. Signal: a uniform/laminar "
                    "inlet on a LES of a turbulent channel "
                    "produces a long laminar entrance "
                    "region (typically 20+ channel-widths) "
                    "before turbulence develops — wastes "
                    "compute. Use Lund-Wu-Squires "
                    "recycling or Jarrin synthetic eddy "
                    "method at the inlet. (Audit "
                    "2026-06-02.)"
                ),
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "les_channel_3d", "description": "LES of turbulent channel flow"}]

    def get_template(self, variant: str = "les_channel_3d") -> str:
        # Not self-contained-runnable: a meaningful LES needs a
        # wall-resolved, periodic, graded HEX8 channel mesh — a coarse
        # inline grid "runs" but produces physically meaningless
        # statistics. Return a valid-YAML reference stub rather than a
        # comment-only one-liner.
        return (
            "# =====================================================\n"
            "# 4C fluid turbulence / LES (variant: les_channel_3d)\n"
            "# =====================================================\n"
            "# Not a self-contained runnable input. Requires:\n"
            "#   * a wall-resolved 3-D HEX8 channel mesh graded to\n"
            "#     the wall (y+ ~ 1) with periodic stream/spanwise\n"
            "#     boundary surfaces\n"
            "#   * FLUID DYNAMIC + a TURBULENCE MODEL section\n"
            "#     (Smagorinsky / dynamic / WALE) + periodic BCs +\n"
            "#     turbulence statistics sampling\n"
            "#   * MAT_fluid at the target Reynolds number\n"
            "# Pitfalls (see knowledge() for the full set):\n"
            "#   * LES on an under-resolved mesh is meaningless — it\n"
            "#     'runs' but the statistics are wrong\n"
            "#   * needs a long sampling time to converge mean /\n"
            "#     Reynolds-stress profiles\n"
            "# =====================================================\n"
            "TITLE:\n"
            "  - \"4C fluid turbulence (LES) reference stub\"\n"
            "PROBLEM TYPE:\n"
            "  PROBLEMTYPE: \"Fluid\"\n"
        )

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
