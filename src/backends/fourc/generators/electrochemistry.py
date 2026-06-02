"""Electrochemistry generator for 4C.

Covers electrochemical transport problems governed by the Nernst-Planck
equation with electroneutrality constraints.  Solves for ionic
concentrations and electric potential in electrolyte systems.  Used for
battery electrolyte modeling, rotating disk electrodes, and
diffusion-migration problems.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class ElectrochemistryGenerator(BaseGenerator):
    """Generator for electrochemistry (Nernst-Planck) problems in 4C."""

    module_key = "electrochemistry"
    display_name = "Electrochemistry (Nernst-Planck / ELCH)"
    problem_type = "Electrochemistry"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The electrochemistry module solves the Nernst-Planck "
                "equation for ionic transport in electrolyte systems.  It "
                "couples diffusion, migration (electric field-driven "
                "transport), and optionally convection of multiple ionic "
                "species.  The electric potential is determined by an "
                "electroneutrality condition.  The PROBLEM TYPE is "
                "'Electrochemistry'.  The module uses SCALAR TRANSPORT "
                "DYNAMIC for the transport equations and ELCH CONTROL for "
                "electrochemistry-specific settings (temperature, "
                "electroneutrality method, diffusion-conduction formulation).  "
                "Materials use MAT_ion for individual ionic species wrapped "
                "in MAT_matlist.  Each ion has a diffusivity and valence."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "SCALAR TRANSPORT DYNAMIC",
                "SCALAR TRANSPORT DYNAMIC/STABILIZATION",
                "SCALAR TRANSPORT DYNAMIC/NONLINEAR",
                "ELCH CONTROL",
                "SOLVER 1",
                "MATERIALS",
            ],
            "optional_sections": [
                "FLUID DYNAMIC",
                "FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES",
                "SCALAR TRANSPORT DYNAMIC/S2I COUPLING",
                "IO/RUNTIME VTK OUTPUT",
            ],
            "materials": {
                "MAT_ion": {
                    "description": (
                        "Single ionic species material.  Defines the "
                        "diffusion coefficient and charge valence of an "
                        "ion in the electrolyte."
                    ),
                    "parameters": {
                        "DIFFUSIVITY": {
                            "description": (
                                "Diffusion coefficient D_i of the ionic "
                                "species [m^2/s]"
                            ),
                            "range": "> 0",
                        },
                        "VALENCE": {
                            "description": (
                                "Charge number z_i of the ionic species "
                                "(positive for cations, negative for anions)"
                            ),
                            "range": "integer, != 0",
                        },
                    },
                },
                "MAT_matlist": {
                    "description": (
                        "Material list that groups multiple MAT_ion species "
                        "into a single material for the scalar transport "
                        "field.  The number of species determines the "
                        "number of transported scalars."
                    ),
                    "parameters": {
                        "LOCAL": {
                            "description": "Local material flag (typically false)",
                            "range": "true/false",
                        },
                        "NUMMAT": {
                            "description": "Number of ionic species in the list",
                            "range": ">= 2",
                        },
                        "MATIDS": {
                            "description": (
                                "List of MAT_ion material IDs for each species"
                            ),
                            "range": "valid MAT IDs",
                        },
                    },
                },
                "MAT_electrode": {
                    "description": (
                        "Electrode material for Butler-Volmer kinetics at "
                        "electrode-electrolyte interfaces (S2I coupling).  "
                        "Defines concentration-dependent diffusion, "
                        "conductivity, and open-circuit potential."
                    ),
                },
            },
            "solver": {
                "direct": {
                    "type": "UMFPACK",
                    "notes": (
                        "Robust direct solver for electrochemistry.  Works "
                        "well for moderate-size problems."
                    ),
                },
            },
            "time_integration": {
                "SOLVERTYPE": (
                    "'nonlinear' is required for electrochemistry due to "
                    "the nonlinear coupling between concentration and "
                    "potential fields."
                ),
                "TIMESTEP": "Time step size for the transport equation.",
                "NUMSTEP": "Total number of time steps.",
                "MAXTIME": "Maximum simulation time.",
                "THETA": (
                    "Time integration parameter for the one-step-theta "
                    "scheme.  theta=0.5 gives Crank-Nicolson (second-order), "
                    "theta=1.0 gives backward Euler (first-order, more stable)."
                ),
            },
            "elch_settings": {
                "TEMPERATURE": (
                    "Thermodynamic temperature in ELCH CONTROL.  In 4C "
                    "units this is often specified as T/F (temperature "
                    "divided by Faraday constant) for non-dimensionalised "
                    "formulations, e.g. 11604.506 for ~1 V."
                ),
                "EQUPOT": (
                    "Electroneutrality method.  Options: "
                    "'ENC' (electroneutrality constraint -- algebraic), "
                    "'divi' (divergence-based closure equation), "
                    "'Laplace' (Laplace equation for potential)."
                ),
                "DIFFCOND_FORMULATION": (
                    "Set true for concentrated solution theory "
                    "(diffusion-conduction formulation).  Set false for "
                    "dilute solution theory (Nernst-Planck)."
                ),
            },
            "pitfalls": [
                (
                    "[Input] EQUPOT in ELCH CONTROL determines "
                    "how the electric potential is computed; "
                    "wrong choice changes the PHYSICS, not "
                    "just the numerics. 'ENC' enforces strict "
                    "electroneutrality (algebraic constraint); "
                    "'divi' solves an additional Poisson "
                    "equation for phi. Signal: an 'ENC' run "
                    "on a problem with charge separation "
                    "across an electrode interface produces "
                    "ZERO potential drop at the interface, "
                    "whereas 'divi' resolves the double-"
                    "layer voltage. Pick by physical "
                    "regime. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] MATID in SCALAR TRANSPORT "
                    "DYNAMIC must reference the MAT_matlist "
                    "material — NOT an individual MAT_ion. "
                    "Signal: pointing MATID at a MAT_ion "
                    "entry directly aborts with 'expected "
                    "matlist, got ion' from "
                    "4C_scatra_factory.cpp; the matlist "
                    "wraps ALL ionic species in MATIDS. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Input] Number of transported scalars = "
                    "NUMMAT in MAT_matlist (one per ionic "
                    "species) plus one for the electric "
                    "potential (when EQUPOT solves a "
                    "potential PDE). Signal: a 3-ion ELCH "
                    "problem with NUMMAT=3 expects 4 scalars "
                    "in INITIALFIELD (3 conc + 1 phi); "
                    "providing only 3 raises 'INITIALFIELD "
                    "component count mismatch' or silently "
                    "leaves phi at 0. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] Initial conditions for "
                    "concentrations + potential should use "
                    "INITIALFIELD: 'field_by_function' with "
                    "INITFUNCNO / STARTFUNCNO. Each scalar "
                    "component needs its OWN COMPONENT entry "
                    "in the FUNCT. Signal: a multi-component "
                    "FUNCT missing one COMPONENT silently "
                    "sets that scalar to 0 — visible as "
                    "discontinuous initial concentration "
                    "plot for the missing species. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Output] CALCFLUX_DOMAIN: 'total' "
                    "enables post-processing of species "
                    "fluxes. Signal: without it, the VTU "
                    "output has only concentrations and "
                    "potential — flux fields show 'not "
                    "computed' or are missing from the "
                    "dataset; downstream visualisation of "
                    "current density is impossible. Always "
                    "enable for ELCH problems. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] For S2I (scatra-scatra "
                    "interface) coupling with Butler-Volmer "
                    "kinetics, BOTH SCALAR TRANSPORT DYNAMIC/"
                    "S2I COUPLING AND DESIGN SURF S2I "
                    "COUPLING CONDITIONS must be specified. "
                    "Signal: omitting the DESIGN SURF "
                    "section produces a setup that compiles "
                    "but never applies the BV kinetics at "
                    "the interface — current across the "
                    "electrode is ~0 regardless of applied "
                    "potential. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] Stabilisation typically set "
                    "to 'no_stabilization' for ELCH "
                    "(diffusion-dominated). Add SUPG only if "
                    "convection is significant (e.g. forced "
                    "electrolyte flow). Signal: SUPG in the "
                    "SCATRA STABILIZATION block on a pure-"
                    "diffusion ELCH (PROBLEMTYPE: "
                    "Scalar_Transport with SCATRATIMINTTYPE: "
                    "Elch) damps physical concentration "
                    "gradients by 5-20% — the BV current is "
                    "under-predicted. For natural convection "
                    "(Re ~ 1), keep no_stabilization; for "
                    "forced flow add SUPG. (Audit "
                    "2026-06-02.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "diffusion_migration_3d",
                    "description": (
                        "Diffusion-migration of binary electrolyte in a 3-D "
                        "domain.  Two ionic species with different "
                        "diffusivities and valences.  Tests electroneutrality "
                        "coupling and flux computation."
                    ),
                    "template_variant": "nernst_planck_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "nernst_planck_3d",
                "description": (
                    "3-D Nernst-Planck diffusion-migration problem with "
                    "binary electrolyte (cation + anion).  Uses MAT_ion "
                    "materials in MAT_matlist, ENC electroneutrality, "
                    "UMFPACK solver."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "nernst_planck_3d") -> str:
        templates = {
            "nernst_planck_3d": self._template_nernst_planck_3d,
        }
        if variant == "default":
            variant = "nernst_planck_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_nernst_planck_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D Nernst-Planck Electrochemistry (Binary Electrolyte)
            #
            # Diffusion-migration of two ionic species (cation and anion)
            # in a 3-D domain with electroneutrality constraint.
            #
            # Mesh: exodus file with:
            #   element_block 1 = electrolyte domain (HEX8 or TET4)
            #   node_set 1 = Dirichlet boundary (fixed concentrations)
            #   node_set 2 = opposite boundary
            # ---------------------------------------------------------------
            TITLE:
              - "3-D electrochemistry (Nernst-Planck) -- generated template"
            PROBLEM TYPE:
              PROBLEMTYPE: "Electrochemistry"

            # == Scalar Transport (carries concentration + potential) ===========
            SCALAR TRANSPORT DYNAMIC:
              SOLVERTYPE: "nonlinear"
              MAXTIME: <end_time>
              NUMSTEP: <number_of_steps>
              TIMESTEP: <timestep>
              RESTARTEVERY: <restart_interval>
              MATID: <matlist_material_id>
              INITIALFIELD: "field_by_function"
              INITFUNCNO: <initial_condition_function_id>
              CALCFLUX_DOMAIN: "total"
              LINEAR_SOLVER: 1
            SCALAR TRANSPORT DYNAMIC/STABILIZATION:
              STABTYPE: "no_stabilization"
            SCALAR TRANSPORT DYNAMIC/NONLINEAR:
              ITEMAX: <max_nonlinear_iterations>
              CONVTOL: <nonlinear_convergence_tolerance>
              EXPLPREDICT: <explicit_predictor_flag>

            # == Electrochemistry control ======================================
            ELCH CONTROL:
              TEMPERATURE: <thermodynamic_temperature>
              EQUPOT: "<electroneutrality_method>"

            # == Solver ========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "elch_solver"

            # == Materials =====================================================
            MATERIALS:
              # Cation
              - MAT: 1
                MAT_ion:
                  DIFFUSIVITY: <cation_diffusivity>
                  VALENCE: <cation_valence>
              # Anion
              - MAT: 2
                MAT_ion:
                  DIFFUSIVITY: <anion_diffusivity>
                  VALENCE: <anion_valence>
              # Material list wrapping all ionic species
              - MAT: <matlist_material_id>
                MAT_matlist:
                  LOCAL: false
                  NUMMAT: <number_of_species>
                  MATIDS: [1, 2]

            # == Initial condition function ====================================
            # One COMPONENT per scalar: species 1, species 2, potential
            FUNCT<initial_condition_function_id>:
              - COMPONENT: 0
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<initial_concentration_1_expression>"
              - COMPONENT: 1
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<initial_concentration_2_expression>"
              - COMPONENT: 2
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<initial_potential_expression>"

            # == Boundary Conditions ===========================================
            DESIGN SURF TRANSPORT DIRICH CONDITIONS:
              - E: <dirichlet_face_id>
                NUMDOF: <num_scalar_dofs>
                ONOFF: [<active_scalar_dofs>]
                VAL: [<boundary_concentrations_and_potential>]
                FUNCT: [<time_functions>]

            # == Geometry ======================================================
            TRANSPORT GEOMETRY:
              FILE: "<mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  TRANSP:
                    HEX8:
                      MAT: <matlist_material_id>
                      TYPE: Std

            RESULT DESCRIPTION:
              - SCATRA:
                  DIS: "scatra"
                  NODE: <result_node_id>
                  QUANTITY: "phi"
                  VALUE: <expected_concentration>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check diffusivity
        diffusivity = params.get("DIFFUSIVITY")
        if diffusivity is not None:
            try:
                d = float(diffusivity)
                if d <= 0:
                    issues.append(
                        f"DIFFUSIVITY must be > 0, got {d}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"DIFFUSIVITY must be a positive number, "
                    f"got {diffusivity!r}."
                )

        # Check valence
        valence = params.get("VALENCE")
        if valence is not None:
            try:
                z = int(valence)
                if z == 0:
                    issues.append(
                        "VALENCE must be non-zero (charge number of the ion)."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"VALENCE must be a non-zero integer, got {valence!r}."
                )

        # Check EQUPOT
        equpot = params.get("EQUPOT")
        if equpot is not None and equpot not in ("ENC", "divi", "Laplace"):
            issues.append(
                f"EQUPOT must be 'ENC', 'divi', or 'Laplace', "
                f"got {equpot!r}."
            )

        # Check TEMPERATURE
        temperature = params.get("TEMPERATURE")
        if temperature is not None:
            try:
                t = float(temperature)
                if t <= 0:
                    issues.append(
                        f"TEMPERATURE must be > 0 (thermodynamic temperature), "
                        f"got {t}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"TEMPERATURE must be a positive number, "
                    f"got {temperature!r}."
                )

        # Check CONVTOL
        convtol = params.get("CONVTOL")
        if convtol is not None:
            try:
                ct = float(convtol)
                if ct <= 0:
                    issues.append(
                        f"CONVTOL must be > 0, got {ct}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"CONVTOL must be a positive number, got {convtol!r}."
                )

        # Check NUMMAT in matlist
        nummat = params.get("NUMMAT")
        if nummat is not None:
            try:
                nm = int(nummat)
                if nm < 2:
                    issues.append(
                        f"NUMMAT in MAT_matlist should be >= 2 (at least "
                        f"two ionic species for electroneutrality), got {nm}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"NUMMAT must be an integer >= 2, got {nummat!r}."
                )

        return issues
