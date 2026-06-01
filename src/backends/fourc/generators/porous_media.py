"""Generator for porous media flow physics module (pressure-based porofluid).

Covers single-phase and multiphase flow through rigid or deformable porous
media using the porofluid_dynamic section (lowercase!) in 4C.  The complex
nested material hierarchy (MAT_FluidPoroMultiPhase -> MAT_FluidPoroSinglePhase
-> density/viscosity/relative-permeability laws) is fully documented.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class PorousMediaGenerator(BaseGenerator):
    """Generator for porous media flow problems in 4C.

    Supports single-phase and multiphase pressure-based porofluid
    simulations with complex material hierarchies.
    """

    module_key = "porous_media"
    display_name = "Porous Media Flow (Pressure-Based Porofluid)"
    problem_type = "porofluid_pressure_based"

    # -- Knowledge -----------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The porous media module in 4C solves single-phase or multiphase "
                "flow through porous media using a pressure-based formulation.  "
                "CRITICAL: this module uses LOWERCASE section names "
                "(porofluid_dynamic, NOT POROFLUID DYNAMIC).  The PROBLEM TYPE is "
                "'porofluid_pressure_based' (for rigid skeleton) or "
                "'porofluid_pressure_based_elasticity' (for deformable skeleton "
                "coupled with structural mechanics).  The material definition "
                "follows a complex nested hierarchy: MAT_FluidPoroMultiPhase -> "
                "MAT_FluidPoroSinglePhase -> density law + viscosity law + "
                "relative permeability law + DoF type.  Each sub-material is "
                "defined as a separate MATERIALS entry with its own MAT ID, "
                "referenced by ID from the parent material."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "porofluid_dynamic",
                "SOLVER 1",
                "MATERIALS",
            ],
            "optional_sections": [
                "STRUCTURAL DYNAMIC (for porofluid_pressure_based_elasticity)",
                "porofluid_elasticity_dynamic (for coupled problems)",
                "CLONING MATERIAL MAP (for coupled problems)",
                "IO",
                "NODE COORDS (inline mesh)",
                "TRANSPORT ELEMENTS (inline mesh)",
            ],
            "section_naming": {
                "CRITICAL": (
                    "Porous media sections use LOWERCASE names unlike most "
                    "other 4C modules.  Use 'porofluid_dynamic' NOT "
                    "'POROFLUID DYNAMIC'.  Sub-sections also lowercase: "
                    "'porofluid_dynamic/time_integration', etc."
                ),
                "porofluid_dynamic": {
                    "total_simulation_time": "Maximum simulation time",
                    "time_integration/number_of_time_steps": "Number of time steps",
                    "time_integration/time_step_size": "Time step size",
                    "time_integration/theta": "Theta for one-step-theta scheme (1.0 = implicit Euler)",
                    "nonlinear_solver/linear_solver_id": "Reference to SOLVER N section",
                    "initial_condition/type": "'by_function' to set IC via FUNCT section",
                    "initial_condition/function_id": "FUNCT ID for initial condition",
                    "output/porosity": "Whether to output porosity field",
                    "flux_reconstruction/active": "Enable flux reconstruction",
                    "flux_reconstruction/solver_id": "Solver for flux reconstruction",
                },
            },
            "materials": {
                "MAT_FluidPoroMultiPhase (top-level)": {
                    "description": (
                        "Top-level porous media material.  Wraps one or more "
                        "single-phase materials and defines the intrinsic "
                        "permeability of the porous medium."
                    ),
                    "parameters": {
                        "LOCAL": {
                            "description": "Local assembly flag (typically false)",
                            "range": "true / false",
                        },
                        "PERMEABILITY": {
                            "description": "Intrinsic permeability of the porous medium [m^2]",
                            "range": "> 0",
                        },
                        "NUMMAT": {
                            "description": "Number of single-phase sub-materials",
                            "range": ">= 1",
                        },
                        "MATIDS": {
                            "description": "List of MAT IDs for each single-phase material",
                            "range": "e.g. [10] or [10, 20, 30]",
                        },
                        "NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE": {
                            "description": "Number of fluid phases sharing the pore space",
                            "range": ">= 1",
                        },
                    },
                },
                "MAT_FluidPoroSinglePhase": {
                    "description": (
                        "Single fluid phase within the pore space.  References "
                        "sub-materials for density law, viscosity law, relative "
                        "permeability law, and DoF type."
                    ),
                    "parameters": {
                        "DENSITYLAWID": {
                            "description": "MAT ID of the density law",
                            "range": "References a MAT_PoroDensityLaw* material",
                        },
                        "DENSITY": {
                            "description": "Reference density of the fluid phase",
                            "range": "> 0",
                        },
                        "RELPERMEABILITYLAWID": {
                            "description": "MAT ID of the relative permeability law",
                            "range": "References a MAT_FluidPoroRelPermeabilityLaw*",
                        },
                        "VISCOSITY_LAW_ID": {
                            "description": "MAT ID of the viscosity law",
                            "range": "References a MAT_FluidPoroViscosityLaw*",
                        },
                        "DOFTYPEID": {
                            "description": "MAT ID defining the primary DoF type for this phase",
                            "range": "References a MAT_FluidPoroSinglePhaseDoF*",
                        },
                    },
                },
                "MAT_PoroDensityLawExp": {
                    "description": (
                        "Exponential density law: rho = rho_0 * exp((p - p_0) / K).  "
                        "For nearly incompressible fluids use a large BULKMODULUS."
                    ),
                    "parameters": {
                        "BULKMODULUS": {
                            "description": "Fluid bulk modulus K",
                            "range": "> 0 (large for incompressible, e.g. 100--1e6)",
                        },
                    },
                },
                "MAT_FluidPoroViscosityLawConstant": {
                    "description": "Constant dynamic viscosity.",
                    "parameters": {
                        "VALUE": {
                            "description": "Dynamic viscosity mu [Pa.s]",
                            "range": "> 0 (e.g. 0.001 for water, 0.01 for oil)",
                        },
                    },
                },
                "MAT_FluidPoroRelPermeabilityLawConstant": {
                    "description": (
                        "Constant relative permeability (k_r = VALUE).  "
                        "Use VALUE = 1.0 for single-phase flow."
                    ),
                    "parameters": {
                        "VALUE": {
                            "description": "Relative permeability k_r (dimensionless)",
                            "range": "0 < k_r <= 1 (1.0 for single-phase)",
                        },
                    },
                },
                "MAT_FluidPoroSinglePhaseDofPressure": {
                    "description": (
                        "DoF type: primary variable is absolute pressure.  "
                        "The simplest choice for single-phase flow."
                    ),
                    "parameters": {
                        "PHASELAWID": {
                            "description": "MAT ID of a phase law (optional for single-phase)",
                            "range": "References a MAT_PhaseLaw* material",
                        },
                    },
                },
            },
            "dof_types": {
                "Pressure": (
                    "MAT_FluidPoroSinglePhaseDofPressure -- primary variable is "
                    "absolute pressure p.  Simplest for single-phase flow."
                ),
                "Saturation": (
                    "MAT_FluidPoroSinglePhaseDofSaturation -- primary variable is "
                    "saturation S.  Used in multiphase flow when one phase's "
                    "saturation is solved directly."
                ),
                "DiffPressure": (
                    "MAT_FluidPoroSinglePhaseDofDiffPressure -- primary variable "
                    "is a pressure difference (p_i - p_j).  Used in multiphase "
                    "flow for capillary pressure formulations."
                ),
            },
            "property_laws": {
                "PoroDensityLawExp": "Exponential density law (compressible fluid)",
                "PoroDensityLawConstant": "Constant density (incompressible)",
                "FluidPoroViscosityLawConstant": "Constant dynamic viscosity",
                "FluidPoroRelPermeabilityLawConstant": "Constant relative permeability",
                "FluidPoroRelPermeabilityLawExp": "Exponential relative permeability (S-dependent)",
            },
            "solver": {
                "recommended": {
                    "SOLVER": "UMFPACK",
                    "description": (
                        "Direct solver, robust for porous media problems.  "
                        "The porous media system can be stiff due to "
                        "permeability contrasts."
                    ),
                },
            },
            "time_integration": {
                "scheme": (
                    "One-step-theta method.  theta = 1.0 gives implicit Euler "
                    "(unconditionally stable, first-order accurate).  "
                    "theta = 0.5 gives Crank-Nicolson (second-order, may "
                    "oscillate)."
                ),
                "time_step_size": "Size of each time step.",
                "number_of_time_steps": "Total number of time steps.",
                "total_simulation_time": "Maximum simulation time.",
            },
            "mesh_format": {
                "note": (
                    "Most porous media examples in 4C use inline mesh format "
                    "(NODE COORDS + TRANSPORT ELEMENTS).  Exodus meshes can also "
                    "be used via TRANSPORT GEOMETRY with POROFLUID element type."
                ),
            },
            "pitfalls": [
                (
                    "[Input] CRITICAL: Section names are LOWERCASE in "
                    "porous media.  Use 'porofluid_dynamic' NOT "
                    "'POROFLUID DYNAMIC'.  Using uppercase will "
                    "silently fall back to defaults. Signal: input "
                    "parser banner shows `unknown section: POROFLUID "
                    "DYNAMIC`, or simulation completes with result "
                    "fields uniformly equal to the (default) initial "
                    "state. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] The material hierarchy is deeply nested: "
                    "MAT_FluidPoroMultiPhase -> MAT_FluidPoroSinglePhase "
                    "-> density law + viscosity law + relative "
                    "permeability law + DoF type.  Each level is a "
                    "separate MATERIALS entry with its own MAT ID.  "
                    "Missing any sub-material causes a crash. Signal: "
                    "4C aborts with `referenced material ID X not "
                    "found in MATERIALS` during material-list "
                    "construction; the missing ID matches one of the "
                    "sub-materials. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] PERMEABILITY in MAT_FluidPoroMultiPhase "
                    "must be > 0.  Zero permeability means no flow is "
                    "possible and the system becomes singular. Signal: "
                    "linear solver fails immediately with "
                    "`zero pivot detected` or "
                    "`solver returned status: -3` (Belos converged but "
                    "true residual is NaN); a Darcy benchmark shows "
                    "zero pressure gradient. (Audit 2026-06-02.)"
                ),
                (
                    "For single-phase flow, set "
                    "NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE: 1 and "
                    "NUMMAT: 1.  Even single-phase flow requires the "
                    "full material hierarchy."
                ),
                (
                    "[Numerical] The initial condition must be set via "
                    "'initial_condition: type: by_function' in "
                    "porofluid_dynamic.  Without an initial condition "
                    "for pressure, the solver starts from zero which "
                    "may be unphysical. Signal: pressure field at "
                    "step 1 jumps from 0 to the BC values in a thin "
                    "boundary layer; for a consolidation benchmark "
                    "the early-time pressure dissipates as if t=0 was "
                    "an instant Dirichlet shock. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] For coupled poroelasticity problems, a "
                    "CLONING MATERIAL MAP section is required to map "
                    "the porofluid material to the structural "
                    "material. Signal: 4C aborts with "
                    "`CloningMaterialMap missing for porofluid -> "
                    "solid` during preprocessing; or the coupled run "
                    "decouples silently with the structure response "
                    "matching a dry-block reference. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] The PROBLEM TYPE string is "
                    "'porofluid_pressure_based' (lowercase, with "
                    "underscores), not 'Porofluid' or "
                    "'POROFLUID_PRESSURE_BASED'. Signal: parser abort "
                    "`unknown PROBLEMTYPE` listing the legal "
                    "lower-case strings. (Audit 2026-06-02.)"
                ),
                (
                    "BULKMODULUS in MAT_PoroDensityLawExp controls "
                    "fluid compressibility.  Very large values (> 1e6) "
                    "may cause numerical issues; very small values "
                    "produce unphysical compressibility."
                ),
            ],
            "typical_experiments": [
                {
                    "name": "single_phase_pressure_driven",
                    "description": (
                        "Single-phase Darcy flow through a 3D porous block.  "
                        "Pressure boundary conditions on inlet and outlet.  "
                        "Simplest porous media test case."
                    ),
                    "template_variant": "single_phase_3d",
                },
                {
                    "name": "multiphase_infiltration",
                    "description": (
                        "Two-phase flow (water displacing oil) in a porous "
                        "medium.  Uses saturation and pressure DoF types.  "
                        "Demonstrates the full material hierarchy."
                    ),
                },
            ],
        }

    # -- Templates -----------------------------------------------------------

    _TEMPLATES: dict[str, str] = {
        "single_phase_3d": """\
# ----------------------------------------------------------------
# Single-phase pressure-driven flow through a 3D porous block
#
# Simple Darcy flow with:
#   - Constant permeability, viscosity, and density
#   - Pressure BC: p = 1.0 at inlet, p = 0.0 at outlet
#   - Initial condition: p = 0.1 everywhere
#
# Material hierarchy:
#   MAT 1: MAT_FluidPoroMultiPhase (top-level, permeability)
#     -> MAT 10: MAT_FluidPoroSinglePhase (single phase)
#          -> MAT 101: MAT_FluidPoroSinglePhaseDofPressure (DoF = pressure)
#          -> MAT 102: MAT_PhaseLawConstraint (trivial phase law)
#          -> MAT 103: MAT_PoroDensityLawExp (density law)
#          -> MAT 104: MAT_FluidPoroViscosityLawConstant (viscosity)
#          -> MAT 105: MAT_FluidPoroRelPermeabilityLawConstant (k_r = 1)
#
# NOTE: Uses inline mesh (NODE COORDS + TRANSPORT ELEMENTS).
#       For Exodus mesh, replace with TRANSPORT GEOMETRY section.
# ----------------------------------------------------------------
TITLE:
  - "Single-phase Darcy flow through 3D porous block"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "porofluid_pressure_based"
DISCRETISATION:
  NUMSTRUCDIS: 0
  NUMALEDIS: 0
  NUMTHERMDIS: 0

# -- Porofluid dynamics (NOTE: lowercase section name!) ----------
porofluid_dynamic:
  total_simulation_time: 1.0
  time_integration:
    number_of_time_steps: 10
    time_step_size: 0.1
    theta: 1
  nonlinear_solver:
    linear_solver_id: 1
  output:
    porosity: false
  initial_condition:
    type: by_function
    function_id: 1
  flux_reconstruction:
    active: true
    solver_id: 1

IO:
  VERBOSITY: "Minimal"

# -- Solver ------------------------------------------------------
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "porofluid_solver"

# -- Material hierarchy ------------------------------------------
MATERIALS:
  # Top-level: multiphase container (single phase here)
  - MAT: 1
    MAT_FluidPoroMultiPhase:
      LOCAL: false
      PERMEABILITY: 1.0
      NUMMAT: 1
      MATIDS: [10]
      NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE: 1
  # Single fluid phase
  - MAT: 10
    MAT_FluidPoroSinglePhase:
      DENSITYLAWID: 103
      DENSITY: 1.0
      RELPERMEABILITYLAWID: 105
      VISCOSITY_LAW_ID: 104
      DOFTYPEID: 101
  # DoF type: absolute pressure
  - MAT: 101
    MAT_FluidPoroSinglePhaseDofPressure:
      PHASELAWID: 102
  # Phase law: trivial constraint (single-phase)
  - MAT: 102
    MAT_PhaseLawConstraint: {}
  # Density law: exponential (nearly incompressible)
  - MAT: 103
    MAT_PoroDensityLawExp:
      BULKMODULUS: 100.0
  # Viscosity: constant
  - MAT: 104
    MAT_FluidPoroViscosityLawConstant:
      VALUE: 0.01
  # Relative permeability: constant (k_r = 1 for single phase)
  - MAT: 105
    MAT_FluidPoroRelPermeabilityLawConstant:
      VALUE: 1.0

# -- Initial condition: uniform pressure p = 0.1 ----------------
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.1"

# -- Boundary conditions (pressure Dirichlet) --------------------
DESIGN SURF DIRICH CONDITIONS:
  # Inlet face: p = 1.0
  - E: 1
    ENTITY_TYPE: node_set_id
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [0]
  # Outlet face: p = 0.0
  - E: 2
    ENTITY_TYPE: node_set_id
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]

# -- Inline mesh (simple 2x2x5 hex8 block) ----------------------
# Replace with TRANSPORT GEOMETRY + Exodus for production meshes
NODE COORDS:
  - "NODE 1  COORD 0.0 0.0 0.0"
  - "NODE 2  COORD 1.0 0.0 0.0"
  - "NODE 3  COORD 1.0 1.0 0.0"
  - "NODE 4  COORD 0.0 1.0 0.0"
  - "NODE 5  COORD 0.0 0.0 1.0"
  - "NODE 6  COORD 1.0 0.0 1.0"
  - "NODE 7  COORD 1.0 1.0 1.0"
  - "NODE 8  COORD 0.0 1.0 1.0"
  - "NODE 9  COORD 0.0 0.0 2.0"
  - "NODE 10 COORD 1.0 0.0 2.0"
  - "NODE 11 COORD 1.0 1.0 2.0"
  - "NODE 12 COORD 0.0 1.0 2.0"
TRANSPORT ELEMENTS:
  - "1 POROFLUIDMULTIPHASE HEX8 1 2 3 4 5 6 7 8 MAT 1 TYPE PoroFluidMultiPhase"
  - "2 POROFLUIDMULTIPHASE HEX8 5 6 7 8 9 10 11 12 MAT 1 TYPE PoroFluidMultiPhase"

# -- Design surface topology for BCs ----------------------------
DSURF-NODE TOPOLOGY:
  # Inlet (z = 0 face)
  - "NODE 1 DSURFACE 1"
  - "NODE 2 DSURFACE 1"
  - "NODE 3 DSURFACE 1"
  - "NODE 4 DSURFACE 1"
  # Outlet (z = 2 face)
  - "NODE 9  DSURFACE 2"
  - "NODE 10 DSURFACE 2"
  - "NODE 11 DSURFACE 2"
  - "NODE 12 DSURFACE 2"

RESULT DESCRIPTION:
  - POROFLUIDMULTIPHASE:
      DIS: "porofluid"
      NODE: 1
      QUANTITY: "phi1"
      VALUE: 1.0
      TOLERANCE: 1e-06
""",
    }

    def get_template(self, variant: str = "default") -> str:
        if variant == "default":
            variant = "single_phase_3d"
        if variant not in self._TEMPLATES:
            available = ", ".join(sorted(self._TEMPLATES))
            raise ValueError(
                f"Unknown template variant {variant!r} for {self.module_key}. "
                f"Available: {available}"
            )
        return self._TEMPLATES[variant]

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "single_phase_3d",
                "description": (
                    "Single-phase pressure-driven Darcy flow through a 3D "
                    "porous block.  Demonstrates the full material hierarchy "
                    "(MAT_FluidPoroMultiPhase -> MAT_FluidPoroSinglePhase -> "
                    "density/viscosity/permeability laws).  Uses inline mesh "
                    "with HEX8 elements and pressure Dirichlet BCs."
                ),
            },
        ]

    # -- Validation ----------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        """Physics-aware validation of porous media parameters.

        Expected keys in *params* (all optional):
            PERMEABILITY       - intrinsic permeability
            BULKMODULUS        - fluid bulk modulus (density law)
            VISCOSITY          - dynamic viscosity
            REL_PERMEABILITY   - relative permeability
            DENSITY            - reference fluid density
            NUMMAT             - number of sub-materials
            MATIDS             - list of sub-material IDs
            NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE - number of fluid phases
            time_step_size     - time step
            theta              - one-step-theta parameter
        """
        errors: list[str] = []

        # Check permeability
        perm = params.get("PERMEABILITY")
        if perm is not None:
            try:
                k = float(perm)
                if k <= 0:
                    errors.append(
                        f"PERMEABILITY must be > 0, got {k}.  "
                        f"Zero permeability means no flow is possible "
                        f"and the system becomes singular."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"PERMEABILITY must be a positive number, got {perm!r}."
                )

        # Check bulk modulus (density law)
        bulk = params.get("BULKMODULUS")
        if bulk is not None:
            try:
                K = float(bulk)
                if K <= 0:
                    errors.append(
                        f"BULKMODULUS must be > 0, got {K}."
                    )
                elif K > 1e8:
                    errors.append(
                        f"WARNING: BULKMODULUS = {K} is very large.  "
                        f"This may cause numerical issues.  Consider "
                        f"using MAT_PoroDensityLawConstant for "
                        f"incompressible fluids."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"BULKMODULUS must be a positive number, got {bulk!r}."
                )

        # Check viscosity
        visc = params.get("VISCOSITY") or params.get("VALUE")
        if visc is not None:
            try:
                mu = float(visc)
                if mu <= 0:
                    errors.append(
                        f"Viscosity must be > 0, got {mu}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"Viscosity must be a positive number, got {visc!r}."
                )

        # Check relative permeability
        rel_perm = params.get("REL_PERMEABILITY")
        if rel_perm is not None:
            try:
                kr = float(rel_perm)
                if kr <= 0 or kr > 1:
                    errors.append(
                        f"Relative permeability must be in (0, 1], got {kr}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"Relative permeability must be a number in (0, 1], "
                    f"got {rel_perm!r}."
                )

        # Check density
        density = params.get("DENSITY")
        if density is not None:
            try:
                rho = float(density)
                if rho <= 0:
                    errors.append(
                        f"DENSITY must be > 0, got {rho}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"DENSITY must be a positive number, got {density!r}."
                )

        # Check material hierarchy consistency
        nummat = params.get("NUMMAT")
        matids = params.get("MATIDS")
        numphases = params.get("NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE")

        if nummat is not None and matids is not None:
            try:
                nm = int(nummat)
                if isinstance(matids, list) and len(matids) != nm:
                    errors.append(
                        f"NUMMAT = {nm} but MATIDS has {len(matids)} entries.  "
                        f"These must match."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"NUMMAT must be a positive integer, got {nummat!r}."
                )

        if numphases is not None:
            try:
                np_ = int(numphases)
                if np_ < 1:
                    errors.append(
                        f"NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE must be >= 1, "
                        f"got {np_}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE must be a "
                    f"positive integer, got {numphases!r}."
                )

        # Check time integration parameter theta
        theta = params.get("theta")
        if theta is not None:
            try:
                th = float(theta)
                if th < 0 or th > 1:
                    errors.append(
                        f"theta must be in [0, 1], got {th}.  "
                        f"Use 1.0 for implicit Euler (stable, first-order) "
                        f"or 0.5 for Crank-Nicolson (second-order)."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"theta must be a number in [0, 1], got {theta!r}."
                )

        # Check time step
        dt = params.get("time_step_size")
        if dt is not None:
            try:
                step = float(dt)
                if step <= 0:
                    errors.append(
                        f"time_step_size must be > 0, got {step}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"time_step_size must be a positive number, got {dt!r}."
                )

        return errors
