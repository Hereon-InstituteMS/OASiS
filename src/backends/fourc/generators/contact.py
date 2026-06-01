"""Generator for contact mechanics physics module (Penalty, Uzawa, Nitsche).

Covers mortar-based contact between deformable bodies using the
CONTACT DYNAMIC + MORTAR COUPLING sections in 4C.  Produces validated,
working .4C.yaml templates for LLM-driven contact simulation setup.
"""

from __future__ import annotations

from typing import Any

from .base import BaseGenerator


class ContactGenerator(BaseGenerator):
    """Generator for contact mechanics problems in 4C.

    Supports penalty, Uzawa, and Nitsche contact enforcement strategies
    with mortar-based interface coupling between deformable 3D bodies.
    """

    module_key = "contact"
    display_name = "Contact Mechanics (Penalty / Uzawa / Nitsche)"
    problem_type = "Structure"

    # -- Knowledge -----------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The contact mechanics module in 4C models frictionless or "
                "frictional contact between deformable bodies using mortar-based "
                "surface coupling.  It adds two mandatory sections on top of the "
                "standard structural analysis: CONTACT DYNAMIC (enforcement "
                "strategy and parameters) and MORTAR COUPLING (mortar algorithm "
                "settings).  Contact surfaces are defined via "
                "DESIGN SURF MORTAR CONTACT CONDITIONS 3D, where each contact "
                "interface consists of a Slave and a Master surface pair sharing "
                "the same InterfaceID.  The PROBLEM TYPE remains 'Structure'.  "
                "Quasi-static contact problems typically require load stepping "
                "(many small time steps with DYNAMICTYPE GenAlpha or Statics) to "
                "achieve convergence as the contact zone evolves."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "STRUCTURAL DYNAMIC",
                "MORTAR COUPLING",
                "CONTACT DYNAMIC",
                "SOLVER 1",
                "MATERIALS",
                "STRUCTURE GEOMETRY",
                "DESIGN SURF MORTAR CONTACT CONDITIONS 3D",
            ],
            "materials": {
                "MAT_Struct_StVenantKirchhoff": {
                    "description": (
                        "Standard structural material for contact bodies.  "
                        "Use different material IDs for different bodies to "
                        "allow distinct stiffness values."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus E",
                            "range": "> 0",
                        },
                        "NUE": {
                            "description": "Poisson's ratio nu",
                            "range": "0 < nu < 0.5",
                        },
                        "DENS": {
                            "description": (
                                "Mass density.  Set > 0 for dynamic contact; "
                                "can be 0 for quasi-static if using Statics."
                            ),
                            "range": ">= 0",
                        },
                    },
                },
                "MAT_ElastHyper + ELAST_CoupNeoHooke": {
                    "description": (
                        "Neo-Hookean hyperelastic material for large-deformation "
                        "contact.  Recommended when large strains are expected "
                        "near the contact zone."
                    ),
                    "parameters": {
                        "YOUNG": {"description": "Young's modulus", "range": "> 0"},
                        "NUE": {"description": "Poisson's ratio", "range": "0 < nu < 0.5"},
                        "DENS": {"description": "Mass density", "range": ">= 0"},
                    },
                },
            },
            "contact_strategies": {
                "Penalty": {
                    "description": (
                        "Penalty enforcement: adds a stiff penalty spring when "
                        "penetration is detected.  Simple and robust for small "
                        "penetration.  Requires tuning PENALTYPARAM: too low "
                        "allows excessive penetration; too high causes "
                        "ill-conditioning."
                    ),
                    "key_parameter": "PENALTYPARAM",
                    "typical_range": "1e2 -- 1e5 (problem-dependent)",
                },
                "Uzawa": {
                    "description": (
                        "Augmented Lagrangian (Uzawa iteration): iteratively "
                        "updates Lagrange multipliers to enforce zero penetration.  "
                        "More accurate than pure penalty but more expensive "
                        "(multiple Newton solves per time step)."
                    ),
                    "key_parameter": "PENALTYPARAM (initial penalty for augmentation)",
                    "typical_range": "1e2 -- 1e4",
                },
                "Nitsche": {
                    "description": (
                        "Nitsche's method: variationally consistent penalty-like "
                        "formulation.  Combines accuracy of Lagrange multipliers "
                        "with the simplicity of penalty.  Requires consistent "
                        "linearization."
                    ),
                    "key_parameter": "PENALTYPARAM (Nitsche stabilization parameter)",
                    "typical_range": "1e2 -- 1e5",
                },
            },
            "mortar_coupling": {
                "ALGORITHM": (
                    "'Mortar' -- the mortar method for contact surface coupling.  "
                    "This is the standard and recommended algorithm."
                ),
                "LM_SHAPEFCN": (
                    "'Standard' -- standard Lagrange multiplier shape functions.  "
                    "Alternative: 'Dual' for dual Lagrange multipliers (more "
                    "robust for non-matching meshes)."
                ),
            },
            "contact_conditions": {
                "format": (
                    "DESIGN SURF MORTAR CONTACT CONDITIONS 3D defines contact "
                    "surface pairs.  Each interface requires TWO entries with "
                    "the SAME InterfaceID: one with Side: Slave and one with "
                    "Side: Master."
                ),
                "slave_selection": (
                    "The slave surface should generally be the finer mesh or "
                    "the softer body.  This ensures better constraint "
                    "satisfaction."
                ),
                "example": (
                    "- NODE_SET_NAME: contact_slave\n"
                    "  InterfaceID: 1\n"
                    "  Side: Slave\n"
                    "- NODE_SET_NAME: contact_master\n"
                    "  InterfaceID: 1\n"
                    "  Side: Master"
                ),
            },
            "solver": {
                "small_problems": {
                    "SOLVER": "UMFPACK",
                    "description": (
                        "Direct solver, robust for contact problems up to ~50k "
                        "DOFs.  Recommended for initial testing and debugging."
                    ),
                },
                "large_problems": {
                    "SOLVER": "Belos",
                    "AZPREC": "MueLu",
                    "description": (
                        "Iterative solver with AMG preconditioner.  Contact "
                        "problems can be ill-conditioned; may need tighter "
                        "tolerances or saddle-point preconditioners."
                    ),
                },
            },
            "time_integration": {
                "quasi_static": (
                    "For quasi-static contact: use DYNAMICTYPE 'GenAlpha' with "
                    "small load steps (NUMSTEP = 10--100) to gradually establish "
                    "contact.  Sudden load application often fails to converge."
                ),
                "dynamic_contact": (
                    "For dynamic contact (impact): use DYNAMICTYPE 'GenAlpha' with "
                    "DENS > 0 in all materials.  Time step must be small enough "
                    "to capture the contact event."
                ),
                "TIMESTEP": "Time step size (load stepping parameter for quasi-static).",
                "NUMSTEP": "Number of load/time steps.",
                "MAXTIME": "Maximum simulation time.",
            },
            "pitfalls": [
                (
                    "[Input] Both MORTAR COUPLING and CONTACT DYNAMIC "
                    "sections are REQUIRED.  Missing either one causes "
                    "4C to crash or silently ignore the contact "
                    "conditions. Signal: input parser aborts with "
                    "`MORTAR COUPLING section required for contact` / "
                    "`CONTACT DYNAMIC section missing`, OR the run "
                    "completes but two bodies pass through each other "
                    "(zero contact pressure, identical displacement to "
                    "no-contact reference). (Audit 2026-06-02.)"
                ),
                (
                    "[Input] Each contact interface needs BOTH a Slave "
                    "and a Master surface with the same InterfaceID.  "
                    "A missing partner surface causes the contact "
                    "search to fail. Signal: log line `no master "
                    "partner found for interface X` / `MortarInterface: "
                    "InterfaceID X has 0 master elements`. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Numerical] PENALTYPARAM tuning is critical.  Too "
                    "low: excessive penetration (inaccurate results).  "
                    "Too high: ill-conditioned system matrix (Newton "
                    "divergence).  Start with 1e3 and adjust based on "
                    "penetration depth. Signal: NOX log shows "
                    "non-monotonic residual or condition-number warning "
                    "above ~1e14; OR max contact penetration in "
                    "post-processing exceeds ~5% of the contact-pair "
                    "element edge length. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] Quasi-static contact MUST use load "
                    "stepping.  Applying the full load in one step "
                    "almost always causes Newton divergence because "
                    "the contact zone changes drastically. Signal: "
                    "NOX hits StatusTest::MaxIters at the first step; "
                    "the residual oscillates between two values "
                    "associated with the contact set toggling each "
                    "iteration. (Audit 2026-06-02.)"
                ),
                (
                    "The slave surface should be the finer mesh or "
                    "the softer body.  Swapping slave and master can "
                    "cause convergence issues or inaccurate contact "
                    "pressure distributions."
                ),
                (
                    "[Numerical] KINEM must be 'nonlinear' for contact "
                    "problems to correctly handle the geometric "
                    "nonlinearity of contact gap computation.  Linear "
                    "kinematics produce wrong results. Signal: contact "
                    "pressure distribution is symmetric about the "
                    "initial geometry instead of following the "
                    "deformed configuration; for a Hertz benchmark "
                    "the contact radius is off by ~20% from the "
                    "analytical (3*F*R/(4*E*))^(1/3). (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] Use node_set_id or NODE_SET_NAME (not "
                    "element set) for contact surface definitions.  "
                    "The mortar integration is surface-based and "
                    "requires node sets. Signal: parser error "
                    "`expected node set for contact surface, got "
                    "element set` / `MortarInterface needs DNODE not "
                    "DELE`. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] Contact surfaces must not overlap or "
                    "intersect initially.  An initial penetration can "
                    "cause the Newton iteration to diverge at the "
                    "first step. Signal: NOX residual at iteration 0 "
                    "is already large (~|PENALTYPARAM * penetration|) "
                    "and the first Newton update overshoots; "
                    "MortarInterface diagnostic prints `initial "
                    "penetration X.X > tolerance`. (Audit 2026-06-02.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "two_block_compression",
                    "description": (
                        "Two 3D blocks pressed together.  Bottom block fixed, "
                        "top block displaced downward.  Penalty contact.  "
                        "Classic benchmark for contact algorithm verification."
                    ),
                    "template_variant": "penalty_3d",
                },
                {
                    "name": "hertz_contact",
                    "description": (
                        "Hertzian contact: cylinder or sphere on flat surface.  "
                        "Analytical solution available for contact pressure "
                        "distribution and contact radius.  Good for validation."
                    ),
                },
            ],
        }

    # -- Templates -----------------------------------------------------------

    _TEMPLATES: dict[str, str] = {
        "penalty_3d": """\
# FORMAT TEMPLATE — all numerical values are placeholders.
# ----------------------------------------------------------------
# 3D Contact: Two blocks in compression with penalty enforcement
#
# Geometry (requires Exodus mesh with named node sets):
#   - Block 1 (bottom): fixed on bottom face
#   - Block 2 (top): pushed downward via prescribed displacement
#   - Contact between top-of-block1 (master) and bottom-of-block2 (slave)
#
# The mesh must define node sets:
#   wall    -- bottom face of block 1 (fully fixed)
#   pushing -- top face of block 2 (prescribed displacement)
#   slave   -- bottom face of block 2 (contact slave)
#   master  -- top face of block 1 (contact master)
# ----------------------------------------------------------------
TITLE:
  - "Contact -- two 3D blocks in compression with penalty method"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"

IO:
  STRUCT_STRESS: "Cauchy"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: <output_interval_steps>
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
  STRESS_STRAIN: true
  ELEMENT_MAT_ID: true

# -- Structural dynamics (quasi-static via load stepping) --------
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "GenAlpha"
  TIMESTEP: <load_step_size>
  NUMSTEP: <number_of_load_steps>
  MAXTIME: <end_time>
  RESTARTEVERY: <restart_interval>
  TOLDISP: <displacement_tolerance>
  TOLRES: <residual_tolerance>
  LINEAR_SOLVER: 1

# -- Mortar coupling (required for contact) ----------------------
MORTAR COUPLING:
  ALGORITHM: Mortar
  LM_SHAPEFCN: Standard

# -- Contact strategy --------------------------------------------
CONTACT DYNAMIC:
  STRATEGY: Penalty
  PENALTYPARAM: <penalty_parameter>
  LINEAR_SOLVER: 1

# -- Solver ------------------------------------------------------
SOLVER 1:
  SOLVER: "UMFPACK"

# -- Materials (two distinct bodies) -----------------------------
MATERIALS:
  # Block 1 (bottom, stiffer)
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: <Young_modulus_block1>
      NUE: <Poisson_ratio_block1>
      DENS: <density_block1>
  # Block 2 (top, softer)
  - MAT: 2
    MAT_Struct_StVenantKirchhoff:
      YOUNG: <Young_modulus_block2>
      NUE: <Poisson_ratio_block2>
      DENS: <density_block2>

# -- Loading function (gradual compression) ----------------------
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "compression"
  - VARIABLE: 0
    NAME: "compression"
    TYPE: "linearinterpolation"
    NUMPOINTS: <number_of_interpolation_points>
    TIMES: [<interpolation_times>]
    VALUES: [<interpolation_values>]

# -- Contact surfaces --------------------------------------------
DESIGN SURF MORTAR CONTACT CONDITIONS 3D:
  - NODE_SET_NAME: slave
    InterfaceID: 1
    Side: Slave
  - NODE_SET_NAME: master
    InterfaceID: 1
    Side: Master

# -- Boundary conditions -----------------------------------------
DESIGN SURF DIRICH CONDITIONS:
  # Bottom of block 1: fully fixed
  - NODE_SET_NAME: wall
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  # Top of block 2: prescribed compression in x
  - NODE_SET_NAME: pushing
    NUMDOF: 3
    ONOFF: [1, 0, 0]
    VAL: [<prescribed_displacement>, 0.0, 0.0]
    FUNCT: [1, 0, 0]

# -- Geometry (Exodus mesh) --------------------------------------
STRUCTURE GEOMETRY:
  FILE: contact_two_blocks.e
  SHOW_INFO: detailed_summary
  ELEMENT_BLOCKS:
    - ID: 1
      SOLID:
        HEX8:
          MAT: 1
          KINEM: nonlinear
    - ID: 2
      SOLID:
        HEX8:
          MAT: 2
          KINEM: nonlinear

RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 1
      QUANTITY: "dispx"
      VALUE: <expected_result_value>
      TOLERANCE: <result_tolerance>
""",
    }

    def get_template(self, variant: str = "default") -> str:
        if variant == "default":
            variant = "penalty_3d"
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
                "name": "penalty_3d",
                "description": (
                    "Two 3D blocks in compression with penalty contact.  "
                    "Bottom block fixed, top block pushed down.  "
                    "Demonstrates MORTAR COUPLING, CONTACT DYNAMIC, and "
                    "DESIGN SURF MORTAR CONTACT CONDITIONS 3D sections.  "
                    "Uses quasi-static load stepping with GenAlpha."
                ),
            },
        ]

    # -- Validation ----------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        """Physics-aware validation of contact parameters.

        Expected keys in *params* (all optional):
            PENALTYPARAM   - penalty stiffness
            STRATEGY       - contact enforcement strategy
            YOUNG          - Young's modulus of materials
            NUE            - Poisson's ratio
            KINEM          - kinematics type
            slave_surface  - whether slave surface is defined
            master_surface - whether master surface is defined
            TIMESTEP       - time step / load step size
            NUMSTEP        - number of load steps
        """
        errors: list[str] = []

        # Check penalty parameter
        penalty = params.get("PENALTYPARAM")
        if penalty is not None:
            try:
                p = float(penalty)
                if p <= 0:
                    errors.append(
                        f"PENALTYPARAM must be > 0, got {p}.  "
                        f"A non-positive penalty parameter is unphysical."
                    )
                elif p < 10:
                    errors.append(
                        f"WARNING: PENALTYPARAM = {p} is very low.  "
                        f"This will allow large penetration.  Typical "
                        f"range: 1e2 to 1e5."
                    )
                elif p > 1e8:
                    errors.append(
                        f"WARNING: PENALTYPARAM = {p} is very high.  "
                        f"This may cause ill-conditioning and Newton "
                        f"divergence.  Typical range: 1e2 to 1e5."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"PENALTYPARAM must be a positive number, got {penalty!r}."
                )

        # Check strategy
        strategy = params.get("STRATEGY")
        valid_strategies = {"Penalty", "Uzawa", "Nitsche"}
        if strategy is not None and strategy not in valid_strategies:
            errors.append(
                f"STRATEGY must be one of {sorted(valid_strategies)}, "
                f"got {strategy!r}."
            )

        # Check slave/master pairs
        has_slave = params.get("slave_surface")
        has_master = params.get("master_surface")
        if has_slave is not None and has_master is not None:
            if has_slave and not has_master:
                errors.append(
                    "Contact interface has a Slave surface but no Master.  "
                    "Each interface requires BOTH Slave and Master surfaces "
                    "with the same InterfaceID."
                )
            elif has_master and not has_slave:
                errors.append(
                    "Contact interface has a Master surface but no Slave.  "
                    "Each interface requires BOTH Slave and Master surfaces "
                    "with the same InterfaceID."
                )

        # Check kinematics
        kinem = params.get("KINEM")
        if kinem is not None and str(kinem).lower() == "linear":
            errors.append(
                "KINEM is 'linear' but contact problems require 'nonlinear' "
                "kinematics to correctly compute the geometric nonlinearity "
                "of the contact gap."
            )

        # Check Young's modulus
        young = params.get("YOUNG")
        if young is not None:
            try:
                e = float(young)
                if e <= 0:
                    errors.append(f"YOUNG must be > 0, got {e}.")
            except (TypeError, ValueError):
                errors.append(f"YOUNG must be a positive number, got {young!r}.")

        # Check Poisson's ratio
        nue = params.get("NUE")
        if nue is not None:
            try:
                nu = float(nue)
                if nu <= 0 or nu >= 0.5:
                    errors.append(
                        f"NUE (Poisson's ratio) must be in (0, 0.5), got {nu}."
                    )
            except (TypeError, ValueError):
                errors.append(f"NUE must be a number in (0, 0.5), got {nue!r}.")

        # Check load stepping
        numstep = params.get("NUMSTEP")
        if numstep is not None:
            try:
                ns = int(numstep)
                if ns < 2:
                    errors.append(
                        f"WARNING: NUMSTEP = {ns}.  Contact problems almost "
                        f"always require load stepping (NUMSTEP >= 5) for "
                        f"Newton convergence.  Consider increasing NUMSTEP."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"NUMSTEP must be a positive integer, got {numstep!r}."
                )

        # Check TIMESTEP
        timestep = params.get("TIMESTEP")
        if timestep is not None:
            try:
                dt = float(timestep)
                if dt <= 0:
                    errors.append(
                        f"TIMESTEP must be > 0, got {dt}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"TIMESTEP must be a positive number, got {timestep!r}."
                )

        return errors
