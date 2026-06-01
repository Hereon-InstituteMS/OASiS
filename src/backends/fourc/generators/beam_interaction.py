"""Beam Interaction generator for 4C.

Covers beam-to-beam and beam-to-solid contact and meshtying.  Beam elements
(BEAM3R, BEAM3EB) can interact with other beams via contact or with solid
elements via embedded meshtying or contact.  Applications include fiber
networks, reinforced composites, stent deployment, and knitted/woven
textile mechanics.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class BeamInteractionGenerator(BaseGenerator):
    """Generator for beam interaction (contact/meshtying) problems in 4C."""

    module_key = "beam_interaction"
    display_name = "Beam Interaction (Beam-to-Beam/Beam-to-Solid Contact)"
    problem_type = "Structure"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The beam interaction module handles contact and meshtying "
                "between beam elements (beam-to-beam) and between beam "
                "and solid elements (beam-to-solid).  Beam-to-beam contact "
                "captures fibre-fibre interactions in networks, knots, and "
                "woven structures.  Beam-to-solid contact and meshtying "
                "couples 1-D beam elements with 3-D solid elements for "
                "applications like rebar in concrete, stent-in-vessel, or "
                "fibre-reinforced composites.  The PROBLEM TYPE is "
                "'Structure' (same as standard structural mechanics) but "
                "with additional BEAM INTERACTION sections.  The dynamics "
                "section is STRUCTURAL DYNAMIC.  Beams use BEAM3R or "
                "BEAM3EB elements; solids use standard SOLID HEX8/TET4.  "
                "The BEAM INTERACTION section configures the contact "
                "algorithm, penalty parameters, search strategy, and "
                "coupling type (contact, meshtying, or tied).  A "
                "BINNING STRATEGY section is typically required for "
                "efficient spatial search."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "STRUCTURAL DYNAMIC",
                "BEAM INTERACTION",
                "SOLVER 1",
                "MATERIALS",
                "BINNING STRATEGY",
            ],
            "optional_sections": [
                "BEAM INTERACTION/BEAM TO BEAM CONTACT",
                "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING",
                "BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT",
                "BEAM INTERACTION/BEAM TO SOLID SURFACE MESHTYING",
                "STRUCTURAL DYNAMIC/GENALPHA",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/BEAMS",
                "IO/RUNTIME VTK OUTPUT/STRUCTURE",
            ],
            "materials": {
                "MAT_BeamReissnerElastHyper": {
                    "description": (
                        "Geometrically exact Reissner beam material.  "
                        "Defines the cross-sectional properties of "
                        "beam elements."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus",
                            "range": "> 0",
                        },
                        "POISSONRATIO": {
                            "description": "Poisson's ratio",
                            "range": "(0, 0.5)",
                        },
                        "DENS": {
                            "description": "Mass density",
                            "range": "> 0",
                        },
                        "CROSSAREA": {
                            "description": "Cross-sectional area",
                            "range": "> 0",
                        },
                        "SHEARCORR": {
                            "description": "Shear correction factor",
                            "range": "> 0 (typically 1.0)",
                        },
                        "MOMINPOL": {
                            "description": "Polar moment of inertia",
                            "range": "> 0",
                        },
                        "MOMIN2": {
                            "description": "Second moment of area (axis 2)",
                            "range": "> 0",
                        },
                        "MOMIN3": {
                            "description": "Second moment of area (axis 3)",
                            "range": "> 0",
                        },
                    },
                },
                "MAT_Struct_StVenantKirchhoff": {
                    "description": (
                        "Standard structural material for solid elements "
                        "in beam-to-solid interaction."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus",
                            "range": "> 0",
                        },
                        "NUE": {
                            "description": "Poisson's ratio",
                            "range": "[0, 0.5)",
                        },
                        "DENS": {
                            "description": "Mass density",
                            "range": "> 0",
                        },
                    },
                },
            },
            "solver": {
                "structure_solver": {
                    "type": "UMFPACK",
                    "notes": (
                        "Direct solver works well for beam-interaction "
                        "problems, which tend to have moderate DOF counts."
                    ),
                },
            },
            "beam_interaction_parameters": {
                "STRATEGY": (
                    "Interaction strategy: 'beam_to_beam_contact', "
                    "'beam_to_solid_volume_meshtying', "
                    "'beam_to_solid_surface_contact', "
                    "'beam_to_solid_surface_meshtying'."
                ),
                "PENALTY_PARAMETER": (
                    "Penalty stiffness for contact enforcement.  "
                    "Controls penetration: larger values reduce "
                    "penetration but worsen conditioning."
                ),
                "SEARCH_RADIUS": (
                    "Search radius for detecting potential beam-beam "
                    "or beam-solid interaction pairs.  Must exceed the "
                    "expected gap + beam radius."
                ),
                "COUPLING_TYPE": (
                    "For beam-to-solid: 'consistent' (mortar-like) or "
                    "'penalty' (simpler, less accurate)."
                ),
            },
            "pitfalls": [
                (
                    "[Input] BINNING STRATEGY is required for spatial "
                    "search.  BIN_SIZE_LOWER_BOUND must be at least as "
                    "large as the element sizes to ensure all "
                    "interaction pairs are detected. Signal: "
                    "BeamInteraction diagnostic prints "
                    "`zero pairs found` for a geometry where "
                    "intersections clearly exist; OR runtime warning "
                    "`bin size smaller than max element size, search "
                    "may miss pairs`. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] The penalty parameter for beam "
                    "contact must be carefully chosen.  Too large "
                    "causes ill-conditioning and convergence failure; "
                    "too small permits excessive penetration. Signal: "
                    "NOX condition-number printout exceeds ~1e14 "
                    "(too high), or post-processed beam-beam distance "
                    "at a contact point goes negative by more than "
                    "5% of beam radius (too low). (Audit 2026-06-02.)"
                ),
                (
                    "Beam-to-beam contact uses a point-to-point or "
                    "line-to-line formulation.  The search algorithm "
                    "must be configured to detect close beam pairs "
                    "efficiently (SEARCH_RADIUS)."
                ),
                (
                    "[Numerical] For beam-to-solid volume meshtying, "
                    "the beam elements must lie within the solid "
                    "mesh volume.  Beams outside the solid domain "
                    "are not coupled. Signal: BeamToSolidMeshtying "
                    "diagnostic prints `0 of N beam segments "
                    "coupled`; the beam displaces freely as if no "
                    "solid mesh existed. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] Beam-to-solid surface contact "
                    "requires the beam to approach a solid surface. "
                    " The contact detection uses the beam centerline "
                    "distance to the surface, not the beam radius. "
                    " The gap offset must account for the beam "
                    "cross-section. Signal: visualize shows the beam "
                    "centerline at the solid surface (gap = 0) "
                    "instead of one beam-radius offset; or contact "
                    "force first activates one element-edge later "
                    "than expected geometrically. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] Beam materials use dedicated types "
                    "(MAT_BeamReissnerElastHyper) with cross-"
                    "sectional properties (CROSSAREA, MOMINPOL, "
                    "MOMIN2, MOMIN3).  These must be geometrically "
                    "consistent with the beam cross-section. Signal: "
                    "input parser aborts with `MAT_BeamReissnerElast"
                    "Hyper requires CROSSAREA` / `MOMINPOL not "
                    "specified`; or beam bends differently from "
                    "Euler-Bernoulli prediction (the eigen-frequency "
                    "of a cantilever differs from sqrt(EI/(rho*A*L^4))"
                    " by more than 10%). (Audit 2026-06-02.)"
                ),
                (
                    "[Output] Use IO/RUNTIME VTK OUTPUT/BEAMS for "
                    "beam visualisation.  Standard STRUCTURE output "
                    "only shows solid elements. Signal: visualize"
                    "('list') under work_dir shows structure-*.vtu "
                    "files but no beams-*.vtu; the LLM-visible "
                    "summary reports an empty beam-field group "
                    "though the run completed. (Audit 2026-06-02.)"
                ),
                (
                    "The PROBLEM TYPE is 'Structure' (not a dedicated "
                    "beam-interaction type).  The BEAM INTERACTION "
                    "section activates the beam contact/meshtying "
                    "on top of the standard structural problem."
                ),
            ],
            "typical_experiments": [
                {
                    "name": "fiber_network_contact",
                    "description": (
                        "Two crossing beams with contact.  Tests "
                        "beam-to-beam contact detection, penalty "
                        "enforcement, and large-deformation beam "
                        "mechanics."
                    ),
                    "template_variant": "beam_contact_3d",
                },
                {
                    "name": "beam_in_solid_meshtying",
                    "description": (
                        "A beam embedded in a solid block via volume "
                        "meshtying.  Tests beam-to-solid coupling, "
                        "constraint enforcement, and stress transfer."
                    ),
                    "template_variant": "beam_solid_meshtying_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "beam_contact_3d",
                "description": (
                    "3-D beam-to-beam contact: crossing fibers with "
                    "penalty contact.  BEAM3R elements, "
                    "MAT_BeamReissnerElastHyper, UMFPACK solver."
                ),
            },
            {
                "name": "beam_solid_meshtying_3d",
                "description": (
                    "3-D beam-to-solid volume meshtying: beam embedded "
                    "in solid block.  BEAM3R + SOLID HEX8, penalty "
                    "or mortar coupling, UMFPACK solver."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "beam_contact_3d") -> str:
        templates = {
            "beam_contact_3d": self._template_beam_contact_3d,
            "beam_solid_meshtying_3d": self._template_beam_solid_meshtying_3d,
        }
        if variant == "default":
            variant = "beam_contact_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_beam_contact_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D Beam-to-Beam Contact
            #
            # Two or more beams in contact.  Penalty-based contact detects
            # and resolves beam-beam interactions (crossing, sliding, etc.).
            #
            # Mesh: requires beam mesh with:
            #   element_block 1 = beam 1 (LINE2)
            #   element_block 2 = beam 2 (LINE2)
            #   node_set 1 = beam 1 clamped end
            #   node_set 2 = beam 2 clamped end
            # ---------------------------------------------------------------
            TITLE:
              - "3-D beam-to-beam contact -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Structure"
            IO:
              STDOUTEVERY: <stdout_interval>
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/BEAMS:
              OUTPUT_BEAMS: true
              DISPLACEMENT: true
              USE_ABSOLUTE_POSITIONS: true
              TRIAD_VISUALISATION_POINT: true

            # == Structural dynamics ===========================================
            STRUCTURAL DYNAMIC:
              DYNAMICTYPE: "Statics"
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              LINEAR_SOLVER: 1
              PREDICT: "ConstDisVelAcc"
              TOLRES: <residual_tolerance>
              TOLDISP: <displacement_tolerance>
              RESULTSEVERY: <results_output_interval>

            # == Beam interaction =============================================
            BEAM INTERACTION:
              STRATEGY: "beam_to_beam_contact"
              SEARCH_RADIUS: <contact_search_radius>
            BEAM INTERACTION/BEAM TO BEAM CONTACT:
              CONTACT_TYPE: "point_to_point"
              PENALTY_PARAMETER: <contact_penalty_parameter>
              GAP_SHIFT: <gap_shift>

            # == Binning for spatial search ===================================
            BINNING STRATEGY:
              BIN_SIZE_LOWER_BOUND: <bin_size_lower_bound>

            # == Solver ========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "structure_solver"

            # == Materials =====================================================
            MATERIALS:
              # Beam 1 material
              - MAT: 1
                MAT_BeamReissnerElastHyper:
                  YOUNG: <beam1_Young_modulus>
                  POISSONRATIO: <beam1_Poisson_ratio>
                  DENS: <beam1_density>
                  CROSSAREA: <beam1_cross_section_area>
                  SHEARCORR: <beam1_shear_correction_factor>
                  MOMINPOL: <beam1_polar_moment_of_inertia>
                  MOMIN2: <beam1_moment_of_inertia_2>
                  MOMIN3: <beam1_moment_of_inertia_3>
              # Beam 2 material
              - MAT: 2
                MAT_BeamReissnerElastHyper:
                  YOUNG: <beam2_Young_modulus>
                  POISSONRATIO: <beam2_Poisson_ratio>
                  DENS: <beam2_density>
                  CROSSAREA: <beam2_cross_section_area>
                  SHEARCORR: <beam2_shear_correction_factor>
                  MOMINPOL: <beam2_polar_moment_of_inertia>
                  MOMIN2: <beam2_moment_of_inertia_2>
                  MOMIN3: <beam2_moment_of_inertia_3>

            # == Boundary Conditions ===========================================

            # Beam 1: clamped end
            DESIGN POINT DIRICH CONDITIONS:
              - E: <beam1_clamped_node_set_id>
                NUMDOF: 6
                ONOFF: [1, 1, 1, 1, 1, 1]
                VAL: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0, 0, 0, 0]
              # Beam 2: clamped end
              - E: <beam2_clamped_node_set_id>
                NUMDOF: 6
                ONOFF: [1, 1, 1, 1, 1, 1]
                VAL: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0, 0, 0, 0]

            # Load: prescribed displacement or force on beam tip
            DESIGN POINT NEUMANN CONDITIONS:
              - E: <loaded_beam_tip_node_set_id>
                NUMDOF: 6
                ONOFF: [<dof1_load>, <dof2_load>, <dof3_load>, 0, 0, 0]
                VAL: [<load_value_1>, <load_value_2>, <load_value_3>, 0.0, 0.0, 0.0]
                FUNCT: [<load_function>, <load_function>, <load_function>, 0, 0, 0]

            # Load ramp function
            FUNCT<load_function>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<load_ramp_expression>"

            # == Geometry ======================================================
            STRUCTURE GEOMETRY:
              FILE: "<beam_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  BEAM3R:
                    LINE2:
                      MAT: 1
                - ID: 2
                  BEAM3R:
                    LINE2:
                      MAT: 2

            RESULT DESCRIPTION:
              - STRUCTURE:
                  DIS: "structure"
                  NODE: <result_node_id>
                  QUANTITY: "dispx"
                  VALUE: <expected_displacement>
                  TOLERANCE: <result_tolerance>
        """)

    @staticmethod
    def _template_beam_solid_meshtying_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D Beam-to-Solid Volume Meshtying
            #
            # A beam element is embedded in a solid block.  The beam and
            # solid are coupled via volume meshtying (penalty or mortar).
            # The beam acts as a reinforcement inside the solid.
            #
            # Mesh: requires TWO meshes:
            #   Solid mesh: "solid.e" with
            #     element_block 1 = solid block (HEX8)
            #     node_set 1 = fixed face
            #     node_set 2 = loaded face
            #   Beam mesh: "beam.e" with
            #     element_block 1 = beam (LINE2)
            # ---------------------------------------------------------------
            TITLE:
              - "3-D beam-to-solid volume meshtying -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Structure"
            IO:
              STDOUTEVERY: <stdout_interval>
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/BEAMS:
              OUTPUT_BEAMS: true
              DISPLACEMENT: true
              USE_ABSOLUTE_POSITIONS: true
            IO/RUNTIME VTK OUTPUT/STRUCTURE:
              OUTPUT_STRUCTURE: true
              DISPLACEMENT: true

            # == Structural dynamics ===========================================
            STRUCTURAL DYNAMIC:
              DYNAMICTYPE: "Statics"
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              LINEAR_SOLVER: 1
              PREDICT: "ConstDisVelAcc"
              TOLRES: <residual_tolerance>
              TOLDISP: <displacement_tolerance>
              RESULTSEVERY: <results_output_interval>

            # == Beam interaction =============================================
            BEAM INTERACTION:
              STRATEGY: "beam_to_solid_volume_meshtying"
              SEARCH_RADIUS: <meshtying_search_radius>
            BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING:
              COUPLING_TYPE: "<coupling_type>"
              PENALTY_PARAMETER: <meshtying_penalty_parameter>
              GAUSS_POINTS: <meshtying_gauss_points>

            # == Binning for spatial search ===================================
            BINNING STRATEGY:
              BIN_SIZE_LOWER_BOUND: <bin_size_lower_bound>

            # == Solver ========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "structure_solver"

            # == Materials =====================================================
            MATERIALS:
              # Beam material
              - MAT: 1
                MAT_BeamReissnerElastHyper:
                  YOUNG: <beam_Young_modulus>
                  POISSONRATIO: <beam_Poisson_ratio>
                  DENS: <beam_density>
                  CROSSAREA: <beam_cross_section_area>
                  SHEARCORR: <beam_shear_correction_factor>
                  MOMINPOL: <beam_polar_moment_of_inertia>
                  MOMIN2: <beam_moment_of_inertia_2>
                  MOMIN3: <beam_moment_of_inertia_3>
              # Solid material
              - MAT: 2
                MAT_Struct_StVenantKirchhoff:
                  YOUNG: <solid_Young_modulus>
                  NUE: <solid_Poisson_ratio>
                  DENS: <solid_density>

            # == Boundary Conditions ===========================================

            # Solid: fixed face
            DESIGN SURF DIRICH CONDITIONS:
              - E: <solid_fixed_face_id>
                NUMDOF: 3
                ONOFF: [1, 1, 1]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [0, 0, 0]

            # Solid: loaded face
            DESIGN SURF NEUMANN CONDITIONS:
              - E: <solid_loaded_face_id>
                NUMDOF: 3
                ONOFF: [<dof1_load>, <dof2_load>, <dof3_load>]
                VAL: [<load_value_1>, <load_value_2>, <load_value_3>]
                FUNCT: [<load_function>, <load_function>, <load_function>]

            # Load ramp
            FUNCT<load_function>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<load_ramp_expression>"

            # == Geometry ======================================================
            STRUCTURE GEOMETRY:
              FILE: "<solid_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  SOLID:
                    HEX8:
                      MAT: 2
                      KINEM: <kinematics>

            BEAM GEOMETRY:
              FILE: "<beam_mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  BEAM3R:
                    LINE2:
                      MAT: 1

            RESULT DESCRIPTION:
              - STRUCTURE:
                  DIS: "structure"
                  NODE: <result_solid_node_id>
                  QUANTITY: "dispx"
                  VALUE: <expected_solid_displacement>
                  TOLERANCE: <result_tolerance>
              - STRUCTURE:
                  DIS: "structure"
                  NODE: <result_beam_node_id>
                  QUANTITY: "dispx"
                  VALUE: <expected_beam_displacement>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check beam Young's modulus
        young = params.get("YOUNG") or params.get("beam_YOUNG")
        if young is not None:
            try:
                e = float(young)
                if e <= 0:
                    issues.append(f"YOUNG must be > 0, got {e}.")
            except (TypeError, ValueError):
                issues.append(
                    f"YOUNG must be a positive number, got {young!r}."
                )

        # Check penalty parameter
        penalty = params.get("PENALTY_PARAMETER")
        if penalty is not None:
            try:
                p = float(penalty)
                if p <= 0:
                    issues.append(
                        f"PENALTY_PARAMETER must be > 0, got {p}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"PENALTY_PARAMETER must be a positive number, "
                    f"got {penalty!r}."
                )

        # Check cross-sectional area
        crossarea = params.get("CROSSAREA")
        if crossarea is not None:
            try:
                a = float(crossarea)
                if a <= 0:
                    issues.append(
                        f"CROSSAREA must be > 0, got {a}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"CROSSAREA must be a positive number, "
                    f"got {crossarea!r}."
                )

        # Check search radius
        search_r = params.get("SEARCH_RADIUS")
        if search_r is not None:
            try:
                r = float(search_r)
                if r <= 0:
                    issues.append(
                        f"SEARCH_RADIUS must be > 0, got {r}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"SEARCH_RADIUS must be a positive number, "
                    f"got {search_r!r}."
                )

        # Check bin size
        bin_size = params.get("BIN_SIZE_LOWER_BOUND")
        if bin_size is not None:
            try:
                b = float(bin_size)
                if b <= 0:
                    issues.append(
                        f"BIN_SIZE_LOWER_BOUND must be > 0, got {b}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"BIN_SIZE_LOWER_BOUND must be a positive number, "
                    f"got {bin_size!r}."
                )

        # Check Poisson's ratio
        nue = params.get("NUE") or params.get("POISSONRATIO")
        if nue is not None:
            try:
                nu = float(nue)
                if nu <= 0 or nu >= 0.5:
                    issues.append(
                        f"Poisson's ratio must be in (0, 0.5), got {nu}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"Poisson's ratio must be a number, got {nue!r}."
                )

        return issues
