"""Beam element generator for 4C.

Covers Reissner (BEAM3R), Euler-Bernoulli (BEAM3EB), and Kirchhoff (BEAM3K)
beam formulations.  All beam problems in 4C use inline mesh format
(NODE COORDS + STRUCTURE ELEMENTS), not Exodus files.
"""

from __future__ import annotations

import math
import textwrap
from typing import Any

from .base import BaseGenerator


# ── Cross-section helpers ─────────────────────────────────────────────


def circular_cross_section(radius: float) -> dict[str, float]:
    """Compute beam cross-section properties for a circular section.

    Parameters
    ----------
    radius : float
        Radius of the circular cross-section.

    Returns
    -------
    dict
        Keys: CROSSAREA, MOMINPOL (polar moment), MOMIN2 (I_yy),
        MOMIN3 (I_zz), SHEARCORR (shear correction factor for circle).
    """
    r = radius
    A = math.pi * r ** 2
    I = math.pi * r ** 4 / 4.0       # I_yy = I_zz (symmetric)
    J = math.pi * r ** 4 / 2.0       # polar moment of inertia
    return {
        "CROSSAREA": A,
        "MOMINPOL": J,
        "MOMIN2": I,
        "MOMIN3": I,
        "SHEARCORR": 6.0 / 7.0,       # Cowper (1966) for circular sections
    }


def rectangular_cross_section(
    width: float, height: float
) -> dict[str, float]:
    """Compute beam cross-section properties for a rectangular section.

    Parameters
    ----------
    width : float
        Section width (b), dimension along local y-axis.
    height : float
        Section height (h), dimension along local z-axis.

    Returns
    -------
    dict
        Keys: CROSSAREA, MOMINPOL, MOMIN2 (about y), MOMIN3 (about z),
        SHEARCORR (5/6 for rectangle).
    """
    b, h = width, height
    A = b * h
    I2 = b * h ** 3 / 12.0           # I about y-axis (bending in z)
    I3 = h * b ** 3 / 12.0           # I about z-axis (bending in y)
    # Torsional constant (exact series, leading term approximation)
    a, b_ = max(b, h) / 2.0, min(b, h) / 2.0
    J = a * b_ ** 3 * (16.0 / 3.0 - 3.36 * b_ / a * (1.0 - b_ ** 4 / (12.0 * a ** 4)))
    return {
        "CROSSAREA": A,
        "MOMINPOL": J,
        "MOMIN2": I2,
        "MOMIN3": I3,
        "SHEARCORR": 5.0 / 6.0,      # Timoshenko for rectangle
    }


# ── Generator class ───────────────────────────────────────────────────


class BeamsGenerator(BaseGenerator):
    """Generator for beam element problems in 4C.

    Covers BEAM3R (Reissner), BEAM3EB (Euler-Bernoulli), and BEAM3K
    (Kirchhoff) formulations.
    """

    module_key = "beams"
    display_name = "Beam Elements (Reissner / Euler-Bernoulli / Kirchhoff)"
    problem_type = "Structure"

    # ── Knowledge ─────────────────────────────────────────────────────

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Geometrically exact beam elements for slender structures.  "
                "4C provides three formulations: BEAM3R (Reissner, shear-"
                "deformable), BEAM3EB (Euler-Bernoulli, torsion-free, "
                "inextensible), and BEAM3K (Kirchhoff, with torsion).  "
                "All beam elements MUST use inline mesh format (NODE COORDS "
                "+ STRUCTURE ELEMENTS), NOT Exodus files."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "STRUCTURAL DYNAMIC",
                "SOLVER 1",
                "MATERIALS",
                "NODE COORDS",
                "STRUCTURE ELEMENTS",
                "DNODE-NODE TOPOLOGY",
                "DLINE-NODE TOPOLOGY",
            ],
            "optional_sections": [
                "STRUCTURAL DYNAMIC/GENALPHA",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/BEAMS",
            ],
            "beam_types": {
                "BEAM3R": {
                    "name": "Reissner beam (shear-deformable)",
                    "topologies": ["LINE2", "LINE3", "LINE4", "LINE5"],
                    "dofs_per_node": "6 (standard) or 9 (with HERMITE_CENTERLINE)",
                    "features": [
                        "Full shear deformation",
                        "Finite rotations (multiplicative update)",
                        "TRIADS keyword for initial orientation",
                        "HERMITE_CENTERLINE true for C1-continuous centerline",
                    ],
                    "element_format": (
                        "<id> BEAM3R <topology> <node1> <node2> [<mid>] "
                        "MAT <mat_id> TRIADS <9 or 6 angles> "
                        "[HERMITE_CENTERLINE true]"
                    ),
                },
                "BEAM3EB": {
                    "name": "Euler-Bernoulli beam (torsion-free)",
                    "topologies": ["LINE2"],
                    "dofs_per_node": "6",
                    "features": [
                        "No shear deformation (Kirchhoff constraint)",
                        "Torsion-free assumption",
                        "Simpler material: only YOUNG, DENS, CROSSAREA, MOMIN",
                    ],
                    "element_format": (
                        "<id> BEAM3EB LINE2 <node1> <node2> MAT <mat_id>"
                    ),
                },
                "BEAM3K": {
                    "name": "Kirchhoff beam (with torsion)",
                    "topologies": ["LINE2", "LINE3", "LINE4"],
                    "dofs_per_node": "6 or 7 (with twist DOF)",
                    "features": [
                        "No shear deformation",
                        "Full torsion support",
                        "Higher regularity than Reissner",
                    ],
                    "element_format": (
                        "<id> BEAM3K <topology> <nodes> MAT <mat_id> "
                        "TRIADS <angles>"
                    ),
                },
            },
            "materials": {
                "MAT_BeamReissnerElastHyper": {
                    "description": (
                        "Hyperelastic Reissner beam material.  Requires "
                        "full cross-section property specification."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus [Pa]",
                            "range": "> 0",
                        },
                        "SHEARMOD": {
                            "description": "Shear modulus G = E / (2(1+nu)) [Pa]",
                            "range": "> 0 (if omitted, use POISSONRATIO)",
                        },
                        "POISSONRATIO": {
                            "description": "Poisson's ratio (alternative to SHEARMOD)",
                            "range": "[0, 0.5)",
                        },
                        "DENS": {
                            "description": "Mass density per unit volume [kg/m^3]",
                            "range": ">= 0",
                        },
                        "CROSSAREA": {
                            "description": "Cross-sectional area A [m^2]",
                            "range": "> 0",
                        },
                        "SHEARCORR": {
                            "description": (
                                "Shear correction factor kappa "
                                "(circle: 6/7, rectangle: 5/6)"
                            ),
                            "range": "> 0 (typically 0.8--1.1)",
                        },
                        "MOMINPOL": {
                            "description": "Polar moment of inertia J [m^4]",
                            "range": "> 0",
                        },
                        "MOMIN2": {
                            "description": "Second moment of area I_yy [m^4]",
                            "range": "> 0",
                        },
                        "MOMIN3": {
                            "description": "Second moment of area I_zz [m^4]",
                            "range": "> 0",
                        },
                    },
                },
                "MAT_BeamKirchhoffTorsionFreeElastHyper": {
                    "description": (
                        "Kirchhoff torsion-free beam material for BEAM3EB.  "
                        "Simplified parameter set (no shear or torsion)."
                    ),
                    "parameters": {
                        "YOUNG": {
                            "description": "Young's modulus [Pa]",
                            "range": "> 0",
                        },
                        "DENS": {
                            "description": "Mass density per unit volume [kg/m^3]",
                            "range": ">= 0",
                        },
                        "CROSSAREA": {
                            "description": "Cross-sectional area A [m^2]",
                            "range": "> 0",
                        },
                        "MOMIN": {
                            "description": "Second moment of area I [m^4]",
                            "range": "> 0",
                        },
                    },
                },
            },
            "dynamics": {
                "statics": {
                    "DYNAMICTYPE": "Statics",
                    "notes": "Quasi-static loading via load steps.",
                },
                "dynamics_lie_group": {
                    "DYNAMICTYPE": "GenAlphaLieGroup",
                    "notes": (
                        "Recommended for beam dynamics.  Lie-group time "
                        "integrator handles finite rotations correctly."
                    ),
                    "key_settings": {
                        "MASSLIN": "rotations (linearise mass matrix wrt rotations)",
                        "MAXITER": "40--80 (beams need more Newton iterations)",
                        "TOLDISP": "1e-8 to 1e-11",
                        "TOLRES": "1e-6 to 1e-8",
                    },
                },
            },
            "mesh_format": {
                "important": (
                    "Beam elements MUST use inline mesh format.  "
                    "They CANNOT use Exodus (.e) files."
                ),
                "node_coords": (
                    'NODE COORDS: list of "NODE <id> COORD <x> <y> <z>"'
                ),
                "elements": (
                    'STRUCTURE ELEMENTS: list of "<id> BEAM3R LINE3 '
                    '<n1> <n3> <n2> MAT <mid> TRIADS 0 0 0 0 0 0 0 0 0"'
                ),
                "topology": (
                    "DNODE-NODE TOPOLOGY maps design nodes to mesh nodes "
                    "(for point BCs).  DLINE-NODE TOPOLOGY maps design "
                    "lines to mesh nodes (for distributed loads)."
                ),
            },
            "pitfalls": [
                (
                    "[Input] Beams CANNOT use Exodus mesh "
                    "files — must use inline NODE COORDS + "
                    "STRUCTURE ELEMENTS format. Signal: "
                    "passing STRUCTURE GEOMETRY: FILE: "
                    "mesh.e with BEAM elements raises "
                    "'beam element type not supported in "
                    "Exodus' from 4C_io_meshreader.cpp. "
                    "Beam elements have nodal triads + "
                    "tangent vectors that the Exodus format "
                    "does not encode; inline YAML is the "
                    "only path. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] NUMDOF must match the element "
                    "type: 6 for standard BEAM3R LINE3 "
                    "(without Hermite), 9 for BEAM3R LINE3 "
                    "with HERMITE_CENTERLINE: true. Signal: "
                    "BEAM3R NUMDOF mismatch raises "
                    "'inconsistent DOF count for beam "
                    "element' at discretisation setup; a "
                    "DESIGN ... DIRICH with wrong NUMDOF "
                    "triggers the same. Always match NUMDOF "
                    "to the BEAM3R configuration (6 "
                    "rotational DOF or 9 Hermite-enriched). "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Input] TRIADS keyword is REQUIRED for "
                    "BEAM3R — specifies initial orientation "
                    "angles at each element node. Signal: "
                    "omitting TRIADS gives zero initial "
                    "rotation reference, but the FIRST "
                    "load step then computes finite "
                    "rotations from an undefined "
                    "configuration — visible as random "
                    "initial twist (NaN-like rotation "
                    "increments on first step). For a beam "
                    "aligned with the x-axis, all TRIADS "
                    "values are 0. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] For LINE3 elements (quadratic), "
                    "node ordering is "
                    "endpoint1-endpoint2-midpoint (NOT "
                    "sequential!). Signal: writing nodes in "
                    "sequential 1-2-3 order along the beam "
                    "makes 4C treat the geometric midpoint "
                    "as endpoint2 — element length is "
                    "halved, stiffness is wrong by factor "
                    "2-4. Specify: endpoint1, endpoint2, "
                    "midpoint (the LINE3 convention). "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Input] HERMITE_CENTERLINE: true adds 3 "
                    "tangent DOFs per node (6 -> 9 DOFs). "
                    "Signal: enabling HERMITE_CENTERLINE on "
                    "BEAM3R without increasing NUMDOF from 6 "
                    "to 9 produces an under-DOF'd element — "
                    "the tangent DOFs are silently allocated "
                    "but DESIGN ... DIRICH BCs with NUMDOF=6 "
                    "leave them free, and you get spurious "
                    "tangent growth at constrained nodes. "
                    "Match NUMDOF to HERMITE_CENTERLINE "
                    "choice. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] Use GenAlphaLieGroup for "
                    "beam dynamics — standard Newmark/"
                    "GenAlpha does NOT handle finite "
                    "rotations correctly. Signal: a "
                    "rotating beam (e.g. spinning rigid "
                    "rod) under standard GenAlpha shows "
                    "monotonically growing angular momentum "
                    "(energy drift); switching to "
                    "GenAlphaLieGroup conserves angular "
                    "momentum to machine precision. The "
                    "Lie-group variant uses the SO(3) "
                    "rotation manifold, not Euclidean "
                    "averaging. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] MASSLIN: rotations is REQUIRED "
                    "for GenAlphaLieGroup time integration. "
                    "Signal: GenAlphaLieGroup with "
                    "MASSLIN: linear aborts with "
                    "'inconsistent mass linearisation for "
                    "Lie-group integrator' at setup; the "
                    "Lie-group needs the rotational-mass "
                    "linearisation to construct the "
                    "tangent-space update correctly. Use "
                    "MASSLIN: rotations for finite-"
                    "rotation dynamics. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] Cross-section properties "
                    "(A, I, J) must be MUTUALLY CONSISTENT "
                    "— e.g. A = pi*r^2, Iyy = Izz = "
                    "pi*r^4/4, J = pi*r^4/2 for a solid "
                    "circular cross-section. Signal: "
                    "specifying A = 1.0 but Iyy = 0.01 for "
                    "a 'circular' section produces a beam "
                    "whose bending stiffness does not match "
                    "axial stiffness — frequency analysis "
                    "shows wrong mode shapes. Use the "
                    "helper functions circular_cross_"
                    "section() or rectangular_cross_"
                    "section() to enforce consistency. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Input] DNODE-NODE TOPOLOGY entries are "
                    "needed for point Dirichlet and Neumann "
                    "conditions. DLINE-NODE TOPOLOGY entries "
                    "are needed for distributed LINE loads. "
                    "Signal: applying a DESIGN POINT NEUMANN "
                    "without DNODE-NODE TOPOLOGY raises "
                    "'no design nodes found' from "
                    "4C_io_input_file.cpp; the design-node "
                    "set is empty. Map node IDs -> design "
                    "node IDs in DNODE-NODE TOPOLOGY first. "
                    "(Audit 2026-06-02.)"
                ),
                "[API] Beam material names use CamelCase WITHOUT inner "
                "underscores: MAT_BeamReissnerElastHyper (NOT "
                "MAT_Beam_Reissner_ElastHyper), MAT_BeamKirchhoffElastHyper, "
                "MAT_BeamKirchhoffTorsionFreeElastHyper, MAT_BeamReissnerElastPlastic, "
                "and four '_ByModes' parameterization variants. The catalog "
                "previously had wrong underscore-separated forms. Signal: "
                "invalid material name fails at YAML parse with "
                "input_spec_builders.cpp 'Could not match this input'. "
                "Verified against 4C 2026.3 MATERIALS schema 2026-06-01.",
                "[API] BEAM3R supports LINE2/LINE3/LINE4/"
                "LINE5; BEAM3K supports LINE2/LINE3/LINE4; "
                "BEAM3EB supports LINE2 only. Signal: a "
                "BEAM3R LINE6 (or any unsupported cell "
                "type) raises 'Unknown beam element cell "
                "type' from beam_factory.cpp at setup; the "
                "factory only registers the cell types "
                "listed. Verified against 4C 2026.3 "
                "STRUCTURE GEOMETRY/ELEMENT_BLOCKS schema "
                "2026-06-01.",
            ],
            "typical_experiments": [
                {
                    "name": "cantilever_reissner",
                    "description": (
                        "Cantilever beam under tip load.  Classic benchmark "
                        "for beam element verification.  Fixed at x=0, "
                        "transverse force or moment at x=L."
                    ),
                },
                {
                    "name": "dynamic_beam",
                    "description": (
                        "Dynamic cantilever under sinusoidal distributed "
                        "load.  Tests GenAlphaLieGroup integrator, energy "
                        "conservation, and large-rotation dynamics."
                    ),
                },
            ],
            "cross_section_helpers": {
                "circular_cross_section(radius)": (
                    "Returns CROSSAREA, MOMINPOL, MOMIN2, MOMIN3, SHEARCORR "
                    "for a circular cross-section."
                ),
                "rectangular_cross_section(width, height)": (
                    "Returns CROSSAREA, MOMINPOL, MOMIN2, MOMIN3, SHEARCORR "
                    "for a rectangular cross-section."
                ),
            },
        }

    # ── Variants ──────────────────────────────────────────────────────

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "cantilever_static",
                "description": (
                    "Static cantilever beam (10 Reissner elements, LINE2).  "
                    "Fixed at x=0, tip force in z-direction.  UMFPACK solver."
                ),
            },
            {
                "name": "cantilever_dynamic",
                "description": (
                    "Dynamic cantilever beam (10 Reissner elements, LINE3 "
                    "with Hermite centerline).  GenAlphaLieGroup time "
                    "integration.  Tip moment loading with ramp."
                ),
            },
        ]

    # ── Templates ─────────────────────────────────────────────────────

    def get_template(self, variant: str = "cantilever_static") -> str:
        templates = {
            "cantilever_static": self._template_cantilever_static,
            "cantilever_dynamic": self._template_cantilever_dynamic,
        }
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_cantilever_static() -> str:
        """Static cantilever with 10 LINE2 Reissner beam elements.

        Beam along x-axis from x=0 to x=10.  11 nodes, 10 elements.
        Circular cross-section r=0.1.  Tip force F_z = 1.0.
        """
        # Compute cross-section properties for a circular beam r=0.1
        r = 0.1
        cs = circular_cross_section(r)

        # Build node coordinate lines
        n_elem = 10
        n_nodes = n_elem + 1  # LINE2: 2 nodes per element, shared
        L = 10.0
        dx = L / n_elem

        node_lines = []
        for i in range(n_nodes):
            nid = i + 1
            x = i * dx
            node_lines.append(f'  - "NODE {nid} COORD {x:.1f} 0.0 0.0"')

        # Build element lines (LINE2: 2 nodes, no midpoint)
        elem_lines = []
        for i in range(n_elem):
            eid = i + 1
            n1 = i + 1
            n2 = i + 2
            elem_lines.append(
                f'  - "{eid} BEAM3R LINE2 {n1} {n2} MAT 1 '
                f'TRIADS 0.0 0.0 0.0 0.0 0.0 0.0"'
            )

        # Build DLINE-NODE TOPOLOGY (all nodes on line 1)
        dline_lines = []
        for i in range(n_nodes):
            nid = i + 1
            dline_lines.append(f'  - "NODE {nid} DLINE 1"')

        # Join blocks with consistent YAML indentation (2-space list items)
        node_block = "\n".join(node_lines)
        elem_block = "\n".join(elem_lines)
        dline_block = "\n".join(dline_lines)

        # Build template parts, then combine (avoids textwrap.dedent + f-string
        # multiline block indentation issues)
        header = textwrap.dedent(f"""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # Static Cantilever Beam -- 10 Reissner LINE2 Elements
            #
            # Geometry: L = 10, circular cross-section r = 0.1
            # Fixed at x = 0 (DNODE 1 = node 1)
            # Tip force F_z at x = L (DNODE 2 = node 11)
            # ---------------------------------------------------------------
            TITLE:
              - "Static cantilever beam -- Reissner BEAM3R LINE2"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Structure"
            IO:
              VERBOSITY: "Standard"
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/BEAMS:
              OUTPUT_BEAMS: true
              DISPLACEMENT: true
              STRAINS_GAUSSPOINT: true

            # -- Structural dynamics (static) ------------------------------
            STRUCTURAL DYNAMIC:
              DYNAMICTYPE: "Statics"
              TIMESTEP: <load_step_size>
              NUMSTEP: <number_of_load_steps>
              MAXTIME: <end_time>
              PREDICT: "TangDis"
              LINEAR_SOLVER: 1
            STRUCT NOX/Printing:
              Error: true
              Details: true

            # -- Solver ----------------------------------------------------
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "Structure_Solver"

            # -- Material (cross-section properties) -----------------------
            MATERIALS:
              - MAT: 1
                MAT_BeamReissnerElastHyper:
                  YOUNG: <Young_modulus>
                  POISSONRATIO: <Poisson_ratio>
                  DENS: <density>
                  CROSSAREA: <cross_section_area>
                  SHEARCORR: <shear_correction_factor>
                  MOMINPOL: <polar_moment_of_inertia>
                  MOMIN2: <second_moment_of_area_Iyy>
                  MOMIN3: <second_moment_of_area_Izz>

            # -- Boundary conditions ---------------------------------------
            # DNODE 1 = clamped end (node 1): fix all 6 DOFs
            DESIGN POINT DIRICH CONDITIONS:
              - E: 1
                NUMDOF: 6
                ONOFF: [1, 1, 1, 1, 1, 1]
                VAL: [0, 0, 0, 0, 0, 0]
                FUNCT: [0, 0, 0, 0, 0, 0]

            # DNODE 2 = tip (node 11): transverse force in z
            DESIGN POINT NEUMANN CONDITIONS:
              - E: 2
                NUMDOF: 6
                ONOFF: [0, 0, 1, 0, 0, 0]
                VAL: [0, 0, <tip_force_z>, 0, 0, 0]
                FUNCT: [0, 0, 1, 0, 0, 0]

            # Load ramp: linear from 0 to 1 over t=[0,1]
            FUNCT1:
              - SYMBOLIC_FUNCTION_OF_TIME: "t"

            # -- Topology (design nodes/lines -> mesh nodes) ---------------
            DNODE-NODE TOPOLOGY:
              - "NODE 1 DNODE 1"
              - "NODE {n_nodes} DNODE 2"
            DLINE-NODE TOPOLOGY:
        """)

        footer = textwrap.dedent("""\

            # -- Inline mesh -----------------------------------------------
            NODE COORDS:
        """)

        return (header + dline_block + footer + node_block
                + "\nSTRUCTURE ELEMENTS:\n" + elem_block + "\n")

    @staticmethod
    def _template_cantilever_dynamic() -> str:
        """Dynamic cantilever with 10 LINE3 Reissner beam elements + Hermite.

        Beam along x-axis from x=0 to x=10.
        LINE3 (quadratic): 3 nodes per element -> 21 nodes total.
        HERMITE_CENTERLINE true -> 9 DOFs per node.
        GenAlphaLieGroup time integration.
        Tip moment loading with smooth ramp.
        """
        # Compute cross-section properties for a circular beam r=0.1
        r = 0.1
        cs = circular_cross_section(r)

        # Build nodes: LINE3 quadratic elements along x-axis
        # For n_elem LINE3 elements: 2*n_elem + 1 nodes total
        n_elem = 10
        L = 10.0
        n_nodes = 2 * n_elem + 1
        dx = L / (n_nodes - 1)

        node_lines = []
        for i in range(n_nodes):
            nid = i + 1
            x = i * dx
            node_lines.append(f'  - "NODE {nid} COORD {x:.10e} 0.0 0.0"')

        # Build elements: LINE3 with node ordering endpoint1-endpoint2-midpoint
        elem_lines = []
        for i in range(n_elem):
            eid = i + 1
            n1 = 2 * i + 1      # first endpoint
            n2 = 2 * i + 3      # second endpoint
            n_mid = 2 * i + 2   # midpoint
            elem_lines.append(
                f'  - "{eid} BEAM3R LINE3 {n1} {n2} {n_mid} MAT 1 '
                f'TRIADS 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 '
                f'HERMITE_CENTERLINE true"'
            )

        # Build DLINE-NODE TOPOLOGY (all nodes on line 1)
        dline_lines = []
        for i in range(n_nodes):
            nid = i + 1
            dline_lines.append(f'  - "NODE {nid} DLINE 1"')

        node_block = "\n".join(node_lines)
        elem_block = "\n".join(elem_lines)
        dline_block = "\n".join(dline_lines)

        header = textwrap.dedent(f"""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # Dynamic Cantilever Beam -- 10 Reissner LINE3 Elements
            #
            # Geometry: L = 10, circular cross-section r = 0.1
            # Fixed at x = 0 (DNODE 1 = node 1)
            # Tip moment M_x at x = L (DNODE 2 = node 21)
            # GenAlphaLieGroup time integration (finite rotations)
            # ---------------------------------------------------------------
            TITLE:
              - "Dynamic cantilever beam -- Reissner BEAM3R LINE3 + Hermite"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Structure"
            IO:
              VERBOSITY: "Standard"
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/BEAMS:
              OUTPUT_BEAMS: true
              DISPLACEMENT: true
              TRIAD_VISUALIZATIONPOINT: true
              STRAINS_GAUSSPOINT: true
              INTERNAL_ENERGY_ELEMENT: true
              KINETIC_ENERGY_ELEMENT: true

            # -- Structural dynamics (Lie group generalized-alpha) ---------
            STRUCTURAL DYNAMIC:
              DYNAMICTYPE: "GenAlphaLieGroup"
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              TOLDISP: <displacement_tolerance>
              TOLRES: <residual_tolerance>
              MAXITER: <max_iterations>
              MASSLIN: "rotations"
              RESTARTEVERY: <restart_interval>
              LINEAR_SOLVER: 1
            STRUCTURAL DYNAMIC/GENALPHA:
              RHO_INF: <spectral_radius_rho_inf>
            STRUCT NOX/Printing:
              Inner Iteration: false
              Outer Iteration StatusTest: false

            # -- Solver ----------------------------------------------------
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "Structure_Solver"

            # -- Material (cross-section properties) -----------------------
            MATERIALS:
              - MAT: 1
                MAT_BeamReissnerElastHyper:
                  YOUNG: <Young_modulus>
                  SHEARMOD: <shear_modulus>
                  DENS: <density>
                  CROSSAREA: <cross_section_area>
                  SHEARCORR: <shear_correction_factor>
                  MOMINPOL: <polar_moment_of_inertia>
                  MOMIN2: <second_moment_of_area_Iyy>
                  MOMIN3: <second_moment_of_area_Izz>

            # -- Boundary conditions ---------------------------------------
            # DNODE 1 = clamped end (node 1): fix all 9 DOFs
            #   (6 standard + 3 Hermite tangent DOFs)
            DESIGN POINT DIRICH CONDITIONS:
              - E: 1
                NUMDOF: 9
                ONOFF: [1, 1, 1, 1, 1, 1, 0, 0, 0]
                VAL: [0, 0, 0, 0, 0, 0, 0, 0, 0]
                FUNCT: [0, 0, 0, 0, 0, 0, 0, 0, 0]

            # DNODE 2 = tip (node {n_nodes}): twist moment M_x
            DESIGN POINT NEUMANN CONDITIONS:
              - E: 2
                NUMDOF: 9
                ONOFF: [0, 0, 0, 1, 0, 0, 0, 0, 0]
                VAL: [0, 0, 0, <tip_moment_x>, 0, 0, 0, 0, 0]
                FUNCT: [0, 0, 0, 1, 0, 0, 0, 0, 0]

            # Smooth ramp: increases from 0 to peak over ramp-up, then zero
            FUNCT1:
              - COMPONENT: 0
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<load_function_expression>"
              - VARIABLE: 0
                NAME: "a"
                TYPE: "linearinterpolation"
                NUMPOINTS: <number_of_interpolation_points>
                TIMES: [<interpolation_times>]
                VALUES: [<interpolation_values>]

            # -- Topology (design nodes/lines -> mesh nodes) ---------------
            DNODE-NODE TOPOLOGY:
              - "NODE 1 DNODE 1"
              - "NODE {n_nodes} DNODE 2"
            DLINE-NODE TOPOLOGY:
        """)

        footer = textwrap.dedent("""\

            # -- Inline mesh -----------------------------------------------
            NODE COORDS:
        """)

        return (header + dline_block + footer + node_block
                + "\nSTRUCTURE ELEMENTS:\n" + elem_block + "\n")

    # ── Validation ────────────────────────────────────────────────────

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        """Validate beam-specific parameters.

        Checks:
        - Cross-section properties are consistent (A, I, J positive)
        - NUMDOF matches element type
        - Mesh format is inline (not Exodus)
        - Material parameters are physically reasonable
        """
        issues: list[str] = []

        # Check cross-section property consistency
        A = params.get("CROSSAREA")
        I2 = params.get("MOMIN2")
        I3 = params.get("MOMIN3")
        J = params.get("MOMINPOL")

        if A is not None:
            try:
                area = float(A)
                if area <= 0:
                    issues.append(f"CROSSAREA must be > 0, got {area}.")
            except (TypeError, ValueError):
                issues.append(f"CROSSAREA must be numeric, got {A!r}.")

        for name, val in [("MOMIN2", I2), ("MOMIN3", I3), ("MOMINPOL", J)]:
            if val is not None:
                try:
                    v = float(val)
                    if v <= 0:
                        issues.append(f"{name} must be > 0, got {v}.")
                except (TypeError, ValueError):
                    issues.append(f"{name} must be numeric, got {val!r}.")

        # Check that I2 + I3 ~ J (perpendicular axis theorem for circular)
        if I2 is not None and I3 is not None and J is not None:
            try:
                i2 = float(I2)
                i3 = float(I3)
                j = float(J)
                expected_j = i2 + i3
                if expected_j > 0 and abs(j - expected_j) / expected_j > 0.1:
                    issues.append(
                        f"MOMINPOL ({j:.6e}) should approximately equal "
                        f"MOMIN2 + MOMIN3 ({expected_j:.6e}) for common "
                        f"cross-sections.  Check consistency."
                    )
            except (TypeError, ValueError):
                pass

        # Check material
        young = params.get("YOUNG") or params.get("young")
        if young is not None:
            try:
                E = float(young)
                if E <= 0:
                    issues.append(f"YOUNG must be > 0, got {E}.")
            except (TypeError, ValueError):
                issues.append(f"YOUNG must be numeric, got {young!r}.")

        shearmod = params.get("SHEARMOD") or params.get("shearmod")
        if shearmod is not None:
            try:
                G = float(shearmod)
                if G <= 0:
                    issues.append(f"SHEARMOD must be > 0, got {G}.")
            except (TypeError, ValueError):
                issues.append(f"SHEARMOD must be numeric, got {shearmod!r}.")

        # Check NUMDOF vs element type
        numdof = params.get("NUMDOF") or params.get("numdof")
        elem_type = params.get("element_type") or params.get("beam_type")
        hermite = params.get("HERMITE_CENTERLINE") or params.get("hermite")

        if numdof is not None and elem_type is not None:
            try:
                nd = int(numdof)
                etype = str(elem_type).upper()
                if etype in ("BEAM3R",):
                    if hermite:
                        if nd != 9:
                            issues.append(
                                f"BEAM3R with HERMITE_CENTERLINE requires "
                                f"NUMDOF = 9, got {nd}."
                            )
                    else:
                        if nd != 6:
                            issues.append(
                                f"BEAM3R (standard) requires NUMDOF = 6, "
                                f"got {nd}."
                            )
                elif etype in ("BEAM3EB",):
                    if nd != 6:
                        issues.append(
                            f"BEAM3EB requires NUMDOF = 6, got {nd}."
                        )
            except (TypeError, ValueError):
                pass

        # Check mesh format
        mesh_file = params.get("mesh_file") or params.get("FILE")
        if mesh_file is not None:
            mf = str(mesh_file).lower()
            if mf.endswith((".e", ".exo", ".exodus")):
                issues.append(
                    "Beam elements CANNOT use Exodus mesh files.  "
                    "Use inline NODE COORDS + STRUCTURE ELEMENTS format."
                )

        # Check dynamics type for beam dynamics
        dyntype = params.get("DYNAMICTYPE")
        if dyntype is not None:
            dt = str(dyntype)
            if dt not in ("Statics", "GenAlphaLieGroup"):
                issues.append(
                    f"For beam elements, use DYNAMICTYPE 'Statics' or "
                    f"'GenAlphaLieGroup', got {dt!r}.  Standard Newmark/"
                    f"GenAlpha does not handle finite rotations correctly."
                )

        return issues
