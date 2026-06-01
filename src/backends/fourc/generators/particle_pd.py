"""Generator for Peridynamics (bond-based PD) simulations in 4C.

Encodes general PD knowledge including pre-cracks, rigid impactors
via boundaryphase, SPH infrastructure requirements, and CFL calculation.

Key insight: PD in 4C rides on top of the SPH particle framework.  Even
though the physics is peridynamic (bond-based), the SPH kernel, boundary
formulation, and transport velocity parameters *must* be specified or the
code crashes with ``pd_neighbor_pairs = 0``.
"""

from __future__ import annotations

import math
import textwrap
from typing import Any

from .base import BaseGenerator


class ParticlePDGenerator(BaseGenerator):
    """Generator for bond-based Peridynamics (PD) particle simulations."""

    module_key = "particle_pd"
    display_name = "Peridynamics (Bond-Based PD)"
    problem_type = "Particle"

    # ── Knowledge ─────────────────────────────────────────────────────

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Bond-based peridynamics in 4C models fracture and fragmentation "
                "by replacing the classical PDE-based continuum mechanics formulation "
                "with a non-local integral equation.  Each material point interacts "
                "with neighbors within a finite 'horizon' via pairwise bonds that "
                "can break when a critical stretch is exceeded.  In 4C the PD module "
                "is built on top of the SPH (Smoothed Particle Hydrodynamics) particle "
                "infrastructure, which means SPH kernel and boundary parameters MUST "
                "be specified even though the physics is purely peridynamic.  Failing "
                "to include the SPH sub-section causes the code to crash with "
                "'pd_neighbor_pairs = 0'."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "IO",
                "BINNING STRATEGY",
                "PARTICLE DYNAMIC",
                "PARTICLE DYNAMIC/INITIAL AND BOUNDARY CONDITIONS",
                "PARTICLE DYNAMIC/SPH",
                "PARTICLE DYNAMIC/PD",
                "MATERIALS",
                "PARTICLES",
            ],
            "section_details": {
                "PROBLEM TYPE": {
                    "PROBLEMTYPE": '"Particle"',
                },
                "BINNING STRATEGY": {
                    "BIN_SIZE_LOWER_BOUND": (
                        "Must be strictly greater than the PD horizon.  "
                        "Recommended: horizon + 1.0 (in length units)."
                    ),
                    "DOMAINBOUNDINGBOX": (
                        'Format: "xmin ymin zmin xmax ymax zmax".  Must enclose ALL '
                        "particles with a margin of at least one particle spacing on "
                        "each side.  A bounding box that is too tight causes particles "
                        "to fall outside bins and the simulation crashes."
                    ),
                },
                "PARTICLE DYNAMIC": {
                    "DYNAMICTYPE": '"VelocityVerlet" (explicit time integration)',
                    "INTERACTION": (
                        '"SPH" -- MUST be SPH even for PD simulations.  PD is '
                        "activated by setting PD_BODY_INTERACTION: true."
                    ),
                    "PD_BODY_INTERACTION": "true  -- enables peridynamic bond interactions",
                    "RIGID_BODY_MOTION": "false  -- set true only for DEM-style rigid bodies",
                    "PHASE_TO_MATERIAL_ID": (
                        'Maps particle phase names to material IDs.  Format: '
                        '"boundaryphase 1 pdphase 2".  Each phase name used in the '
                        "PARTICLES section must appear here."
                    ),
                    "PHASE_TO_DYNLOADBALFAC": (
                        "Load-balancing factors per phase.  Typically 1.0 for all."
                    ),
                    "GRAVITY_ACCELERATION": '"0.0 0.0 0.0" (default, set if needed)',
                    "TIMESTEP": "Must satisfy CFL: dt < dx / sqrt(E/rho), safety factor 0.5",
                    "NUMSTEP": "Total number of time steps",
                    "MAXTIME": "Maximum simulation time (simulation stops at min(MAXTIME, NUMSTEP*TIMESTEP))",
                    "RESULTSEVERY": "Output frequency in steps",
                    "RESTARTEVERY": "Restart file frequency in steps",
                },
                "PARTICLE DYNAMIC/SPH": {
                    "KERNEL": (
                        "QuinticSpline  -- REQUIRED even for PD.  The SPH kernel is "
                        "used internally for neighbor search infrastructure."
                    ),
                    "KERNEL_SPACE_DIM": (
                        "Kernel2D for 2D problems, Kernel3D for 3D.  Must match the "
                        "physical dimension of the problem."
                    ),
                    "INITIALPARTICLESPACING": "The grid spacing dx between particles.",
                    "BOUNDARYPARTICLEFORMULATION": (
                        "AdamiBoundaryFormulation  -- REQUIRED.  Handles interaction "
                        "between boundaryphase and pdphase particles."
                    ),
                    "TRANSPORTVELOCITYFORMULATION": (
                        "StandardTransportVelocity  -- REQUIRED for PD simulations."
                    ),
                },
                "PARTICLE DYNAMIC/PD": {
                    "INTERACTION_HORIZON": (
                        "The PD horizon delta.  Recommended: m * dx where m = 3 "
                        "(horizon ratio).  All bonds within this radius interact."
                    ),
                    "PERIDYNAMIC_GRID_SPACING": (
                        "Must equal the particle spacing dx.  Used to compute the "
                        "PD influence function."
                    ),
                    "PD_DIMENSION": (
                        "Peridynamic_2DPlaneStrain for 2D plane strain, "
                        "Peridynamic_3D for full 3D.  Affects bond-force calculation."
                    ),
                    "NORMALCONTACTLAW": (
                        "NormalLinearSpring  -- penalty contact between PD bodies "
                        "and boundary particles."
                    ),
                    "NORMAL_STIFF": (
                        "Normal contact stiffness for inter-body contact.  Typical "
                        "range: 1e3 to 1e5.  Too high causes instability; too low "
                        "causes excessive penetration."
                    ),
                    "PRE_CRACKS": (
                        'Line segments defining pre-existing cracks.  Format: '
                        '"x1 y1 x2 y2 ; x3 y3 x4 y4".  Bonds crossing these line '
                        "segments are broken at initialization (visibility condition).  "
                        "Multiple crack segments separated by semicolons.  This is the "
                        "mechanism for modeling notches and initial damage without "
                        "removing particles."
                    ),
                },
                "PARTICLE DYNAMIC/INITIAL AND BOUNDARY CONDITIONS": {
                    "DIRICHLET_BOUNDARY_CONDITION": (
                        '"boundaryphase 1"  -- applies FUNCT1 as prescribed '
                        "displacement to all particles of the named phase.  "
                        "The integer is the function ID."
                    ),
                    "CONSTRAINT": (
                        '"Projection2D" for 2D problems.  Constrains out-of-plane '
                        "motion."
                    ),
                },
            },
            "materials": {
                "MAT_ParticlePD": {
                    "description": (
                        "Bond-based peridynamic material for deformable PD bodies.  "
                        "Used with pdphase particles.  Bonds break when stretch "
                        "exceeds CRITICAL_STRETCH."
                    ),
                    "parameters": {
                        "INITRADIUS": {
                            "description": "Initial particle radius = dx/2",
                            "range": "Problem-dependent (half of particle spacing)",
                        },
                        "INITDENSITY": {
                            "description": "Mass density in consistent units",
                            "range": "e.g. 8e-3 g/mm^3 for steel in mm/ms/g system",
                        },
                        "YOUNG": {
                            "description": "Young's modulus",
                            "range": "e.g. 190e3 MPa for steel in mm/ms/g system",
                        },
                        "CRITICAL_STRETCH": {
                            "description": (
                                "Bond breaking criterion.  A bond breaks irreversibly "
                                "when its stretch s = (|xi+eta| - |xi|)/|xi| exceeds "
                                "this value.  Related to fracture energy: "
                                "s_c = sqrt(5 * G_c / (9 * E * delta)) for 2D plane "
                                "stress.  Typical range: 0.001 to 0.05."
                            ),
                            "range": "0.001 -- 0.05 (material-dependent)",
                        },
                    },
                },
                "MAT_ParticleSPHBoundary": {
                    "description": (
                        "Boundary particle material for rigid walls and impactors.  "
                        "Used with boundaryphase particles.  These particles do NOT "
                        "deform -- their motion is prescribed via Dirichlet BCs."
                    ),
                    "parameters": {
                        "INITRADIUS": {
                            "description": "Initial particle radius = dx/2",
                            "range": "Same as PD material spacing",
                        },
                        "INITDENSITY": {
                            "description": (
                                "Density for boundary particles.  Can be set to 1 "
                                "(arbitrary) since boundary particles are rigid."
                            ),
                            "range": "Typically 1 (does not affect PD physics)",
                        },
                    },
                },
            },
            "solver": {
                "type": "Explicit (VelocityVerlet)",
                "notes": (
                    "PD uses explicit time integration exclusively.  There is no "
                    "implicit solver option.  The time step is governed by the CFL "
                    "condition."
                ),
            },
            "time_integration": {
                "scheme": "Velocity Verlet (symplectic, second-order)",
                "CFL_condition": (
                    "dt < dx / c_wave where c_wave = sqrt(E / rho).  "
                    "A safety factor of 0.5 is recommended: dt = 0.5 * dx / sqrt(E/rho)."
                ),
                "example_steel": (
                    "Compute wave speed c = sqrt(E/rho) for your material. "
                    "Then dt < 0.5 * dx / c. Choose dt with a safety factor."
                ),
            },
            "particle_types": {
                "pdphase": (
                    "Peridynamic body particles.  These are deformable and can form "
                    "and break bonds.  Format: "
                    '"TYPE pdphase POS x y z PDBODYID 0".  '
                    "The PDBODYID groups particles into distinct PD bodies (use 0 "
                    "for a single body, increment for multiple bodies)."
                ),
                "boundaryphase": (
                    "Rigid boundary particles.  Their motion is prescribed via "
                    "Dirichlet BCs (FUNCT).  Used for rigid impactors, walls, and "
                    "loading platens.  Format: "
                    '"TYPE boundaryphase POS x y z".  '
                    "NOTE: Do NOT use PDBODYID for boundaryphase particles."
                ),
                "WARNING_rigidphase": (
                    "NEVER use rigidphase (DEM rigid bodies) with PD simulations.  "
                    "rigidphase is for DEM granular mechanics and is incompatible "
                    "with peridynamics.  For rigid impactors, ALWAYS use "
                    "boundaryphase + DIRICHLET_BOUNDARY_CONDITION."
                ),
            },
            "rigid_impactor_recipe": {
                "step_1": "Create boundaryphase particles filling the impactor geometry.",
                "step_2": (
                    "Set PHASE_TO_MATERIAL_ID to map boundaryphase to a "
                    "MAT_ParticleSPHBoundary material."
                ),
                "step_3": (
                    'Set DIRICHLET_BOUNDARY_CONDITION: "boundaryphase 1" to apply '
                    "FUNCT1 as prescribed displacement."
                ),
                "step_4": (
                    "Define FUNCT1 with SYMBOLIC_FUNCTION_OF_SPACE_TIME giving "
                    "displacement = velocity * t for constant-velocity impact."
                ),
            },
            "pre_crack_mechanism": {
                "description": (
                    "Pre-cracks are implemented via the visibility condition: at "
                    "initialization, any bond whose reference line segment (connecting "
                    "two particles) crosses a pre-crack line segment is broken.  This "
                    "models notches, saw-cuts, and initial damage without removing "
                    "particles from the discretization."
                ),
                "format": (
                    '"x1 y1 x2 y2 ; x3 y3 x4 y4"  -- each segment is defined by '
                    "its two endpoints (2D coordinates).  Multiple segments separated "
                    "by semicolons."
                ),
                "example": (
                    '"x1 y1 x2 y2 ; x3 y3 x4 y4"  -- multiple line segments separated by semicolons'
                ),
            },
            "unit_systems": {
                "mm_ms_g (recommended)": {
                    "Length": "mm",
                    "Time": "ms",
                    "Mass": "g",
                    "Force": "N (= g*mm/ms^2)",
                    "Stress": "MPa (= N/mm^2)",
                    "Density": "g/mm^3 (= 1e-3 * kg/m^3 value)",
                    "Velocity": "mm/ms (= m/s)",
                    "Energy": "mJ (= N*mm)",
                },
                "SI (m_s_kg)": {
                    "Length": "m",
                    "Time": "s",
                    "Mass": "kg",
                    "Force": "N",
                    "Stress": "Pa",
                    "Density": "kg/m^3",
                    "Velocity": "m/s",
                    "Energy": "J",
                },
            },
            "pitfalls": [
                (
                    "CRITICAL: The PARTICLE DYNAMIC/SPH section is MANDATORY for PD "
                    "simulations even though the physics is peridynamic, not SPH.  "
                    "Missing SPH parameters (KERNEL, KERNEL_SPACE_DIM, "
                    "BOUNDARYPARTICLEFORMULATION, TRANSPORTVELOCITYFORMULATION) causes "
                    "the code to crash with 'pd_neighbor_pairs = 0' because the "
                    "neighbor search infrastructure is not initialized."
                ),
                (
                    "DOMAINBOUNDINGBOX must enclose ALL particles with margin.  If a "
                    "particle (including the moving impactor at any time step) falls "
                    "outside the bounding box, the simulation crashes."
                ),
                (
                    "Use boundaryphase (NOT rigidphase) for rigid impactors.  "
                    "rigidphase is DEM-only and incompatible with PD.  boundaryphase "
                    "particles interact correctly with pdphase via the Adami "
                    "boundary formulation."
                ),
                (
                    "Horizon ratio m = delta/dx should be at least 3 for convergence.  "
                    "m=2 gives poor accuracy; m=4+ is more expensive but more accurate."
                ),
                (
                    "BIN_SIZE_LOWER_BOUND must be > horizon.  If bins are smaller "
                    "than the horizon, some neighbors will not be found."
                ),
                (
                    "Pre-cracks (PRE_CRACKS) must use 2D coordinates (x, y) matching "
                    "the particle positions.  The visibility check is geometric -- it "
                    "tests whether the line segment connecting two particles crosses "
                    "the crack segment."
                ),
                (
                    "CFL violation: if dt >= dx / sqrt(E/rho), the explicit time "
                    "integration becomes unstable.  Use a safety factor of 0.5."
                ),
                (
                    "INITRADIUS in the material must equal dx/2.  An inconsistent "
                    "value causes incorrect mass and volume computation."
                ),
                (
                    "Bond-based PD restricts Poisson's ratio to nu=0.25 (2D) or "
                    "nu=1/3 (3D).  This is a fundamental limitation of the pairwise "
                    "force model."
                ),
                (
                    "For 2D problems, particles must still have z=0.0 coordinates "
                    "and the DOMAINBOUNDINGBOX must have a small z-extent (e.g., "
                    "-0.01 to 0.01)."
                ),
                (
                    "LOADING: boundaryphase particles interact with pdphase via "
                    "REPULSIVE contact only (std::min(0.0, ...)).  They CANNOT apply "
                    "tensile loads — only compressive impact.  For tension/opening "
                    "problems (DCB, fracture), use INITIAL_VELOCITY_FIELD to impart "
                    "kinetic energy, or PDFIXED flag on boundary particles with "
                    "prescribed Dirichlet displacement."
                ),
                (
                    "PDFIXED: per-particle flag that fixes a particle in place "
                    "(zero displacement).  Add 'PDFIXED 1' to the particle definition "
                    "string.  Use for clamped supports in fracture problems."
                ),
                (
                    "IO/RUNTIME VTK OUTPUT sections are INCOMPATIBLE with particle "
                    "problems — they crash 4C.  Remove them.  4C writes particle VTU "
                    "files automatically via the PARTICLE DYNAMIC output mechanism."
                ),
                (
                    "Critical stretch formula differs between plane stress and plane "
                    "strain.  Plane strain (2D): s_c = sqrt(5*G_Ic/(9*K_b*delta)) "
                    "where K_b = E/(3*(1-2*nu)).  Plane stress: s_c = sqrt(5*G_Ic/(6*E*delta))."
                ),
            ],
            "typical_experiments": [
                {
                    "name": "Plate under uniaxial tension",
                    "description": (
                        "A rectangular plate with a central notch pulled in tension.  "
                        "Crack propagates from the notch tip perpendicular to the "
                        "loading direction.  Good first test for PD fracture."
                    ),
                    "template_variant": "plate_2d",
                },
                {
                    "name": "Plate impact (1D wave propagation)",
                    "description": (
                        "Simplest PD test: a 1D bar or thin plate impacted at one "
                        "end.  Validates wave speed and bond force computation."
                    ),
                },
            ],
        }

    # ── Templates ─────────────────────────────────────────────────────

    def get_template(self, variant: str = "plate_2d") -> str:
        if variant == "plate_2d":
            return self._template_plate_2d()
        raise ValueError(
            f"Unknown variant {variant!r} for {self.module_key}.  "
            f"Available: {[v['name'] for v in self.list_variants()]}"
        )

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "plate_2d",
                "description": (
                    "2D plate with a horizontal pre-crack under prescribed "
                    "velocity impact from the left.  Demonstrates all essential PD "
                    "features: pdphase body, boundaryphase impactor, pre-cracks, "
                    "CFL-safe time stepping.  Uses mm/ms/g unit system."
                ),
            },
        ]

    # ── Validation ────────────────────────────────────────────────────

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        """Physics-aware validation of PD parameters.

        Expected keys in *params* (all optional):
            dx              - particle spacing
            horizon         - PD interaction horizon
            young           - Young's modulus
            density         - material density
            dt              - time step
            critical_stretch - bond breaking criterion
            bin_size        - BIN_SIZE_LOWER_BOUND
            domain_bbox     - [xmin, ymin, zmin, xmax, ymax, zmax]
            particles_bbox  - [xmin, ymin, zmin, xmax, ymax, zmax] (actual particle extent)
        """
        warnings: list[str] = []

        dx = params.get("dx")
        horizon = params.get("horizon")
        young = params.get("young")
        density = params.get("density")
        dt = params.get("dt")
        critical_stretch = params.get("critical_stretch")
        bin_size = params.get("bin_size")
        domain_bbox = params.get("domain_bbox")
        particles_bbox = params.get("particles_bbox")

        # Horizon vs dx check
        if horizon is not None and dx is not None:
            m = horizon / dx
            if m < 2.0:
                warnings.append(
                    f"ERROR: Horizon ratio m = delta/dx = {m:.2f} < 2.  "
                    f"PD requires m >= 2 (recommended m = 3)."
                )
            elif m < 2.5:
                warnings.append(
                    f"WARNING: Horizon ratio m = {m:.2f} is low.  "
                    f"m = 3 is recommended for accuracy."
                )

        # CFL condition
        if young is not None and density is not None and dt is not None and dx is not None:
            if density <= 0:
                warnings.append("ERROR: Density must be positive.")
            elif young <= 0:
                warnings.append("ERROR: Young's modulus must be positive.")
            else:
                c_wave = math.sqrt(young / density)
                dt_cfl = dx / c_wave
                ratio = dt / dt_cfl
                if ratio >= 1.0:
                    warnings.append(
                        f"ERROR: CFL VIOLATION.  dt={dt} >= dt_CFL={dt_cfl:.6e} "
                        f"(ratio={ratio:.3f}).  The simulation WILL be unstable.  "
                        f"Reduce dt to at most {0.5 * dt_cfl:.6e} (safety factor 0.5)."
                    )
                elif ratio > 0.8:
                    warnings.append(
                        f"WARNING: CFL ratio = {ratio:.3f} is dangerously high.  "
                        f"Recommended: dt <= {0.5 * dt_cfl:.6e} (safety factor 0.5)."
                    )

        # Critical stretch sanity
        if critical_stretch is not None:
            if critical_stretch <= 0:
                warnings.append("ERROR: CRITICAL_STRETCH must be positive.")
            elif critical_stretch > 0.1:
                warnings.append(
                    f"WARNING: CRITICAL_STRETCH = {critical_stretch} is unusually "
                    f"large.  Typical values: 0.001 to 0.05."
                )
            elif critical_stretch < 1e-4:
                warnings.append(
                    f"WARNING: CRITICAL_STRETCH = {critical_stretch} is very small.  "
                    f"Bonds will break almost immediately."
                )

        # Bin size vs horizon
        if bin_size is not None and horizon is not None:
            if bin_size <= horizon:
                warnings.append(
                    f"ERROR: BIN_SIZE_LOWER_BOUND ({bin_size}) must be > horizon "
                    f"({horizon}).  Neighbors outside the bin will be missed."
                )

        # Bounding box encloses particles
        if domain_bbox is not None and particles_bbox is not None:
            if len(domain_bbox) == 6 and len(particles_bbox) == 6:
                labels = ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]
                for i in range(3):  # min dimensions
                    if domain_bbox[i] > particles_bbox[i]:
                        warnings.append(
                            f"ERROR: DOMAINBOUNDINGBOX {labels[i]}={domain_bbox[i]} "
                            f"is greater than particle {labels[i]}={particles_bbox[i]}.  "
                            f"Particles will fall outside the domain."
                        )
                for i in range(3, 6):  # max dimensions
                    if domain_bbox[i] < particles_bbox[i]:
                        warnings.append(
                            f"ERROR: DOMAINBOUNDINGBOX {labels[i]}={domain_bbox[i]} "
                            f"is less than particle {labels[i]}={particles_bbox[i]}.  "
                            f"Particles will fall outside the domain."
                        )

        return warnings

    # ── Private template builders ─────────────────────────────────────

    @staticmethod
    def _template_plate_2d() -> str:
        """Template showing the FORMAT of a 2D PD input. All values are placeholders."""
        return textwrap.dedent("""\
            # 2D Peridynamics: FORMAT TEMPLATE
            # ALL numerical values below are PLACEHOLDERS — they must be determined
            # by the user based on the specific problem geometry, material, and
            # required resolution. Consult the literature and 4C test files
            # via examples(keyword, solver='fourc', action='search') for
            # appropriate values.
            #
            # Units: choose a consistent system (e.g., mm-ms-g or SI)

            PROBLEM TYPE:
              PROBLEMTYPE: "Particle"

            IO:
              STDOUTEVERY: <OUTPUT_FREQUENCY>
              VERBOSITY: "Standard"

            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <VTK_OUTPUT_FREQUENCY>
            IO/RUNTIME VTK OUTPUT/PARTICLES:
              PARTICLE_OUTPUT: true
              DISPLACEMENT: true
              VELOCITY: true
              OWNER: true

            BINNING STRATEGY:
              BIN_SIZE_LOWER_BOUND: <HORIZON + margin>
              DOMAINBOUNDINGBOX: "<xmin> <ymin> <zmin> <xmax> <ymax> <zmax>"

            PARTICLE DYNAMIC:
              DYNAMICTYPE: "VelocityVerlet"
              INTERACTION: "SPH"
              RESULTSEVERY: <OUTPUT_FREQUENCY>
              RESTARTEVERY: <RESTART_FREQUENCY>
              TIMESTEP: "<dt-from-CFL: dt < 0.5 * dx / sqrt(E/rho)>"
              NUMSTEP: <total steps>
              MAXTIME: <end time>
              GRAVITY_ACCELERATION: "0.0 0.0 0.0"
              PHASE_TO_DYNLOADBALFAC: "boundaryphase 1.0 pdphase 1.0"
              PHASE_TO_MATERIAL_ID: "boundaryphase 1 pdphase 2"
              RIGID_BODY_MOTION: false
              PD_BODY_INTERACTION: true

            PARTICLE DYNAMIC/INITIAL AND BOUNDARY CONDITIONS:
              DIRICHLET_BOUNDARY_CONDITION: "boundaryphase 1"
              CONSTRAINT: "Projection2D"

            FUNCT1:
              - COMPONENT: 0
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<velocity>*t"
              - COMPONENT: 1
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
              - COMPONENT: 2
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"

            # CRITICAL: SPH section is REQUIRED even for pure PD simulations
            PARTICLE DYNAMIC/SPH:
              KERNEL: QuinticSpline
              KERNEL_SPACE_DIM: Kernel2D
              INITIALPARTICLESPACING: <dx — choose based on problem and required resolution>
              BOUNDARYPARTICLEFORMULATION: AdamiBoundaryFormulation
              TRANSPORTVELOCITYFORMULATION: StandardTransportVelocity

            PARTICLE DYNAMIC/PD:
              INTERACTION_HORIZON: <m * dx, typically m=3>
              PERIDYNAMIC_GRID_SPACING: <dx — must match INITIALPARTICLESPACING>
              PD_DIMENSION: Peridynamic_2DPlaneStrain
              NORMALCONTACTLAW: NormalLinearSpring
              NORMAL_STIFF: <contact stiffness>
              PRE_CRACKS: "<x1> <y1> <x2> <y2> ; <x3> <y3> <x4> <y4>"

            MATERIALS:
              - MAT: 1
                MAT_ParticleSPHBoundary:
                  INITRADIUS: <dx/2>
                  INITDENSITY: <density>
              - MAT: 2
                MAT_ParticlePD:
                  INITRADIUS: <dx/2>
                  INITDENSITY: <density>
                  YOUNG: <Young's modulus>
                  CRITICAL_STRETCH: <critical stretch for bond breaking>

            # Generate particle positions programmatically:
            # Regular grid with uniform spacing dx covering the domain
            PARTICLES:
              - "TYPE pdphase POS <x> <y> 0.0 PDBODYID 0"
              - "TYPE boundaryphase POS <x> <y> 0.0"
        """)
