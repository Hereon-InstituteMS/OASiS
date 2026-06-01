"""Generator for Smoothed Particle Hydrodynamics (SPH) simulations in 4C.

SPH is a meshfree Lagrangian method primarily used for fluid dynamics
(dam breaks, sloshing, free-surface flows) but also applicable to
solid-mechanics problems.  In 4C it shares the particle infrastructure
with peridynamics (PD) but uses fundamentally different physics:
kernel-weighted summation/integration for field quantities rather than
pairwise bond forces.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class ParticleSPHGenerator(BaseGenerator):
    """Generator for SPH particle simulations."""

    module_key = "particle_sph"
    display_name = "Smoothed Particle Hydrodynamics (SPH)"
    problem_type = "Particle"

    # ── Knowledge ─────────────────────────────────────────────────────

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Smoothed Particle Hydrodynamics (SPH) in 4C is a meshfree "
                "Lagrangian particle method for simulating fluid dynamics and, "
                "with appropriate constitutive models, solid mechanics.  The "
                "method approximates continuous fields by kernel-weighted "
                "summation over neighboring particles.  4C supports weakly "
                "compressible SPH with various kernels, density evaluation "
                "strategies, and boundary formulations.  Common applications "
                "include dam break, Poiseuille flow, sloshing, and free-surface "
                "flows."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "IO",
                "BINNING STRATEGY",
                "PARTICLE DYNAMIC",
                "PARTICLE DYNAMIC/SPH",
                "MATERIALS",
                "PARTICLES",
            ],
            "section_details": {
                "PROBLEM TYPE": {
                    "PROBLEMTYPE": '"Particle"',
                },
                "BINNING STRATEGY": {
                    "BIN_SIZE_LOWER_BOUND": (
                        "Must be >= the kernel support radius (typically 3*dx for "
                        "QuinticSpline).  Controls the spatial hashing bin size."
                    ),
                    "DOMAINBOUNDINGBOX": (
                        '"xmin ymin zmin xmax ymax zmax" -- must enclose ALL particles '
                        "with margin.  Include space for fluid expansion."
                    ),
                },
                "PARTICLE DYNAMIC": {
                    "DYNAMICTYPE": (
                        '"VelocityVerlet" (default explicit integrator for SPH)'
                    ),
                    "INTERACTION": '"SPH"',
                    "PHASE_TO_MATERIAL_ID": (
                        'Maps phase names to material IDs.  Example: '
                        '"phase1 1 boundaryphase 2"'
                    ),
                    "PHASE_TO_DYNLOADBALFAC": "Load-balancing factors per phase.",
                    "TIMESTEP": "Time step size (must satisfy CFL for SPH)",
                    "NUMSTEP": "Total number of time steps",
                    "MAXTIME": "Maximum simulation time",
                    "RESULTSEVERY": "Output frequency in steps",
                    "RESTARTEVERY": "Restart checkpoint frequency",
                    "GRAVITY_ACCELERATION": (
                        '"gx gy gz" -- gravitational acceleration vector.  '
                        "Essential for dam break and hydrostatic problems."
                    ),
                },
                "PARTICLE DYNAMIC/SPH": {
                    "KERNEL": (
                        "QuinticSpline (recommended, C^2 smooth, compact support) or "
                        "CubicSpline (simpler, C^1 smooth).  QuinticSpline gives "
                        "better accuracy for most problems."
                    ),
                    "KERNEL_SPACE_DIM": (
                        "Kernel1D, Kernel2D, or Kernel3D.  MUST match the physical "
                        "dimension of the problem.  A mismatch causes incorrect "
                        "kernel normalization and wrong results."
                    ),
                    "INITIALPARTICLESPACING": (
                        "The initial uniform spacing dx between particles."
                    ),
                    "DENSITYEVALUATION": (
                        "DensitySummation (direct kernel sum, conserves mass) or "
                        "DensityIntegration (continuity equation, smoother).  "
                        "DensityIntegration is more common for dynamic problems."
                    ),
                    "BOUNDARYPARTICLEFORMULATION": (
                        "AdamiBoundaryFormulation (recommended).  Handles wall "
                        "boundary conditions via mirrored pressure/velocity."
                    ),
                    "TRANSPORTVELOCITYFORMULATION": (
                        "StandardTransportVelocity -- used with some formulations "
                        "to reduce tensile instability.  Optional for pure SPH."
                    ),
                },
                "PARTICLE DYNAMIC/INITIAL AND BOUNDARY CONDITIONS": {
                    "INITIAL_VELOCITY_FIELD": (
                        '"phase1 1" -- applies FUNCT1 as initial velocity to '
                        "all particles of the named phase."
                    ),
                    "DIRICHLET_BOUNDARY_CONDITION": (
                        '"boundaryphase 1" -- prescribed displacement for boundary '
                        "particles."
                    ),
                },
            },
            "materials": {
                "MAT_ParticleSPHFluid": {
                    "description": (
                        "Weakly compressible SPH fluid material.  The equation of "
                        "state relates density to pressure via a Tait-type EOS: "
                        "p = (BULK_MODULUS / EXPONENT) * "
                        "((rho/rho0)^EXPONENT - 1) + BACKGROUNDPRESSURE.  "
                        "REFDENSFAC scales the reference density."
                    ),
                    "parameters": {
                        "INITRADIUS": {
                            "description": (
                                "Initial particle radius.  For consistent kernel "
                                "support, use INITRADIUS = 3 * dx for QuinticSpline."
                            ),
                            "range": "Problem-dependent",
                        },
                        "INITDENSITY": {
                            "description": "Reference fluid density",
                            "range": "e.g. 1000 kg/m^3 for water (SI) or 1 (normalized)",
                        },
                        "REFDENSFAC": {
                            "description": (
                                "Reference density factor.  Multiplied with "
                                "INITDENSITY to get the EOS reference density."
                            ),
                            "range": "Typically 1.0",
                        },
                        "BULK_MODULUS": {
                            "description": (
                                "Artificial bulk modulus for weakly compressible SPH.  "
                                "Controls the speed of sound: c = sqrt(BULK_MODULUS/rho).  "
                                "Should be large enough that density variations stay < 1%."
                            ),
                            "range": "Problem-dependent (usually >> rho * v_max^2)",
                        },
                        "DYNAMIC_VISCOSITY": {
                            "description": "Physical dynamic viscosity of the fluid",
                            "range": "e.g. 1e-3 Pa*s for water (SI)",
                        },
                        "BULK_VISCOSITY": {
                            "description": (
                                "Bulk viscosity for numerical stabilization.  "
                                "Often set to 0."
                            ),
                            "range": "0 to small positive value",
                        },
                        "ARTIFICIAL_VISCOSITY": {
                            "description": (
                                "Monaghan-style artificial viscosity coefficient "
                                "for shock capturing.  Set 0 for viscous flows."
                            ),
                            "range": "0 to 1.0 (0.1 typical for shock problems)",
                        },
                        "BACKGROUNDPRESSURE": {
                            "description": (
                                "Background pressure to prevent tensile instability.  "
                                "Set > 0 for free-surface flows."
                            ),
                            "range": "0 or small positive value",
                        },
                        "EXPONENT": {
                            "description": (
                                "Exponent in the Tait equation of state.  "
                                "Typical values: 1 (linear EOS) or 7 (water)."
                            ),
                            "range": "1 to 7",
                        },
                    },
                },
                "MAT_ParticleSPHBoundary": {
                    "description": (
                        "Boundary particle material for rigid walls.  These "
                        "particles are fixed (or prescribed) and provide wall "
                        "boundary conditions via the Adami formulation."
                    ),
                    "parameters": {
                        "INITRADIUS": {
                            "description": "Initial particle radius (same as fluid)",
                            "range": "Same as fluid particle spacing",
                        },
                        "INITDENSITY": {
                            "description": "Density for boundary particles",
                            "range": "Same as fluid density (for Adami formulation)",
                        },
                    },
                },
            },
            "solver": {
                "type": "Explicit (VelocityVerlet)",
                "notes": (
                    "SPH uses explicit time integration.  The time step is "
                    "governed by the CFL condition, viscous condition, and "
                    "body-force condition."
                ),
            },
            "time_integration": {
                "scheme": "Velocity Verlet (symplectic, second-order)",
                "CFL_condition": (
                    "dt < 0.25 * h / c_s where h is the smoothing length "
                    "(~ 3*dx for QuinticSpline) and c_s = sqrt(BULK_MODULUS / rho).  "
                    "Additional viscous constraint: dt < 0.125 * h^2 / nu."
                ),
            },
            "pitfalls": [
                (
                    "[Input] KERNEL_SPACE_DIM must match the physical "
                    "problem dimension.  A 2D problem with Kernel3D "
                    "gives wrong kernel normalization and incorrect "
                    "pressure/density fields. Signal: hydrostatic "
                    "pressure at the bottom of a still column equals "
                    "(2D-expected) / (3D-normalization) instead of "
                    "rho*g*h; density field deviates ~30% from "
                    "INITDENSITY at the first step. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Numerical] INITRADIUS for MAT_ParticleSPHFluid "
                    "is the kernel support radius, NOT half the "
                    "particle spacing.  For QuinticSpline, use "
                    "INITRADIUS = 3 * dx. Signal: density field is "
                    "uniformly low (~0.5 * INITDENSITY) because most "
                    "particles are outside each others' kernels; "
                    "neighbor-count diagnostic prints ~5-10 instead "
                    "of the expected ~50-80 in 2D. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] DOMAINBOUNDINGBOX must be large "
                    "enough to contain all particles throughout the "
                    "simulation, including splashing and fluid "
                    "expansion. Signal: same as PD case — abort "
                    "`particle outside DOMAINBOUNDINGBOX at step N`; "
                    "in SPH this typically triggers when a splash "
                    "particle exits the bounding y-extent on a "
                    "dam-break case. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] BULK_MODULUS must be large enough "
                    "that density variations stay below ~1%.  Rule "
                    "of thumb: BULK_MODULUS >= 100 * rho * v_max^2 "
                    "where v_max is the maximum expected velocity. "
                    "Signal: density field at peak compression "
                    "exceeds 1.01 * INITDENSITY; pressure spikes "
                    "noisy, post-impact oscillation does not damp; "
                    "Mach number = v_max/sqrt(BULK/rho) > 0.1. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "Boundary particles (boundaryphase) should use "
                    "the same INITDENSITY as the fluid for the Adami "
                    "boundary formulation to work correctly."
                ),
                (
                    "[Numerical] Particle spacing must be uniform at "
                    "initialization.  Non-uniform spacing causes "
                    "zeroth-order kernel approximation errors. "
                    "Signal: pressure field has visible block-stripe "
                    "artifacts aligned with the spacing transitions; "
                    "the hydrostatic test (column under gravity) "
                    "shows pressure deviating from rho*g*h by more "
                    "than 5% even at t=0. (Audit 2026-06-02.)"
                ),
                (
                    "[Numerical] For free-surface problems, set "
                    "BACKGROUNDPRESSURE > 0 to prevent tensile "
                    "instability at the free surface. Signal: "
                    "particles at the free surface cluster together "
                    "or 'snap' into chains; visualize shows "
                    "non-physical voids beneath the free surface and "
                    "pressure goes negative there. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] GRAVITY_ACCELERATION must be set "
                    "explicitly for hydrostatic and dam-break "
                    "problems.  The default is zero. Signal: a "
                    "dam-break column stays static — no collapse, "
                    "no flow; visualize shows particle velocities "
                    "remain ~0 throughout. (Audit 2026-06-02.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "Dam break (2D)",
                    "description": (
                        "Classic SPH benchmark: a column of fluid collapses under "
                        "gravity and splashes against a wall.  Validates free-surface "
                        "tracking and pressure computation."
                    ),
                },
                {
                    "name": "Poiseuille flow (2D)",
                    "description": (
                        "Pressure-driven flow between parallel plates.  Steady-state "
                        "parabolic velocity profile validates viscous force "
                        "implementation.  Good for convergence studies."
                    ),
                    "template_variant": "poiseuille_2d",
                },
                {
                    "name": "Hydrostatic tank",
                    "description": (
                        "Fluid at rest in a container under gravity.  The pressure "
                        "must be linear with depth (p = rho*g*h).  Tests density "
                        "evaluation and boundary conditions."
                    ),
                },
                {
                    "name": "1D pressure wave",
                    "description": (
                        "A Gaussian velocity perturbation propagates as a pressure "
                        "wave in a 1D column.  Validates the equation of state and "
                        "wave speed."
                    ),
                },
            ],
        }

    # ── Templates ─────────────────────────────────────────────────────

    def get_template(self, variant: str = "poiseuille_2d") -> str:
        if variant == "poiseuille_2d":
            return self._template_poiseuille_2d()
        raise ValueError(
            f"Unknown variant {variant!r} for {self.module_key}.  "
            f"Available: {[v['name'] for v in self.list_variants()]}"
        )

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "poiseuille_2d",
                "description": (
                    "2D Poiseuille flow between parallel plates.  Demonstrates "
                    "SPH fluid setup with boundary particles, viscous flow, and "
                    "pressure-driven steady state.  Good introductory SPH example."
                ),
            },
        ]

    # ── Validation ────────────────────────────────────────────────────

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        """Physics-aware validation of SPH parameters.

        Expected keys in *params* (all optional):
            dx              - particle spacing
            kernel          - kernel name (QuinticSpline / CubicSpline)
            kernel_dim      - Kernel1D / Kernel2D / Kernel3D
            problem_dim     - 1, 2, or 3
            bulk_modulus    - artificial bulk modulus
            density         - reference density
            v_max           - expected maximum velocity
            dt              - time step
            init_radius     - INITRADIUS from material
            domain_bbox     - [xmin, ymin, zmin, xmax, ymax, zmax]
            particles_bbox  - [xmin, ymin, zmin, xmax, ymax, zmax]
        """
        warnings: list[str] = []

        dx = params.get("dx")
        kernel_dim = params.get("kernel_dim", "")
        problem_dim = params.get("problem_dim")
        bulk_modulus = params.get("bulk_modulus")
        density = params.get("density")
        v_max = params.get("v_max")
        dt = params.get("dt")
        init_radius = params.get("init_radius")
        domain_bbox = params.get("domain_bbox")
        particles_bbox = params.get("particles_bbox")

        # Kernel dimension vs problem dimension
        if kernel_dim and problem_dim is not None:
            expected = f"Kernel{problem_dim}D"
            if kernel_dim != expected:
                warnings.append(
                    f"ERROR: KERNEL_SPACE_DIM is {kernel_dim!r} but the problem "
                    f"is {problem_dim}D.  Expected {expected!r}.  This causes "
                    f"incorrect kernel normalization."
                )

        # Init radius check for QuinticSpline
        if init_radius is not None and dx is not None:
            expected_radius = 3.0 * dx
            ratio = init_radius / expected_radius
            if ratio < 0.8 or ratio > 1.2:
                warnings.append(
                    f"WARNING: INITRADIUS = {init_radius} does not match expected "
                    f"3*dx = {expected_radius} for QuinticSpline kernel.  "
                    f"This may cause incorrect neighbor counts."
                )

        # Bulk modulus adequacy
        if bulk_modulus is not None and density is not None and v_max is not None:
            if density > 0 and v_max > 0:
                min_bulk = 100.0 * density * v_max ** 2
                if bulk_modulus < min_bulk:
                    warnings.append(
                        f"WARNING: BULK_MODULUS = {bulk_modulus} may be too small.  "
                        f"For density variations < 1%, need BULK_MODULUS >= "
                        f"{min_bulk:.2e} (100 * rho * v_max^2)."
                    )

        # CFL for SPH
        if dt is not None and dx is not None and bulk_modulus is not None and density is not None:
            if density > 0 and bulk_modulus > 0:
                import math
                c_s = math.sqrt(bulk_modulus / density)
                h = 3.0 * dx  # smoothing length for QuinticSpline
                dt_cfl = 0.25 * h / c_s
                if dt > dt_cfl:
                    warnings.append(
                        f"ERROR: CFL VIOLATION.  dt = {dt} > dt_CFL = {dt_cfl:.6e}.  "
                        f"Reduce time step."
                    )

        # Bounding box encloses particles
        if domain_bbox is not None and particles_bbox is not None:
            if len(domain_bbox) == 6 and len(particles_bbox) == 6:
                labels = ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]
                for i in range(3):
                    if domain_bbox[i] > particles_bbox[i]:
                        warnings.append(
                            f"ERROR: DOMAINBOUNDINGBOX {labels[i]}={domain_bbox[i]} "
                            f"> particle {labels[i]}={particles_bbox[i]}."
                        )
                for i in range(3, 6):
                    if domain_bbox[i] < particles_bbox[i]:
                        warnings.append(
                            f"ERROR: DOMAINBOUNDINGBOX {labels[i]}={domain_bbox[i]} "
                            f"< particle {labels[i]}={particles_bbox[i]}."
                        )

        return warnings

    # ── Private template builders ─────────────────────────────────────

    @staticmethod
    def _template_poiseuille_2d() -> str:
        """Minimal 2D Poiseuille flow between parallel plates."""
        return textwrap.dedent("""\
            # 2D SPH Poiseuille Flow
            # Pressure-driven flow between two parallel plates.
            # Steady-state: parabolic velocity profile  u(y) = dp/dx / (2*mu) * y*(H-y)
            #
            # FORMAT TEMPLATE — all values are placeholders.
            # Determine domain size, material properties, and particle spacing
            # based on the specific problem. Consult the solver's test files.
            #
            # NOTE: Replace the PARTICLES section with actual particle positions
            # generated by a script.

            PROBLEM TYPE:
              PROBLEMTYPE: "Particle"

            IO:
              STDOUTEVERY: <output_frequency>
              VERBOSITY: "Standard"

            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <vtk_output_frequency>
            IO/RUNTIME VTK OUTPUT/PARTICLES:
              PARTICLE_OUTPUT: true
              VELOCITY: true
              OWNER: true

            BINNING STRATEGY:
              BIN_SIZE_LOWER_BOUND: <must be > kernel support radius>
              DOMAINBOUNDINGBOX: "<xmin> <ymin> <zmin> <xmax> <ymax> <zmax>"

            PARTICLE DYNAMIC:
              INTERACTION: "SPH"
              RESULTSEVERY: <output_frequency>
              RESTARTEVERY: <restart_frequency>
              TIMESTEP: <dt — must satisfy CFL and viscous stability>
              NUMSTEP: <total_steps>
              MAXTIME: <end_time>
              GRAVITY_ACCELERATION: "0.0 0.0 0.0"
              PHASE_TO_DYNLOADBALFAC: "phase1 1.0 boundaryphase 1.0"
              PHASE_TO_MATERIAL_ID: "phase1 1 boundaryphase 2"

            PARTICLE DYNAMIC/INITIAL AND BOUNDARY CONDITIONS:
              INITIAL_VELOCITY_FIELD: "phase1 1"

            FUNCT1:
              - COMPONENT: 0
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<initial_vx>"
              - COMPONENT: 1
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<initial_vy>"
              - COMPONENT: 2
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"

            PARTICLE DYNAMIC/SPH:
              KERNEL: QuinticSpline
              KERNEL_SPACE_DIM: Kernel2D
              INITIALPARTICLESPACING: <dx — choose based on problem>
              DENSITYEVALUATION: DensityIntegration
              BOUNDARYPARTICLEFORMULATION: AdamiBoundaryFormulation

            MATERIALS:
              - MAT: 1
                MAT_ParticleSPHFluid:
                  INITRADIUS: <kernel_support = 3*dx for QuinticSpline>
                  INITDENSITY: <fluid_density>
                  REFDENSFAC: 1
                  EXPONENT: 1
                  BACKGROUNDPRESSURE: 0
                  BULK_MODULUS: <bulk_modulus>
                  DYNAMIC_VISCOSITY: <viscosity>
                  BULK_VISCOSITY: 0
                  ARTIFICIAL_VISCOSITY: 0
              - MAT: 2
                MAT_ParticleSPHBoundary:
                  INITRADIUS: <same as fluid INITRADIUS>
                  INITDENSITY: <same as fluid INITDENSITY>

            # Generate particle positions programmatically
            PARTICLES:
              - "TYPE phase1 POS <x> <y> 0.0"
              - "TYPE boundaryphase POS <x> <y> 0.0"
        """)
