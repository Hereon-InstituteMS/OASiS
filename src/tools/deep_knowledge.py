"""
MCP-side deep domain knowledge for SELECTED backends.

This file is NOT a complete brain — it carries prose-style knowledge for
only **4 of 8** registered backends, in two roles:

  1. **Canonical** for FEniCSx (`_FENICS_KNOWLEDGE`, 31 keys, ~91 KB):
     fenics/backend.py reads this FIRST and falls back to its own
     generator KNOWLEDGE only when a key is missing. Touch with care
     — adding/renaming keys here can shadow generator pitfalls.

  2. **Last-fallback supplement** for deal.II (`_DEALII_KNOWLEDGE`,
     14 keys, ~20 KB): dealii/backend.py reads
     data/dealii_knowledge.py first, then generator KNOWLEDGE, then
     this dict. Useful only for keys NOT covered upstream.

  3. **Workflows.py merge source** for 4C/fourc (`_4C_KNOWLEDGE`,
     9 keys, ~11 KB): the `module_knowledge` workflow overlays
     backend pitfalls onto this dict for prose like description /
     weak_form / problem_type / materials.

Cross-solver hints (`_CROSS_SOLVER_KNOWLEDGE`) supply small
benchmark-verification notes and live independently of any backend.

**NOT covered here** (their canonical knowledge lives elsewhere):
skfem, NGSolve, Kratos, DUNE, FEBio — these backends carry their
catalogs in `src/backends/<name>/generators/` plus
`data/<name>_knowledge.py` (where applicable). Do not add new
entries for those backends here; extend the per-backend module
instead.

(`_FEBIO_KNOWLEDGE` was removed 2026-06-02: it had only 4 keys
while the febio backend's own generators carry 16 physics with
matching `description` fields, and the workflows.py merge shape
allowed deep_knowledge to *shadow* the more-comprehensive backend
catalog. The backend is now the single source of truth for febio.)
"""

import json
from mcp.server.fastmcp import FastMCP
from core.registry import get_backend, available_backends


# ═══════════════════════════════════════════════════════════════════════════════
# 4C MULTIPHYSICS — COMPREHENSIVE DOMAIN KNOWLEDGE
# Ported from 4c-ai-interface generators (9 physics modules, 30+ material types)
# ═══════════════════════════════════════════════════════════════════════════════

_4C_KNOWLEDGE = {
    "scalar_transport": {
        "description": "Solves advection-diffusion equation for scalar transport. Special cases: Poisson (stationary, zero velocity), heat conduction, SUPG-stabilised advection.",
        "problem_type": "Scalar_Transport",
        "required_sections": ["PROBLEM TYPE", "SCALAR TRANSPORT DYNAMIC", "SOLVER 1", "MATERIALS", "TRANSPORT GEOMETRY"],
        "materials": {
            "MAT_scatra": {"DIFFUSIVITY": "Isotropic diffusion coefficient > 0 (typical 0.01-100)"},
            "MAT_Fourier": {"CAPA": "Volumetric heat capacity (rho*c_p) > 0", "CONDUCT": "Thermal conductivity (YAML: constant: [value]) > 0"},
        },
        "time_integration": {
            "TIMEINTEGR": "Stationary | BDF2 | OneStepTheta",
            "SOLVERTYPE": "linear_full (linear) | nonlinear (nonlinear terms)",
            "VELOCITYFIELD": "zero (pure diffusion) | function (prescribed) | Navier_Stokes",
        },
        "solver": {"small": "UMFPACK (direct, ~50k DOFs)", "large": "Belos + MueLu (iterative, scalable)"},
        "pitfalls": [
            "Section name is 'SCALAR TRANSPORT DYNAMIC', NOT 'SCATRA DYNAMIC'",
            "VELOCITYFIELD must be 'zero' (not omitted) for pure diffusion",
            "VTK path: SCALAR TRANSPORT DYNAMIC/RUNTIME VTK OUTPUT (NOT IO/RUNTIME VTK OUTPUT/SCATRA)",
            "Geometry section: TRANSPORT GEOMETRY with TRANSP element category",
            "NUMDOF=1, all arrays (ONOFF/VAL/FUNCT) have exactly 1 entry",
        ],
        "variants": ["poisson_2d", "heat_transient_2d"],
    },
    "solid_mechanics": {
        "description": "Quasi-static structural problems. DYNAMICTYPE: Statics, small/large deformation, 2D (WALL) / 3D (SOLID).",
        "problem_type": "Structure",
        "required_sections": ["PROBLEM TYPE", "STRUCTURAL DYNAMIC", "SOLVER 1", "MATERIALS", "STRUCTURE GEOMETRY"],
        "materials": {
            "MAT_Struct_StVenantKirchhoff": {"YOUNG": "> 0 (steel 210e3 MPa)", "NUE": "0 < nu < 0.5", "DENS": "Optional for statics"},
            "MAT_ElastHyper + ELAST_CoupNeoHooke": {"NUMMAT": "1", "MATIDS": "[id]", "DENS": "> 0"},
            "MAT_Struct_PlasticNlnLogNeoHooke": {"YOUNG": "> 0", "NUE": "0 < nu < 0.5", "YIELD": "Initial yield > 0", "SATHARDENING": ">= 0", "HARDEXPO": "> 0"},
        },
        "time_integration": {
            "DYNAMICTYPE": "Statics (quasi-static, incremental loading)",
            "KINEM": "linear (small def) vs nonlinear (large def)",
            "MAXITER": "1 for linear, 20-50 for nonlinear",
            "TOLDISP": "1e-6 to 1e-10", "TOLRES": "1e-6 to 1e-10",
        },
        "solver": {"small": "UMFPACK (direct, ~50k DOFs)", "large": "Belos + MueLu (GMRES + AMG)"},
        "pitfalls": [
            "KINEM must match material: Neo-Hookean/plasticity REQUIRE nonlinear",
            "MAXITER=1 only for truly linear problems",
            "HEX8 suffers locking — use TECH: eas_full, fbar, or higher-order elements",
            "2D uses WALL category (not SOLID), requires THICK and STRESS_STRAIN",
            "Neumann BCs have NUMDOF: 6 (forces + moments)",
        ],
        "variants": ["linear_2d", "nonlinear_3d"],
    },
    "fluid": {
        "description": "Incompressible Navier-Stokes with SUPG/PSPG stabilisation. Fixed Eulerian (NA: Euler) or ALE (for FSI).",
        "problem_type": "Fluid",
        "required_sections": ["PROBLEM TYPE", "PROBLEM SIZE", "FLUID DYNAMIC", "SOLVER 1", "MATERIALS", "FLUID GEOMETRY"],
        "materials": {
            "MAT_fluid": {"DYNVISCOSITY": "Dynamic viscosity [Pa*s] > 0 (water 1e-3, air 1.8e-5)", "DENSITY": "Fluid density [kg/m^3] > 0 (water 1000, air 1.2)"},
        },
        "time_integration": {
            "schemes": ["Np_Gen_Alpha (RECOMMENDED)", "BDF2", "OneStepTheta", "Stationary"],
            "TIMESTEP": "Time step size", "NUMSTEP": "Number of steps", "ITEMAX": "Max nonlinear iters (default 10)",
        },
        "solver": {"small_2d": "UMFPACK (< ~50k DOFs)", "large_or_3d": "Belos with block preconditioner"},
        "pitfalls": [
            "NUMDOF INCLUDES pressure: 3 in 2D (vx,vy,p), 4 in 3D (vx,vy,vz,p)",
            "Stabilisation (SUPG/PSPG) critical — without it, equal-order elements oscillate",
            "Fully Dirichlet velocity: pressure up to constant — PIN at one node",
            "FLUID GEOMETRY uses FLUID category (not SOLID)",
            "Use NA: Euler for pure fluid, NA: ALE only for FSI mesh motion",
        ],
        "variants": ["channel_2d", "cavity_2d"],
    },
    "fsi": {
        "description": "Monolithic/partitioned coupling of incompressible Navier-Stokes with geometrically nonlinear structures via ALE mesh motion. Most complex problem type in 4C.",
        "problem_type": "Fluid_Structure_Interaction",
        "required_sections": [
            "PROBLEM TYPE", "STRUCTURAL DYNAMIC", "STRUCTURAL DYNAMIC/GENALPHA",
            "FLUID DYNAMIC", "ALE DYNAMIC", "FSI DYNAMIC", "FSI DYNAMIC/MONOLITHIC SOLVER",
            "SOLVER 1, 2, 3", "MATERIALS", "STRUCTURE GEOMETRY", "FLUID GEOMETRY",
            "CLONING MATERIAL MAP", "DESIGN FSI COUPLING CONDITIONS",
        ],
        "materials": {
            "MAT_fluid": "Newtonian (DYNVISCOSITY, DENSITY)",
            "MAT_ElastHyper": "Hyperelastic structure (Neo-Hooke)",
            "ALE clone": "Spring-based ALE via CLONING MATERIAL MAP",
        },
        "coupling": {
            "recommended": "iter_mortar_monolithicfluidsplit",
            "alternatives": ["iter_monolithicfluidsplit", "iter_stagg_AITKEN_rel_force"],
        },
        "pitfalls": [
            "Fluid MUST use NA: ALE (NOT Euler!) for FSI",
            "ALE Dirichlet BCs on ALL outer fluid boundaries (not FSI interface) — missing = mesh distortion",
            "CLONING MATERIAL MAP is MANDATORY (fluid mat → ALE pseudo-mat)",
            "SHAPEDERIVATIVES: true in MONOLITHIC SOLVER",
            "Each field (structure, fluid, ALE) needs own SOLVER N entry",
            "2D: DESIGN FSI COUPLING LINE CONDITIONS, 3D: SURF CONDITIONS",
            "Structure NUMDOF = dim, Fluid NUMDOF = dim+1 (includes pressure)",
        ],
        "variants": ["fsi_2d"],
    },
    "beams": {
        "description": "Geometrically exact beam elements: BEAM3R (Reissner, shear-deformable), BEAM3EB (Euler-Bernoulli), BEAM3K (Kirchhoff). CRITICAL: MUST use inline mesh (NODE COORDS + STRUCTURE ELEMENTS), NOT Exodus.",
        "problem_type": "Structure",
        "required_sections": ["PROBLEM TYPE", "STRUCTURAL DYNAMIC", "SOLVER 1", "MATERIALS", "NODE COORDS", "STRUCTURE ELEMENTS", "DNODE-NODE TOPOLOGY", "DLINE-NODE TOPOLOGY"],
        "beam_types": {
            "BEAM3R": {"name": "Reissner (shear-deformable)", "topologies": ["LINE2", "LINE3", "LINE4"], "dofs": "6 or 9 (HERMITE)"},
            "BEAM3EB": {"name": "Euler-Bernoulli (torsion-free)", "topologies": ["LINE2"], "dofs": "6"},
            "BEAM3K": {"name": "Kirchhoff (with torsion)", "topologies": ["LINE2", "LINE3"], "dofs": "6 or 7"},
        },
        "materials": {
            "MAT_BeamReissnerElastHyper": {"YOUNG": "> 0", "SHEARMOD": "G = E/(2(1+nu))", "CROSSAREA": "> 0", "MOMINPOL": "J", "MOMIN2": "I_yy", "MOMIN3": "I_zz", "SHEARCORR": "circle: 6/7, rect: 5/6"},
        },
        "pitfalls": [
            "Beams CANNOT use Exodus — must use inline NODE COORDS + STRUCTURE ELEMENTS",
            "TRIADS required for BEAM3R/K (initial orientation)",
            "LINE3: endpoint1-endpoint2-midpoint ordering (NOT sequential!)",
            "GenAlphaLieGroup REQUIRED for dynamics (not standard GenAlpha)",
            "MASSLIN: rotations required with GenAlphaLieGroup",
            "Cross-section properties must be mutually consistent",
        ],
        "variants": ["cantilever_static", "cantilever_dynamic"],
    },
    "contact": {
        "description": "Mortar-based contact between deformable bodies. Penalty / Uzawa / Nitsche. Adds CONTACT DYNAMIC + MORTAR COUPLING on top of structure.",
        "problem_type": "Structure",
        "required_sections": ["PROBLEM TYPE", "STRUCTURAL DYNAMIC", "MORTAR COUPLING", "CONTACT DYNAMIC", "SOLVER 1", "MATERIALS", "STRUCTURE GEOMETRY", "DESIGN SURF MORTAR CONTACT CONDITIONS 3D"],
        "materials": {
            "MAT_Struct_StVenantKirchhoff": {"YOUNG": "> 0", "NUE": "0 < nu < 0.5", "DENS": ">= 0 (0 for quasi-static)"},
            "MAT_ElastHyper": "For large-deformation contact",
        },
        "strategies": {
            "Penalty": "Stiff spring on penetration. PENALTYPARAM (1e2-1e5): too low=penetration, too high=ill-conditioning",
            "Uzawa": "Augmented Lagrangian. Accurate, expensive.",
            "Nitsche": "Variationally consistent penalty. Accuracy + simplicity.",
        },
        "pitfalls": [
            "Both MORTAR COUPLING and CONTACT DYNAMIC required — missing either crashes/ignores",
            "Each interface needs BOTH Slave and Master with same InterfaceID",
            "PENALTYPARAM tuning critical: start 1e3, adjust by penetration depth",
            "Quasi-static MUST use load stepping — full load in 1 step → Newton divergence",
            "Slave surface = finer mesh or softer body",
            "KINEM must be nonlinear for correct gap computation",
            "Contact surfaces must NOT overlap initially",
        ],
        "variants": ["penalty_3d"],
    },
    "structural_dynamics": {
        "description": "Time-dependent structural: impact, vibration, wave propagation. GenAlpha (implicit, recommended) or ExplEuler.",
        "problem_type": "Structure",
        "required_sections": ["PROBLEM TYPE", "STRUCTURAL DYNAMIC", "SOLVER 1", "MATERIALS", "STRUCTURE GEOMETRY"],
        "materials": {
            "MAT_Struct_StVenantKirchhoff": {"YOUNG": "> 0", "NUE": "0 < nu < 0.5", "DENS": "MANDATORY > 0 for dynamics (zero = singular mass matrix!)"},
            "MAT_ElastHyper + ELAST_CoupNeoHooke": {"YOUNG": "> 0", "NUE": "0 < nu < 0.5", "DENS": "> 0 in wrapper"},
        },
        "time_integration": {
            "GenAlpha": "Implicit, 2nd order, RHO_INF [0,1]: 1=energy-conserving, 0=max damping (typical 0.8-0.9)",
            "GenAlphaLieGroup": "Lie-group variant for beams (rotational DOFs on SO(3))",
            "ExplEuler": "Explicit, CFL-constrained (dt < h/c where c=sqrt(E/rho))",
        },
        "damping": {"Rayleigh": "M_DAMP (low freq) + K_DAMP (high freq)", "None": "Numerical dissipation only"},
        "pitfalls": [
            "DENS MANDATORY and > 0 — zero/missing = zero mass matrix (singular)",
            "Time step must resolve highest frequency of interest",
            "Explicit: CFL violation = immediate divergence",
            "RHO_INF=1: energy-conserving but may show spurious ringing — reduce to 0.8",
        ],
        "variants": ["genalpha_2d"],
    },
    "particle_pd": {
        "description": "Bond-based peridynamics for fracture. Non-local integral equations. CRITICAL: SPH section MANDATORY even for pure PD (else 'pd_neighbor_pairs=0' crash).",
        "problem_type": "Particle",
        "required_sections": ["PROBLEM TYPE", "IO", "BINNING STRATEGY", "PARTICLE DYNAMIC", "PARTICLE DYNAMIC/SPH", "PARTICLE DYNAMIC/PD", "MATERIALS", "PARTICLES"],
        "materials": {
            "MAT_ParticlePD": {"INITRADIUS": "dx/2", "INITDENSITY": "e.g. 8e-3 g/mm^3 steel", "YOUNG": "e.g. 190e3 MPa steel", "CRITICAL_STRETCH": "Bond break 0.001-0.05"},
            "MAT_ParticleSPHBoundary": {"INITRADIUS": "Same as PD", "INITDENSITY": "Can be 1 (rigid)"},
        },
        "solver": "Explicit only (VelocityVerlet). dt < dx/sqrt(E/rho), safety factor 0.5.",
        "pre_cracks": "Visibility condition: bonds crossing line segments broken at init. Format: 'x1 y1 x2 y2 ; x3 y3 x4 y4'",
        "pitfalls": [
            "PARTICLE DYNAMIC/SPH section MANDATORY for PD — missing causes crash",
            "DOMAINBOUNDINGBOX must enclose ALL particles including moving impactor",
            "Use boundaryphase (NOT rigidphase) for rigid impactors",
            "Horizon ratio m=delta/dx >= 3 for convergence",
            "BIN_SIZE_LOWER_BOUND > horizon (else neighbors missed)",
            "Bond-based PD restricts Poisson's ratio: nu=0.25 (2D), nu=1/3 (3D)",
            "CFL violation = UNSTABLE",
        ],
        "unit_systems": {"mm_ms_g": "Length=mm, Time=ms, Mass=g, Stress=MPa", "SI": "Length=m, Time=s, Mass=kg, Stress=Pa"},
        "variants": ["plate_2d", "impact_2d"],
    },
    "particle_sph": {
        "description": "Smoothed Particle Hydrodynamics for free-surface flows (dam break, sloshing). Meshfree Lagrangian, kernel-weighted summation.",
        "problem_type": "Particle",
        "required_sections": ["PROBLEM TYPE", "IO", "BINNING STRATEGY", "PARTICLE DYNAMIC", "PARTICLE DYNAMIC/SPH", "MATERIALS", "PARTICLES"],
        "materials": {
            "MAT_ParticleSPHFluid": {
                "INITRADIUS": "Kernel support (3*dx for QuinticSpline)", "INITDENSITY": "Reference density",
                "BULK_MODULUS": "Artificial (>> rho*v_max^2)", "DYNAMIC_VISCOSITY": "Physical viscosity",
                "ARTIFICIAL_VISCOSITY": "Monaghan shock capturing (0-1, typical 0.1)",
                "BACKGROUNDPRESSURE": "> 0 for free-surface", "EXPONENT": "Tait EOS (1=linear, 7=water)",
            },
        },
        "solver": "Explicit (VelocityVerlet). CFL: dt < 0.25*h/c_s where c_s=sqrt(K/rho).",
        "pitfalls": [
            "KERNEL_SPACE_DIM MUST match physical dimension — mismatch = wrong normalization",
            "INITRADIUS is kernel support radius (3*dx for QuinticSpline), NOT half spacing",
            "DOMAINBOUNDINGBOX must accommodate fluid expansion/splashing",
            "BULK_MODULUS >= 100*rho*v_max^2 for <1% density variation",
            "Boundary particles MUST use same INITDENSITY as fluid (Adami formulation)",
            "BACKGROUNDPRESSURE > 0 for free-surface problems",
        ],
        "variants": ["poiseuille_2d", "dam_break_2d"],
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# FENICSX (DOLFINX) — COMPREHENSIVE DOMAIN KNOWLEDGE
# ═══════════════════════════════════════════════════════════════════════════════


def get_deep_fenics_knowledge(physics: str) -> dict:
    """Get deep FEniCSx knowledge for a specific physics type."""
    return _FENICS_KNOWLEDGE.get(physics, {})


_FENICS_KNOWLEDGE = {
    # ═══════════════════════════════════════════════════════════════════════════
    # ELEMENT CATALOG — Complete Basix/UFL element families
    # ═══════════════════════════════════════════════════════════════════════════
    "element_catalog": {
        "description": "Complete catalog of finite element families available in FEniCSx via Basix. Elements are created with basix.ufl.element() or basix.ufl.blocked_element().",
        "basix_element_families": {
            "P (Lagrange)": {
                "basix_name": "basix.ElementFamily.P",
                "ufl_name": "'Lagrange' or 'P'",
                "continuity": "C0 (continuous across facets)",
                "orders": "1, 2, 3, ... (arbitrary order); pyramid is capped at degree 2 — degree 3 raises RuntimeError 'Non-equispaced points on pyramids not supported yet.' (verified basix 0.10.0, 2026-08-03)",
                "cell_types": "interval, triangle, quadrilateral, tetrahedron, hexahedron, prism, pyramid",
                "api": "basix.ufl.element('Lagrange', cell, degree)",
                "variants": {
                    "equispaced": "basix.LagrangeVariant.equispaced (equally spaced points, default for low order)",
                    "gll_warped": "basix.LagrangeVariant.gll_warped (GLL points, lower Lebesgue constant for high order)",
                    "gll_isaac": "basix.LagrangeVariant.gll_isaac (GLL with Isaac warp on simplices)",
                    "gll_centroid": "basix.LagrangeVariant.gll_centroid (GLL with centroid warp)",
                    "chebyshev_warped": "basix.LagrangeVariant.chebyshev_warped (Chebyshev points) — DISCONTINUOUS ONLY: on a continuous Lagrange element basix 0.10 raises RuntimeError 'This variant of Lagrange is only supported for discontinuous elements'. Use it on 'DG'. (Verified 2026-08-03.)",
                    "chebyshev_isaac": "basix.LagrangeVariant.chebyshev_isaac — discontinuous only (same RuntimeError on continuous Lagrange)",
                    "chebyshev_centroid": "basix.LagrangeVariant.chebyshev_centroid — discontinuous only (same RuntimeError on continuous Lagrange)",
                },
                "notes": "Use gll_warped for degree >= 5 to avoid Runge phenomenon. DG variant: 'DG' or basix.ElementFamily.P with discontinuous=True.",
            },
            "DG (Discontinuous Lagrange)": {
                "basix_name": "basix.ElementFamily.P (with discontinuous=True)",
                "ufl_name": "'DG' or 'Discontinuous Lagrange'",
                "continuity": "Discontinuous (no inter-element continuity)",
                "orders": "0, 1, 2, ... (arbitrary order, DG0 = piecewise constant)",
                "api": "basix.ufl.element('DG', cell, degree)",
                "use_cases": "Advection-dominated problems, conservation laws, DG methods, interior penalty",
            },
            "RT (Raviart-Thomas)": {
                "basix_name": "basix.ElementFamily.RT",
                "ufl_name": "'RT' or 'Raviart-Thomas'",
                "continuity": "H(div) — normal component continuous across facets",
                "orders": "1, 2, 3, ...",
                "cell_types": "triangle, quadrilateral, tetrahedron, hexahedron",
                "api": "basix.ufl.element('RT', cell, degree)",
                "use_cases": "Mixed Poisson (Darcy flow), flux-conservative methods",
                "notes": "Pair with DG(k-1) for stable mixed Poisson. Normal component preserved by contravariant Piola map.",
            },
            "BDM (Brezzi-Douglas-Marini)": {
                "basix_name": "basix.ElementFamily.BDM",
                "ufl_name": "'BDM' or 'Brezzi-Douglas-Marini'",
                "continuity": "H(div) — normal component continuous",
                "orders": "1, 2, 3, ...",
                "cell_types": "triangle, quadrilateral, tetrahedron, hexahedron",
                "api": "basix.ufl.element('BDM', cell, degree)",
                "notes": "Full polynomial space on each cell (more DOFs than RT but better approximation).",
            },
            "N1E (Nedelec 1st kind)": {
                "basix_name": "basix.ElementFamily.N1E",
                "ufl_name": "'N1curl' or 'Nedelec 1st kind H(curl)'",
                "continuity": "H(curl) — tangential component continuous across facets",
                "orders": "1, 2, 3, ...",
                "cell_types": "triangle, quadrilateral, tetrahedron, hexahedron",
                "api": "basix.ufl.element('N1curl', cell, degree)",
                "use_cases": "Maxwell equations, electromagnetic wave propagation, curl-curl problems",
                "notes": "Tangential component preserved by covariant Piola map. Essential for electromagnetics.",
            },
            "N2E (Nedelec 2nd kind)": {
                "basix_name": "basix.ElementFamily.N2E",
                "ufl_name": "'N2curl' or 'Nedelec 2nd kind H(curl)'",
                "continuity": "H(curl)",
                "orders": "1, 2, ...",
                "cell_types": "triangle, quadrilateral, tetrahedron, hexahedron",
                "api": "basix.ufl.element('N2curl', cell, degree)",
                "notes": "Full polynomial space (more DOFs than N1E, better approximation).",
            },
            "CR (Crouzeix-Raviart)": {
                "basix_name": "basix.ElementFamily.CR",
                "ufl_name": "'CR' or 'Crouzeix-Raviart'",
                "continuity": "Nonconforming — continuous at facet midpoints only",
                "orders": "1 only (degree 2 raises RuntimeError 'Degree must be 1 for Crouzeix-Raviart')",
                # 2026-08-03: quadrilateral/hexahedron were listed here but
                # basix 0.10 rejects them —
                # ValueError: Unknown element family: CR with cell type
                # quadrilateral. Simplices only.
                "cell_types": "triangle, tetrahedron ONLY — quadrilateral/hexahedron raise ValueError 'Unknown element family: CR with cell type quadrilateral' (verified 2026-08-03)",
                "api": "basix.ufl.element('CR', cell, 1)",
                "use_cases": "Stokes (CR/DG0 pair is inf-sup stable), nonconforming methods",
            },
            "bubble": {
                "basix_name": "basix.ElementFamily.bubble",
                "ufl_name": "'Bubble'",
                "continuity": "Zero on element boundaries (vanishes on facets)",
                # 2026-08-03: minimum degrees re-measured on basix 0.10 —
                # triangle 3, tetrahedron 4, quadrilateral 2, hexahedron 2
                # (the old '3 for hex' was wrong; hex degree 2 builds a
                # 1-dof bubble, degree 3 gives 8 dofs).
                "orders": "Minimum degree per cell type: 2 for interval, 3 for triangle, 4 for tet, 2 for quad, 2 for hex (below that: RuntimeError 'Bubble element on a <cell> must have degree at least N'). Verified 2026-08-03; the interval minimum was added 2026-08-03 after a full degree sweep — cell_types listed interval but its minimum was missing.",
                "cell_types": "interval, triangle, quadrilateral, tetrahedron, hexahedron",
                "api": "basix.ufl.element('Bubble', cell, degree)",
                "use_cases": "MINI element for Stokes (Lagrange + Bubble enrichment), stabilization",
            },
            "Regge": {
                "basix_name": "basix.ElementFamily.Regge",
                "ufl_name": "'Regge'",
                "continuity": "Tangent-tangent component continuous",
                "orders": "0, 1, 2, ...",
                "cell_types": "triangle, tetrahedron",
                "api": "basix.ufl.element('Regge', cell, degree)",
                "use_cases": "Linearized general relativity, metric tensors, elasticity complexes",
            },
            "HHJ (Hellan-Herrmann-Johnson)": {
                "basix_name": "basix.ElementFamily.HHJ",
                "ufl_name": "'HHJ'",
                "continuity": "Normal-normal component continuous",
                "orders": "0, 1, 2, ...",
                # 2026-08-03: tetrahedron works too in basix 0.10
                # (element('HHJ','tetrahedron',1).dim == 24).
                "cell_types": "triangle, tetrahedron (verified 2026-08-03)",
                "api": "basix.ufl.element('HHJ', cell, degree)",
                "use_cases": "Kirchhoff plates, biharmonic equation (symmetric tensor field for moments)",
            },
            "serendipity": {
                "basix_name": "basix.ElementFamily.serendipity",
                "ufl_name": "'S' or 'serendipity'",
                "continuity": "C0",
                "orders": "1, 2, 3, ...",
                "cell_types": "quadrilateral, hexahedron (interval also builds: dim 2 at degree 1, 3 at degree 2). Simplices/prism/pyramid raise ValueError 'Unknown element family: serendipity with cell type triangle'. Cell types swept by execution 2026-08-03.",
                "api": "basix.ufl.element('S', cell, degree)",
                "notes": "Fewer DOFs than tensor-product Lagrange on quads/hexes. S2 has no interior node on quad.",
            },
            "DPC (Discontinuous Piecewise Complete)": {
                "basix_name": "basix.ElementFamily.DPC",
                "ufl_name": "'DPC'",
                "continuity": "Discontinuous",
                "orders": "0, 1, 2, ...",
                "cell_types": "quadrilateral, hexahedron (interval also builds: dim 1 at degree 0, 2 at degree 1). Simplices/prism/pyramid raise ValueError 'Unknown element family: DPC with cell type triangle'. Cell types swept by execution 2026-08-03.",
                "api": "basix.ufl.element('DPC', cell, degree)",
                "notes": "Complete polynomial on quads/hexes (not tensor-product). Used in compatible DG schemes.",
            },
            "Hermite": {
                "basix_name": "basix.ElementFamily.Hermite",
                "ufl_name": "'Hermite'",
                "continuity": "C1 (value and gradient continuous at vertices)",
                "orders": "3",
                "cell_types": "interval, triangle, tetrahedron",
                # 2026-08-03: the STRING form is not accepted by basix 0.10 —
                # basix.ufl.element('Hermite', 'triangle', 3) raises
                # ValueError: Unknown element family: Hermite with cell type
                # triangle. The ENUM form works (dim 10 on triangle,
                # 20 on tet, 4 on interval).
                "api": "basix.ufl.element(basix.ElementFamily.Hermite, basix.CellType.triangle, 3) — the ENUM is required; the string 'Hermite' raises ValueError 'Unknown element family: Hermite with cell type triangle' in basix 0.10 (verified 2026-08-03)",
                "use_cases": "Beam/plate problems requiring C1 continuity, Kirchhoff theory",
            },
            "iso (isoparametric/macro)": {
                "basix_name": "basix.ElementFamily.iso",
                "ufl_name": "'iso'",
                "continuity": "C0 (piecewise on sub-cells)",
                # 2026-08-03: measured limits on basix 0.10 — degree 2 only
                # unless a LagrangeVariant is supplied ('Lagrange elements of
                # degree > 2 need to be given a variant'), and tetrahedron is
                # not implemented at all ('Only degree 0 and 1 macro polysets
                # are currently implemented on a tetrahedron').
                "orders": "2 (degree > 2 raises RuntimeError 'Lagrange elements of degree > 2 need to be given a variant' unless a LagrangeVariant is passed). Verified 2026-08-03.",
                "cell_types": "interval, triangle, quadrilateral, hexahedron — NOT tetrahedron (RuntimeError 'Only degree 0 and 1 macro polysets are currently implemented on a tetrahedron'). Verified 2026-08-03.",
                "api": "basix.ufl.element('iso', cell, degree)",
                "notes": "Macro element: cell is split into sub-cells, lower-order polynomial on each. Fewer DOFs than standard high-order.",
            },
        },
        "compound_elements": {
            "blocked_element": {
                "api": "basix.ufl.blocked_element(sub_element, shape=(gdim,))",
                "use": "Vector/tensor function spaces from scalar elements. E.g., vector Lagrange for elasticity.",
                "example": "Ve = basix.ufl.element('Lagrange', cell, 2); basix.ufl.blocked_element(Ve, shape=(3,))",
            },
            "mixed_element": {
                "api": "basix.ufl.mixed_element([el1, el2, ...])",
                "use": "Combine different elements for mixed formulations (Taylor-Hood, Stokes, etc.)",
                "example": "P2 = basix.ufl.element('Lagrange', cell, 2, shape=(gdim,)); P1 = basix.ufl.element('Lagrange', cell, 1); ME = basix.ufl.mixed_element([P2, P1])",
            },
            "enriched_element": {
                "api": "basix.ufl.enriched_element([el1, el2])",
                "use": "Combine elements to enrich approximation space. Used for MINI element.",
                "example": "P1 = basix.ufl.element('Lagrange', cell, 1, shape=(gdim,)); B = basix.ufl.element('Bubble', cell, 3, shape=(gdim,)); MINI = basix.ufl.enriched_element([P1, B])",
            },
        },
        "cell_types": {
            "interval": "1D line segment",
            "triangle": "2D simplex (3 vertices)",
            "quadrilateral": "2D quad (4 vertices)",
            "tetrahedron": "3D simplex (4 vertices)",
            "hexahedron": "3D brick (8 vertices)",
            "prism": "3D triangular prism (6 vertices)",
            "pyramid": "3D pyramid (5 vertices)",
        },
        "pitfalls": [
            "In dolfinx >= 0.8, use basix.ufl.element() NOT ufl.FiniteElement() — the legacy names are GONE, not merely deprecated: ufl.FiniteElement / ufl.VectorElement / ufl.MixedElement all raise AttributeError \"module 'ufl' has no attribute 'FiniteElement'\" on ufl 2025.2.1 (verified 2026-08-03)",
            "For vector elements use blocked_element or shape= parameter, NOT VectorElement (removed)",
            "For mixed spaces use basix.ufl.mixed_element, NOT ufl.MixedElement (removed)",
            "Element variant matters for high order (>= 5): use gll_warped to avoid ill-conditioning. The chebyshev_* variants are DISCONTINUOUS-ONLY — asking for them on continuous Lagrange raises RuntimeError 'This variant of Lagrange is only supported for discontinuous elements' (verified 2026-08-03)",
            "Not all element families support all cell types — check Basix docs for compatibility. Measured on basix 0.10: CR and Regge are simplex-only; iso is not implemented on tetrahedra; Lagrange on pyramid stops at degree 2",
            "Bubble element minimum degree depends on cell type: 3 for triangle, 4 for tet, 2 for quad, 2 for hex",
            "Serendipity and DPC elements are the tensor-product families: quadrilateral and hexahedron (plus the degenerate interval case, which also builds). Both raise ValueError 'Unknown element family: <fam> with cell type triangle' on simplices, prisms and pyramids. (Cell-type support swept by execution on basix 0.10.0, 2026-08-03 — the earlier wording 'only available on quads/hexes' omitted interval.)",
            "[API] Some families are reachable ONLY through the basix.ElementFamily ENUM, not the family string. Hermite is the concrete case. Signal: basix.ufl.element('Hermite', 'triangle', 3) raises ValueError 'Unknown element family: Hermite with cell type triangle', while basix.ufl.element(basix.ElementFamily.Hermite, basix.CellType.triangle, 3) builds the 10-dof C1 element. (Verified empirically 2026-08-03, basix 0.10.0.)",
            "[API] 'CG' still resolves. Signal: basix.ufl.element('CG', 'triangle', 1) returns a valid element and only emits a DeprecationWarning ( '\"CG\" element name is deprecated. Consider using \"Lagrange\" or \"P\" instead') — it is NOT rejected. 'P' also resolves. The name that genuinely raises ValueError 'Unknown element family: P1 with cell type triangle' is the old DOLFIN degree-suffixed form 'P1'. (Verified empirically 2026-08-03 — corrects an older catalog claim that 'CG' raises.)",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # MESH CAPABILITIES
    # ═══════════════════════════════════════════════════════════════════════════
    "mesh_catalog": {
        "description": "Complete mesh creation, import, and manipulation capabilities in DOLFINx.",
        "built_in_meshes": {
            "create_unit_square": {
                "api": "dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=CellType.triangle)",
                "geometry": "[0,1] x [0,1]",
                "cell_types": "CellType.triangle (default), CellType.quadrilateral",
            },
            "create_unit_cube": {
                "api": "dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, nx, ny, nz, cell_type=CellType.tetrahedron)",
                "geometry": "[0,1]^3",
                "cell_types": "CellType.tetrahedron (default), CellType.hexahedron",
            },
            "create_rectangle": {
                "api": "dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [p0, p1], [nx, ny], cell_type=...)",
                "geometry": "Arbitrary rectangle [p0, p1]",
                "cell_types": "CellType.triangle, CellType.quadrilateral",
            },
            "create_box": {
                "api": "dolfinx.mesh.create_box(MPI.COMM_WORLD, [p0, p1], [nx, ny, nz], cell_type=...)",
                "geometry": "Arbitrary box [p0, p1]",
                "cell_types": "CellType.tetrahedron, CellType.hexahedron",
            },
            "create_unit_interval": {
                "api": "dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, n)",
                "geometry": "[0,1] interval",
            },
            "create_interval": {
                "api": "dolfinx.mesh.create_interval(MPI.COMM_WORLD, n, [a, b])",
                "geometry": "[a,b] interval",
            },
        },
        "gmsh_integration": {
            "api_0_9": "dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0)",
            "api_0_10": "dolfinx.io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0) — returns a MeshData object with .mesh/.cell_tags/.facet_tags/.ridge_tags/.peak_tags/.physical_groups (verified 2026-08-03)",
            # 2026-08-03: the gmshio module is GONE in 0.10 —
            # `import dolfinx.io.gmshio` raises ModuleNotFoundError.
            "read_from_msh": "dolfinx.io.gmsh.read_from_msh('file.msh', MPI.COMM_WORLD, rank=0) — NOTE the module is `gmsh`, not `gmshio`: `import dolfinx.io.gmshio` raises ModuleNotFoundError: No module named 'dolfinx.io.gmshio' on 0.10 (verified 2026-08-03)",
            "workflow": "1. Build geometry with gmsh Python API, 2. Mesh with gmsh.model.mesh.generate(dim), 3. Convert with model_to_mesh()",
            "returns": "MeshData with mesh, cell_tags (codim 0), facet_tags (codim 1), ridge/peak tags, physical group lookup",
            "notes": "Gmsh model processed on rank 0, DOLFINx mesh distributed across all ranks automatically.",
        },
        "xdmf_import": {
            "read_mesh": "with dolfinx.io.XDMFFile(MPI.COMM_WORLD, 'mesh.xdmf', 'r') as f: mesh = f.read_mesh()",
            "read_tags": "f.read_meshtags(mesh, name='facets')",
            "notes": "Good for pre-generated meshes. Geometry order <= 2 supported.",
        },
        "vtkhdf_import": {
            "api": "dolfinx.io.vtkhdf.read_mesh('mesh.vtkhdf', MPI.COMM_WORLD) — new in 0.10",
            "notes": "Kitware's future-proof format. Transition from XDMF has started. Writing is present too in 0.10: dolfinx.io.vtkhdf exposes write_mesh, write_point_data, write_cell_data (verified 2026-08-03).",
        },
        "mesh_refinement": {
            # 2026-08-03: BOTH refine entry points need the edges to exist
            # first. On a freshly created mesh:
            #   uniform_refine -> RuntimeError: Missing entities of dimension 1,
            #                     need to call create_entities(1)
            #   refine         -> RuntimeError: Missing IndexMap in Topology.
            #                     Maybe you need to create_entities(1).
            "PREREQUISITE": "mesh.topology.create_entities(1) MUST be called before either refine entry point on a freshly built mesh, otherwise RuntimeError 'Missing entities of dimension 1, need to call create_entities(1)' (uniform_refine) / 'Missing IndexMap in Topology. Maybe you need to create_entities(1).' (refine). Verified 2026-08-03.",
            "uniform_refine": "mesh.topology.create_entities(1); m2 = dolfinx.mesh.uniform_refine(mesh) — refines all cells uniformly, returns a Mesh (32 -> 128 cells on a 4x4 unit square)",
            "refine": "mesh.topology.create_entities(1); m2, parent_cells, parent_facets = dolfinx.mesh.refine(mesh, edges=None) — returns a 3-TUPLE, not a bare Mesh. Measured element types on 0.10.0 (2026-08-03): (Mesh, ndarray, NoneType) for the default call — the third slot is None unless facet parents are requested, so do not assume it is an array",
            "partitioner": "Optional custom partitioner for distributing refined mesh",
        },
        "mesh_operations": {
            "create_submesh": "dolfinx.mesh.create_submesh(mesh, dim, entities) — extract subdomain mesh",
            "meshtags": "dolfinx.mesh.meshtags(mesh, dim, entities, values) — tag entities with integer markers",
            "locate_entities": "dolfinx.mesh.locate_entities(mesh, dim, marker_fn) — find entities satisfying geometric condition",
            "locate_entities_boundary": "dolfinx.mesh.locate_entities_boundary(mesh, dim, marker_fn) — boundary entities only",
            "exterior_facet_indices": "dolfinx.mesh.exterior_facet_indices(mesh.topology) — all exterior facets",
        },
        "pitfalls": [
            "MUST pass MPI.COMM_WORLD (or appropriate communicator) to all mesh creation functions",
            "Gmsh model_to_mesh: module renamed from gmshio to gmsh in dolfinx 0.10",
            "For parallel: gmsh model built on rank 0 only (if gmsh.isInitialized())",
            "Topology connectivity must be created before use: mesh.topology.create_connectivity(dim1, dim2). Measured scope on 0.10 (2026-08-03): locate_entities_boundary, locate_dofs_topological and ds/dS assembly all build it LAZILY and work without the call; dolfinx.mesh.exterior_facet_indices(mesh.topology) does NOT and raises RuntimeError 'Facet to cell connectivity has not been computed.'",
            "Both refine entry points need mesh.topology.create_entities(1) first — see mesh_refinement.PREREQUISITE",
            "Branching meshes (T-joints, 3+ cells per facet) supported since 0.10",
            "create_unit_square default is triangles — use CellType.quadrilateral explicitly for quads",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # SOLVER CATALOG
    # ═══════════════════════════════════════════════════════════════════════════
    "solver_catalog": {
        "description": "Complete PETSc/SLEPc solver and preconditioner catalog for DOLFINx.",
        "linear_solvers": {
            "high_level_api": {
                "LinearProblem": {
                    "api": (
                        "dolfinx.fem.petsc.LinearProblem(a, L, "
                        "petsc_options_prefix='myprob_', "
                        "bcs=bcs, petsc_options={...})"
                    ),
                    "usage": (
                        "Simplest interface: problem.solve() "
                        "returns Function. ALL non-form args are "
                        "keyword-only in dolfinx 0.10; "
                        "petsc_options_prefix is REQUIRED — "
                        "omitting it raises TypeError "
                        "'missing 1 required keyword-only "
                        "argument: petsc_options_prefix'."
                    ),
                    "returns": "problem.solve() returns a dolfinx.fem.Function (NOT a tuple) on 0.10; the KSP is reachable as problem.solver for getConvergedReason() / getIterationNumber(). Same for NonlinearProblem.solve(). (Verified 2026-08-03.)",
                    "0_10_note": "Now supports blocked problems via kind='mpi' or kind='nest'",
                },
                "DIAGNOSTIC_TRAP": (
                    "[API] When a LinearProblem / NonlinearProblem "
                    "CONSTRUCTOR raises, dolfinx 0.10 immediately "
                    "emits a SECOND, misleading traceback from "
                    "__del__ on the half-built object: "
                    "\"Exception ignored in: <function "
                    "LinearProblem.__del__ ...> AttributeError: "
                    "'LinearProblem' object has no attribute "
                    "'_solver'\" (and '_snes' for "
                    "NonlinearProblem). The REAL error is the "
                    "first one — e.g. TypeError missing "
                    "petsc_options_prefix. Do not chase the "
                    "_solver / _snes AttributeError; it is "
                    "garbage-collection noise and appears on "
                    "stderr even when the real exception was "
                    "caught and handled. (Verified empirically "
                    "2026-08-03, dolfinx 0.10.0.)"
                ),
            },
            "direct_solvers": {
                "mumps": {"options": {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}, "use": "General sparse, parallel, recommended default direct solver"},
                "superlu_dist": {"options": {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "superlu_dist"}, "use": "Alternative parallel direct solver"},
                "umfpack": {"options": {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "umfpack"}, "use": "Sequential only, good for small problems"},
            },
            "iterative_solvers": {
                "CG": {"options": {"ksp_type": "cg"}, "use": "Symmetric positive definite (Poisson, elasticity, heat)", "requires": "SPD matrix and SPD preconditioner"},
                "GMRES": {"options": {"ksp_type": "gmres"}, "use": "Non-symmetric systems (advection, Navier-Stokes)", "notes": "Restarted, set ksp_gmres_restart for large problems"},
                "BiCGStab": {"options": {"ksp_type": "bcgs"}, "use": "Non-symmetric alternative to GMRES"},
                "MinRes": {"options": {"ksp_type": "minres"}, "use": "Symmetric indefinite (saddle-point: Stokes, mixed Poisson)"},
                "Richardson": {"options": {"ksp_type": "richardson"}, "use": "Simple iteration, often as smoother"},
            },
            "preconditioners": {
                "ILU": {"options": {"pc_type": "ilu"}, "use": "General-purpose incomplete LU (sequential)"},
                "ICC": {"options": {"pc_type": "icc"}, "use": "Incomplete Cholesky for SPD systems (sequential)"},
                "Jacobi": {"options": {"pc_type": "jacobi"}, "use": "Diagonal scaling, cheap, for DG mass matrices"},
                "SOR": {"options": {"pc_type": "sor"}, "use": "Successive over-relaxation"},
                "GAMG": {"options": {"pc_type": "gamg"}, "use": "PETSc native smoothed aggregation AMG — good for Poisson, elasticity", "notes": "Provide near-nullspace (rigid body modes) for elasticity"},
                "hypre_boomeramg": {
                    "options": {"pc_type": "hypre", "pc_hypre_type": "boomeramg"},
                    "use": "Classical AMG via hypre — excellent for Poisson, good for elasticity",
                    "tuning": {"pc_hypre_boomeramg_strong_threshold": "0.25 (2D) or 0.5-0.7 (3D)", "pc_hypre_boomeramg_agg_nl": "2-4 (aggressive coarsening levels)"},
                },
                "BDDC": {"options": {"pc_type": "bddc"}, "use": "Balancing domain decomposition by constraints — scalable parallel",
                         "caveat": "NOT usable as a bare option dict: PCBDDC requires a MATIS-format operator, and a plain dolfinx-assembled AIJ matrix makes KSPSetUp abort with PETSc error code 62 (verified 2026-08-03)."},
                "fieldsplit": {"options": {"pc_type": "fieldsplit"}, "use": "Block preconditioner for saddle-point (Stokes, mixed)",
                               "caveat": "NOT usable as a bare option dict: the splits must be defined (IS fields via pc.setFieldSplitIS / a blocked LinearProblem with kind='nest'), otherwise KSPSetUp aborts with PETSc error code 77 (verified 2026-08-03)."},
            },
        },
        "nonlinear_solvers": {
            "SNES_via_NonlinearProblem": {
                "api_0_9": "problem = NonlinearProblem(F, u, bcs); solver = NewtonSolver(MPI.COMM_WORLD, problem)",
                "api_0_10": (
                    "problem = dolfinx.fem.petsc.NonlinearProblem("
                    "F, u, bcs=bcs, "
                    "petsc_options_prefix='myprob_', "
                    "petsc_options={...}); problem.solve()"
                ),
                "0_10_signature_pitfalls": (
                    "ALL kwargs are keyword-only (after the * in "
                    "the signature). NonlinearProblem(F, u, bcs) "
                    "as positional fails with TypeError 'takes 3 "
                    "positional arguments but 4 were given'. "
                    "Omitting petsc_options_prefix fails with "
                    "TypeError 'missing 1 required keyword-only "
                    "argument: petsc_options_prefix'. (Empirically "
                    "verified 2026-06-01 — Tier-2 fixture "
                    "nonlinear_problem_signature_kwargs.)"
                ),
                "note": "dolfinx.nls.petsc.NewtonSolver deprecated in 0.10 in favor of NonlinearProblem wrapping SNES directly",
            },
            "snes_types": {
                "newtonls": {"options": {"snes_type": "newtonls"}, "description": "Newton with line search (default, most common)"},
                "newtontr": {"options": {"snes_type": "newtontr"}, "description": "Newton with trust region (more robust for difficult problems)"},
                "nrichardson": {"options": {"snes_type": "nrichardson"}, "description": "Nonlinear Richardson (fixed-point)"},
                "ngmres": {"options": {"snes_type": "ngmres"}, "description": "Nonlinear GMRES (Anderson acceleration)"},
            },
            "convergence": {
                "snes_atol": "Absolute tolerance on residual norm (default 1e-50, set to 1e-8 or 1e-10)",
                "snes_rtol": "Relative tolerance (default 1e-8)",
                "snes_stol": "Step tolerance for ||delta_x||/||x|| (default 1e-8)",
                "snes_max_it": "Maximum nonlinear iterations (default 50)",
                "snes_monitor": "Print convergence info (set to None/empty string)",
            },
            "custom_newton": {
                "description": "Hand-written Newton loop for full control (jsdokken tutorial chapter 4)",
                "approach": "Assemble F and J manually, solve J*du=-F, update u, check convergence",
                "api": "dolfinx.fem.petsc.assemble_matrix(a), dolfinx.fem.petsc.assemble_vector(L), apply_lifting, set_bc",
                "convergence_criterion": "'residual' (default) or 'incremental'",
            },
        },
        "eigenvalue_solvers": {
            "SLEPc_EPS": {
                "api": "from slepc4py import SLEPc; eps = SLEPc.EPS().create(MPI.COMM_WORLD)",
                "use": "Generalized eigenvalue problem A*x = lambda*B*x",
                "methods": "krylovschur (default, recommended), arnoldi, lanczos, power, jd (Jacobi-Davidson)",
                "spectral_transform": "ST for shift-and-invert to find eigenvalues near a target",
                "demo": "Electromagnetic modal analysis (waveguide demo)",
            },
        },
        "block_solvers": {
            "description": "For saddle-point problems (Stokes, mixed Poisson)",
            # 2026-08-03: assemble_matrix_block / assemble_matrix_nest are GONE
            # in dolfinx 0.10 — dir(dolfinx.fem.petsc) contains only
            # assemble_matrix / assemble_vector / assemble_jacobian /
            # assemble_residual. The blocked path is now the `kind` kwarg.
            "api_0_10": "dolfinx.fem.petsc.assemble_matrix(a_block, bcs=bcs, kind='mpi'|'nest') — the separate assemble_matrix_block / assemble_matrix_nest functions were REMOVED in 0.10 and raise AttributeError (verified 2026-08-03)",
            "high_level": "dolfinx.fem.petsc.LinearProblem(a_block, L_block, kind='nest'|'mpi', petsc_options_prefix=...) handles blocked problems directly in 0.10",
            "nullspace": "Build nullspace for pressure (constant) or rigid body modes (elasticity), attach to matrix with A.setNullSpace(PETSc.NullSpace().create(constant=True)) — note dolfinx.la has NO create_petsc_nullspace_constants helper (verified 2026-08-03)",
        },
        "alternative_backends": {
            "pyamg": {
                "api": "Convert DOLFINx matrix to scipy sparse, use pyamg.ruge_stuben_solver() or pyamg.smoothed_aggregation_solver()",
                "note": "Serial only (not MPI-parallel), good for rapid prototyping",
                "demo": "demo_pyamg.py",
            },
            "scipy": {
                "api": "mat.to_scipy() to convert DOLFINx matrix, then use scipy.sparse.linalg",
                "note": "Useful for interfacing with optimization (scipy.optimize)",
            },
        },
        "pitfalls": [
            "Always set petsc_options as dict: {'ksp_type': 'cg', 'pc_type': 'gamg'}",
            "For elasticity AMG: MUST provide near-nullspace (6 rigid body modes in 3D) via setNearNullSpace()",
            "For Stokes: pressure nullspace (constant) must be set via setNullSpace()",
            "GAMG/hypre strong_threshold: 0.25 for 2D, 0.5-0.7 for 3D (wrong value = poor convergence)",
            "Direct solvers fail silently for very large problems — check ksp_monitor for divergence",
            "NewtonSolver deprecated in 0.10 — use NonlinearProblem.solve() instead. It is not merely advisory: constructing dolfinx.nls.petsc.NewtonSolver(comm, problem) around a 0.10 NonlinearProblem emits a DeprecationWarning AND then fails with AttributeError: 'NonlinearProblem' object has no attribute 'a' (verified 2026-08-03). Any pitfall text below that names NewtonSolver.solve as its signal is describing a 0.9-era code path.",
            "snes_atol default is 1e-50 (effectively disabled) — you MUST set it explicitly. Measured defaults on petsc4py 3.24.4: (rtol, atol, stol, max_it) = (1e-8, 1e-50, 1e-8, 50) (verified 2026-08-03)",
            "[API] pc_type 'bddc' / 'fieldsplit' / hypre 'ams' cannot be used as bare option dicts — each needs extra setup (MATIS operator, field splits, discrete-gradient operator respectively). Signal: LinearProblem(..., petsc_options={'pc_type': 'bddc'}) aborts inside KSPSetUp with PETSc error code 62; 'fieldsplit' with error 77; hypre 'ams' with error 83 (verified 2026-08-03)",
        ],
        "by_physics": {
            "poisson": "CG + hypre/GAMG (or LU for small)",
            "elasticity": "CG + GAMG with near-nullspace (or LU for small)",
            "heat_transient": "CG + hypre per time step",
            "stokes": "MinRes + fieldsplit (AMG for velocity block, mass matrix for Schur complement)",
            "navier_stokes": "SNES newtonls + GMRES + AMG (or LU for small)",
            "helmholtz": "GMRES + LU (complex-valued, direct often needed)",
            "maxwell": "GMRES + AMS (from hypre) for H(curl) problems — AMS needs the discrete gradient + vertex coordinates attached to the PC; {'pc_type':'hypre','pc_hypre_type':'ams'} alone aborts with PETSc error 83 (verified 2026-08-03)",
            "cahn_hilliard": "SNES + LU per time step",
        },
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # BOUNDARY CONDITIONS
    # ═══════════════════════════════════════════════════════════════════════════
    "boundary_conditions": {
        "description": "Complete boundary condition types and API in DOLFINx.",
        "dirichlet": {
            "api": "dolfinx.fem.dirichletbc(value, dofs, V=None)",
            "locate_topological": "dolfinx.fem.locate_dofs_topological(V, entity_dim, entities)",
            "locate_geometrical": "dolfinx.fem.locate_dofs_geometrical(V, marker_fn)",
            "component_wise": "V0, _ = V.sub(0).collapse(); dofs = locate_dofs_topological((V.sub(0), V0), fdim, facets)",
            "enforcement": "Strong enforcement via lifting (modify RHS, zero rows/cols in matrix)",
            "notes": "DOLFINx uses the lifting approach internally, not identity rows",
        },
        "neumann": {
            "api": "L += g * v * ds(marker)",
            "description": "Natural BC: specified flux, added as surface integral in weak form",
            "notes": "Zero Neumann (insulated/free) = do nothing (natural condition). Non-zero: integrate over ds with marker.",
        },
        "robin": {
            "api": "a += r * u * v * ds(marker); L += r * s * v * ds(marker)",
            "description": "Mixed BC: -k*du/dn = r*(u - s) where r=transfer coefficient, s=ambient value",
            "use_cases": "Convective heat transfer, radiation, absorbing boundary",
        },
        "periodic": {
            "library": "dolfinx_mpc (extension by Jørgen S. Dokken)",
            "api": "mpc = dolfinx_mpc.MultiPointConstraint(V); mpc.create_periodic_constraint_geometrical(V, indicator, relation, bcs, scale)",
            "notes": "NOT built into DOLFINx core — requires separate dolfinx_mpc package",
            "topological": "mpc.create_periodic_constraint_topological(V, meshtag, tag, relation, bcs, scale)",
        },
        "point_constraints": {
            "approach": "Use locate_dofs_geometrical with a function checking point proximity",
            "lagrange_multiplier": "Possible via real-valued function space (workaround for integral constraints)",
        },
        "outlet_do_nothing": {
            "description": "Natural (do-nothing) BC at outlet: zero stress condition",
            "api": "Simply do not specify any BC on the outlet boundary — it is naturally satisfied",
        },
        "pitfalls": [
            # 2026-08-03: this used to say "MUST create connectivity before
            # locating boundary". On dolfinx 0.10 that is no longer true for
            # the locate_* path (lazy build); it IS still true for
            # exterior_facet_indices. Narrowed to what actually reproduces.
            "Connectivity: mesh.topology.create_connectivity(fdim, tdim) is NO LONGER required before locate_entities_boundary / locate_dofs_topological / ds / dS on dolfinx 0.10 — connectivity is built lazily and all four work without it. It IS still required before dolfinx.mesh.exterior_facet_indices(mesh.topology), which raises RuntimeError 'Facet to cell connectivity has not been computed.' Calling it explicitly remains harmless and is the safer tutorial pattern. (Verified empirically 2026-08-03.)",
            "For sub-space BCs: locate_dofs_topological needs BOTH the sub-space AND collapsed sub-space as tuple",
            "Periodic BCs require dolfinx_mpc extension — not natively in DOLFINx (confirmed: no periodic-constraint API anywhere in dolfinx 0.10)",
            "Dirichlet value type must match: np.array([0.0]*gdim, dtype=default_scalar_type) for a vector space, scalar for a scalar space. A scalar on a vector space raises RuntimeError 'Rank mismatch between Constant and function space in DirichletBC' (verified 2026-08-03 — it is a dolfinx rank check, NOT a numpy broadcast error)",
            "For enclosed flows (all Dirichlet velocity): pin pressure at one DOF to remove nullspace",
            "[API] A strong DirichletBC on a DG FunctionSpace is a silent no-op — impose inflow/boundary data weakly through a ds integral instead. Signal: fem.locate_dofs_topological on a DG1 space with the x=0 boundary facets of an 8x8 unit square returns an EMPTY array (0 dofs), so the BC constrains nothing and no error is raised. (Verified empirically 2026-08-03.)",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # I/O AND OUTPUT
    # ═══════════════════════════════════════════════════════════════════════════
    "io_catalog": {
        "description": "Complete I/O capabilities in DOLFINx for visualization, checkpointing, and data exchange.",
        "vtx_writer": {
            "api": "dolfinx.io.VTXWriter(MPI.COMM_WORLD, 'output.bp', [u], engine='BP4')",
            "write": "writer.write(t)",
            "close": "writer.close()",
            "features": "Arbitrary-order Lagrange, time series, parallel",
            "viewer": "ParaView (open .bp directory)",
            "notes": "Requires ADIOS2. Best for Lagrange elements. VTXMeshPolicy controls mesh update frequency.",
        },
        "xdmf_file": {
            "api": "dolfinx.io.XDMFFile(MPI.COMM_WORLD, 'output.xdmf', 'w')",
            "write_mesh": "f.write_mesh(mesh)",
            "write_function": "f.write_function(u, t)",
            "read_mesh": "f.read_mesh()",
            "features": "XML+HDF5, parallel, read/write meshes and functions",
            "notes": "Geometry order <= 2 supported. Good for meshes. For functions, VTX preferred.",
        },
        "vtkhdf": {
            "api": "dolfinx.io.vtkhdf.read_mesh('file.vtkhdf', comm) — new in 0.10",
            # 2026-08-03: writing has landed; the module exports write_mesh,
            # write_point_data, write_cell_data, write_vtkhdf_mesh,
            # write_vtkhdf_data.
            "notes": "Kitware's future format. Reading AND writing are both available on 0.10: read_mesh, write_mesh, write_point_data, write_cell_data (verified 2026-08-03 — the older 'writing in progress' note is stale).",
        },
        "checkpointing": {
            "library": "adios4dolfinx (extension by Jørgen S. Dokken)",
            "api": "adios4dolfinx.write_mesh(mesh, filename); adios4dolfinx.write_function(u, filename)",
            "read": "adios4dolfinx.read_mesh(filename, comm); adios4dolfinx.read_function(V, filename)",
            "features": "N-to-M checkpointing (write on N ranks, read on M ranks), function + mesh + meshtags",
            "notes": "Requires ADIOS2. Essential for restart/continuation simulations.",
        },
        "function_evaluation": {
            "at_points": "u.eval(points, cells) — evaluate function at arbitrary points (must find containing cells first)",
            "find_cells": "dolfinx.geometry.bb_tree + compute_collisions + compute_colliding_cells",
            "interpolation": "u.interpolate(expr) — interpolate expression or function into FE space",
            "nonmatching": "dolfinx.fem.Function.interpolate_nonmatching() — interpolate between different meshes",
        },
        "visualization": {
            "pyvista": {
                "api": "grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(V))",
                "scalar_warp": "grid.warp_by_scalar()",
                "vector_glyphs": "grid.glyph(orient='vectors', factor=0.1)",
                "streamlines": "grid.streamlines(vectors='vectors')",
            },
        },
        "pitfalls": [
            "VTXWriter requires ADIOS2 — check dolfinx.io.has_adios2",
            "XDMFFile: only geometry order <= 2; for high-order elements, use VTX. Concretely, XDMFFile.write_function on a P2 Function over a P1 mesh raises RuntimeError 'Degree of output Function must be same as mesh degree. Maybe the Function needs to be interpolated?' while VTXWriter writes the same P2 Function fine (verified 2026-08-03)",
            "VTXWriter only works with (discontinuous) Lagrange elements — not RT, Nedelec, etc. Exact message: RuntimeError 'Only (discontinuous) Lagrange functions are supported. Interpolate Functions before output.' (verified 2026-08-03)",
            "Function eval requires finding containing cell first — use dolfinx.geometry.bb_tree + compute_collisions_points + compute_colliding_cells",
            "Checkpointing (restart) requires adios4dolfinx extension — not built into DOLFINx",
            "Close writers explicitly (writer.close()) to flush data to disk",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # UFL FORM LANGUAGE
    # ═══════════════════════════════════════════════════════════════════════════
    "ufl_reference": {
        "description": "Unified Form Language (UFL) reference for expressing variational forms in FEniCSx.",
        "differential_operators": {
            "grad(f)": "Gradient: scalar -> vector, vector -> tensor",
            "div(v)": "Divergence: vector -> scalar, tensor -> vector",
            "curl(v)": "Curl: vector -> vector (3D) or scalar (2D)",
            "nabla_grad(f)": "Same as grad but with different index convention for tensors",
            "nabla_div(v)": "Same as div but with different index convention",
            "Dx(f, i)": "Partial derivative df/dx_i",
        },
        "algebraic_operators": {
            "inner(a, b)": "Full contraction (all indices). For vectors: dot product. Complex: conjugates 2nd arg.",
            "dot(a, b)": "Contracts last index of a with first of b",
            "outer(a, b)": "Outer product (tensor product)",
            "cross(a, b)": "Cross product (3D vectors)",
            "det(A)": "Determinant of matrix",
            "tr(A)": "Trace of matrix",
            "sym(A)": "Symmetric part: 0.5*(A + A^T)",
            "skew(A)": "Skew part: 0.5*(A - A^T)",
            "dev(A)": "Deviatoric part: A - tr(A)/dim * I",
            "inv(A)": "Matrix inverse (use cofac for better numerical stability)",
            "cofac(A)": "Cofactor matrix: det(A) * inv(A)^T",
            "transpose(A)": "Matrix transpose",
        },
        "measures": {
            "dx": "Volume (cell) integration",
            "ds": "Exterior facet (boundary) integration",
            "dS": "Interior facet integration (DG methods)",
            "dx(marker)": "Integration over subdomain with given marker",
            "ds(marker)": "Integration over boundary with given marker",
        },
        "special_functions": {
            "ufl.variable(expr)": "Declare expression as differentiable variable",
            "ufl.diff(f, var)": "Differentiate f with respect to variable var",
            "ufl.derivative(F, u, v)": "Gateaux derivative of form F w.r.t. u in direction v (for Newton Jacobian)",
            "ufl.adjoint(a)": "Adjoint of bilinear form (swap trial/test)",
            "ufl.action(a, f)": "Replace trial function with coefficient f",
            "ufl.replace(form, {old: new})": "Substitute expressions in form",
            "ufl.lhs(F)": "Extract bilinear (left) part from equation",
            "ufl.rhs(F)": "Extract linear (right) part from equation",
            "ufl.system(F)": "Split into (lhs, rhs) pair",
        },
        "dg_operators": {
            "jump(v)": "Jump across interior facet: v('+') - v('-')",
            "jump(v, n)": "Jump with normal: v('+')*n('+') + v('-')*n('-')",
            "avg(v)": "Average across interior facet: 0.5*(v('+') + v('-'))",
            "v('+'), v('-')": "Restriction to positive/negative side of interior facet",
        },
        "form_compilation": {
            "form_compiler_options": "Passed to FFCx: run 'ffcx --help' for all options",
            "jit_options": "Passed to CFFI JIT compilation of generated C code",
            "quadrature_degree": "Set via metadata: dx(metadata={'quadrature_degree': q})",
            "example": "dolfinx.fem.form(a, form_compiler_options={'optimize': True}, jit_options={'timeout': 120})",
        },
        "automatic_differentiation": {
            "description": "UFL supports symbolic differentiation for deriving Jacobians, sensitivities, adjoint operators",
            "jacobian_example": "F = inner(sigma(u), grad(v)) * dx; J = ufl.derivative(F, u, du) — auto-derive Newton Jacobian",
            "material_tangent": "c = ufl.variable(c); psi = f(c); dpsi_dc = ufl.diff(psi, c) — material law differentiation",
            "adjoint_optimization": "Use ufl.adjoint() and ufl.action() for PDE-constrained optimization",
        },
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # PHYSICS: POISSON
    # ═══════════════════════════════════════════════════════════════════════════
    "poisson": {
        "description": "Poisson equation -div(kappa * grad(u)) = f. Foundation of all elliptic PDEs. Covers steady-state diffusion, electrostatics, potential flow.",
        "weak_form": "kappa * inner(grad(u), grad(v)) * dx = inner(f, v) * dx + inner(g, v) * ds",
        "function_space": "Lagrange order 1 or 2 (higher order for smooth solutions)",
        "demo_url": "https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_poisson.html",
        "code_skeleton": {
            "imports": "from mpi4py import MPI; from dolfinx import fem, mesh, io; from dolfinx.fem.petsc import LinearProblem; import ufl; import numpy as np",
            "mesh": "domain = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)",
            "space": "V = fem.functionspace(domain, ('Lagrange', 1))",
            "bc": "fdim = domain.topology.dim - 1; boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True)); bc = fem.dirichletbc(0.0, fem.locate_dofs_topological(V, fdim, boundary_facets), V)",
            "forms": "u, v = ufl.TrialFunction(V), ufl.TestFunction(V); a = inner(grad(u), grad(v)) * ufl.dx; L = f * v * ufl.dx",
            "solve": "problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix='solve', petsc_options={'ksp_type': 'cg', 'pc_type': 'hypre'}); uh = problem.solve()",
        },
        "solver": {"direct": "ksp_type: preonly, pc_type: lu, pc_factor_mat_solver_type: mumps", "iterative": "ksp_type: cg, pc_type: hypre (BoomerAMG)"},
        "mixed_formulation": {
            "description": "Mixed Poisson: introduce flux sigma = -grad(u), solve for (sigma, u) simultaneously",
            "elements": "Raviart-Thomas for sigma + DG(k-1) for u",
            "demo_url": "https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_mixed-poisson.html",
            "block_preconditioner": "Block-diagonal Riesz-map preconditioner for the saddle-point system",
        },
        "matrix_free": {
            "description": "Matrix-free CG solver using action of bilinear form (no explicit matrix assembly)",
            "demo_url": "https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_poisson_matrix_free.html",
            "notes": "Computes matrix-vector product on-the-fly. Diagonal assembly available for Jacobi preconditioning.",
        },
        "pitfalls": [
            "[API] In recent dolfinx, mesh.topology.create_connectivity(fdim, tdim) is no longer a hard prerequisite for locate_entities_boundary / locate_dofs_topological — connectivity is built lazily on first need. Calling it explicitly is harmless and is the safer tutorial pattern, but its ABSENCE no longer triggers an exception in current dolfinx. Signal: in older dolfinx (pre-0.7), locate_dofs_topological raised RuntimeError mentioning 'connectivity has not been computed'; current dolfinx returns dof indices without that step. EXCEPTION worth knowing (re-verified 2026-08-03 on 0.10.0): dolfinx.mesh.exterior_facet_indices(mesh.topology) is still eager and raises RuntimeError 'Facet to cell connectivity has not been computed.' on a fresh mesh — so the common 'grab all boundary facets' idiom DOES need create_connectivity(fdim, tdim) even though locate_* does not. ds and dS assembly do not. (Verified empirically 2026-06-01; scope re-measured 2026-08-03.)",
            "[API] dolfinx.default_scalar_type for Constants and Function arrays so dtype matches the PETSc build (float64 if PETSc is real, complex128 if PETSc is complex). Signal: [re-measured 2026-08-03 on a REAL conda-forge build] assembling a ufl form that carries an imaginary coefficient raises ValueError 'Unexpected complex value in real expression.' from fem.form, and writing a complex value into a real Function array raises TypeError \"float() argument must be a string or a real number, not 'complex'\". Note fem.Constant(mesh, 1+2j) itself does NOT raise — the failure surfaces at form compilation / array assignment, not at Constant construction.",
            "[API] VTXWriter (ADIOS2 backend) supports only Lagrange / DG element families. Mixed / Nedelec / BDM / RT Functions cannot be written. Signal: [exact text, re-measured 2026-08-03 on 0.10.0] VTXWriter construction raises RuntimeError 'Only (discontinuous) Lagrange functions are supported. Interpolate Functions before output.' — the older quoted strings 'Cannot interpolate function to the VTX output basis' / 'ADIOS2 VTX only supports Lagrange elements' do NOT appear in current dolfinx, and the error fires at VTXWriter(...) construction, not at .write().",
            "[Physics] Pure-Neumann Poisson admits the constant null space — the solution is determined only up to a constant. Either pin one DOF (DirichletBC on a single point) or add a Lagrange multiplier enforcing mean(u) = 0. Signal: LinearProblem.solve returns successfully (CG with pc_type='none' even converges without raising), but the resulting Function array has a HUGE additive offset accommodating the null space — np.array shows max ≈ min ≈ O(1e6) with tiny std (e.g. max=2.18e+06, std=112 on an 8x8 unit square with f=1). The 'KSP fails' alternative does NOT typically fire; you observe the bug as the un-pinned constant. (Verified empirically 2026-06-01.)",
            "[Syntax] For non-unit kappa coefficients: define as fem.Constant for spatially uniform, or fem.Function (interpolated) for spatially varying. Plain Python floats inside ufl forms work for unit coefficients but lose unit metadata. Signal: ufl form runs but the assembled stiffness scale disagrees with the analytic kappa-scaled stiffness by exactly the kappa value (when float coefficient was forgotten).",
        ],
        "materials": {"kappa": {"range": [0.001, 1e6], "unit": "W/(m*K) or dimensionless"}},
        "reference_solutions": {
            "unit_square_f1": "max(u) ~ 0.0737 for -laplacian(u)=1 on [0,1]^2, u=0 on boundary (re-measured 2026-08-03 on dolfinx 0.10.0: 0.073657 with P1 on a 64x64 mesh)",
            "mms_convergence": "Manufactured u = sin(pi x) sin(pi y), f = 2 pi^2 sin(pi x) sin(pi y): observed L2 rates on N = 8,16,32,64 are P1 -> 1.97/1.99/2.00, P2 -> 3.00/3.00/3.00, P3 -> 4.06/4.03/4.01, i.e. the textbook O(h^(k+1)) (verified 2026-08-03)",
        },
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # PHYSICS: LINEAR ELASTICITY
    # ═══════════════════════════════════════════════════════════════════════════
    "linear_elasticity": {
        "description": "Linear elasticity with Lame parameters. Small strain assumption. Plane strain, plane stress, or full 3D.",
        "weak_form": "inner(sigma(u), epsilon(v)) * dx = dot(f, v) * dx + dot(t, v) * ds",
        "function_space": "Vector Lagrange: element('Lagrange', cell, 1, shape=(gdim,))",
        "demo_url": "https://jsdokken.com/dolfinx-tutorial/chapter2/linearelasticity.html",
        "constitutive": {
            "sigma(u)": "lambda_ * nabla_div(u) * Identity(d) + 2*mu * epsilon(u)",
            "epsilon(u)": "ufl.sym(ufl.grad(u)) = 0.5*(grad(u) + grad(u)^T)",
            "mu": "E / (2*(1+nu))",
            "lambda_": "E*nu / ((1+nu)*(1-2*nu))",
            "plane_stress_lambda": "2*mu*lambda_ / (2*mu + lambda_)",
        },
        "code_skeleton": {
            "space": "V = fem.functionspace(domain, ('Lagrange', 1, (gdim,)))",
            "sigma": "def sigma(u): return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2*mu*ufl.sym(ufl.grad(u))",
            "forms": "a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx; L = ufl.dot(f, v) * ufl.dx",
        },
        "solver": {
            "recommended": "CG + GAMG with near-nullspace (rigid body modes)",
            "alternative": "LU (MUMPS) for small problems",
            "near_nullspace": "6 modes in 3D: 3 translations + 3 rotations. Set via matrix.setNearNullSpace()",
            "demo_url": "https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_elasticity.html",
        },
        "static_condensation": {
            "description": "Mixed stress-displacement formulation with condensation of internal stress DOFs",
            "demo_url": "https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_static-condensation.html",
            "notes": "Uses numba for efficient condensation of block forms. Cook's membrane benchmark.",
        },
        "pitfalls": [
            "[Syntax] Vector function space for elasticity in "
            "dolfinx is created with ('Lagrange', 1, (gdim,)) — "
            "the trailing shape tuple marks it vector-valued. "
            "Passing ('Lagrange', 1) gives a SCALAR space; the "
            "weak form fails at construction when ufl.sym(ufl.grad) "
            "is invoked on the scalar trial. Signal: ufl.sym "
            "raises ValueError 'Symmetric part of tensor with "
            "rank != 2 is undefined.' inside the form definition "
            "(before assemble). (Verified empirically 2026-06-01 "
            "— prior wording 'Invalid ranks' / 'expected rank 1 "
            "trial' does not appear in current dolfinx.)",
            "[Syntax] Dirichlet BC value for a vector elasticity "
            "space must be np.array([0.0]*gdim, dtype="
            "default_scalar_type) — not scalar 0. Signal: ["
            "[re-measured 2026-08-03 on dolfinx 0.10.0] "
            "fem.dirichletbc(0.0, dofs, V) on a vector space "
            "raises RuntimeError 'Rank mismatch between Constant "
            "and function space in DirichletBC'. This is a "
            "dolfinx rank check — the older quoted numpy "
            "ValueError 'could not broadcast input array from "
            "shape () into shape (gdim,)' does NOT appear.",
            "[Physics] Plane strain vs plane stress: lambda must "
            "be adjusted. Plane stress uses lambda_star = "
            "2*lambda*mu/(lambda+2*mu). Signal: [MEASURED "
            "2026-08-03, cantilever 1.0 x 0.2, P2 vector "
            "Lagrange, 40x8 triangles, end traction] the "
            "plane-STRAIN tip deflection is a factor (1 - nu^2) "
            "of the plane-STRESS one — 0.90941 measured vs "
            "0.9100 predicted at nu=0.3, and 0.79100 vs 0.7975 "
            "at nu=0.45. So plane strain is only ~9% stiffer at "
            "nu=0.3, NOT ~30%, and the factor is (1-nu^2), NOT "
            "(1-nu). (Corrects the previous wording, which "
            "quoted both '~30%' and '(1-nu)' — neither "
            "reproduces.)",
            "[Numerical] Near-incompressible (nu > 0.49): a mixed "
            "(u, p) formulation is the robust choice, but the "
            "severity of locking depends strongly on ELEMENT "
            "ORDER and mesh, not just on nu. Signal: [MEASURED 2026-08-03] "
            "(same cantilever, tip deflection vs a P2/P1 "
            "Taylor-Hood reference): "
            "P1 triangles lock by 7.2x / 16.5x / 19.5x at "
            "nu = 0.49 / 0.499 / 0.4999 on a coarse 10x2 mesh, "
            "improving to 1.2x / 2.4x / 9.2x on 80x16 — i.e. "
            "locking is real but mesh-dependent and is a factor "
            "of ~2-20, NOT 'orders of magnitude' and NOT the "
            "'~1e-3 of analytic' quoted previously. "
            "P2 triangles do NOT meaningfully lock at all here: "
            "TaylorHood/P2 = 1.00-1.06 across every nu and mesh "
            "tested. Recommend mixed (or P2+) for nu > 0.49; do "
            "not expect the P1 catastrophe from P2.",
            "[Numerical] For GAMG/AMG: provide the near-nullspace "
            "(rigid body modes — 3 translations + 3 rotations in "
            "3D) via A.setNearNullSpace(PETSc.NullSpace()."
            "create(vectors=rbm)). Signal: [MEASURED 2026-08-03, "
            "dolfinx 0.10.0; P1 tetrahedral cantilever 2x1x1, "
            "CG + GAMG, rtol 1e-8] WITHOUT the near-nullspace "
            "the solve still CONVERGES (reason 2) in 31 / 38 / "
            "41 iterations at 1911 / 7623 / 19575 dofs; WITH the "
            "6 rigid-body modes it takes 16 / 16 / 23. "
            "IMPORTANT CORRECTION: at these sizes the "
            "near-nullspace is a 1.8x-2.4x iteration-count win, "
            "NOT the 10x-50x previously claimed, and its absence "
            "does NOT produce 'KSP did not converge' / iteration "
            "count = max_it. The claim of outright failure was "
            "not reproduced up to ~20k dofs; treat 'MUST' as "
            "'strongly recommended, and increasingly so with "
            "problem size' and measure your own iteration counts "
            "before quoting a speedup.",
            "[Physics] There is no dolfinx 'default': whichever "
            "lambda you put in sigma() decides it, and the "
            "standard 3D Lame lambda = E*nu/((1+nu)(1-2nu)) "
            "written in a 2D form gives PLANE STRAIN. Forgetting "
            "this is a silent source of wrong answers for thin "
            "structures. Signal: a 2D VectorH1 dolfinx Function "
            "plate deflection differs from the plane-stress "
            "reference by factor (1-nu^2) — measured 0.90941 at "
            "nu=0.3 against a predicted 0.9100 (verified "
            "empirically 2026-08-03).",
            "[API] dolfinx.fem.functionspace rejects element "
            "family names that basix does not register. Signal: ["
            "[re-measured 2026-08-03 on dolfinx 0.10.0 / basix "
            "0.10.0] the DOLFIN degree-suffixed name 'P1' "
            "raises ValueError 'Unknown element family: P1 with "
            "cell type triangle'. 'CG' however STILL WORKS — it "
            "only emits a DeprecationWarning ('\"CG\" element "
            "name is deprecated. Consider using \"Lagrange\" or "
            "\"P\" instead') and builds the space. The previous "
            "wording, which claimed ('CG', 1) raises ValueError "
            "'Unknown element family CG', does not reproduce.",
            "[API] dolfinx XDMFFile.write_function requires the "
            "Function degree to match the mesh degree. P2 on a P1 "
            "mesh (the common case) is rejected — interpolate to a "
            "matching-degree space, or use VTKFile / VTXWriter. "
            "Signal: XDMFFile.write_function raises RuntimeError "
            "'Degree of output Function must be same as mesh "
            "degree. Maybe the Function needs to be interpolated?'. "
            "(Verified empirically 2026-06-01 — prior wording "
            "'XDMF mesh must be P1' does not appear.)",
        ],
        "materials": {
            "E": {"range": [1.0, 1e12], "unit": "Pa", "examples": {"steel": 210e9, "aluminum": 70e9, "rubber": 1e6}},
            "nu": {"range": [0.0, 0.499], "unit": "dimensionless", "examples": {"steel": 0.3, "rubber": 0.49}},
        },
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # PHYSICS: STOKES FLOW
    # ═══════════════════════════════════════════════════════════════════════════
    "stokes": {
        "description": "Stokes flow (Re -> 0). Linear saddle-point problem. Mixed P2/P1 (Taylor-Hood) or MINI element.",
        "weak_form": "nu*inner(grad(u),grad(v))*dx - p*div(v)*dx - q*div(u)*dx = dot(f,v)*dx",
        "function_space": "Mixed: Taylor-Hood P2/P1 (inf-sup stable). Alternative: MINI (P1+Bubble/P1), CR/DG0.",
        "demo_url": "https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_stokes.html",
        "element_construction": {
            "taylor_hood": "P2v = basix.ufl.element('Lagrange', cell, 2, shape=(gdim,)); P1 = basix.ufl.element('Lagrange', cell, 1); TH = basix.ufl.mixed_element([P2v, P1])",
            "mini": "P1v = basix.ufl.element('Lagrange', cell, 1, shape=(gdim,)); B = basix.ufl.element('Bubble', cell, gdim+1, shape=(gdim,)); V_el = basix.ufl.enriched_element([P1v, B]); P1 = basix.ufl.element('Lagrange', cell, 1); MINI = basix.ufl.mixed_element([V_el, P1])",
        },
        "solver": {
            "direct": "LU (MUMPS) for small problems (linear system, no Newton)",
            "iterative": "MinRes + fieldsplit block preconditioner",
            "block_precon": "AMG for velocity block, pressure mass matrix for Schur complement approximation",
        },
        "pitfalls": [
            "[Numerical] MUST use an inf-sup stable velocity-"
            "pressure pair. Taylor-Hood (P2v + P1) and MINI "
            "(P1v + Bubble enriched + P1) are stable; equal-"
            "order P1/P1 constructs a valid mixed FunctionSpace "
            "but the discrete LBB condition is violated, so the "
            "pressure field develops checkerboard oscillations "
            "in the kernel direction. Signal: with the same "
            "4x4 unit-square triangulation in dolfinx 0.10, "
            "basix.ufl.mixed_element returns FunctionSpaces with "
            "dim 187 (TH), 139 (MINI), 75 (P1/P1); the P1/P1 "
            "system assembles but the pressure null space has "
            "more vectors than just the constant pressure. "
            "(Verified empirically 2026-06-01 — Tier-2 fixture "
            "stokes_basix_element_construction in scripts/"
            "tier2_fixtures/fenics/. Re-verified 2026-08-03 on "
            "dolfinx 0.10.0: on the 4x4 mesh the three dims are "
            "still exactly 187 / 139 / 75, and the instability is "
            "now measured rather than advisory — SVD of the "
            "bc-applied Stokes matrix on an 8x8 mesh gives "
            "numerical null dimension 1 for Taylor-Hood (the "
            "constant pressure alone) versus 8 for P1/P1. "
            "MIND THE MESH: the dims and the SVD were measured on "
            "DIFFERENT meshes. On 8x8 the same three pairs give "
            "659 / 499 / 243, not 187 / 139 / 75. Corrected "
            "2026-08-03 — the earlier wording said 'the same 8x8 "
            "mesh' for both, which does not reproduce.)",
            "[Numerical] Pressure for enclosed (all-Dirichlet on "
            "velocity) flows is determined only up to an "
            "additive constant. Pin one pressure DOF with a "
            "DirichletBC at a chosen vertex, or build the "
            "constant nullspace yourself with "
            "PETSc.NullSpace().create(constant=True) and call "
            "A.setNullSpace(ns) on the PETSc matrix. NOTE "
            "(2026-08-03): there is NO "
            "dolfinx.la.create_petsc_nullspace_constants helper "
            "in dolfinx 0.10 — dolfinx.la exposes only "
            "BlockMode / IndexMap / InsertMode / MatrixCSR / "
            "Norm / Vector / is_orthonormal / matrix_csr / norm "
            "/ orthonormalize / petsc / vector. The previously "
            "quoted call does not exist. "
            "Skipping this leaves PETSc to handle "
            "a singular system — MUMPS will either complain or "
            "return a solution with arbitrary global pressure "
            "shift. Signal: PETSc KSP iteration converges "
            "trivially with zero pressure correction, or MUMPS "
            "emits 'INFOG(1)=-9' (singular matrix) from the "
            "factorisation. (Catalog claim inherited; not "
            "separately Tier-2 falsified this iteration.)",
            "[API] basix.ufl.element supports quadrilateral cells "
            "(CellType.quadrilateral) for Taylor-Hood-style "
            "Q2/Q1: pass cell=msh.basix_cell() from a "
            "create_unit_square(..., cell_type=CellType."
            "quadrilateral) mesh and the same 'Lagrange' family "
            "string + degree=2/1. Triangle-mesh helpers like "
            "the default cell from create_unit_square use "
            "CellType.triangle; the cell type must match. "
            "Signal: msh.basix_cell() returns "
            "'CellType.triangle' or 'CellType.quadrilateral' "
            "consistent with the mesh constructor. (Catalog "
            "claim inherited; not separately Tier-2 falsified "
            "this iteration.)",
            "[Numerical] Block preconditioner is essential for "
            "iterative MinRes / GMRES solves beyond ~100k dofs. "
            "Use fieldsplit with PETSc PCFIELDSPLIT: type="
            "Schur, with A^-1 on the velocity block (AMG via "
            "PCHYPRE / GAMG) and a pressure mass matrix M_p as "
            "the Schur-complement approximation. Without "
            "fieldsplit the saddle-point spectrum forces MinRes "
            "iteration counts to scale with mesh refinement. "
            "Signal: PETSc KSPSolve iteration count grows like "
            "O(h^-2) without fieldsplit and stays O(1) with the "
            "block preconditioner. (Catalog claim inherited; "
            "not separately Tier-2 falsified this iteration.)",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # PHYSICS: NAVIER-STOKES
    # ═══════════════════════════════════════════════════════════════════════════
    "navier_stokes": {
        "description": "Incompressible Navier-Stokes. Two approaches: (1) Monolithic Newton on mixed formulation, (2) IPCS fractional-step splitting.",
        "weak_form_monolithic": "nu*inner(grad(u),grad(v))*dx + inner(dot(u,nabla_grad(u)),v)*dx - p*div(v)*dx - q*div(u)*dx = dot(f,v)*dx",
        "function_space": "Mixed: P2 velocity + P1 pressure (Taylor-Hood, inf-sup stable)",
        "ipcs_splitting": {
            "description": "Incremental Pressure Correction Scheme (IPCS) — Chorin's splitting, 2nd order",
            "step1": "Tentative velocity: solve momentum with old pressure",
            "step2": "Pressure correction: pressure Poisson equation using tentative velocity divergence",
            "step3": "Velocity correction: project velocity to be divergence-free",
            "demo_url": "https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code1.html",
            "advantage": "Decouples velocity and pressure — smaller systems, easier to precondition",
            "disadvantage": "Splitting error, requires small time step for accuracy",
        },
        "dg_navier_stokes": {
            "description": "Divergence-conforming DG method using BDM elements for exactly divergence-free velocity",
            "demo_url": "https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_navier-stokes.html",
        },
        "benchmarks": {
            "poiseuille_channel": "https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code1.html",
            "dfg_cylinder_benchmark": "https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html — DFG 2D-3, T=8, dt=1/1600, Re=100",
        },
        "solver": {
            "monolithic": "NonlinearProblem with SNES newtonls + MUMPS (small) or GMRES+AMG (large)",
            "ipcs": "Three sequential LinearProblem solves per time step",
        },
        "pitfalls": [
            "[API] READ FIRST — several signals in this topic name "
            "classes that do not exist on dolfinx 0.10 and can "
            "therefore never fire. Signal: PETScKrylovSolver is "
            "not part of dolfinx at all (hasattr(dolfinx, "
            "'PETScKrylovSolver') is False, and it is absent from "
            "dolfinx.fem.petsc) — it is a legacy DOLFIN name; and "
            "dolfinx.nls.petsc.NewtonSolver still imports but "
            "wrapping a 0.10 NonlinearProblem raises AttributeError: "
            "'NonlinearProblem' object has no attribute 'a'. Read "
            "both as PETSc SNES/KSP signals instead: build "
            "NonlinearProblem(..., petsc_options_prefix=..., "
            "petsc_options={'snes_monitor': ''}) and inspect "
            "problem.solver.getConvergedReason() / "
            "getIterationNumber(); for linear solves use "
            "LinearProblem(...).solver, which is a petsc4py KSP. "
            "(Verified by execution 2026-08-03, dolfinx 0.10.0 — "
            "same correction the hyperelasticity topic already "
            "carries; it was missing here.)",
            "[Numerical] Must use an inf-sup stable element pair "
            "(Taylor-Hood P2/P1 is the canonical choice in "
            "dolfinx; basix.ufl.mixed_element([P2, P1])). "
            "Equal-order P1/P1 fails the LBB condition. Signal: "
            "PETScKrylovSolver reports residual stalling far "
            "from tolerance OR the pressure field shows visible "
            "checkerboard mode patterns on the assembled "
            "Function. (Claim inherited — not yet empirically "
            "separated.)",
            "[Physics] Enclosed-flow incompressible Stokes / "
            "Navier-Stokes admits the constant pressure null "
            "space — pin one DoF (dirichletbc on a single point) "
            "or attach a null space via "
            "PETScKrylovSolver.setNullSpace. Signal: "
            "LinearProblem.solve / SNES Newton returns "
            "successfully but the post-processed pressure "
            "Function has a large additive offset (max ≈ min "
            "≈ O(1e6), tiny std) — same family as poisson "
            "pure-Neumann (fenics poisson#3). (Claim inherited — "
            "not yet empirically separated for navier_stokes "
            "specifically.)",
            "[Numerical] High Re (>500) requires finer mesh or "
            "continuation in Re for Newton convergence. Naively "
            "running Re=1000 from a zero initial guess often "
            "fails to converge. Signal: dolfinx.nls.petsc."
            "NewtonSolver.solve reports 'Failed to converge' / "
            "iteration count = max_it; switching to a "
            "continuation loop in Re recovers convergence. "
            "(Claim inherited — not yet empirically verified.)",
            "[API] Dirichlet BCs on sub-spaces of a mixed "
            "FunctionSpace require a Function on the COLLAPSED "
            "sub-space, NOT a raw numpy constant. Passing a "
            "constant array to dolfinx.fem.dirichletbc with "
            "(V_sub_dofs, V_sub_full) raises TypeError "
            "'incompatible function arguments'. Correct: "
            "u_bc = dolfinx.fem.Function(V_sub.collapse()[0]); "
            "u_bc.x.array[:] = 0.0; dolfinx.fem.dirichletbc("
            "u_bc, boundary_dofs, V_sub). Signal: TypeError "
            "'incompatible function arguments' from "
            "dirichletbc.__init__ at the moment the BC is "
            "constructed with a raw constant on a sub-space. "
            "(Verified empirically 2026-06-01.)",
            "[API] P2 velocity Function cannot be written "
            "directly via XDMFFile.write_function — same degree-"
            "mismatch as fenics linear_elasticity#3. Interpolate "
            "to a P1 space first, or use VTKFile / VTXWriter. "
            "Signal: XDMFFile.write_function raises RuntimeError "
            "'Degree of output Function must be same as mesh "
            "degree. Maybe the Function needs to be "
            "interpolated?'. (Cross-referenced from the fenics "
            "linear_elasticity#3 fixture — same failure mode.)",
            "[Numerical] Newton may not converge for hard NS "
            "cases — inspect snes_monitor (set "
            "'snes_monitor_short' in petsc_options), reduce Re, "
            "refine the mesh, or switch to an IPCS time-split "
            "scheme. Signal: NewtonSolver.solve raises "
            "'Failed to converge' with snes_monitor showing "
            "non-monotonic residual; IPCS does not require "
            "Newton at all (three sequential LinearProblem "
            "solves per step). (Claim inherited.)",
            "[Numerical] IPCS time step dt must respect the "
            "splitting accuracy: the splitting error is O(dt) "
            "per step, so dt > (target_l2_error) / Re is "
            "necessary for first-order splitting and tighter "
            "for higher-order projections. Signal: integrated "
            "L2 error — computed as dolfinx.fem.assemble_scalar"
            "(dolfinx.fem.form(ufl.inner(u_h-u_exact, "
            "u_h-u_exact)*ufl.dx)) — of u_h (the Function from "
            "LinearProblem.solve) vs an analytic reference "
            "saturates as dt is reduced because the splitting "
            "error dominates the spatial error; switching back "
            "to monolithic dolfinx.nls.petsc.NonlinearProblem + "
            "NewtonSolver recovers the spatial-error regime. "
            "(Note 2026-06-02: dolfin's ufl.errornorm helper is "
            "NOT available in dolfinx; assemble the inner-product "
            "form manually as shown.)",
        ],
        "materials": {
            "Re": {"range": [1, 10000], "unit": "dimensionless", "description": "Reynolds number"},
            "nu": {"range": [1e-6, 1.0], "unit": "m^2/s", "description": "Kinematic viscosity = 1/Re for unit domain"},
        },
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # PHYSICS: HEAT EQUATION
    # ═══════════════════════════════════════════════════════════════════════════
    "heat": {
        "description": "Heat equation (steady or transient). Fourier's law: rho*cp*dT/dt - div(k*grad(T)) = Q.",
        "weak_form_steady": "k * inner(grad(T), grad(v)) * dx = Q * v * dx",
        "weak_form_transient": "(T - T_n)/dt * v * dx + k * inner(grad(T), grad(v)) * dx = Q * v * dx",
        "function_space": "Lagrange order 1 or 2",
        "demo_url": "https://jsdokken.com/dolfinx-tutorial/chapter2/heat_equation.html",
        "time_integration": {
            "backward_euler": "Implicit, 1st order, unconditionally stable. theta=1 in theta-method.",
            "crank_nicolson": "theta=0.5, 2nd order, may oscillate near discontinuities.",
            "bdf2": "2nd order backward difference, requires 2 previous solutions.",
            "implementation": "LHS matrix is time-independent — assemble once, update RHS each step.",
        },
        "code_skeleton": {
            "time_loop": "for n in range(num_steps): t += dt; update_bcs(t); assemble L; solve Au=b; u_n.x.array[:] = u.x.array",
        },
        "solver": {"direct": "LU (small)", "iterative": "CG + hypre per time step"},
        "pitfalls": [
            (
                "[API] Insulated boundary = natural BC — DO "
                "NOTHING (zero flux is built into the weak "
                "form). Signal: applying a DirichletBC with "
                "value=0 on a wall meant to be insulated "
                "OVER-constrains the temperature (forces T=0 "
                "there, not dT/dn=0); the simulated temperature "
                "is pulled toward zero at the boundary instead "
                "of merely having no heat flux. Compare the "
                "no-BC run vs Dirichlet=0 — the difference "
                "exposes the misapplied BC. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] For transient: update BCs and source "
                "term at each time step. Signal: result at step N "
                "matches the steady solution of the FIRST step's BCs "
                "instead of evolving — typical bug when a time-"
                "dependent fem.Constant(T0) is created once outside "
                "the loop and the Dirichlet value never gets "
                "rewritten via Constant.value. (Audit 2026-06-02.)"
            ),
            (
                "[API] Mass matrix assembly for time derivative: "
                "(T/dt)*v*dx on LHS, (T_n/dt)*v*dx on RHS. Signal: "
                "wrong sign / placement gives wildly oscillating "
                "temperature with magnitude growing geometrically; "
                "energy is not conserved in an insulated cell test "
                "(temperature should be constant). (Audit "
                "2026-06-02.)"
            ),
            (
                "[Input] Temperature units must be consistent with "
                "material properties. Signal: an SI material "
                "(k=W/(m*K), rho*cp=J/(m^3*K)) wired through "
                "fem.Constant on the dolfinx Function and fed "
                "degrees-Celsius + degrees-Celsius/s data gives "
                "wildly wrong diffusion timescales — the "
                "characteristic time L^2 * rho*cp / k is off by "
                "orders of magnitude when K vs C are mixed. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Backward Euler is diffusive but stable; "
                "Crank-Nicolson is more accurate but may oscillate. "
                "Signal: CN with sharp transients (e.g. step source "
                "or step BC) shows 10-30% over/undershoot at the "
                "transient location that does not damp with mesh "
                "refinement; switching to Backward Euler removes "
                "the oscillation at the cost of phase error. "
                "(Audit 2026-06-02.)"
            ),
        ],
        "materials": {"conductivity": {"range": [0.01, 1000], "unit": "W/(m*K)"}, "rho_cp": {"description": "Volumetric heat capacity"}},
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # PHYSICS: CONVECTION-DIFFUSION (SUPG)
    # ═══════════════════════════════════════════════════════════════════════════
    "convection_diffusion": {
        "description": "Advection-diffusion equation with SUPG (Streamline Upwind Petrov-Galerkin) stabilization for advection-dominated transport.",
        "weak_form": "inner(b, grad(u))*v*dx + kappa*inner(grad(u), grad(v))*dx = f*v*dx",
        "supg_stabilization": {
            "description": "Add stabilization term: tau * inner(b, grad(v)) * (inner(b, grad(u)) + kappa*div(grad(u)) - f) * dx",
            "tau": "h / (2*|b|) * (coth(Pe_h) - 1/Pe_h) where Pe_h = |b|*h/(2*kappa) is cell Peclet number",
            "implementation": "Modify test function: v_stab = v + tau * inner(b, grad(v))",
        },
        "alternative_stabilizations": {
            "DG": "Discontinuous Galerkin with upwind flux — naturally handles advection",
            "GLS": "Galerkin Least Squares — similar to SUPG but also stabilizes reaction",
        },
        "pitfalls": [
            (
                "[Numerical] Without stabilization, the Galerkin "
                "method oscillates whenever the CELL Peclet "
                "number Pe_h = |b|*h/(2*kappa) exceeds ~1. "
                "Signal: [MEASURED 2026-08-03, dolfinx 0.10.0, "
                "P1 on the unit square, b=(1,0), kappa=1e-3, "
                "u=0 at x=0 and u=1 at x=1 — an exponential "
                "outflow layer] the Galerkin undershoot is "
                "-6.28 / -3.00 / -1.72 / -1.41 / -1.02 / -0.56 "
                "/ -0.08 at N = 8 / 16 / 32 / 64 / 128 / 256 / "
                "512 (Pe_h = 62.5 down to 0.98). "
                "IMPORTANT CORRECTION: the oscillation DOES "
                "damp under refinement — it disappears once the "
                "mesh resolves the layer (Pe_h < 1). The "
                "previous wording ('oscillation amplitude does "
                "not damp with mesh refinement') is falsified. "
                "The real argument for stabilisation is COST: "
                "resolving the layer needs h < 2*kappa/|b|, "
                "which is unaffordable for small kappa. SUPG "
                "with the same meshes keeps the undershoot at "
                "-0.042 down to -0.000 throughout."
            ),
            (
                "[Numerical] SUPG tau parameter depends on mesh "
                "size h and velocity magnitude — must compute "
                "PER CELL via tau = h/(2*|b|) * (coth(Pe_h) - "
                "1/Pe_h) using ufl.CellDiameter inside the "
                "dolfinx fem.form. A single global tau Constant "
                "does not vanish as h -> 0 and leaves a fixed "
                "streamline-diffusion floor. Signal: [MEASURED "
                "2026-08-03; MMS u = sin(pi x) sin(pi y), "
                "b=(1,1), kappa=0.01, N = 8..128] with the "
                "per-cell CellDiameter tau the L2 rates are "
                "1.91 / 1.57 / 1.62 / 1.84 (P1, approaching 2 "
                "as Pe_h drops below 1) — O(h^2) as claimed. "
                "With a FIXED tau Constant (0.0575, the coarse-"
                "mesh value) the errors are 9.76e-3, 4.12e-3, "
                "3.94e-3, 4.03e-3, 4.06e-3 — rates 1.24, 0.06, "
                "-0.03, -0.01. "
                "IMPORTANT CORRECTION: a constant tau does not "
                "degrade the rate to ~O(h); it STALLS "
                "convergence entirely — the error plateaus at a "
                "fixed floor and further refinement buys "
                "nothing. Same behaviour on P2 (rates 1.76, "
                "-0.42, -0.19). Look for a flat error curve, "
                "not a halved slope."
            ),
            (
                "[Numerical] DG methods are a cleaner alternative "
                "for pure advection (no diffusion). Signal: for "
                "vanishing diffusion kappa -> 0, the SUPG dolfinx "
                "ufl form's tau degenerates (tau -> h/|b|, but "
                "stabilisation residual scales with kappa) and "
                "the LinearProblem solution oscillates between "
                "elements; an upwind DG basix.ufl element on the "
                "same mesh produces a smooth Function with no "
                "parameter tuning. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] For time-dependent: SUPG in "
                "space + implicit time stepping. Mixing SUPG "
                "with explicit Euler can break: SUPG injects "
                "time-derivative coupling via the residual, "
                "which needs implicit treatment. Signal: "
                "explicit dolfinx fem.assemble + SUPG "
                "diverges to NaN within a few steps even "
                "below the convective CFL; switching to "
                "implicit (theta=1 or BDF2) inside a "
                "NonlinearProblem restores stability. "
                "(Audit 2026-06-02.)"
            ),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # PHYSICS: HYPERELASTICITY
    # ═══════════════════════════════════════════════════════════════════════════
    "hyperelasticity": {
        "description": "Nonlinear hyperelasticity with large deformations. Stored energy function approach.",
        "weak_form": "delta_Pi(u;v) = 0 where Pi = integral(psi(F) dx - T.u ds), solved as F(u,v) = dPi/du[v] = 0",
        "function_space": "Vector Lagrange order 1 or 2",
        "demo_url": "https://jsdokken.com/dolfinx-tutorial/chapter2/hyperelasticity.html",
        "kinematics": {
            "F": "ufl.variable(ufl.Identity(d) + ufl.grad(u)) — deformation gradient",
            "C": "F.T * F — right Cauchy-Green tensor",
            "J": "ufl.det(F) — volume ratio (J>0 required)",
            "I_C": "ufl.tr(C) — first invariant",
            "I_Cbar": "J^(-2/d) * I_C — isochoric first invariant",
        },
        "material_models": {
            "neo_hookean": {
                "psi": "(mu/2)*(I_C - 3) - mu*ln(J) + (lambda_/2)*(ln(J))**2",
                "parameters": "mu = E/(2*(1+nu)), lambda_ = E*nu/((1+nu)*(1-2*nu))",
            },
            "mooney_rivlin": {
                "psi": "c1*(I_C - 3) + c2*(II_C - 3) + (K/2)*(J-1)**2",
                "parameters": "c1, c2 (material constants), K (bulk modulus)",
                "notes": "II_C = 0.5*(tr(C)^2 - tr(C^2)) is second invariant",
            },
        },
        "code_skeleton": {
            "F": "F = ufl.variable(ufl.Identity(d) + ufl.grad(u))",
            "psi": "psi = (mu/2)*(ufl.tr(F.T*F) - 3) - mu*ufl.ln(ufl.det(F)) + (lmbda/2)*(ufl.ln(ufl.det(F)))**2",
            "P": "P = ufl.diff(psi, F)  # First Piola-Kirchhoff stress via automatic differentiation",
            "F_form": "F_form = ufl.inner(P, ufl.grad(v)) * ufl.dx - ufl.dot(traction, v) * ufl.ds",
        },
        "solver": {
            "nonlinear": "NonlinearProblem with SNES newtonls",
            "petsc_options": {"snes_type": "newtonls", "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
            "load_stepping": "For large deformations: apply load in increments, solving at each step",
        },
        "pitfalls": [
            "[API] Several signals below name dolfinx.nls.petsc.NewtonSolver.solve, which is a dolfinx 0.9-era code path. Signal: on dolfinx 0.10.0, NewtonSolver(MPI.COMM_WORLD, problem) around a 0.10 NonlinearProblem emits a DeprecationWarning and then raises AttributeError: 'NonlinearProblem' object has no attribute 'a'. (Verified empirically 2026-08-03.) Read those signals as PETSc SNES signals instead: use NonlinearProblem(..., petsc_options_prefix=..., petsc_options={'snes_monitor': ''}) and inspect problem.solver.getConvergedReason() / getIterationNumber().",
            "[Numerical] Large load steps cause Newton divergence in hyperelasticity. Use incremental load stepping: ramp the dirichletbc value or body-force fem.Constant across N steps, solving at each level. Signal: the SNES converged reason goes negative (DIVERGED_LINE_SEARCH / DIVERGED_MAX_IT) with the residual at the last iter still O(1); reducing the per-step load increment by 2-4x recovers convergence. (Claim inherited; the NewtonSolver-specific wording was corrected 2026-08-03 — see the version note above.)",
            "[Numerical] Near-incompressible regime (nu > 0.49) can make the pure-displacement formulation lock — use a dolfinx mixed (u, p) basix.ufl.mixed_element([P2-vector, P1]) FunctionSpace or the F-bar method (uniform-pressure projection). Signal: [CAUTION on magnitude, 2026-08-03] the previously quoted signal ('Cook-membrane tip deflection at nu = 0.4999 with pure P2 displacement is O(1e-3) of analytic') is not supported by measurement in the linear analogue — a P2 cantilever at nu=0.4999 came within 0-6% of the P2/P1 Taylor-Hood reference on every mesh tested, while P1 was 9x-20x too stiff. Expect the severe locking at P1, not at P2; measure the ratio for your own geometry rather than assuming orders of magnitude. (Linear-elasticity measurement 2026-08-03; the hyperelastic Cook membrane itself was NOT re-run.)",
            "[Physics] Neo-Hookean / any compressible hyperelastic model requires J = det(F) > 0 everywhere. A locally inverted element gives J <= 0 and the log(J) term blows up. Signal: NewtonSolver.solve raises RuntimeError / FloatingPointError, or the residual jumps to nan, when det(F) at any quadrature point hits 0 or goes negative. Defensive check: ufl.conditional(J > 0, ..., raise_an_error). (Claim inherited.)",
            "[API] ufl.variable() + ufl.diff() automate stress computation from a stored energy W. Wrap F in ufl.variable to mark it as the differentiation target, define W(F_var), then P = ufl.diff(W, F_var) yields the 1st Piola-Kirchhoff stress as a ufl.VariableDerivative expression directly usable inside the residual ufl.inner(P, grad(v))*dx form. Signal: type(ufl.variable(F)) is ufl.classes.Variable; type(ufl.diff(W, F_var)).__name__ == 'VariableDerivative'. NOTE the spelling: the class lives in the ufl.variable MODULE, so repr(type(...)) prints \"<class 'ufl.variable.Variable'>\", but the attribute ufl.variable is the FUNCTION variable() and shadows the submodule — writing `ufl.variable.Variable` raises AttributeError: 'function' object has no attribute 'Variable'. Use ufl.classes.Variable. Hand-coding the gradient bypasses ufl's analytic differentiation and is error-prone. (Verified empirically 2026-06-01; spelling re-checked by execution 2026-08-03 on ufl 2025.2.1.)",
            "[Numerical] Near-incompressibility split: decompose F = F_iso * F_vol where F_vol = (J^(1/3))*I (via ufl.det and ufl.Identity); then W = W_iso(F_iso) + U(J) with a quadratic-in-(J-1) volumetric penalty U(J) = kappa/2 * (J - 1)^2. Avoids volumetric locking in pure-displacement settings AND retains a well-conditioned tangent. Signal: dolfinx fem.assemble_scalar of the post-processed pressure (= dU/dJ) gives a bounded value; without the split, the discrete pressure Function at Gauss points oscillates wildly element-to-element. (Claim inherited.)",
            "[API] PETSc SNES newtonls residual monitor: pass 'snes_monitor': '' (or 'snes_monitor_short') in dolfinx.nls.petsc.NewtonSolver options. The monitor prints the residual norm per iter to stderr; if it stalls, halve the load increment and re-run. Signal: stderr shows '0 SNES Function norm ...' lines from PETSc; a stalled iteration shows the norm plateauing at a fixed O(1) value over many iterations rather than dropping by 10x per step. (Claim inherited.)",
        ],
        "materials": {
            "E": {"range": [1e2, 1e12], "unit": "Pa"},
            "nu": {"range": [0.0, 0.499], "unit": "dimensionless"},
        },
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # PHYSICS: THERMAL-STRUCTURAL COUPLING
    # ═══════════════════════════════════════════════════════════════════════════
    "thermal_structural": {
        "description": "Coupled thermal-structural: solve heat -> apply thermal strain -> solve elasticity. Sequential (one-way) or iterative (two-way).",
        "weak_form": "Step 1: k*(grad(T),grad(v))*dx = Q*v*dx. Step 2: sigma(u)=C:(eps(u) - alpha*DeltaT*I), inner(sigma,eps(v))*dx = 0",
        "function_space": "Scalar Lagrange for T, Vector Lagrange for u (two separate function spaces)",
        "coupling_approach": {
            "one_way": "Sequential: solve thermal first, feed temperature to structural as thermal load",
            "two_way": "Iterative: solve thermal, solve mechanical, update thermal conductivity with deformation, repeat",
        },
        "solver": {"thermal": "CG + hypre", "structural": "CG + GAMG"},
        "pitfalls": [
            (
                "[Numerical] Thermal strain = alpha * DeltaT * "
                "Identity is isotropic (equal expansion in all "
                "directions). Signal: applying alpha as a "
                "scalar inside sigma = C:eps(u) but FORGETTING "
                "to subtract alpha*DeltaT*I from the elastic "
                "strain gives a free-expansion temperature "
                "field that produces ZERO mechanical "
                "displacement at unconstrained boundaries; the "
                "expected uniform expansion u = alpha * DeltaT "
                "* x is missing. The correct form is "
                "sigma = C : (eps(u) - alpha * DeltaT * I), "
                "with the subtraction applied INSIDE the "
                "constitutive law. (Audit 2026-06-02.)"
            ),
            (
                "[Input] Reference temperature T_ref matters: "
                "DeltaT = T - T_ref. Signal: leaving the "
                "T_ref ufl.Constant at 0 with an SI material "
                "at room temperature gives an initial "
                "thermal pre-strain of order alpha*T_room "
                "(~3e-3 for steel at 300 K) that the "
                "dolfinx NonlinearProblem structure must "
                "equilibrate against — first-step "
                "displacement Function is huge compared to "
                "the actual loading and the Newton iteration "
                "may oscillate. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Plane strain: use full 3D Lame "
                "parameters (not plane stress modification). "
                "Signal: a dolfinx ufl plane-strain run that "
                "swaps in the plane-stress E' = E/(1-nu^2) on "
                "the fem.Constant lambda under-predicts stress "
                "by a factor of ~(1+nu)/(1-nu) and the VectorH1 "
                "displacement Function diverges from the 3D "
                "reference by ~20-50% at nu=0.3. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Mechanical BC needed to prevent "
                "rigid body motion (over-constrained = locking). "
                "Signal: a dolfinx fem.petsc.LinearProblem solve "
                "without a dirichletbc hangs / reports near-zero "
                "pivot; the stiffness matrix has 3 (2D) / 6 (3D) "
                "zero eigenvalues corresponding to translation + "
                "rotation. Add a minimal set of 3 (or 6) "
                "dirichletbc entries to kill the null space. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Two-way coupling (thermoelastic) "
                "requires Picard iteration between fields. "
                "Signal: one-shot (no-iteration) two-way "
                "solve via two dolfinx fem.NonlinearProblem "
                "calls shows a delta-T-dependent error in "
                "displacement of order |alpha*DeltaT*L| "
                "because the structural response affects the "
                "heat-conduction geometry but the back-"
                "influence was never iterated. The Picard "
                "residual ||T_new - T_old|| / ||T_old|| "
                "computed via dolfinx assemble_vector should "
                "drop below ~1e-3 across coupling "
                "iterations. (Audit 2026-06-02.)"
            ),
        ],
        "materials": {
            "E": {"range": [1e3, 1e12], "unit": "Pa"},
            "nu": {"range": [0.0, 0.499], "unit": "dimensionless"},
            "alpha": {"range": [1e-7, 1e-4], "unit": "1/K", "description": "Thermal expansion coefficient",
                      "examples": {"steel": 12e-6, "aluminum": 23e-6, "concrete": 10e-6}},
        },
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # PHYSICS: BIHARMONIC / KIRCHHOFF PLATE
    # ═══════════════════════════════════════════════════════════════════════════
    "biharmonic": {
        "description": "Biharmonic equation (4th order): laplacian^2(u) = f. Used for Kirchhoff plates, stream function formulation. Requires DG or C1 elements.",
        "weak_form_ip": "inner(div(grad(u)), div(grad(v)))*dx - inner(avg(div(grad(u))), jump(grad(v),n))*dS - inner(jump(grad(u),n), avg(div(grad(v))))*dS + alpha/h*inner(jump(grad(u),n), jump(grad(v),n))*dS",
        "method": "Interior Penalty (IP-DG): C0 elements with penalty on gradient jumps",
        "function_space": "Lagrange order 2 (with interior penalty for C0 elements)",
        "demo_url": "https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_biharmonic.html",
        "alternative": "Hermite elements (C1 conforming) — avoids DG penalty terms but limited to simplices",
        "solver": "LU (direct) for moderate sizes, GMRES for large",
        "pitfalls": [
            (
                "[Numerical] Penalty parameter alpha must be large "
                "enough for coercivity, but the standard "
                "'alpha = 4*(k+1)^2' rule of thumb is a "
                "STABILITY floor, not an accuracy optimum. "
                "Signal: [MEASURED 2026-08-03, dolfinx 0.10.0; "
                "C0-IP MMS u = sin(pi x) sin(pi y) on the unit "
                "square, simply-supported, L2 error at N = "
                "8/16/32/64] P2 with alpha = 36 (= 4*(k+1)^2) "
                "gives 8.40e-2 -> 1.80e-3 at rate ~1.96, while "
                "alpha = 1 gives 1.92e-2 -> 2.96e-4 at rate "
                "2.00 — same order, ~6x smaller error. "
                "IMPORTANT CORRECTION: too-small alpha does NOT "
                "make the solution norm diverge under mesh "
                "refinement. Measured the other way round — "
                "alpha = 1e-6 gives a HUGE coarse-mesh error "
                "(1.25e+1 at N=16) that then converges at rate "
                "~4.0 as the mesh is refined (7.69e-1, 4.78e-2, "
                "2.98e-3). The observable for an under-"
                "penalised C0-IP scheme is a blown-up error "
                "CONSTANT on coarse meshes, not divergence "
                "under refinement."
            ),
            (
                "[API] h_E (cell-size measure for the penalty "
                "weight) should use ufl.CellDiameter / "
                "ufl.FacetArea so the penalty tracks the local "
                "element size on graded / locally refined "
                "meshes. Signal: [MEASURED 2026-08-03] on a "
                "UNIFORM refinement sequence, hard-coding "
                "h = 1/8 as a fem.Constant while refining from "
                "N=16 to N=128 does NOT break convergence — the "
                "measured P2 L2 rates are 2.65 / 2.59 / 2.43, "
                "at least as good as the CellDiameter form. "
                "IMPORTANT CORRECTION: the previously quoted "
                "signal ('convergence rate degrades from O(h^2) "
                "to ~O(h) or stagnates') is NOT reproducible on "
                "uniform meshes. Treat this as a graded-mesh "
                "concern only, and diagnose it by comparing "
                "local penalty magnitudes rather than by "
                "watching a global rate."
            ),
            (
                "[Performance] Interior penalty requires interior "
                "facet integrals (dS) — more expensive than "
                "standard FEM (each facet visited from both "
                "sides). Signal: assembly time per step in a "
                "biharmonic problem is 5-10x the equivalent "
                "Poisson; profile shows dolfinx.fem.assemble_matrix "
                "spending most time in facet kernels. Mixed "
                "method (u + auxiliary sigma) avoids dS at the "
                "cost of doubling the DOF count. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Alternative: split into two "
                "2nd-order equations (mixed method with auxiliary "
                "variable). Signal: [MEASURED 2026-08-03, dolfinx "
                "0.10.0] writing the naive single 4th-order "
                "form inner(div(grad(u)), div(grad(v)))*dx on a "
                "C0 Lagrange space raises NOTHING — it compiles "
                "and assembles cleanly (P2 on an 8x8 unit "
                "square: 3073 nonzeros), and on P1 it assembles "
                "an IDENTICALLY ZERO matrix (497 stored "
                "nonzeros, max |entry| = 0.0) because "
                "div(grad(.)) of a P1 function vanishes "
                "cell-wise. "
                "IMPORTANT CORRECTION: dolfinx does NOT raise "
                "`NotImplementedError: H2 conformity required` "
                "(no such error exists) and it does NOT "
                "silently substitute the interior-penalty form. "
                "The failure is silent and numerical: a "
                "singular/inconsistent operator. You must write "
                "the dS interior-penalty terms yourself, or use "
                "the mixed (u, sigma) split with sigma = "
                "Laplacian(u) on P1 x P1."
            ),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # PHYSICS: HELMHOLTZ
    # ═══════════════════════════════════════════════════════════════════════════
    "helmholtz": {
        "description": "Helmholtz equation: -laplacian(u) - k^2*u = f. Acoustic/optical wave propagation. Can be complex-valued.",
        "weak_form": "inner(grad(u), grad(v))*dx - k**2 * inner(u, v)*dx = inner(f, v)*dx",
        "function_space": "Lagrange order 2+ (need ~10 points per wavelength for accuracy)",
        "demo_url": "https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_helmholtz.html",
        "complex_valued": {
            "description": "Helmholtz with complex source/solution requires complex-valued PETSc build",
            "scalar_type": "np.complex128",
            "notes": "DOLFINx supports float32, float64, complex64, complex128 scalar types",
        },
        "absorbing_bc": {
            "description": "First-order absorbing BC: du/dn = -ik*u on artificial boundary",
            "implementation": "Add -1j*k*inner(u,v)*ds to bilinear form",
        },
        "solver": "GMRES + LU (direct) for moderate sizes. Indefinite system — CG does NOT work.",
        "pitfalls": [
            (
                "[Numerical] Need a fine mesh, and '~10 points "
                "per wavelength' is a FLOOR that is far too "
                "loose for P1. Signal: [MEASURED 2026-08-03, "
                "dolfinx 0.10.0; MMS u = sin(k x/sqrt2) "
                "sin(k y/sqrt2) on the unit square with "
                "Dirichlet data interpolated from the exact "
                "solution, P1, non-resonant k chosen so k^2/pi^2 "
                "stays away from any m^2+n^2] "
                "at k = 12.17 the relative L2 error is 5.8e-1 at "
                "8.3 pts/wavelength, 1.0e-1 at 16.5, 2.6e-2 at "
                "33, 1.6e-3 at 132 (rates 2.46/2.01/2.00/2.00). "
                "At k = 27.57 the errors are 3.04 / 8.18 / 0.73 "
                "/ 9.6e-2 / 2.2e-2 at 3.6 / 7.3 / 14.6 / 29.2 / "
                "58.3 pts/wavelength; at k = 54.55, 1.81 / 2.26 "
                "/ 9.63 / 1.09 / 1.4e-1. "
                "IMPORTANT CORRECTION: for k*h >~ 1 the scheme "
                "does not converge at ~O(h) — it does not "
                "converge AT ALL (relative error >= 1 with "
                "NEGATIVE measured rates). Clean O(h^2) only "
                "returns once k*h <~ 0.5. The pollution effect "
                "is visible as the required points-per-"
                "wavelength for a fixed accuracy growing with k "
                "(33 pts/wave gives 2.6% at k=12 but 29 "
                "pts/wave still gives 14% at k=55). "
                "PRACTICAL WARNING for MMS tests: pick k away "
                "from resonance — k^2 near pi^2*(m^2+n^2) makes "
                "the discrete system near-singular and the error "
                "study meaningless."
            ),
            (
                "[Numerical] System is INDEFINITE — standard CG "
                "diverges. Use GMRES or direct solver. Signal: ["
                "[re-verified 2026-08-03, dolfinx 0.10.0, "
                "k = 20 on a 32x32 unit square, ksp_type='cg' "
                "pc_type='icc'] PETSc stops after 3 iterations "
                "with converged reason -10 = "
                "KSP_DIVERGED_INDEFINITE_PC. For ~< 100k DOFs "
                "use LU; for larger meshes use GMRES + a shifted-"
                "Laplacian preconditioner."
            ),
            (
                "[Numerical] High wavenumber k: requires "
                "specialized preconditioners (shifted "
                "Laplacian). Signal: PETSc GMRES with "
                "default ILU/Jacobi PC on a k > 100 "
                "problem stagnates at residual ~1e-2 "
                "after 1000 iterations; the shifted-"
                "Laplacian preconditioner (PC with "
                "k_shift = k + i*epsilon) applied to the "
                "dolfinx LinearProblem restores ~10 "
                "iterations per convergence. (Audit "
                "2026-06-02.)"
            ),
            (
                "[API] Complex mode: PETSc must be compiled with "
                "--with-scalar-type=complex. Signal: importing "
                "PETSc into a real-mode build and trying to "
                "assemble a complex Helmholtz form raises "
                "`TypeError: cannot convert complex to real` or "
                "the imaginary part is silently dropped. Verify "
                "with PETSc.ScalarType == complex before running. "
                "(Audit 2026-06-02.)"
            ),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # PHYSICS: MAXWELL / ELECTROMAGNETICS
    # ═══════════════════════════════════════════════════════════════════════════
    "maxwell": {
        "description": "Maxwell's equations for electromagnetic wave propagation. Curl-curl formulation. Requires H(curl) (Nedelec) elements.",
        "weak_form_curl_curl": "inner(curl(E), curl(v))*dx - k0**2 * epsilon_r * inner(E, v)*dx = inner(J, v)*dx",
        "function_space": "Nedelec 1st kind (N1curl) — H(curl) conforming, tangential continuity",
        "demos": {
            "scattering_wire": "https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_scattering_boundary_conditions.html",
            "scattering_pml": "https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_pml.html",
            "waveguide_modes": "https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_half_loaded_waveguide.html",
            "axisymmetric_sphere": "https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_axis.html",
        },
        "pml": {
            "description": "Perfectly Matched Layer — artificial absorbing boundary layer",
            "implementation": "Complex-valued coordinate stretching transforms Maxwell equations in PML region",
        },
        "eigenvalue": {
            "description": "Electromagnetic modal analysis — find waveguide modes using SLEPc EPS",
            "elements": "N1curl (Nedelec) for transverse + Lagrange for axial component on quads",
            "solver": "SLEPc Krylov-Schur with spectral transformation (shift-and-invert)",
        },
        "solver": "GMRES + AMS (auxiliary-space Maxwell solver from hypre) for curl-curl",
        "pitfalls": [
            "[Physics] MUST use H(curl) elements (Nedelec / N1curl) for Maxwell — standard Lagrange spaces lack the tangential continuity that the physical fields require. Signal: dolfinx.fem.form does NOT fail at form construction (ufl.curl is accepted on vector Lagrange and even on scalar Lagrange in 2D), so the bug is silent at compile/assemble time. The observable failure is numerical: the post-processed B = curl(A) field has spurious normal jumps at element interfaces, and convergence against an analytic test (e.g., uniform B in a cavity) plateaus at ~10% error regardless of refinement. (Verified empirically 2026-06-01 — prior catalog wording 'violates physical constraints' implied a syntactic/assembly-time rejection; in current dolfinx the form compiles fine and the bug surfaces in the field values.)",
            "[Syntax] Complex-valued Maxwell: PETSc must be compiled with --with-scalar-type=complex. Signal: [exact text re-measured 2026-08-03 on a REAL conda-forge build, dolfinx 0.10.0 / PETSc 3.24.4] building the form raises ValueError 'Unexpected complex value in real expression.' at dolfinx.fem.form(...) — before assemble_vector is ever reached. Writing a complex value into a real Function array raises TypeError \"float() argument must be a string or a real number, not 'complex'\". The previously quoted strings 'cannot convert complex to float' / 'imaginary part discarded' do NOT appear. Check with numpy.issubdtype(dolfinx.default_scalar_type, numpy.complexfloating) before building the form.",
            "[Numerical] PML (Perfectly Matched Layer): requires coordinate stretching of the form x_i → x_i*(1 + i*sigma(x_i)/omega) inside the PML region. A real-only stretching (real sigma) gives a lossy real boundary, NOT a radiating PML. Signal: a fem.Function evaluated in the PML region decays by orders of magnitude only when the coordinate-stretch coefficient is constructed with numpy.complex128 ScalarType — with a real-only stretch the dolfinx.fem.assemble_vector output shows a standing-wave reflection back into the domain.",
            "[Numerical] Low-frequency breakdown: curl-curl + omega^2-mass formulation becomes ill-conditioned as omega → 0 because the gradient kernel of curl is no longer regularised by the mass term. Use mixed (A, phi) formulation with a Lagrange multiplier on the divergence. Signal: KSP iteration count for GMRES + AMS preconditioner explodes as omega is reduced below ~10^-3 of the lowest cavity eigenvalue; condition number printed by PETSc grows as 1/omega^2.",
            "[API] Edge elements (basix.ElementFamily.N1E / 'Nedelec 1st kind H(curl)') have DOF ordering by edge, not by node. Setting tangential BCs requires interpolating onto the edge basis, not the nodal basis. Signal: dirichletbc on an HCurl space defined with a vector-valued function silently sets only the first component on each edge, leaving the tangential trace 90 degrees off from intended; post-processed E field has non-zero normal component on the conductor boundary.",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # PHYSICS: CAHN-HILLIARD (PHASE FIELD)
    # ═══════════════════════════════════════════════════════════════════════════
    "cahn_hilliard": {
        "description": "Cahn-Hilliard equation: nonlinear, time-dependent 4th-order PDE for phase separation in binary mixtures. Split into two 2nd-order equations.",
        "equations": "dc/dt = div(M * grad(mu)), mu = f'(c) - lambda*laplacian(c), f(c) = 100*c^2*(1-c)^2",
        "weak_form": "(c-c_n)/dt * q * dx + M * inner(grad(mu), grad(q)) * dx = 0; mu*v*dx - df/dc*v*dx - lambda*inner(grad(c),grad(v))*dx = 0",
        "function_space": "Mixed element: two copies of Lagrange for (c, mu)",
        "demo_url": "https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_cahn-hilliard.html",
        "code_skeleton": {
            "element": "P1 = basix.ufl.element('Lagrange', cell, 1); ME = basix.ufl.mixed_element([P1, P1])",
            "differentiation": "c = ufl.variable(c); f = 100*c**2*(1-c)**2; dfdc = ufl.diff(f, c)",
            "time_stepping": "theta-method with theta=0.5 (Crank-Nicolson) for time integration",
        },
        "solver": "SNES Newton + LU per time step",
        "parameters": {
            "lmbda": "Surface parameter (controls interface width) ~ 1e-2",
            "dt": "Time step ~ 5e-6 (must be small for stability)",
            "M": "Mobility coefficient",
        },
        "pitfalls": [
            (
                "[Numerical] Very stiff system — requires small time "
                "step especially initially. Signal: starting from a "
                "random initial condition with dt ~ 1.0 gives "
                "SNES `DIVERGED_FNORM_NAN` within the first 1-3 "
                "steps; using dt ~ 1e-5 for the first ~100 steps "
                "and ramping to dt ~ 1e-2 afterwards is the "
                "standard recipe. (Audit 2026-06-02.)"
            ),
            (
                "[API] Chemical potential df/dc must use "
                "ufl.variable() and ufl.diff() for automatic "
                "differentiation. Signal: hand-coding the Cahn-"
                "Hilliard chemical potential (12 * c * (c-1) * "
                "(2c-1) for the double-well derivative) and "
                "missing a factor or sign gives sublinear "
                "convergence; ufl.diff guarantees the analytic "
                "exact derivative. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Random initial condition: "
                "c_0 = 0.63 + 0.02*(random - 0.5) for spinodal "
                "decomposition. Signal: setting the dolfinx "
                "fem.Function via interpolate(lambda x: "
                "np.full(...)) at c_0 = 0.5 exactly (the "
                "unstable symmetric mean) gives no phase "
                "separation — the Function stays uniformly at "
                "0.5 because there's no symmetry-breaking "
                "perturbation. The XDMFFile output at t = 1 "
                "should show interface formation; if not, the "
                "IC is too symmetric. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Newton convergence sensitive "
                "to time step — reduce dt if diverging. "
                "Signal: the dolfinx NonlinearProblem SNES "
                "emits `step rejected, reducing dt`; or "
                "the residual norm from assemble_vector "
                "diverges within 2-3 Newton iterations. "
                "Cahn-Hilliard becomes singular at fast-"
                "evolving interfaces; dt ~ eps^4 / M is "
                "the conservative stability limit (eps = "
                "interface thickness, M = mobility). "
                "(Audit 2026-06-02.)"
            ),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # PHYSICS: EIGENVALUE PROBLEMS
    # ═══════════════════════════════════════════════════════════════════════════
    "eigenvalue": {
        "description": "Eigenvalue problems A*x = lambda*B*x using SLEPc. Vibration modes, buckling, electromagnetic modes.",
        "function_space": "Depends on physics: Lagrange for scalar, Nedelec for EM, vector Lagrange for structural",
        "demo_url": "https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_half_loaded_waveguide.html",
        "code_skeleton": {
            "imports": "from slepc4py import SLEPc",
            "setup": "eps = SLEPc.EPS().create(MPI.COMM_WORLD); eps.setOperators(A, B); eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)",
            "target": "eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE); eps.setTarget(sigma)",
            "spectral_transform": "st = eps.getST(); st.setType(SLEPc.ST.Type.SINVERT)  # shift-and-invert",
            "solve": "eps.solve(); nconv = eps.getConverged()",
            "extract": "eigval = eps.getEigenvalue(i); eps.getEigenvector(i, xr, xi)",
        },
        "solver_types": {
            "krylovschur": "Default, recommended for most problems",
            "arnoldi": "Standard Arnoldi iteration",
            "lanczos": "For symmetric (Hermitian) problems",
            "power": "Power iteration (only for dominant eigenvalue)",
            "jd": "Jacobi-Davidson (interior eigenvalues)",
        },
        "pitfalls": [
            "[Integration] Eigenvalue problems in dolfinx use SLEPc (the eigenvalue counterpart of PETSc). SLEPc must be installed; PETSc must be configured with --download-slepc (or built against an external SLEPc). The Python binding is slepc4py.SLEPc.EPS. Signal: 'from slepc4py import SLEPc; SLEPc.EPS' resolves successfully when properly installed; ImportError 'No module named slepc4py' (or similar) when missing. (Verified empirically 2026-06-01 in the ofa-fenicsx conda env — slepc4py is present with EPS.)",
            "[Numerical] Shift-and-invert spectral transformation (SINVERT) is essential for interior eigenvalues. SLEPc.EPS().setST(...) with a SLEPc.ST configured to SINVERT centers the spectrum on the target value. Signal: searching for eigenvalues near k^2_estimate on the dolfinx-assembled stiffness Matrix without SINVERT returns extreme eigenvalues (highest or lowest) instead; with SINVERT and target = k^2_estimate the returned eigenvalues cluster near the target. (Claim inherited — not yet empirically separated.)",
            "[API] eps.setDimensions(nev, ncv) requests nev eigenvalues with ncv search-space size (ncv >= 2*nev is the SLEPc default heuristic). Too-small ncv slows convergence or fails. Signal: eps.solve() reports 'converged' with fewer than requested eigenvalues, or returns an error code != 0 from eps.getConvergedReason(); doubling ncv typically fixes it. (Claim inherited.)",
            "[Numerical] For a generalised eigenvalue problem A*x = lambda*B*x with Dirichlet BC, the Dirichlet rows of A and of the mass matrix B must not both be given the SAME diagonal value, or every constrained DOF contributes a spurious eigenvalue at A_ii/B_ii. Signal: [MEASURED 2026-08-03, dolfinx 0.10.0, P1 Laplacian on a 24x24 unit square, SLEPc krylovschur + shift-and-invert; analytic Dirichlet eigenvalues 19.739, 49.348, 49.348, 78.957] assembling A with bcs (default diag=1) and B ALSO with bcs (default diag=1) returns 1, 1, 1, 1, 19.82, 49.71, 49.92, 80.3 — the spurious modes sit at lambda = 1, NOT at lambda = 0 as previously written, and the true spectrum follows them. Two clean recipes, both verified: assemble B with bcs=[] (no BC at all), or assemble B with bcs=[bc] and diag=0.0 to push the constrained modes to infinity — both give 19.82, 49.71, 49.92, 80.3 with no spurious entries. NOTE the kwarg is `diag`, not `diagonal`: dolfinx 0.10's assemble_matrix signature is (a, bcs=None, diag=1, constants=None, coeffs=None, kind=None), and `diagonal=0.0` raises TypeError 'assemble_matrix() got an unexpected keyword argument'. (Verified empirically 2026-08-03 — corrects the previous 'spurious ZERO eigenvalues' wording.)",
            "[Integration] Complex-valued eigenvalues require dolfinx + PETSc + SLEPc all compiled with --with-scalar-type=complex. The default conda-forge fenics-dolfinx build is REAL: dolfinx.default_scalar_type is numpy.float64 (verified empirically 2026-06-01). For complex Helmholtz / Maxwell eigenproblems either rebuild with complex scalar OR split into (re, im) real-pair formulation. Signal: dolfinx.default_scalar_type returns numpy.float64 in a real build; numpy.issubdtype(dolfinx.default_scalar_type, np.complexfloating) is False — assembling a ufl form with an imaginary coefficient then yields a wrong real-valued Function with the imaginary part silently dropped. (Verified empirically in the ofa-fenicsx env.)",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # PHYSICS: REACTION-DIFFUSION SYSTEMS
    # ═══════════════════════════════════════════════════════════════════════════
    "reaction_diffusion": {
        "description": "Systems of coupled reaction-diffusion equations. Nonlinear reaction terms, multiple species.",
        "weak_form": "For species i: d(c_i)/dt * v_i * dx + D_i*inner(grad(c_i),grad(v_i))*dx = R_i(c)*v_i*dx",
        "function_space": "Mixed element with one Lagrange component per species",
        "demo_url": "https://jsdokken.com/dolfinx-tutorial/chapter2/intro.html (advection-diffusion-reaction systems)",
        "solver": "SNES Newton for nonlinear reaction terms",
        "pitfalls": [
            (
                "[Numerical] Nonlinear reaction terms require "
                "Newton iteration. Signal: a single-Picard-"
                "step solve on a quadratic reaction R(u) = "
                "u^2 via dolfinx LinearProblem converges "
                "linearly (residual ratio ~0.5 per "
                "iteration) instead of quadratically; SNES "
                "NonlinearProblem with the UFL-derived "
                "Jacobian restores quadratic convergence. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Stiff reactions (fast kinetics) may "
                "need implicit time stepping with small dt. "
                "Signal: explicit Euler / theta < 0.5 on a "
                "Damkohler-number-100 problem requires dt < "
                "2/lambda_max ~ 1e-3, otherwise the dolfinx "
                "Function explodes to NaN within a few steps; "
                "switching to backward Euler or BDF2 in the "
                "dolfinx NonlinearProblem restores stability, "
                "and for very stiff systems (Da > 1000) "
                "external SUNDIALS coupling is required. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Species concentrations should "
                "remain non-negative — check solution and "
                "add constraints if needed. Signal: "
                "visualizing the c dolfinx Function shows "
                "pockets of negative concentration (often "
                "near steep gradients) — unphysical. "
                "Standard fix: SUPG + shock-capturing in "
                "the BilinearForm-equivalent dolfinx fem "
                "form, or projection onto the non-negative "
                "cone via NonlinearProblem after each "
                "step. (Audit 2026-06-02.)"
            ),
            (
                "[API] Use ufl.variable() and ufl.diff() for "
                "automatic Jacobian of reaction terms. Signal: "
                "hand-coding the Jacobian and forgetting a "
                "df/dv coupling between species causes Newton "
                "to converge linearly instead of quadratically; "
                "ufl.diff(R(u), u) emits the exact partials. "
                "(Audit 2026-06-02.)"
            ),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # PHYSICS: NEARLY INCOMPRESSIBLE ELASTICITY
    # ═══════════════════════════════════════════════════════════════════════════
    "nearly_incompressible_elasticity": {
        "description": "Mixed methods for nearly incompressible elasticity (nu -> 0.5) to avoid volumetric locking.",
        "weak_form": "2*mu*inner(eps_dev(u),eps(v))*dx + p*div(v)*dx + (div(u) - p/kappa)*q*dx = dot(f,v)*dx",
        "function_space": "Mixed: Vector Lagrange for displacement + DG(k-1) for pressure",
        "approach": {
            "displacement_pressure": "u-p formulation: displacement (vector) + pressure (scalar) as independent unknowns",
            "three_field": "u-p-theta: displacement + pressure + dilatation (for Neo-Hookean)",
        },
        "solver": "MinRes or GMRES with block preconditioner (saddle-point structure)",
        "pitfalls": [
            (
                "[Numerical] Low-order displacement formulations "
                "LOCK as nu -> 0.5 — a mixed (u, p) method is "
                "the robust fix. Signal: [MEASURED 2026-08-03, "
                "dolfinx 0.10.0; cantilever 1.0 x 0.2 with end "
                "traction, tip deflection against a P2/P1 "
                "Taylor-Hood reference] "
                "P1 triangles are 7.2x / 3.2x / 1.6x / 1.2x too "
                "stiff at nu=0.49 on 10x2 / 20x4 / 40x8 / 80x16 "
                "meshes; 16.5x / 11.4x / 5.5x / 2.4x at "
                "nu=0.499; 19.5x / 18.7x / 15.4x / 9.2x at "
                "nu=0.4999. P2 triangles are within 0-6% of the "
                "mixed reference at EVERY nu and mesh tested. "
                "IMPORTANT CORRECTION: the locking ratio is NOT "
                "~1/(1-2nu) — that formula predicts 500x at "
                "nu=0.499 where 2.4x-16.5x is measured, "
                "depending on mesh. Locking magnitude depends "
                "jointly on nu, element order and h; quote a "
                "measured ratio, not the 1/(1-2nu) rule."
            ),
            (
                "[Numerical] Inf-sup (LBB) condition: "
                "pressure FunctionSpace must be STRICTLY "
                "SMALLER than the displacement "
                "FunctionSpace. Signal: [measured on the "
                "Stokes analogue 2026-08-03, dolfinx "
                "0.10.0] SVD of the bc-applied saddle-point "
                "matrix on an 8x8 unit square gives "
                "numerical null dimension 1 for P2/P1 "
                "Taylor-Hood (the constant pressure alone) "
                "but 8 for equal-order P1/P1 — the extra "
                "kernel vectors ARE the checkerboard modes. "
                "The LBB constant collapsing with h is the "
                "diagnostic for inf-sup failure."
            ),
            (
                "[Numerical] Taylor-Hood (P2/P1) or (P2/"
                "DG0) satisfy inf-sup; P1/P0 does NOT. "
                "Signal: convergence-rate test with P1/P0 "
                "dolfinx mixed FunctionSpace stagnates at "
                "first-order in displacement while P2/P1 "
                "achieves second-order; cross-check via "
                "the Mandel benchmark — P2/P1 recovers "
                "the analytic result to within 0.5%, "
                "P1/P0 differs by 5-10%. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Penalty method (large kappa) is "
                "alternative but introduces parameter sensitivity. "
                "Signal: penalty too small -> volumetric "
                "locking returns (det(F)-1 deviates by > 1% from "
                "0); penalty too large -> condition number "
                "exceeds 1e14 and Newton stalls. Mixed method "
                "is parameter-free and preferred for production "
                "runs. (Audit 2026-06-02.)"
            ),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # PHYSICS: CONTACT PROBLEMS
    # ═══════════════════════════════════════════════════════════════════════════
    "contact": {
        "description": "Contact mechanics in FEniCSx. Not built into DOLFINx core — requires custom implementation or extensions.",
        "approaches": {
            "penalty_method": "Add penalty energy for penetration: 1/2 * epsilon * max(0, -gap)^2. Simple but parameter-sensitive.",
            "nitsche_method": "Variationally consistent weak enforcement of contact. No additional unknowns.",
            "lagrange_multiplier": "Introduce multiplier for contact pressure. Exact but increases system size.",
            "dolfinx_contact": "github.com/jorgensd/dolfinx_contact — extension package for contact in DOLFINx",
        },
        "pitfalls": [
            (
                "[API] No built-in contact in DOLFINx — must "
                "implement penalty/Nitsche manually OR use the "
                "dolfinx_contact extension package. Signal: "
                "searching dolfinx.fem for `ContactBoundary` or "
                "`ContactProblem` returns nothing; the catalog "
                "ships hand-coded penalty / Nitsche snippets the "
                "user copies — there is no single-call contact "
                "API. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Penalty parameter: too small = "
                "penetration, too large = ill-conditioning. "
                "Signal: penetration > 5% of element edge "
                "indicates the penalty is too low; PETSc condition-"
                "number warning > 1e14 indicates too high. Rule "
                "of thumb: penalty = 1e2 * E / h for solid contact "
                "where E is the softer material's Young modulus. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Contact detection (gap computation) "
                "requires geometric search — naive O(N^2) "
                "all-pairs is fine for small problems but "
                "dominates wall-clock past ~10k surface points. "
                "Use bounding-volume hierarchies (BVH) from "
                "dolfinx.geometry. Signal: assembly time per "
                "Newton iteration grows quadratically with mesh "
                "size; using "
                "dolfinx.geometry.bb_tree(mesh, dim) keeps it "
                "near-linear. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Self-contact requires careful "
                "implementation of contact pairs — a node on the "
                "surface can contact another part of the SAME "
                "surface (not just the partner body). Signal: "
                "a buckling problem (post-bifurcation cylinder, "
                "ring crush) shows surfaces passing through "
                "themselves; visualize confirms intersecting "
                "geometry; need to flag the surface as both "
                "slave AND master in the contact pair list. "
                "(Audit 2026-06-02.)"
            ),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # PHYSICS: PHASE-FIELD FRACTURE
    # ═══════════════════════════════════════════════════════════════════════════
    "fracture": {
        "description": "Phase-field approach to fracture mechanics. Diffuse crack representation avoids remeshing. Extensions: PhaseFieldX library.",
        "equations": "Coupled: (1) mechanical equilibrium with degraded stiffness g(d)*sigma, (2) phase-field evolution for damage d",
        "function_space": "Vector Lagrange for displacement, scalar Lagrange for damage field d in [0,1]",
        "approach": {
            "AT1": "Standard phase-field model with linear dissipation",
            "AT2": "Phase-field model with quadratic dissipation (most common)",
        },
        "parameters": {
            "Gc": "Critical energy release rate [J/m^2]",
            "l0": "Length scale parameter (regularization width) — mesh must resolve l0",
            "irreversibility": "d_new >= d_old (crack cannot heal) — enforce via history variable or penalty",
        },
        "solver": "Staggered scheme (alternate between mechanical and damage) or monolithic Newton",
        "libraries": {
            "phasefieldx": "github.com/CastillonMiguel/phasefieldx — open-source DOLFINx phase-field framework",
        },
        "pitfalls": [
            (
                "[Numerical] Mesh must be fine enough to resolve "
                "length scale l0 (rule: h << l0, typically h "
                "< l0/3). Signal: the dolfinx damage Function d "
                "in the XDMFFile output shows visible staircase "
                "patterns following element edges (below-"
                "resolution diffuse-crack); the predicted "
                "fracture energy from fem.assemble_scalar(...) "
                "under-shoots Griffith's G_c * area by ~30-50% "
                "when h ~ l0. Refining the crack-path region "
                "recovers the analytic G_c. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Irreversibility constraint: must "
                "enforce d_new >= d_old (cracks cannot heal). "
                "Signal: the dolfinx damage Function visualised "
                "across time steps in XDMFFile shows d "
                "DECREASING in some elements between steps — "
                "unphysical. Standard fix: history-field "
                "projection max(d, d_prev) after each "
                "minimisation step. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Staggered scheme: simple but "
                "slow convergence; monolithic: fast but "
                "needs good initial guess. Signal: "
                "staggered iteration count per load step "
                "exceeds ~50 for moderately-loaded "
                "specimens (each step alternates between "
                "solving u-subproblem and d-subproblem via "
                "two dolfinx NonlinearProblem calls); "
                "monolithic mixed-FunctionSpace requires "
                "<10 Newton iters per step but diverges "
                "from the trivial u=0, d=0 initial guess "
                "past first crack nucleation. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Tension-compression split (Miehe) "
                "needed to prevent crack closure under "
                "compression. Signal: without the split, a "
                "compressive load nucleates spurious damage "
                "d > 0 in the loaded region (cracks 'form' "
                "under compression — physically wrong); with "
                "the split, uniaxial-compression test gives "
                "max(d) ~ 0. (Audit 2026-06-02.)"
            ),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # PHYSICS: COUPLED STOKES-DARCY
    # ═══════════════════════════════════════════════════════════════════════════
    "stokes_darcy": {
        "description": "Coupled Stokes-Darcy for free fluid / porous medium interaction. Interface conditions: Beavers-Joseph-Saffman.",
        "equations": {
            "stokes_region": "-div(2*mu*eps(u) - p*I) = f, div(u) = 0",
            "darcy_region": "u_D = -(K/mu)*grad(p_D), div(u_D) = g",
            "interface": "Continuity of normal flux, balance of normal stress, Beavers-Joseph-Saffman tangential condition",
        },
        "function_space": "Taylor-Hood for Stokes, RT+DG for Darcy (or unified mixed formulation)",
        "implementation_approaches": {
            "monolithic": "Single mesh with subdomain markers, different weak forms per region",
            "partitioned": "Separate meshes coupled via interface conditions (submesh approach)",
            "submesh": "DOLFINx create_submesh() to extract regions, couple via restriction operators",
        },
        "pitfalls": [
            (
                "[API] No built-in Stokes-Darcy demo in DOLFINx "
                "— must assemble custom weak forms. Signal: "
                "searching dolfinx.fem for `StokesDarcy` returns "
                "nothing; the user must hand-build the block "
                "system [[A_Stokes, C_interface], [C^T, "
                "A_Darcy]] and condense via "
                "dolfinx.fem.petsc.LinearProblem with explicit "
                "block layout. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Interface conditions (Beavers-"
                "Joseph-Saffman) require careful "
                "implementation. Signal: omitting the BJS "
                "slip-velocity term in the dolfinx fem "
                "FacetIntegrals gives a Stokes-Darcy result "
                "that disagrees with experiments by ~30% "
                "near the porous interface; including BJS "
                "with alpha_BJ ~ 0.1-1 as a Constant and "
                "proper normal-flux continuity restores "
                "the empirical match. (Audit 2026-06-02.)"
            ),
            (
                "[API] Different function spaces in different "
                "regions: use submesh or subdomain-restricted "
                "forms. Signal: putting a single H1 space over "
                "both Stokes and Darcy domains gives the wrong "
                "regularity in the porous side (Darcy requires "
                "H(div) flux, not H1 velocity). Use "
                "dolfinx.mesh.create_submesh() to carve out the "
                "porous subregion and assemble per-region. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Permeability K Constant can "
                "vary by orders of magnitude — use "
                "appropriate preconditioners. Signal: a "
                "coarse-grained block-Jacobi PETSc PC "
                "applied via the dolfinx LinearProblem to "
                "a Darcy block with K=1 Constant on one "
                "half and K=1e-6 Constant on the other "
                "stalls with residual ratio ~1; switching "
                "to a domain-decomposition or AMG-on-each-"
                "region preconditioner restores ~10 "
                "iterations to convergence. (Audit "
                "2026-06-02.)"
            ),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # PROVENANCE — adversarial re-verification pass
    # ═══════════════════════════════════════════════════════════════════════════
    "_provenance": {
        "description": "Record of what was actually EXECUTED against an installed dolfinx to substantiate the entries in this catalog. Entries not listed here were not re-run in that pass and carry their older audit tag.",
        "2026-08-03_adversarial_reverification": {
            "environment": (
                "dolfinx 0.10.0, basix 0.10.0, ufl 2025.2.1, "
                "petsc4py 3.24.4 (real scalars, 32-bit indices, "
                "MUMPS + SuperLU_DIST + UMFPACK all present), "
                "slepc4py 3.24.3, mpi4py 4.1.1, gmsh 4.15.1, "
                "Python 3.12.13 conda-forge, Linux x86_64. "
                "NOT installed in that env: dolfinx_mpc, "
                "adios4dolfinx, pyamg."
            ),
            "method": (
                "Every 'works' claim was executed; every 'fails' "
                "/ pitfall claim was reproduced by running the "
                "WRONG variant and comparing the observed error "
                "text or numerical misbehaviour to the "
                "documented signal. Numerical claims (element "
                "orders, stabilisation requirements, locking, "
                "penalty scaling, pollution) were checked with "
                "manufactured-solution convergence studies: "
                "prescribe an exact u, derive f symbolically "
                "(sympy) or analytically, refine, and fit the "
                "observed L2 rate."
            ),
            "mms_studies_run": [
                "Poisson u=sin(pi x)sin(pi y), P1/P2/P3, N=8..64 -> rates 2.00 / 3.00 / 4.01",
                "Convection-diffusion b=(1,1), kappa=0.01, P1 and P2, N=8..128: plain Galerkin, per-cell CellDiameter SUPG tau, and fixed global tau",
                "Advection boundary layer kappa=1e-3, N=8..512, Galerkin vs SUPG undershoot",
                "Biharmonic C0-IP u=sin(pi x)sin(pi y) (simply supported), P2/P3, alpha in {4(k+1)^2, 8, 1, 0.1, 0.01, 1e-4, 1e-6}, plus hard-coded vs CellDiameter h, N=8..128",
                "Helmholtz u=sin(kx/sqrt2)sin(ky/sqrt2), P1, non-resonant k in {12.17, 27.57, 54.55}, N=16..256",
                "Elasticity cantilever: plane-strain vs plane-stress tip deflection at nu=0.3/0.45; volumetric locking P1 vs P2 vs P2/P1 Taylor-Hood at nu=0.49/0.499/0.4999 on four meshes",
                "Stokes: SVD null-space dimension of the bc-applied saddle-point matrix, Taylor-Hood vs equal-order P1/P1, 8x8",
            ],
            "generator_execution": (
                "All 35 (physics, variant) pairs exposed by "
                "src/backends/fenics/generators were generated "
                "and RUN on the installed dolfinx; all 35 exit "
                "with return code 0. RETURN CODE WAS THE ONLY "
                "CRITERION APPLIED IN THAT PASS — the outputs "
                "were not checked for physical sanity, and a "
                "2026-08-03 audit re-run found THREE templates "
                "that exit 0 while producing wrong numbers. Do "
                "not treat 'rc=0' from this catalog as evidence "
                "that a template is correct:\n"
                "  - dg_methods/2d: prints 'DG advection-"
                "diffusion solved' and u: min=inf, max=inf. The "
                "raw PETSc KSP returns converged reason -11 "
                "(KSP_DIVERGED_PC_FAILED) and the script never "
                "inspects it.\n"
                "  - mixed_poisson/2d: imposes sigma.n = 0 on the "
                "ENTIRE boundary, so the pressure is determined "
                "only up to a constant and no nullspace is "
                "attached; it reports min(p) = max(p) = "
                "-1.035309e+13 (KSP reason 4, LU on a singular "
                "saddle-point system).\n"
                "  - eigenvalue/2d: assembles BOTH A and the mass "
                "matrix M with the same Dirichlet diagonal "
                "(diag=1.0) and asks for SMALLEST_REAL, so the "
                "two lowest reported eigenvalues are the spurious "
                "constraint modes 1.0000000000003 and "
                "1.0000000000031; the true fundamental 19.79 is "
                "third. This is exactly the failure documented in "
                "the eigenvalue pitfall of this catalog, and the "
                "generator's inline comment claiming diag=0.0 "
                "leaves M singular is wrong — assembling M with "
                "diag=0.0, or with bcs=[], both give the clean "
                "spectrum 19.82 / 49.71 / 49.92 / 80.30.\n"
                "Timing note: navier_stokes/3d converges (3 "
                "Newton iterations, 49072 dofs). The earlier "
                "claim that it 'needs more than 300 s' is "
                "hardware-dependent and did not reproduce — it "
                "completed in 230 s on the audit machine. Budget "
                "by measurement, not by this number."
            ),
            "corrections_landed": (
                "element_catalog (CR cells, Hermite api, Bubble "
                "hex minimum, chebyshev variants, iso cells, HHJ "
                "cells, pyramid degree, 'CG' vs 'P1'), "
                "mesh_catalog (gmshio module removed, refine "
                "prerequisites and return shapes, vtkhdf "
                "writing), solver_catalog "
                "(assemble_matrix_block/nest removed, "
                "bddc/fieldsplit/ams need extra setup, "
                "NewtonSolver hard failure), boundary_conditions "
                "(connectivity scope, BC rank-mismatch message, "
                "DG no-op), io_catalog (VTX exact message), "
                "poisson (VTX message, complex messages, "
                "exterior_facet_indices exception), "
                "linear_elasticity (plane-stress factor, locking "
                "magnitude, 'CG' family, BC message), stokes "
                "(nullspace helper does not exist), "
                "convection_diffusion (oscillation damping, "
                "constant-tau stagnation), biharmonic (alpha "
                "behaviour, hard-coded h, no H2 "
                "NotImplementedError), helmholtz (no convergence "
                "at all for k*h>1), "
                "nearly_incompressible_elasticity (1/(1-2nu) "
                "rule), hyperelasticity (NewtonSolver-era "
                "signals, P2 locking magnitude), maxwell "
                "(complex message), parallel_computing "
                "(assemble_scalar is not collective), plus the "
                "generator-level catalogs for dg_methods, "
                "mixed_poisson, nonlinear_pde and magnetostatics."
            ),
            "not_re-run": (
                "Claims requiring hardware/scale or absent "
                "packages were left with their inherited tag and "
                "NOT upgraded: fieldsplit iteration-count "
                "scaling beyond ~100k dofs, GAMG near-nullspace "
                "benefit at large scale, complex-PETSc "
                "behaviour (this build is real), "
                "dolfinx_mpc / adios4dolfinx / pyamg behaviour "
                "(not installed), demo_catalog and "
                "tutorial_catalog URLs (documentation links, not "
                "executed), and the hyperelastic Cook-membrane "
                "locking figure (only its linear analogue was "
                "measured)."
            ),
        },
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # ADVANCED: MULTIPHYSICS ON SUBMESHES
    # ═══════════════════════════════════════════════════════════════════════════
    "multiphysics_submeshes": {
        "description": "Solving PDEs on subdomains with different physics using DOLFINx submeshes (0.10+ feature).",
        "demo_url": "https://jsdokken.com/FEniCS-workshop/src/multiphysics/submeshes.html",
        "approach": {
            "create_submesh": "Extract subdomain mesh from parent mesh",
            "restriction": "Integration over subdomains using measures dx(marker)",
            "coupling": "Transfer data between submeshes via interpolation or shared DOFs",
        },
        "use_cases": "Different materials, different physics (FSI), domain decomposition",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # ADVANCED: OPTIMAL CONTROL / ADJOINT
    # ═══════════════════════════════════════════════════════════════════════════
    "optimal_control": {
        "description": "PDE-constrained optimization and adjoint methods in FEniCSx.",
        "demo_url": "https://jsdokken.com/FEniCS-workshop/src/applications/optimal_control.html",
        "approach": {
            "derive_adjoint": "Use UFL adjoint() and action() to derive adjoint PDE",
            "interface_scipy": "Extract gradient via adjoint solve, pass to scipy.optimize for minimization",
            "dolfin_adjoint": "Algorithmic differentiation tool (github.com/dolfin-adjoint/dolfin-adjoint) — automatic tape-based AD",
        },
        "use_cases": "Shape optimization, topology optimization, parameter estimation, inverse problems",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # ADVANCED: COMPLEX-VALUED PROBLEMS
    # ═══════════════════════════════════════════════════════════════════════════
    "complex_valued": {
        "description": "Solving PDEs with complex-valued solutions in DOLFINx (Helmholtz, Maxwell, wave scattering).",
        "demo_url": "https://jsdokken.com/dolfinx-tutorial/chapter1/complex_mode.html",
        "scalar_types": {
            "float32": "Single precision real",
            "float64": "Double precision real (default)",
            "complex64": "Single precision complex",
            "complex128": "Double precision complex",
        },
        "api": "dolfinx.default_scalar_type — check/switch between real/complex builds",
        "demo_types_url": "https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_types.html",
        "pitfalls": [
            "PETSc must be compiled with --with-scalar-type=complex for complex problems",
            "Cannot mix real and complex in same session — it is a build-time choice",
            "Some solvers (CG) do not work with complex arithmetic — use GMRES",
            "inner(a,b) in UFL conjugates the second argument for complex-valued problems",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # ADVANCED: PARALLEL COMPUTING
    # ═══════════════════════════════════════════════════════════════════════════
    "parallel_computing": {
        "description": "MPI-based parallel computing in DOLFINx. First-class parallel from ground up.",
        "api": {
            "communicator": "All mesh/solver creation takes MPI.COMM_WORLD (or sub-communicator)",
            "run": "mpirun -np N python script.py",
            "partitioning": "Automatic mesh partitioning on creation (configurable partitioner)",
            # 2026-08-03: this said assemble_scalar "sums across ranks
            # automatically". It does NOT. dolfinx's own docstring: "The
            # returned value is local and not accumulated across processes."
            # Measured on 2 ranks, assembling 1*dx over a 16x16 unit square:
            # rank 0 -> 0.505859, rank 1 -> 0.494141 (true area 1.0).
            "assembly": "dolfinx.fem.assemble_scalar() returns the RANK-LOCAL contribution ONLY — it does NOT reduce across ranks. You MUST wrap it: comm.allreduce(fem.assemble_scalar(fem.form(M)), op=MPI.SUM). (Verified empirically 2026-08-03 on 2 ranks: 0.505859 + 0.494141 for a functional whose true value is 1.0. Any error norm / integral computed without the allreduce is silently wrong in parallel.)",
        },
        "performance": {
            "scaling": "Strong and weak scaling demonstrated up to thousands of cores",
            "mesh_partitioning": "Graph-based (ParMETIS, SCOTCH, or KaHIP) for load balancing",
            "ghost_layer": "DOLFINx manages ghost cells/DOFs automatically",
            "neighbourhood_collectives": "MPI Neighbourhood collectives for efficient halo exchange",
        },
        "pitfalls": [
            "[Numerical] fem.assemble_scalar is NOT collective — it returns each rank's local piece, so forgetting comm.allreduce(..., op=MPI.SUM) makes every L2 error, energy and volume integral wrong in parallel while looking perfectly plausible in serial. Signal: running fem.assemble_scalar(fem.form(1.0*ufl.dx(domain=msh))) on a 16x16 unit square under mpirun -np 2 returns 0.505859 on rank 0 and 0.494141 on rank 1 instead of 1.0 on both; dolfinx's own docstring states 'The returned value is local and not accumulated across processes.' (Verified empirically 2026-08-03.)",
            "MUST use MPI communicator consistently — do not mix serial and parallel operations",
            "Output: only rank 0 should print; use if MPI.COMM_WORLD.rank == 0:",
            "Some operations (e.g., Gmsh model creation) should be done on rank 0 only",
            "pyamg is serial-only — use PETSc AMG for parallel",
            "Function evaluation at points requires parallel geometric search (BoundingBoxTree)",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # API CHANGES TRACKER (0.9 -> 0.10)
    # ═══════════════════════════════════════════════════════════════════════════
    "api_changes": {
        "description": "Critical API changes between DOLFINx versions. Essential for writing version-portable code.",
        "0_9_to_0_10": {
            "NewtonSolver_deprecated": "dolfinx.nls.petsc.NewtonSolver deprecated -> use dolfinx.fem.petsc.NonlinearProblem wrapping PETSc SNES directly",
            "gmsh_module_renamed": "dolfinx.io.gmshio -> dolfinx.io.gmsh (module rename)",
            "gmsh_returns_MeshData": "model_to_mesh() returns MeshData dataclass (with cell_tags, facet_tags by codimension) instead of tuple",
            "LinearProblem_blocked": "dolfinx.fem.petsc.LinearProblem now supports blocked problems (kind='mpi' or kind='nest')",
            "ZeroBaseForm": "ufl.ZeroBaseForm removes need for dummy 0*v*dx to compile empty forms",
            "uniform_refine": "dolfinx.mesh.uniform_refine() added (all CellTypes supported)",
            "vtkhdf_reader": "dolfinx.io.vtkhdf.read_mesh() added (Kitware's next-gen format)",
            "branching_meshes": "T-joints (3+ cells per facet) now supported as input meshes",
        },
        "0_7_to_0_8": {
            "basix_ufl_element": "Use basix.ufl.element() instead of ufl.FiniteElement()",
            "mixed_element": "Use basix.ufl.mixed_element() instead of ufl.MixedElement()",
            "blocked_element": "Use basix.ufl.blocked_element() for vector/tensor elements",
            "functionspace": "fem.functionspace() (lowercase) replaces fem.FunctionSpace()",
        },
        "0_9_to_0_10_MEASURED_ADDITIONS": {
            # 2026-08-03: found by executing against 0.10.0; these were
            # not in the tracker and each one breaks 0.9-era code.
            "gmshio_module_removed": "`import dolfinx.io.gmshio` now raises ModuleNotFoundError — it is not an alias, the module is gone. Use dolfinx.io.gmsh.",
            "assemble_matrix_block_nest_removed": "dolfinx.fem.petsc.assemble_matrix_block / assemble_matrix_nest no longer exist; use assemble_matrix(..., kind='mpi'|'nest') or LinearProblem(..., kind=...).",
            "NewtonSolver_hard_break": "dolfinx.nls.petsc.NewtonSolver still imports and warns, but wrapping a 0.10 NonlinearProblem raises AttributeError: 'NonlinearProblem' object has no attribute 'a'. The 0.9 two-step Newton pattern is dead code, not merely deprecated.",
            "interpolation_points_removed": "basix.ufl elements have no `interpolation_points` attribute at all (neither property nor method) — AttributeError: '_BasixElement' object has no attribute 'interpolation_points'. The points are element.basix_element.points.",
            "refine_needs_entities": "dolfinx.mesh.uniform_refine / refine require mesh.topology.create_entities(1) first, and refine returns a 3-tuple (Mesh, parent_cells, parent_facets).",
            "create_form_signature": "dolfinx.fem.create_form(form, function_spaces, msh, subdomains, coefficient_map, constant_map, entity_maps=None) — no parent_mesh= / coefficients= / constants= kwargs.",
            "vtkhdf_writing": "dolfinx.io.vtkhdf now exports write_mesh / write_point_data / write_cell_data, not just read_mesh.",
        },
        "pitfalls": [
            "Online tutorials may use old API (ufl.FiniteElement, FunctionSpace) — translate to new API. ufl.FiniteElement / VectorElement / MixedElement are REMOVED, so old scripts fail with AttributeError on the ufl module rather than a deprecation warning (verified 2026-08-03)",
            "The jsdokken tutorial is updated for latest version — use it as primary reference",
            "DOLFINx version in Docker images may differ from pip install — check dolfinx.__version__",
            "There is no errornorm helper in dolfinx or ufl — assemble inner(uh-uex, uh-uex)*dx yourself, and remember to comm.allreduce the result in parallel (verified 2026-08-03)",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # DEMO CATALOG — All official DOLFINx demos
    # ═══════════════════════════════════════════════════════════════════════════
    "demo_catalog": {
        "description": "Complete catalog of official DOLFINx demos (docs.fenicsproject.org/dolfinx/main/python/demos.html).",
        "demos": {
            "demo_poisson": "Poisson equation — fundamental elliptic PDE",
            "demo_mixed-poisson": "Mixed Poisson with Raviart-Thomas elements and block preconditioner",
            "demo_stokes": "Stokes equations with Taylor-Hood elements",
            "demo_navier-stokes": "Divergence-conforming DG for Navier-Stokes",
            "demo_elasticity": "Linear elasticity with algebraic multigrid (GAMG)",
            "demo_static-condensation": "Static condensation of mixed elasticity (Cook's membrane)",
            "demo_cahn-hilliard": "Cahn-Hilliard phase-field equation (spinodal decomposition)",
            "demo_biharmonic": "Biharmonic equation with interior penalty DG",
            "demo_helmholtz": "Helmholtz equation (complex-valued)",
            "demo_scattering_boundary_conditions": "EM scattering from wire (scattering BCs)",
            "demo_pml": "EM scattering from wire (perfectly matched layer)",
            "demo_half_loaded_waveguide": "Electromagnetic modal analysis (SLEPc eigenvalue)",
            "demo_axis": "Axisymmetric EM scattering from sphere",
            "demo_poisson_matrix_free": "Matrix-free CG solver for Poisson",
            "demo_types": "Solving PDEs with different scalar types (float32/64, complex64/128)",
            "demo_lagrange_variants": "Lagrange element variants (equispaced, GLL, Chebyshev)",
            "demo_gmsh": "Mesh generation with Gmsh integration",
            "demo_interpolation-io": "Interpolation and I/O operations",
            "demo_pyvista": "Visualization with PyVista",
            "demo_pyamg": "Poisson and elasticity with pyamg (serial AMG)",
        },
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # TUTORIAL CATALOG — jsdokken FEniCSx tutorial chapters
    # ═══════════════════════════════════════════════════════════════════════════
    "tutorial_catalog": {
        "description": "Complete catalog of jsdokken.com/dolfinx-tutorial chapters.",
        "chapter1_fundamentals": {
            "fundamentals": "Solving the Poisson equation — basic FEniCSx workflow",
            "complex_mode": "Poisson with complex numbers",
        },
        "chapter2_gallery": {
            "heat_equation": "Transient heat equation (backward Euler)",
            "diffusion_code": "Diffusion of a Gaussian function",
            "nonlinpoisson": "Nonlinear Poisson (Newton method)",
            "linearelasticity": "Linear elasticity (cantilever beam)",
            "hyperelasticity": "Hyperelasticity (Neo-Hookean beam bending)",
            "navierstokes": "Navier-Stokes theory (IPCS splitting)",
            "ns_code1": "Channel flow (Poiseuille, IPCS)",
            "ns_code2": "Flow past cylinder (DFG 2D-3 benchmark)",
        },
        "chapter3_bcs_subdomains": {
            "neumann_dirichlet": "Combining Dirichlet and Neumann BCs",
            "robin_neumann_dirichlet": "Multiple Dirichlet, Neumann, and Robin conditions",
            "multiple_dirichlet": "Setting multiple Dirichlet conditions",
            "component_bc": "Component-wise Dirichlet BC (vector problems)",
            "subdomains": "Defining subdomains for different materials",
            "em": "Electromagnetics example (curl-curl with subdomains)",
        },
        "chapter4_advanced": {
            "solvers": "Solver configuration (PETSc options)",
            "newton_solver": "Custom Newton solver implementation",
            "compiler_parameters": "JIT options and visualization (Pandas)",
            "convergence": "Error control — computing convergence rates",
        },
        "fenics_workshop": {
            "url": "https://jsdokken.com/FEniCS-workshop/",
            "topics": "UFL elements, form compilation, advanced elements (Nedelec, RT), mixed problems, restriction/submeshes, optimal control, multiphysics",
        },
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# DEAL.II — COMPREHENSIVE DOMAIN KNOWLEDGE
# ═══════════════════════════════════════════════════════════════════════════════

_DEALII_KNOWLEDGE = {
    "poisson": {
        "description": "Poisson equation solved with deal.II (step-3/4/5). Foundation of all elliptic PDEs.",
        "tutorial_steps": {"step-3": "Basic Poisson on hyper_cube", "step-4": "Dim-independent with non-constant coefficients", "step-5": "Adaptive refinement with Kelly estimator", "step-6": "Higher-order elements + automatic adaptivity", "step-7": "Helmholtz + convergence tables"},
        "function_space": "FE_Q<dim>(degree) — tensor-product Lagrange on quads/hexes",
        "element_catalog": {
            "FE_Q(1)": "Bilinear (2D) / trilinear (3D), standard choice",
            "FE_Q(2)": "Biquadratic, better accuracy for smooth solutions",
            "FE_SimplexP(1)": "Linear on triangles/tets (for simplex meshes)",
            "FE_DGQ(p)": "Discontinuous Galerkin variant",
        },
        "solver": {
            "small": "SolverCG + PreconditionSSOR (or PreconditionIdentity for debugging)",
            "medium": "SolverCG + SparseMIC (incomplete Cholesky)",
            "large": "SolverCG + TrilinosWrappers::PreconditionAMG (algebraic multigrid)",
            "matrix_free": "SolverCG + PreconditionChebyshev (step-37 pattern, fastest)",
        },
        "grid_generators": {
            "hyper_cube": "[0,1]^dim, all boundary_id=0 (use colorize=true for distinct IDs)",
            "hyper_rectangle": "Box [p1,p2], boundary_ids: 0=left,1=right,2=bottom,3=top,4=back,5=front",
            "subdivided_hyper_rectangle": "Box with per-axis subdivision control",
            "hyper_ball": "Circular disk / ball with SphericalManifold",
            "hyper_shell": "Annulus / spherical shell (inner/outer radius)",
            "hyper_L": "L-shaped domain — classic corner singularity benchmark",
            "plate_with_a_hole": "Rectangle with cylindrical hole — stress concentration",
            "channel_with_cylinder": "Flow channel with obstacle — DFG benchmark geometry",
            "cheese": "Rectangle with square holes",
            "hyper_cube_slit": "Square with slit for singularity testing",
        },
        "output": "DataOut → VTU (standard), also VTK, gnuplot, SVG",
        "pitfalls": [
            "Call triangulation.refine_global() BEFORE distributing DOFs",
            "Boundary IDs on hyper_cube: ALL faces = 0 by default; use colorize=true or hyper_rectangle",
            "hyper_rectangle colorized: left=0, right=1, bottom=2, top=3, back=4, front=5",
            "Use DynamicSparsityPattern → copy_from → SparsityPattern (two-step)",
            "QGauss degree should be fe.degree + 1 for optimal convergence",
            "For Neumann-only: solution up to constant — need mean-value constraint",
            "Hanging node constraints MUST be applied on adaptively refined meshes (AffineConstraints)",
            "Forgetting update_values|update_gradients|update_JxW_values in FEValues → silent wrong results",
            "DataOut: must call build_patches() before writing",
        ],
    },
    "linear_elasticity": {
        "description": "Linear elasticity (step-8/17). Vector-valued FESystem with Lamé parameters.",
        "tutorial_steps": {
            "step-8": "Elasticity with FESystem, body forces, component-wise assembly",
            "step-17": "Parallel elasticity with PETSc",
            "step-18": "Quasi-static large-deformation (incremental loading, Lagrangian mesh)",
            "step-44": "Nonlinear solid mechanics — compressible Neo-Hookean, three-field formulation",
        },
        "function_space": "FESystem<dim>(FE_Q<dim>(1), dim) — vector Lagrange",
        "constitutive": {
            "lame": "mu = E/(2(1+nu)), lambda = E*nu/((1+nu)(1-2*nu))",
            "plane_stress": "lambda_star = 2*mu*lambda/(2*mu + lambda)",
        },
        "solver": {
            "small": "SolverCG + PreconditionSSOR",
            "large": "SolverCG + TrilinosWrappers::PreconditionAMG (provide rigid body modes for near-nullspace!)",
        },
        "pitfalls": [
            "Use system_to_component_index() to map local DOF to physical component",
            "For plane stress: use modified lambda_star = 2*mu*lambda/(2*mu + lambda)",
            "Near-incompressible (nu→0.5): MUST use mixed methods to avoid volumetric locking",
            "Providing rigid body modes to AMG dramatically improves convergence for elasticity",
            "Component mask needed for applying BC to individual displacement components",
            "VectorTools::interpolate_boundary_values needs ZeroFunction<dim>(dim) for vector BC",
            "Boundary IDs depend on GridGenerator — check docs for each generator",
        ],
    },
    "heat": {
        "description": "Heat equation — transient diffusion (step-26). Adaptive mesh in time.",
        "tutorial_steps": {
            "step-26": "Transient heat with adaptive mesh refinement, solution interpolation between meshes",
            "step-86": "Heat equation with PETSc time-stepping (TS) framework",
        },
        "function_space": "FE_Q<dim>(1) or FE_Q<dim>(2) — scalar Lagrange",
        "time_integration": "Backward Euler (stable) or Crank-Nicolson (2nd order, theta=0.5)",
        "solver": "SolverCG + PreconditionSSOR per time step",
        "pitfalls": [
            "Mass matrix assembly needed for transient terms",
            "When using adaptive refinement in time: MUST interpolate solution from old to new mesh",
            "Lumped mass matrix can introduce oscillations near steep gradients",
            "Initial condition via VectorTools::interpolate or VectorTools::project",
        ],
    },
    "stokes": {
        "description": "Stokes flow (step-22). Mixed FE with Schur complement preconditioning.",
        "tutorial_steps": {
            "step-22": "Stokes with block preconditioner, Schur complement",
            "step-45": "Parallel Stokes with periodic BCs using Trilinos",
            "step-55": "Parallel Stokes with AMG for velocity block",
            "step-56": "Stokes with geometric multigrid",
        },
        "function_space": "Taylor-Hood: FESystem(FE_Q<dim>(2)^dim, FE_Q<dim>(1)) — Q2/Q1",
        "solver": {
            "recommended": "SolverGMRES or SolverMinRes with block preconditioner",
            "block_precon": "AMG for velocity block + pressure mass matrix for Schur complement",
            "alternative_elements": "FE_BernardiRaugel + FE_DGP(0) for low-order stable pair",
        },
        "pitfalls": [
            "MUST use inf-sup stable pair — Q1/Q1 (equal-order) is UNSTABLE",
            "Taylor-Hood Q2/Q1 is the standard stable pair",
            "Pressure unique only up to constant for enclosed flows — pin one pressure DOF",
            "Schur complement preconditioning essential for efficiency at scale",
            "Pressure mass matrix is a good Schur complement approximation",
        ],
    },
    "navier_stokes": {
        "description": "Navier-Stokes (step-57). Nonlinear extension of Stokes with Newton iteration.",
        "tutorial_steps": {
            "step-57": "Stationary incompressible NS, Newton + continuation in Reynolds number",
            "step-35": "NS via projection/pressure-correction method (time-dependent)",
        },
        "function_space": "Same as Stokes: Taylor-Hood Q2/Q1",
        "solver": "Newton outer loop + direct solve (UMFPACK) per Newton step for small problems",
        "pitfalls": [
            "Newton convergence depends critically on initial guess — use continuation in Re",
            "Start from Stokes solution (Re→0) and gradually increase Re",
            "For Re > ~500, need very fine mesh or stabilization",
        ],
    },
    "advection_dg": {
        "description": "Advection with DG elements (step-9/12). Discontinuous Galerkin for transport.",
        "tutorial_steps": {
            "step-9": "Advection with DG-like stabilization + adaptive refinement",
            "step-12": "DG for linear advection with MeshWorker framework",
            "step-30": "Anisotropic mesh refinement for DG advection",
        },
        "function_space": "FE_DGQ<dim>(p) — discontinuous Lagrange, degree 1-3",
        "solver": "SolverGMRES + PreconditionBlockJacobi (ILU per block)",
        "pitfalls": [
            (
                "[API] Sparsity pattern must include face-coupling: "
                "DoFTools::make_flux_sparsity_pattern(). Signal: "
                "using the regular make_sparsity_pattern() on a DG "
                "discretization gives a matrix with missing off-"
                "diagonal entries for face-coupling DOFs; assembly "
                "then aborts with `SparseMatrix::add() requires "
                "row/col to be in pattern` for every facet "
                "contribution. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Interior penalty parameter (alpha "
                "fed to MeshWorker / FEInterfaceValues face "
                "integrators) must be large enough for stability "
                "(scales with p^2). Signal: alpha too small -> "
                "coercivity loss and the computed L^2 norm from "
                "VectorTools::integrate_difference diverges with "
                "mesh refinement; alpha too large -> condition "
                "number > 1e14 and SolverGMRES stagnates. Rule: "
                "alpha = 4 * (p+1)^2 for SIPG with FE_DGQ<dim>(p). "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Face integrals require careful "
                "normal orientation handling via FEValues / "
                "FEInterfaceValues. Signal: FEFaceValues / "
                "FEInterfaceValues evaluated on the (-) side "
                "gives normal pointing FROM (-) TO (+); "
                "swapping +/- in the jump integral gives a "
                "SIGN ERROR — the assembled SparseMatrix is "
                "the TRANSPOSE of what was intended, and "
                "convergence rate degrades from O(h^(p+1)) to "
                "O(1) (no convergence, diagnosable via "
                "KellyErrorEstimator). (Audit 2026-06-02.)"
            ),
            (
                "[Performance] Streamline ordering of DOFs can "
                "help GMRES convergence. Signal: default DoF "
                "renumbering (Cuthill-McKee) gives GMRES "
                "convergence in ~50-100 iters for advection-"
                "dominated flow; switching to "
                "DoFRenumbering::downstream(b) cuts it to ~10-20 "
                "iters because the upwind sweep matches the "
                "matrix sparsity structure. (Audit 2026-06-02.)"
            ),
        ],
    },
    "wave_equation": {
        "description": "Wave equation (step-23/24/25). Time-dependent hyperbolic PDE.",
        "tutorial_steps": {
            "step-23": "Wave equation in bounded domain",
            "step-24": "Thermoacoustic tomography with absorbing BCs",
            "step-25": "Nonlinear wave (sine-Gordon soliton)",
            "step-48": "Parallel wave equation, matrix-free",
            "step-62": "Elastic wave propagation in phononic crystals",
        },
        "function_space": "FE_Q<dim>(1) — scalar Lagrange per time step",
        "solver": "SolverCG + PreconditionJacobi per time step (mass matrix is SPD)",
    },
    "nonlinear_elasticity": {
        "description": "Nonlinear solid mechanics (step-44). Neo-Hookean, three-field formulation.",
        "tutorial_steps": {"step-44": "Compressible Neo-Hookean with quasi-incompressible three-field formulation"},
        "function_space": "FESystem for displacement + pressure + dilatation (3-field)",
        "solver": "Newton iteration with direct solver",
        "pitfalls": [
            (
                "[Numerical] Three-field formulation needed for "
                "quasi-incompressible materials. Signal: single-"
                "field displacement formulation locks for nu > "
                "0.49 — incompressible Neo-Hookean block under "
                "uniaxial extension shows displacement ~500x too "
                "small. step-44 uses (u, p_tilde, J_tilde) three-"
                "field FESystem to recover the correct response. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Newton convergence requires good "
                "initial guess and small load steps. Signal: a "
                "single load step from zero to full deformation "
                "diverges within 2-3 Newton iterations (visible "
                "in SolverControl::log_history) for stretch "
                "ratios > 1.1; subdividing into 10-20 load "
                "increments with the previous AffineConstraints-"
                "constrained solution as initial guess brings "
                "each step inside Newton's convergence basin. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Performance] Automatic differentiation "
                "(step-71/72) avoids hand-coding Jacobians. "
                "Signal: hand-coded tangent for Mooney-Rivlin or "
                "Holzapfel-Gasser-Ogden has dozens of lines of "
                "tensor index arithmetic — easy to miss a term "
                "and get linear-rate Newton convergence; "
                "Differentiation::AD::ResidualLinearization "
                "(Sacado backend) generates the exact tangent "
                "and restores quadratic convergence. (Audit "
                "2026-06-02.)"
            ),
        ],
    },
    "compressible_euler": {
        "description": "Compressible Euler equations (step-33/67/69). Hyperbolic conservation laws.",
        "tutorial_steps": {
            "step-33": "Compressible Euler, basic conservation law framework",
            "step-67": "High-order DG + explicit time stepping + matrix-free (fastest)",
            "step-69": "Euler with first-order viscous stabilization",
            "step-76": "Cell-centric matrix-free with MPI-3.0 shared memory",
        },
        "function_space": "FE_DGQ<dim>(2-5) — high-order DG",
        "solver": "Explicit Runge-Kutta (no linear solve needed, matrix-free)",
        "pitfalls": [
            (
                "[Numerical] MUST use DG elements — continuous "
                "(CG / FE_Q) elements are unstable for Euler. "
                "Signal: a Sod shock-tube benchmark with FE_Q "
                "develops uncontrolled oscillations that propagate "
                "across the domain within a few time steps; the "
                "same setup with FE_DGQ produces sharp shocks. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Numerical flux choice: Lax-Friedrichs "
                "(simple), HLLC (better shock resolution). Signal: "
                "Lax-Friedrichs over-smears contact discontinuities "
                "in a Sod problem by ~30% of the analytical jump; "
                "HLLC resolves them to <5% smearing. Use LF for "
                "robustness, HLLC for accuracy. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] CFL condition mandatory for explicit "
                "time stepping. Signal: dt > h / (|u| + c) gives "
                "NaN within ~10 steps because the explicit "
                "stencil cannot propagate information faster than "
                "one element per step. CFL safety factor ~0.3 is "
                "typical for SSP-RK3. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Shock capturing / slope limiting "
                "needed for discontinuous solutions. Signal: "
                "high-order DG without limiting produces "
                "Gibbs-style oscillations around shocks (over/"
                "undershoot 5-20% of the jump); applying a "
                "TVB-Minmod limiter or entropy-viscosity "
                "stabilisation eliminates them at the cost of "
                "local accuracy reduction near the shock. (Audit "
                "2026-06-02.)"
            ),
        ],
    },
    "contact": {
        "description": "Contact / variational inequalities (step-41/42). Active set strategy.",
        "tutorial_steps": {
            "step-41": "Obstacle problem (variational inequality)",
            "step-42": "3D elasto-plastic contact with isotropic hardening (parallel)",
        },
        "solver": "Projected CG with AMG preconditioner + active set iteration",
        "pitfalls": [
            (
                "[Numerical] Active set changes require iterating "
                "between constraint detection (AffineConstraints / "
                "PETScWrappers::MPI::Vector test) and SolverGMRES "
                "solve. Signal: a single-shot SolverCG / SolverGMRES "
                "call where the active set is predicted from the "
                "initial guess gives the wrong contact zone for "
                "typical Hertz benchmarks (~30-50% wrong contact "
                "radius); the outer loop should iterate until two "
                "consecutive active sets are identical, usually "
                "3-10 iterations. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Penalty parameter: too small = "
                "constraint violation, too large = ill-"
                "conditioning. Signal: too low -> max penetration "
                "> 5% element edge; too high -> SolverGMRES "
                "stagnates with condition number > 1e14. Rule of "
                "thumb: penalty = 1e3 * E / h for typical Hertz "
                "contact with elastic Young modulus E. (Audit "
                "2026-06-02.)"
            ),
            (
                "[API] Use AffineConstraints to enforce contact "
                "constraints. Signal: hand-modifying the system "
                "matrix (zeroing rows + setting diag = 1) for "
                "constrained nodes is brittle and breaks "
                "parallel assembly. AffineConstraints<double> "
                "with constraints.add_line + add_entries handles "
                "the matrix modifications consistently across "
                "MPI ranks. (Audit 2026-06-02.)"
            ),
        ],
    },
    "grid_generator_catalog": {
        "description": "Complete catalog of deal.II GridGenerator functions for mesh creation.",
        "generators": {
            "hyper_cube": {"geometry": "Unit cube [0,1]^dim", "dims": "1D,2D,3D", "boundary_ids": "All = 0 (colorize=true for distinct)"},
            "hyper_rectangle": {"geometry": "Axis-aligned box [p1,p2]", "dims": "1D,2D,3D", "boundary_ids": "x-=0,x+=1,y-=2,y+=3,z-=4,z+=5"},
            "subdivided_hyper_rectangle": {"geometry": "Box with per-axis subdivision control", "dims": "1D,2D,3D"},
            "hyper_ball": {"geometry": "Circular disk / ball", "dims": "2D,3D", "notes": "SphericalManifold attached"},
            "hyper_shell": {"geometry": "Annulus / spherical shell", "dims": "2D,3D", "notes": "Inner + outer radius"},
            "hyper_L": {"geometry": "L-shaped domain", "dims": "2D", "notes": "Classic corner singularity benchmark"},
            "plate_with_a_hole": {"geometry": "Rectangle with cylindrical hole", "dims": "2D", "notes": "Stress concentration factor"},
            "channel_with_cylinder": {"geometry": "Flow channel with obstacle", "dims": "2D,3D", "notes": "DFG benchmark (Schäfer-Turek)"},
            "cylinder": {"geometry": "Cylinder (circular cross-section)", "dims": "3D"},
            "cylinder_shell": {"geometry": "Hollow cylinder (pipe wall)", "dims": "3D"},
            "truncated_cone": {"geometry": "Cone frustum", "dims": "3D"},
            "cheese": {"geometry": "Rectangle with square holes", "dims": "2D,3D"},
            "hyper_cross": {"geometry": "Cross/plus shape", "dims": "2D,3D"},
            "pipe_junction": {"geometry": "Pipe bifurcation", "dims": "3D"},
            "Airfoil::create_triangulation": {"geometry": "NACA/Joukowski airfoil", "dims": "2D"},
            "extrude_triangulation": {"geometry": "Extrude 2D → 3D", "notes": "Layered 3D from 2D base"},
            "merge_triangulations": {"geometry": "Union of two meshes", "notes": "Combine separate grids"},
        },
    },
    "solver_catalog": {
        "description": "Complete deal.II solver and preconditioner catalog.",
        "solvers": {
            "SolverCG": "Conjugate Gradient — SPD systems (Poisson, elasticity, heat)",
            "SolverGMRES": "Restarted GMRES — non-symmetric (advection, NS)",
            "SolverFGMRES": "Flexible GMRES — variable preconditioner per iteration",
            "SolverBicgstab": "BiCGStab — non-symmetric alternative",
            "SolverMinRes": "MinRes — symmetric indefinite (Stokes, saddle-point)",
            "SparseDirectUMFPACK": "Direct — small/medium, complex-valued, debugging",
        },
        "preconditioners": {
            "PreconditionIdentity": "None — debugging only",
            "PreconditionJacobi": "Diagonal scaling — DG mass matrices",
            "PreconditionSSOR": "Symmetric SOR — CG-compatible, general purpose",
            "PreconditionChebyshev": "Polynomial — matrix-free multigrid smoothers (step-37)",
            "SparseMIC": "Incomplete Cholesky — SPD systems",
            "SparseILU": "Incomplete LU — general non-symmetric",
            "TrilinosWrappers::PreconditionAMG": "Algebraic multigrid (ML/MueLu) — large elliptic/elasticity",
        },
        "by_physics": {
            "poisson": "CG + SSOR (small) or CG + AMG (large) or CG + Chebyshev+GMG (fastest)",
            "elasticity": "CG + AMG (provide rigid body modes for near-nullspace)",
            "heat_transient": "CG + SSOR per time step",
            "stokes": "GMRES/MinRes + block preconditioner (AMG for velocity, mass-matrix for Schur)",
            "navier_stokes": "GMRES + block precon, Newton outer loop",
            "advection_dg": "GMRES + ILU or block-Jacobi",
            "euler_dg": "Explicit RK (no linear solve) — matrix-free",
            "wave": "CG + Jacobi per time step (mass matrix is SPD)",
        },
    },
    "element_catalog": {
        "description": "Complete deal.II finite element catalog.",
        "elements": {
            "FE_Q(p)": {"type": "Lagrange Qp", "continuity": "C0", "use": "Poisson, heat, elasticity — standard choice"},
            "FE_DGQ(p)": {"type": "DG Lagrange", "continuity": "Discontinuous", "use": "Advection, Euler, transport"},
            "FESystem(FE_Q(p), dim)": {"type": "Vector Lagrange", "continuity": "C0", "use": "Elasticity, displacement"},
            "FE_RaviartThomas(p)": {"type": "H(div) conforming", "continuity": "Normal continuous", "use": "Darcy flow, mixed Poisson"},
            "FE_Nedelec(p)": {"type": "H(curl) conforming", "continuity": "Tangential continuous", "use": "Maxwell, electromagnetics"},
            "FE_SimplexP(p)": {"type": "Simplex Lagrange", "continuity": "C0", "use": "Triangle/tet meshes"},
            "FE_BernardiRaugel": {"type": "Enriched velocity", "continuity": "C0", "use": "Low-order inf-sup stable Stokes"},
            "FE_Bernstein(p)": {"type": "Bernstein polynomials", "continuity": "C0", "use": "Positivity-preserving"},
        },
    },
    "tutorial_catalog": {
        "description": "Complete deal.II tutorial step catalog — maps step numbers to physics types and key features.",
        "step-1": "Grid generation and output",
        "step-2": "DOF setup and sparsity patterns",
        "step-3": "Poisson equation (basic)",
        "step-4": "Non-constant coefficients (dim-independent)",
        "step-5": "Adaptive refinement (Kelly estimator)",
        "step-6": "Higher order elements + automatic adaptivity",
        "step-7": "Helmholtz + Neumann BCs + convergence tables",
        "step-8": "Elasticity (vector FE, FESystem)",
        "step-9": "Advection with DG + adaptive refinement",
        "step-12": "DG advection (MeshWorker framework)",
        "step-15": "Minimal surface (nonlinear, Newton)",
        "step-16": "Geometric multigrid for Laplace",
        "step-17": "Parallel elasticity with PETSc",
        "step-18": "Quasi-static large-deformation elasticity",
        "step-20": "Mixed Darcy flow (Raviart-Thomas)",
        "step-22": "Stokes flow (Schur complement preconditioning)",
        "step-23": "Wave equation (time-dependent hyperbolic)",
        "step-26": "Heat equation (transient, adaptive mesh in time)",
        "step-27": "hp-FEM (combined h- and p-refinement)",
        "step-29": "Complex Helmholtz / scattering",
        "step-31": "Boussinesq convection (2D)",
        "step-33": "Compressible Euler equations",
        "step-35": "Navier-Stokes (projection method)",
        "step-36": "Eigenvalue problems (SLEPc)",
        "step-37": "Matrix-free methods (Laplace, fastest pattern)",
        "step-40": "Parallel with Trilinos (distributed)",
        "step-41": "Obstacle / contact problem",
        "step-42": "3D elasto-plastic contact",
        "step-44": "Nonlinear solid mechanics (Neo-Hookean, 3-field)",
        "step-45": "Parallel Stokes with periodic BCs",
        "step-47": "Biharmonic / Kirchhoff plate (C0 interior penalty)",
        "step-49": "Complex mesh generation + external mesh import",
        "step-51": "HDG (hybridizable DG) for convection-diffusion",
        "step-55": "Parallel Stokes + AMG",
        "step-56": "Stokes with geometric multigrid",
        "step-57": "Navier-Stokes (stationary, Newton + continuation)",
        "step-59": "DG + matrix-free (interior penalty)",
        "step-62": "Elastic wave propagation (phononic crystals)",
        "step-67": "Compressible Euler (high-order DG, matrix-free, explicit RK)",
        "step-70": "Particle FSI (immersed boundary method)",
        "step-71": "Automatic differentiation (magneto-mechanical coupling)",
        "step-72": "AD for Jacobians (nonlinear PDEs)",
        "step-74": "SIPG DG for Poisson",
        "step-77": "SUNDIALS KINSOL nonlinear solver",
        "step-79": "Topology optimization (SIMP)",
        "step-81": "Time-harmonic Maxwell equations",
        "step-85": "CutFEM for Poisson on circular domain",
        "step-87": "Remote point evaluation on distributed meshes",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# FEBIO — REMOVED 2026-06-02
# The previous `_FEBIO_KNOWLEDGE` covered only 4 of febio's 16 live
# supported physics, while the febio backend's own generators carry
# matching `description` fields for all 16. workflows.py's merge
# (`dict(deep_knowledge); ... if k not in knowledge`) allowed the
# 4-key deep_knowledge entry to SHADOW the backend's larger catalog.
# The febio backend is now the single source of truth for febio
# knowledge — extend src/backends/febio/generators/ to grow it.
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-SOLVER VALIDATION KNOWLEDGE
# Verified results from 10 benchmarks across FEniCS, deal.II, and 4C.
# This knowledge helps fresh agents set up correct simulations and verify results.
# ═══════════════════════════════════════════════════════════════════════════════

_CROSS_SOLVER_KNOWLEDGE = {
    "cross_validation_principles": {
        "description": (
            "Cross-solver validation means running the same problem on multiple independent "
            "solvers and checking that they produce consistent results. This is a powerful "
            "verification technique — if two solvers agree, it's strong evidence both are correct."
        ),
        "methodology": [
            "Define the problem precisely (domain, BCs, material, source term)",
            "Run on 2+ solvers with comparable discretizations",
            "Compare key output quantities (max field value, tip displacement, etc.)",
            "Expect small differences (1-3%) from different element types — this is normal",
            "Large differences (>5%) indicate a setup error in one of the solvers",
        ],
    },
    "element_type_effects": {
        "description": (
            "Different solvers use different default element types. P1 triangles and Q1 "
            "quadrilaterals give slightly different results on the same mesh density. Both "
            "converge to the same solution under refinement. Differences of 1-3% between "
            "tri and quad elements are expected and normal — not a sign of error."
        ),
    },
    "4c_inline_mesh_notes": {
        "description": (
            "4C inline mesh (NODE COORDS + ELEMENTS) creates self-contained input files "
            "without external Exodus mesh dependencies."
        ),
        "key_pitfalls": [
            "Elasticity NUMDOF=3 even in 2D (z-dof constrained to 0)",
            "Element ordering: node IDs counter-clockwise for QUAD4",
            "IO/RUNTIME VTK OUTPUT section required for ParaView output",
        ],
    },
}
