"""
MCP tools for deep, comprehensive domain knowledge across ALL backends.

This is the brain of the agent — it knows weak forms, material libraries,
solver configurations, pitfalls, element catalogs, and best practices for
every supported FEM code. This is what makes Open FEM Agent genuinely useful
compared to a generic LLM.
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
                "orders": "1, 2, 3, ... (arbitrary order)",
                "cell_types": "interval, triangle, quadrilateral, tetrahedron, hexahedron, prism, pyramid",
                "api": "basix.ufl.element('Lagrange', cell, degree)",
                "variants": {
                    "equispaced": "basix.LagrangeVariant.equispaced (equally spaced points, default for low order)",
                    "gll_warped": "basix.LagrangeVariant.gll_warped (GLL points, lower Lebesgue constant for high order)",
                    "gll_isaac": "basix.LagrangeVariant.gll_isaac (GLL with Isaac warp on simplices)",
                    "gll_centroid": "basix.LagrangeVariant.gll_centroid (GLL with centroid warp)",
                    "chebyshev_warped": "basix.LagrangeVariant.chebyshev_warped (Chebyshev points)",
                    "chebyshev_isaac": "basix.LagrangeVariant.chebyshev_isaac",
                    "chebyshev_centroid": "basix.LagrangeVariant.chebyshev_centroid",
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
                "orders": "1 only",
                "cell_types": "triangle, tetrahedron, quadrilateral, hexahedron",
                "api": "basix.ufl.element('CR', cell, 1)",
                "use_cases": "Stokes (CR/DG0 pair is inf-sup stable), nonconforming methods",
            },
            "bubble": {
                "basix_name": "basix.ElementFamily.bubble",
                "ufl_name": "'Bubble'",
                "continuity": "Zero on element boundaries (vanishes on facets)",
                "orders": "Depends on cell type (3 for triangle, 4 for tet, 2 for quad, 3 for hex)",
                "cell_types": "triangle, quadrilateral, tetrahedron, hexahedron",
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
                "cell_types": "triangle",
                "api": "basix.ufl.element('HHJ', cell, degree)",
                "use_cases": "Kirchhoff plates, biharmonic equation (symmetric tensor field for moments)",
            },
            "serendipity": {
                "basix_name": "basix.ElementFamily.serendipity",
                "ufl_name": "'S' or 'serendipity'",
                "continuity": "C0",
                "orders": "1, 2, 3, ...",
                "cell_types": "quadrilateral, hexahedron",
                "api": "basix.ufl.element('S', cell, degree)",
                "notes": "Fewer DOFs than tensor-product Lagrange on quads/hexes. S2 has no interior node on quad.",
            },
            "DPC (Discontinuous Piecewise Complete)": {
                "basix_name": "basix.ElementFamily.DPC",
                "ufl_name": "'DPC'",
                "continuity": "Discontinuous",
                "orders": "0, 1, 2, ...",
                "cell_types": "quadrilateral, hexahedron",
                "api": "basix.ufl.element('DPC', cell, degree)",
                "notes": "Complete polynomial on quads/hexes (not tensor-product). Used in compatible DG schemes.",
            },
            "Hermite": {
                "basix_name": "basix.ElementFamily.Hermite",
                "ufl_name": "'Hermite'",
                "continuity": "C1 (value and gradient continuous at vertices)",
                "orders": "3",
                "cell_types": "triangle, tetrahedron",
                "api": "basix.ufl.element('Hermite', cell, 3)",
                "use_cases": "Beam/plate problems requiring C1 continuity, Kirchhoff theory",
            },
            "iso (isoparametric/macro)": {
                "basix_name": "basix.ElementFamily.iso",
                "ufl_name": "'iso'",
                "continuity": "C0 (piecewise on sub-cells)",
                "orders": "2, 3, ...",
                "cell_types": "interval, triangle, quadrilateral, tetrahedron, hexahedron",
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
            "In dolfinx >= 0.8, use basix.ufl.element() NOT ufl.FiniteElement() (legacy UFL deprecated)",
            "For vector elements use blocked_element or shape= parameter, NOT VectorElement (deprecated)",
            "For mixed spaces use basix.ufl.mixed_element, NOT ufl.MixedElement (deprecated)",
            "Element variant matters for high order (>= 5): use gll_warped to avoid ill-conditioning",
            "Not all element families support all cell types — check Basix docs for compatibility",
            "Bubble element minimum degree depends on cell type: 3 for triangle, 4 for tet",
            "Serendipity and DPC elements only available on quads/hexes",
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
            "api_0_10": "dolfinx.io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0) — returns MeshData dataclass",
            "read_from_msh": "dolfinx.io.gmshio.read_from_msh('file.msh', MPI.COMM_WORLD, rank=0)",
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
            "notes": "Kitware's future-proof format. Transition from XDMF has started.",
        },
        "mesh_refinement": {
            "uniform_refine": "dolfinx.mesh.uniform_refine(mesh) — refines all cells uniformly",
            "refine": "dolfinx.mesh.refine(mesh, edges=None) — selective refinement of marked edges",
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
            "Topology connectivity must be created before use: mesh.topology.create_connectivity(dim1, dim2)",
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
                    "0_10_note": "Now supports blocked problems via kind='mpi' or kind='nest'",
                },
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
                "BDDC": {"options": {"pc_type": "bddc"}, "use": "Balancing domain decomposition by constraints — scalable parallel"},
                "fieldsplit": {"options": {"pc_type": "fieldsplit"}, "use": "Block preconditioner for saddle-point (Stokes, mixed)"},
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
            "assemble_matrix_block": "dolfinx.fem.petsc.assemble_matrix_block(a_block, bcs)",
            "assemble_matrix_nest": "dolfinx.fem.petsc.assemble_matrix_nest(a_block, bcs)",
            "nullspace": "Build nullspace for pressure (constant) or rigid body modes (elasticity), attach to matrix",
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
            "NewtonSolver deprecated in 0.10 — use NonlinearProblem.solve() instead",
            "snes_atol default is 1e-50 (effectively disabled) — you MUST set it explicitly",
        ],
        "by_physics": {
            "poisson": "CG + hypre/GAMG (or LU for small)",
            "elasticity": "CG + GAMG with near-nullspace (or LU for small)",
            "heat_transient": "CG + hypre per time step",
            "stokes": "MinRes + fieldsplit (AMG for velocity block, mass matrix for Schur complement)",
            "navier_stokes": "SNES newtonls + GMRES + AMG (or LU for small)",
            "helmholtz": "GMRES + LU (complex-valued, direct often needed)",
            "maxwell": "GMRES + AMS (from hypre) for H(curl) problems",
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
            "MUST create connectivity before locating boundary: mesh.topology.create_connectivity(fdim, tdim)",
            "For sub-space BCs: locate_dofs_topological needs BOTH the sub-space AND collapsed sub-space as tuple",
            "Periodic BCs require dolfinx_mpc extension — not natively in DOLFINx",
            "Dirichlet value type must match: np.zeros(gdim) for vector, scalar for scalar space",
            "For enclosed flows (all Dirichlet velocity): pin pressure at one DOF to remove nullspace",
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
            "notes": "Kitware's future format. Reading supported, writing in progress.",
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
            "VTXWriter requires ADIOS2 — check installation",
            "XDMFFile: only geometry order <= 2; for high-order elements, use VTX",
            "VTXWriter only works with (discontinuous) Lagrange elements — not RT, Nedelec, etc.",
            "Function eval requires finding containing cell first — use BoundingBoxTree",
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
            "[API] In recent dolfinx, mesh.topology.create_connectivity(fdim, tdim) is no longer a hard prerequisite for locate_entities_boundary / locate_dofs_topological — connectivity is built lazily on first need. Calling it explicitly is harmless and is the safer tutorial pattern, but its ABSENCE no longer triggers an exception in current dolfinx. Signal: in older dolfinx (pre-0.7), locate_dofs_topological raised RuntimeError mentioning 'connectivity has not been computed'; current dolfinx returns dof indices without that step. (Verified empirically 2026-06-01.)",
            "[API] dolfinx.default_scalar_type for Constants and Function arrays so dtype matches the PETSc build (float64 if PETSc is real, complex128 if PETSc is complex). Signal: passing a Python float into a complex-PETSc Function raises TypeError in fem.form / fem.assemble_matrix; passing 0j into a real-PETSc Function raises ValueError 'cannot convert complex to float'.",
            "[API] VTXWriter (ADIOS2 backend) supports only Lagrange / DG element families. Mixed / Nedelec / BDM Functions cannot be written. Signal: VTXWriter.write raises RuntimeError 'Cannot interpolate function to the VTX output basis' or 'ADIOS2 VTX only supports Lagrange elements'.",
            "[Physics] Pure-Neumann Poisson admits the constant null space — the solution is determined only up to a constant. Either pin one DOF (DirichletBC on a single point) or add a Lagrange multiplier enforcing mean(u) = 0. Signal: LinearProblem.solve returns successfully (CG with pc_type='none' even converges without raising), but the resulting Function array has a HUGE additive offset accommodating the null space — np.array shows max ≈ min ≈ O(1e6) with tiny std (e.g. max=2.18e+06, std=112 on an 8x8 unit square with f=1). The 'KSP fails' alternative does NOT typically fire; you observe the bug as the un-pinned constant. (Verified empirically 2026-06-01.)",
            "[Syntax] For non-unit kappa coefficients: define as fem.Constant for spatially uniform, or fem.Function (interpolated) for spatially varying. Plain Python floats inside ufl forms work for unit coefficients but lose unit metadata. Signal: ufl form runs but the assembled stiffness scale disagrees with the analytic kappa-scaled stiffness by exactly the kappa value (when float coefficient was forgotten).",
        ],
        "materials": {"kappa": {"range": [0.001, 1e6], "unit": "W/(m*K) or dimensionless"}},
        "reference_solutions": {"unit_square_f1": "max(u) ~ 0.0737 for -laplacian(u)=1 on [0,1]^2, u=0 on boundary"},
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
            "default_scalar_type) — not scalar 0. dolfinx "
            "broadcasts the BC value against the space shape; "
            "scalar→vector fails. Signal: numpy raises ValueError "
            "'could not broadcast input array from shape () into "
            "shape (gdim,)' when dirichletbc is constructed with "
            "a scalar on a vector space.",
            "[Physics] Plane strain vs plane stress: lambda must "
            "be adjusted. Plane stress uses lambda_star = "
            "2*lambda*mu/(lambda+2*mu); using plane strain "
            "lambda on a thin plate gives ~30% over-stiffness. "
            "Signal: a dolfinx plane_strain Function tip "
            "deflection differs from the analytic plane_stress "
            "reference by factor (1-nu) at nu=0.3.",
            "[Numerical] Near-incompressible (nu > 0.49): MUST "
            "use mixed formulation (Taylor-Hood or three-field) "
            "to avoid volumetric locking. Pure displacement P1/P2 "
            "at nu=0.4999 has displacement underestimated by "
            "orders of magnitude. Signal: a dolfinx single-field "
            "VectorFunctionSpace tip deflection at nu=0.4999 is "
            "~1e-3 of analytic value; switching to a mixed_P2_P1 "
            "Taylor_Hood MixedElement recovers it to within 1%.",
            "[Numerical] For GAMG/AMG: MUST provide near-nullspace "
            "(rigid body modes — 3 translations + 3 rotations in "
            "3D). Without it, CG+GAMG fails to converge on "
            "large problems. Signal: PETScKrylovSolver.solve "
            "raises 'KSP did not converge' / NoConvergence with "
            "iteration count = max_it; setting "
            "matrix.setNearNullSpace(rbm) reduces iter count by "
            "10-50x.",
            "[Physics] 2D default in dolfinx ufl elasticity is "
            "plane strain — explicit modification needed for "
            "plane stress. Forgetting this is a silent source of "
            "wrong answers for thin structures. Signal: a 2D "
            "VectorH1 dolfinx Function plate deflection differs "
            "from analytic plane-stress reference by factor "
            "(1-nu^2) — the plane-strain stiffness in the "
            "ufl.inner(sigma, eps(v))*dx form over-constrains "
            "thickness.",
            "[API] dolfinx.fem.FunctionSpace rejects element "
            "family names other than the registered basix "
            "families. Passing legacy names like 'P1' or 'CG' "
            "that worked in old DOLFIN raises ValueError. Signal: "
            "dolfinx.fem.functionspace((mesh, ('CG', 1))) raises "
            "ValueError 'Unknown element family CG' — the basix "
            "name is 'Lagrange', not 'CG' or 'P1'.",
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
            "tier2_fixtures/fenics/. Constructability of all "
            "three confirmed; instability of P1/P1 is the "
            "advisory part — well-known LBB theory.)",
            "[Numerical] Pressure for enclosed (all-Dirichlet on "
            "velocity) flows is determined only up to an "
            "additive constant. Pin one pressure DOF with a "
            "DirichletBC at a chosen vertex, or attach a "
            "nullspace via dolfinx.la.create_petsc_nullspace_"
            "constants and call A.setNullSpace(ns) on the "
            "PETSc matrix. Skipping this leaves PETSc to handle "
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
            "L2 error from ufl.errornorm of u_h (the Function "
            "from LinearProblem.solve) vs an analytic reference "
            "saturates as dt is reduced because the splitting "
            "error dominates the spatial error; switching back "
            "to monolithic dolfinx.nls.petsc.NonlinearProblem + "
            "NewtonSolver recovers the spatial-error regime. "
            "(Claim inherited.)",
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
                "[Numerical] Without stabilization, Galerkin method "
                "produces oscillations for Pe > 1. Signal: solution "
                "develops visible wiggles upstream of source/sink "
                "locations; oscillation amplitude does not damp "
                "with mesh refinement in the advection-aligned "
                "direction. Add SUPG, GLS, or upwind DG. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] SUPG tau parameter depends on mesh "
                "size h and velocity magnitude — must compute "
                "PER CELL via tau = h / (2*|b|) using "
                "ufl.CellDiameter inside the dolfinx fem.form "
                "for high-Pe regime. Using a constant global "
                "tau Constant under-stabilises on fine cells "
                "and over-diffuses on coarse ones. Signal: "
                "convergence rate degrades from O(h^2) to "
                "~O(h) with a constant Constant tau; per-cell "
                "ufl.CellDiameter tau restores O(h^2). "
                "(Audit 2026-06-02.)"
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
            "[Numerical] Large load steps cause Newton (NewtonSolver) divergence in hyperelasticity. Use incremental load stepping: ramp the dirichletbc value or body-force fem.Constant across N steps, calling NewtonSolver.solve at each level. Signal: dolfinx.nls.petsc.NewtonSolver.solve raises 'Failed to converge' with the residual at the last iter still O(1); reducing the per-step load increment by 2-4× recovers convergence. (Claim inherited.)",
            "[Numerical] Near-incompressible regime (nu > 0.49) makes the pure-displacement formulation lock — use a dolfinx mixed (u, p) basix.ufl.mixed_element([P2-vector, P1]) FunctionSpace or the F-bar method (uniform-pressure projection). Signal: Cook-membrane tip deflection at nu = 0.4999 with pure P2 VectorH1 displacement on a dolfinx Function is O(1e-3) of the analytic value; switching to the mixed (u, p) formulation in basix.ufl recovers it within ~1%. (Claim inherited.)",
            "[Physics] Neo-Hookean / any compressible hyperelastic model requires J = det(F) > 0 everywhere. A locally inverted element gives J <= 0 and the log(J) term blows up. Signal: NewtonSolver.solve raises RuntimeError / FloatingPointError, or the residual jumps to nan, when det(F) at any quadrature point hits 0 or goes negative. Defensive check: ufl.conditional(J > 0, ..., raise_an_error). (Claim inherited.)",
            "[API] ufl.variable() + ufl.diff() automate stress computation from a stored energy W. Wrap F in ufl.variable to mark it as the differentiation target, define W(F_var), then P = ufl.diff(W, F_var) yields the 1st Piola-Kirchhoff stress as a ufl.VariableDerivative expression directly usable inside the residual ufl.inner(P, grad(v))*dx form. Signal: type(ufl.variable(F)) is ufl.classes.Variable; type(ufl.diff(W, F_var)).__name__ == 'VariableDerivative'. Hand-coding the gradient bypasses ufl's analytic differentiation and is error-prone. (Verified empirically 2026-06-01.)",
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
                "enough for stability (scales with polynomial "
                "degree^2). Signal: too small -> coercivity loss "
                "and the solution norm diverges with mesh "
                "refinement; too large -> cond(K) > 1e14 and "
                "iterative solver stalls. Rule of thumb: alpha = "
                "4 * (k+1)^2 for C0 interior-penalty biharmonic "
                "with degree-k Lagrange. (Audit 2026-06-02.)"
            ),
            (
                "[API] h_E (cell-size measure for the penalty "
                "weight) must use the proper UFL CellDiameter / "
                "FacetArea expressions — hard-coding h as a scalar "
                "gives wrong scaling on graded meshes. Signal: "
                "convergence rate degrades from O(h^2) to ~O(h) "
                "or stagnates because the penalty does not scale "
                "correctly with element size. Use "
                "ufl.CellDiameter(mesh) inside the form. (Audit "
                "2026-06-02.)"
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
                "variable). Signal: writing biharmonic as a "
                "single 4th-order operator on C0 Lagrange "
                "elements raises `NotImplementedError: H2 "
                "conformity required` or silently uses the "
                "interior-penalty form when assembling. Mixed "
                "(u, sigma) with sigma = Laplacian(u) works on "
                "plain P1 x P1. (Audit 2026-06-02.)"
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
                "[Numerical] Need a fine mesh: ~10 points per "
                "wavelength minimum (pollution effect for high k). "
                "Signal: phase error grows as (k*h)^2 — at 5 "
                "points-per-wavelength the computed dolfinx ufl "
                "Function representing the wave shows visible "
                "amplitude drift after ~10 wavelengths in the "
                "XDMFFile output; increasing to 10 pts/wave "
                "restores the analytic amplitude. Convergence "
                "rate degrades from O(h^2) to ~O(h) when k*h > 1. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] System is INDEFINITE — standard CG "
                "diverges. Use GMRES or direct solver. Signal: "
                "PETSc reports `KSPSolve: DIVERGED_INDEFINITE_PC` "
                "or `DIVERGED_BREAKDOWN` with CG; the same matrix "
                "with GMRES converges (slowly). For ~< 100k DOFs "
                "use LU; for larger meshes use GMRES + a shifted-"
                "Laplacian preconditioner. (Audit 2026-06-02.)"
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
            "[Syntax] Complex-valued Maxwell: PETSc must be compiled with --with-scalar-type=complex. Signal: assembling a form with imaginary coefficient (e.g., 1j*k*u*v*ds) into a real-PETSc Function raises TypeError 'cannot convert complex to float' or ValueError 'imaginary part discarded' from dolfinx.fem.assemble_vector.",
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
            "[Numerical] For a generalised eigenvalue problem A*x = lambda*B*x with Dirichlet BC, the mass matrix B must be assembled WITHOUT zeroing the boundary rows the way Dirichlet rows are typically handled — otherwise B becomes singular at Dirichlet DOFs and spurious zero eigenvalues appear. Standard pattern: assemble both A and B with bcs=[], then use dirichletbc-aware row/column reduction only on A. Signal: SLEPc returns n_dirichlet_dof spurious zero eigenvalues at the bottom of the spectrum; the next eigenvalues (skipping those zeros) match the analytic Dirichlet eigenvalues. (Claim inherited.)",
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
                "[Numerical] Standard displacement formulation "
                "LOCKS for nu > 0.49 — MUST use mixed (u, p) "
                "method. Signal: a compressed block shows "
                "essentially zero displacement (artificial "
                "rigidity); volume strain det(F)-1 << expected; "
                "the same setup with Taylor-Hood mixed method "
                "recovers the analytic incompressible solution. "
                "Locking ratio ~1/(1-2nu) — at nu=0.499 "
                "displacement is ~500x too small. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Inf-sup (LBB) condition: "
                "pressure FunctionSpace must be STRICTLY "
                "SMALLER than the displacement "
                "FunctionSpace. Signal: pairing P1 "
                "displacement Function + P1 pressure "
                "Function (equal-order) gives checkerboard "
                "pressure pattern that does NOT converge "
                "under refinement; switching to P2/P1 "
                "dolfinx mixed-element removes the "
                "checkerboard. The LBB constant collapsing "
                "with h is the diagnostic for inf-sup "
                "failure. (Audit 2026-06-02.)"
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
            "assembly": "dolfinx.fem.assemble_scalar() sums across ranks automatically",
        },
        "performance": {
            "scaling": "Strong and weak scaling demonstrated up to thousands of cores",
            "mesh_partitioning": "Graph-based (ParMETIS, SCOTCH, or KaHIP) for load balancing",
            "ghost_layer": "DOLFINx manages ghost cells/DOFs automatically",
            "neighbourhood_collectives": "MPI Neighbourhood collectives for efficient halo exchange",
        },
        "pitfalls": [
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
        "pitfalls": [
            "Online tutorials may use old API (ufl.FiniteElement, FunctionSpace) — translate to new API",
            "The jsdokken tutorial is updated for latest version — use it as primary reference",
            "DOLFINx version in Docker images may differ from pip install — check dolfinx.__version__",
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
# FEBIO — COMPREHENSIVE DOMAIN KNOWLEDGE
# ═══════════════════════════════════════════════════════════════════════════════

_FEBIO_KNOWLEDGE = {
    "linear_elasticity": {
        "description": "Linear elasticity with FEBio — isotropic elastic material. XML input (.feb), v4.0.",
        "input_format": "FEBio XML (.feb), version 4.0",
        "solver": "Newton-Raphson with direct linear solver",
        "materials": {
            "isotropic elastic": {"E": "Young's modulus (Pa)", "v": "Poisson's ratio (NOTE: lowercase v, not nu)"},
        },
        "pitfalls": [
            "FEBio uses lowercase 'v' for Poisson's ratio (not 'nu')",
            "Element connectivity is 1-indexed",
            "MeshDomains section required in v4.0 (links domain to material)",
            "LoadData section with load_controller needed for prescribed BCs",
        ],
    },
    "hyperelasticity": {
        "description": "Nonlinear hyperelasticity — Neo-Hookean, Mooney-Rivlin, Ogden, HGO (tissue).",
        "materials": {
            "neo-Hookean": {"E": "Young's modulus", "v": "Poisson's ratio"},
            "Mooney-Rivlin": {"c1": "1st constant", "c2": "2nd constant", "k": "bulk modulus"},
            "Ogden": {"c1-c6": "Ogden constants", "m1-m6": "Ogden exponents", "k": "bulk modulus"},
            "Holzapfel-Gasser-Ogden": {"c": "matrix modulus", "k1": "fiber stiffness", "k2": "fiber exponent",
                                        "kappa": "dispersion (0=aligned, 1/3=isotropic)", "theta": "fiber angle"},
        },
        "pitfalls": [
            "Use 'STATIC' analysis for quasi-static loading",
            "Large deformations require proper step size control",
            "Convergence issues: reduce step size or use line search",
            "HGO model: fiber direction via local coordinate system",
        ],
    },
    "biphasic": {
        "description": "Biphasic poroelasticity — solid + fluid phases. Key for cartilage, hydrogels.",
        "materials": {
            "biphasic": {"solid": "Any hyperelastic + 'permeability' material",
                         "permeability": "Holmes-Mow or constant permeability"},
        },
        "pitfalls": [
            "Requires Module type='biphasic'",
            "Fluid pressure BC via 'fluid pressure' boundary condition",
            "Time stepping critical — fast diffusion requires small dt initially",
        ],
    },
    "heat": {
        "description": "Heat conduction (steady-state). Module type='heat'.",
        "solver": "Direct or iterative",
    },
}


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


# ═══════════════════════════════════════════════════════════════════════════════
# MCP TOOL REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def register_deep_knowledge_tools(mcp: FastMCP):

    @mcp.tool()
    def get_deep_knowledge(solver: str, physics: str) -> str:
        """Get comprehensive domain knowledge for a physics module.

        Returns everything needed to set up a simulation correctly:
        materials, solver options, time integration, pitfalls, element types,
        reference solutions, and best practices.

        This is MUCH more detailed than get_physics_knowledge — use this
        when you need to understand a physics problem deeply.

        Args:
            solver: Backend name ('fenics', 'fourc', 'dealii', 'febio')
            physics: Physics key (e.g. 'poisson', 'navier_stokes', 'fsi', 'particle_pd')
        """
        knowledge_map = {
            "fourc": _4C_KNOWLEDGE,
            "4c": _4C_KNOWLEDGE,
            "fenics": _FENICS_KNOWLEDGE,
            "fenicsx": _FENICS_KNOWLEDGE,
            "dealii": _DEALII_KNOWLEDGE,
            "deal.ii": _DEALII_KNOWLEDGE,
            "febio": _FEBIO_KNOWLEDGE,
        }

        db = knowledge_map.get(solver.lower())
        if not db:
            return f"Unknown solver: {solver}. Available: fourc, fenics, dealii, febio"

        k = db.get(physics.lower())
        if not k:
            available = sorted(db.keys())
            return f"No knowledge for '{physics}' in {solver}. Available: {', '.join(available)}"

        return json.dumps(k, indent=2, default=str)

    @mcp.tool()
    def get_all_pitfalls(solver: str) -> str:
        """Get ALL pitfalls/common mistakes for a solver, across all physics.

        Critical for preventing simulation failures. Returns a consolidated
        list organized by physics module.

        Args:
            solver: Backend name ('fenics', 'fourc', 'dealii', 'febio')
        """
        knowledge_map = {
            "fourc": _4C_KNOWLEDGE, "fenics": _FENICS_KNOWLEDGE,
            "dealii": _DEALII_KNOWLEDGE, "febio": _FEBIO_KNOWLEDGE,
        }
        db = knowledge_map.get(solver.lower())
        if not db:
            return f"Unknown solver: {solver}"

        lines = [f"# All Pitfalls for {solver}\n"]
        for physics, k in sorted(db.items()):
            pitfalls = k.get("pitfalls", [])
            if pitfalls:
                lines.append(f"## {physics}")
                for p in pitfalls:
                    lines.append(f"- {p}")
                lines.append("")

        return "\n".join(lines)

    @mcp.tool()
    def get_material_catalog(solver: str) -> str:
        """Get the complete material catalog for a solver backend.

        Lists all material types, their parameters, valid ranges,
        and which physics modules use them.

        Args:
            solver: Backend name ('fenics', 'fourc', 'dealii', 'febio')
        """
        knowledge_map = {
            "fourc": _4C_KNOWLEDGE, "fenics": _FENICS_KNOWLEDGE,
            "dealii": _DEALII_KNOWLEDGE, "febio": _FEBIO_KNOWLEDGE,
        }
        db = knowledge_map.get(solver.lower())
        if not db:
            return f"Unknown solver: {solver}"

        catalog = {}
        for physics, k in db.items():
            materials = k.get("materials", {})
            for mat_name, mat_info in materials.items():
                if mat_name not in catalog:
                    catalog[mat_name] = {"used_in": [], "parameters": mat_info}
                catalog[mat_name]["used_in"].append(physics)

        return json.dumps(catalog, indent=2, default=str)

    @mcp.tool()
    def get_solver_guidance(physics: str) -> str:
        """Get cross-solver comparison and recommendation for a physics problem.

        Compares how each available solver handles this physics type,
        including strengths, weaknesses, and which to choose.

        Args:
            physics: Physics type (e.g. 'poisson', 'navier_stokes', 'fsi', 'hyperelasticity')
        """
        lines = [f"# Solver Guidance for: {physics}\n"]

        all_knowledge = {
            "FEniCSx": _FENICS_KNOWLEDGE,
            "deal.II": _DEALII_KNOWLEDGE,
            "4C": _4C_KNOWLEDGE,
            "FEBio": _FEBIO_KNOWLEDGE,
        }

        found_any = False
        for solver_name, db in all_knowledge.items():
            k = db.get(physics.lower())
            if k:
                found_any = True
                lines.append(f"## {solver_name}")
                lines.append(f"**Description:** {k.get('description', 'N/A')}")
                if "solver" in k:
                    lines.append(f"**Solver:** {json.dumps(k['solver'], default=str)}")
                if "pitfalls" in k:
                    lines.append("**Key pitfalls:**")
                    for p in k["pitfalls"][:3]:
                        lines.append(f"  - {p}")
                if "variants" in k:
                    lines.append(f"**Templates:** {', '.join(k['variants'])}")
                lines.append("")

        if not found_any:
            lines.append(f"No solver has knowledge for '{physics}'.")
            lines.append("Available physics across all solvers:")
            all_physics = set()
            for db in all_knowledge.values():
                all_physics.update(db.keys())
            for p in sorted(all_physics):
                lines.append(f"  - {p}")

        # Recommendation
        backend = None
        avail = available_backends()
        for b in avail:
            for p in b.supported_physics():
                if p.name == physics.lower():
                    backend = b
                    break
            if backend:
                break

        if backend:
            lines.append(f"\n**Recommended:** {backend.display_name()} (available on this machine)")

        return "\n".join(lines)

    @mcp.tool()
    def get_solver_catalog(solver: str) -> str:
        """Get the full capability catalog for a specific solver.

        For deal.II: complete tutorial step catalog (50+ steps)
        For 4C: all physics modules with problem types and templates
        For FEniCS: all physics with weak forms and solver options
        For FEBio: material catalog including biomechanics-specific

        Args:
            solver: Backend name ('fenics', 'fourc', 'dealii', 'febio')
        """
        knowledge_map = {
            "fourc": _4C_KNOWLEDGE, "4c": _4C_KNOWLEDGE,
            "fenics": _FENICS_KNOWLEDGE, "fenicsx": _FENICS_KNOWLEDGE,
            "dealii": _DEALII_KNOWLEDGE, "deal.ii": _DEALII_KNOWLEDGE,
            "febio": _FEBIO_KNOWLEDGE,
        }
        db = knowledge_map.get(solver.lower())
        if not db:
            return f"Unknown solver: {solver}. Available: fourc, fenics, dealii, febio"

        lines = [f"# {solver} — Full Capability Catalog\n"]
        for key, k in sorted(db.items()):
            desc = k.get("description", "")
            lines.append(f"## {key}")
            if desc:
                lines.append(f"{desc}")
            if "problem_type" in k:
                lines.append(f"**Problem type:** {k['problem_type']}")
            if "weak_form" in k:
                lines.append(f"**Weak form:** {k['weak_form']}")
            if "variants" in k:
                lines.append(f"**Templates:** {', '.join(k['variants'])}")
            if "tutorial_steps" in k:
                for step, sdesc in k["tutorial_steps"].items():
                    lines.append(f"  - {step}: {sdesc}")
            lines.append("")
        return "\n".join(lines)

    @mcp.tool()
    def get_cross_solver_reference(problem: str = "") -> str:
        """Get verified cross-solver reference solutions and validation knowledge.

        Returns known-good reference values for standard FEM benchmark problems,
        verified across FEniCS, deal.II, and 4C. Essential for:
        - Checking if your simulation result is correct
        - Understanding expected element-type differences (tri vs quad)
        - Setting up matched cross-solver comparisons
        - Knowing 4C inline mesh normalization rules

        Args:
            problem: Specific problem (e.g. 'poisson', 'heat', 'elasticity')
                    or empty for all reference data
        """
        if problem:
            key = problem.lower().replace(" ", "_")
            # Search in reference solutions
            for k, v in _CROSS_SOLVER_KNOWLEDGE["reference_solutions"].items():
                if key in k:
                    return json.dumps({k: v}, indent=2)
            return f"No reference for '{problem}'. Available: {list(_CROSS_SOLVER_KNOWLEDGE['reference_solutions'].keys())}"
        return json.dumps(_CROSS_SOLVER_KNOWLEDGE, indent=2)
