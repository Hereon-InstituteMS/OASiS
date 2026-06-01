"""
Comprehensive 4C Multiphysics knowledge catalogue.

Based on systematic reading of ALL 4C source code.
73 modules, 40 problem types, 120+ materials, 130+ conditions, 20+ cell types.

This is the single source of truth for 4C domain knowledge in the Open FEM Agent.
"""

FOURC_KNOWLEDGE = {
    # ═══════════════════════════════════════════════════════════════════════
    # OVERVIEW
    # ═══════════════════════════════════════════════════════════════════════
    "overview": {
        "description": "4C is a large-scale parallel C++20 multiphysics FEM code developed at TU Munich",
        "source": "$FOURC_ROOT/src/ (73 modules) — set FOURC_ROOT env var",
        "input_format": "YAML (.4C.yaml) with inline mesh or Exodus mesh references",
        "execution": "mpirun -np N $FOURC_BINARY input.4C.yaml (or just 4C if on PATH)",
        "output": "VTU via IO/RUNTIME VTK OUTPUT sections",
        "build": "CMake (cd build && cmake --build . -j$(nproc))",
        "modules": 73,
        "problem_types": 40,
        "material_models": "120+",
        "condition_types": "130+",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # STRUCTURAL MECHANICS (structure, structure_new, solid_3D_ele)
    # ═══════════════════════════════════════════════════════════════════════
    "structural_mechanics": {
        "description": "Full nonlinear structural mechanics — the core of 4C",
        "problemtype": "Structure",
        "yaml_section": "STRUCTURAL DYNAMIC",

        "time_integration": {
            "Statics": "Static analysis (one step, equilibrium)",
            "GenAlpha": "Generalized-alpha implicit time integration",
            "GenAlphaLieGroup": "Generalized-alpha for SO(3) rotation group (beams/shells)",
            "OneStepTheta": "One-step-theta implicit scheme",
            "ExplEuler": "Explicit forward Euler",
            "CentrDiff": "Explicit central differences (wave propagation)",
            "AdamsBashforth2": "Explicit 2nd order Adams-Bashforth",
            "AdamsBashforth4": "Explicit 4th order Adams-Bashforth",
        },

        "kinematics": {
            "linear": "Small strain / linear kinematics (ε = sym(∇u))",
            "nonlinearTotLag": "Total Lagrangian / finite deformation (F = I + ∇u)",
        },

        "element_types": {
            "3D_solid": {
                "SOLID HEX8": "8-node hexahedron (Q1, standard or F-bar or EAS)",
                "SOLID HEX20": "20-node hexahedron (serendipity Q2)",
                "SOLID HEX27": "27-node hexahedron (full Q2)",
                "SOLID TET4": "4-node tetrahedron (P1)",
                "SOLID TET10": "10-node tetrahedron (P2)",
                "SOLID WEDGE6": "6-node wedge/prism",
                "SOLID WEDGE15": "15-node wedge (P2)",
                "SOLID PYRAMID5": "5-node pyramid",
                "SOLIDSCATRA HEX8": "8-node hex with scalar transport coupling (for TSI)",
            },
            "2D_wall": {
                "WALL QUAD4": "4-node quadrilateral (plane strain/stress, EAS option)",
                "WALL QUAD8": "8-node serendipity quad",
                "WALL QUAD9": "9-node full biquadratic quad",
                "WALL TRI3": "3-node triangle",
                "WALL TRI6": "6-node quadratic triangle",
            },
            "1D_beam": {
                "BEAM3R": "Simo-Reissner beam (shear-deformable, geometrically exact)",
                "BEAM3K": "Kirchhoff beam (shear-rigid, inextensible option)",
                "BEAM3EB": "Euler-Bernoulli beam (classical)",
            },
            "shell": {
                "SHELL7P": "7-parameter shell (EAS, ANS options, thickness locking-free)",
                "SHELL_KL_NURBS": "Kirchhoff-Love NURBS shell (isogeometric)",
            },
            "other": {
                "MEMBRANE": "Membrane element (no bending stiffness)",
                "TRUSS3": "Truss element (axial force only)",
                "TORSION3": "Torsional spring element",
                "RIGIDSPHERE": "Rigid sphere for DEM contact",
            },
        },

        "element_technologies": {
            "none": "Standard displacement-based formulation",
            "fbar": "F-bar method (volumetric locking treatment for hex8)",
            "eas_mild": "Enhanced Assumed Strain (mild enrichment, 7 modes for hex8)",
            "eas_full": "Enhanced Assumed Strain (full enrichment, 21 modes for hex8)",
            "shell_ans": "Assumed Natural Strain for shells (shear locking treatment)",
            "shell_eas": "EAS for shells",
            "shell_eas_ans": "Combined EAS + ANS for shells",
        },

        "nonlinear_solvers": {
            "newtonfull": "Full Newton-Raphson (assemble tangent every iteration)",
            "newtonmod": "Modified Newton (reuse tangent, cheaper per iteration)",
            "newtonls": "Newton with line search (backtracking)",
            "newtonuzawalin": "Linear Uzawa for constrained problems",
            "newtonuzawanonlin": "Nonlinear Uzawa",
            "ptc": "Pseudo-transient continuation (robust for difficult convergence)",
            "nox_nln": "NOX nonlinear solver framework (Trilinos)",
        },

        "wall_element_params": "KINEM linear/nonlinear, EAS none/full, THICK 1.0, STRESS_STRAIN plane_strain/plane_stress, GP 2 2",

        "pitfalls": [
            "WALL QUAD4 needs: MAT 1 KINEM linear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 2 2",
            "SOLID elements need: MAT 1 KINEM linear/nonlinearTotLag",
            "SOLIDSCATRA: REQUIRED for TSI coupling — plain SOLID cannot couple with thermal",
            "For statics: MAXITER 1 for linear problems, 10+ for nonlinear",
            "PREDICT TangDis recommended for nonlinear Newton convergence",
            "Body forces: DESIGN SURF NEUMANN (2D) or DESIGN VOL NEUMANN (3D) with NUMDOF 6",
            "Beam elements need special BEAM3* type, not SOLID or WALL",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # MATERIALS (120+ models)
    # ═══════════════════════════════════════════════════════════════════════
    "materials": {
        "description": "120+ material models spanning all physics disciplines",

        "basic_structural": {
            "MAT_Struct_StVenantKirchhoff": {
                "params": "YOUNG, NUE, DENS",
                "use": "Linear elastic (small strain) or geometric nonlinear",
            },
            "MAT_Struct_ThermoStVenantK": {
                "params": "YOUNG (array), NUE, DENS, THEXPANS, INITTEMP, THERMOMAT",
                "use": "Linear elastic with thermal expansion coupling (for TSI)",
                "notes": "THERMOMAT links to a MAT_Fourier for thermal properties",
            },
        },

        "hyperelastic": {
            "MAT_ElastHyper": "Toolbox: combine summands (NeoHooke + volumetric, etc.)",
            "summands": {
                "coupNeoHooke": "Neo-Hooke (coupled form): W = C1*(I1-3) + 1/(2*D1)*(J-1)^2",
                "couploganeohooke": "Logarithmic Neo-Hooke: W = mu/2*(I1-3) - mu*ln(J) + lam/2*ln(J)^2",
                "coupMooneyRivlin": "Mooney-Rivlin (coupled): W = C1*(I1-3) + C2*(I2-3)",
                "isoNeoHooke": "Isochoric Neo-Hooke (incompressible split)",
                "isoOgden": "Isochoric Ogden (stretch-based)",
                "isoYeoh": "Isochoric Yeoh (polynomial in I1)",
                "coupBlatzKo": "Blatz-Ko (compressible rubber-like)",
                "coupSimoPister": "Simo-Pister model",
                "coupAnisoExpo": "Anisotropic exponential fiber model (soft tissue)",
                "coupAnisoNeoHooke": "Anisotropic Neo-Hooke fiber",
            },
            "volumetric": {
                "volOgden": "Ogden volumetric penalty",
                "volPenalty": "Standard penalty: κ/2*(J-1)^2",
                "volSussmanBathe": "Sussman-Bathe volumetric",
            },
        },

        "viscoelastic": {
            "MAT_ViscoElastHyper": "Viscohyperelastic with Maxwell branches",
            "generalizedMaxwell": "Generalized Maxwell (Standard Linear Solid)",
            "fractionalSLS": "Fractional Standard Linear Solid",
        },

        "plasticity": {
            "MAT_PlLinElast": "Small-strain von Mises plasticity (YOUNG, NUE, YIELD, SATHARDENING, etc.)",
            "MAT_PlNlnLogNeoHooke": "Finite strain von Mises + logarithmic Neo-Hooke",
            "MAT_PlDruckPrag": "Drucker-Prager plasticity (pressure-dependent yield)",
            "MAT_PlGTN": "Gurson-Tvergaard-Needleman (ductile damage)",
            "MAT_CrystPlast": "Crystal plasticity (single crystal, multiple slip systems)",
            "MAT_PlElastHyper": "Hyperelastic + finite strain von Mises (semi-smooth Newton)",
        },

        "biological": {
            "MAT_ConstraintMixture": "Constrained mixture model for arterial growth/remodeling",
            "MAT_GrowthRemodelElastHyper": "Growth and remodeling hyperelastic",
            "MAT_Muscle_Combo": "Active strain muscle model (combo)",
            "MAT_Muscle_Giantesio": "Giantesio active strain muscle",
            "MAT_Myocard": "Myocardial tissue with electrophysiology (FHN, TenTusscher, etc.)",
        },

        "fluid": {
            "MAT_Fluid": "Newtonian fluid (DYNVISCOSITY, DENSITY)",
            "MAT_CarreauYasuda": "Carreau-Yasuda shear-thinning",
            "MAT_HerschelBulkley": "Herschel-Bulkley yield stress fluid",
            "MAT_Sutherland": "Temperature-dependent viscosity (Sutherland law)",
        },

        "thermal": {
            "MAT_Fourier": "Fourier heat conduction (CAPA=heat capacity, CONDUCT=conductivity)",
            "MAT_Soret": "Soret effect (thermodiffusion coupling)",
        },

        "scalar_transport": {
            "MAT_scatra": "General scalar transport (DIFFUSIVITY parameter)",
            "MAT_scatra_reaction": "Reactive scalar transport",
            "MAT_scatra_chemotaxis": "Chemotactic scalar transport",
        },

        "porous_media": {
            "MAT_FluidPoro": "Darcy fluid in porous medium",
            "MAT_StructPoro": "Structural skeleton for poroelasticity",
            "phase_laws": "Linear, tangent, constraint, by-function",
            "permeability_laws": "Constant, exponential",
        },

        "particle": {
            "MAT_Particle_SPH_Fluid": "SPH fluid particle",
            "MAT_Particle_DEM": "DEM particle",
            "MAT_Particle_PD": "Peridynamic particle (bond-based)",
        },

        "beam": {
            "MAT_Beam_Reissner_ElastHyper": "Simo-Reissner beam hyperelastic",
            "MAT_Beam_Kirchhoff_ElastHyper": "Kirchhoff beam hyperelastic",
            "MAT_Beam_Reissner_ElastPlastic": "Beam with plasticity",
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # FLUID MECHANICS (fluid, fluid_ele, fluid_turbulence)
    # ═══════════════════════════════════════════════════════════════════════
    "fluid": {
        "description": "Incompressible Navier-Stokes with stabilized FEM",
        "problemtype": "Fluid",
        "yaml_section": "FLUID DYNAMIC",

        "time_integration": {
            "GenAlpha": "Generalized-alpha (default for incompressible NS)",
            "BDF2": "2nd order backward difference formula",
            "OneStepTheta": "One-step-theta",
            "Stationary": "Steady-state RANS or Stokes",
        },

        "stabilization": {
            "SUPG": "Streamline upwind Petrov-Galerkin",
            "GLS": "Galerkin least squares",
            "VMS": "Variational multiscale (recommended)",
            "PSPG": "Pressure stabilization Petrov-Galerkin",
        },

        "turbulence_models": [
            "Dynamic Smagorinsky (LES)",
            "Dynamic Vreman (LES)",
            "k-epsilon (RANS, via additional scatra equations)",
        ],

        "ale": "ALE formulation for moving meshes (ale2, ale3 elements)",

        "pitfalls": [
            "Fluid uses its own element section: FLUID ELEMENTS (not STRUCTURE)",
            "Stabilization parameter needs tuning for high Re",
            "ALE requires separate ALE mesh movement problem",
            "X-wall: extended wall functions for near-wall treatment",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # SCALAR TRANSPORT (scatra, scatra_ele)
    # ═══════════════════════════════════════════════════════════════════════
    "scalar_transport": {
        "description": "Convection-diffusion / scalar transport — the workhorse for Poisson, heat, electrochemistry",
        "problemtype": "Scalar_Transport",
        "yaml_section": "SCALAR TRANSPORT DYNAMIC",

        "time_integration": ["GenAlpha", "BDF2", "OneStepTheta", "Stationary"],

        "physics_variants": {
            "standard": "Pure convection-diffusion-reaction",
            "electrochemistry": "Nernst-Planck ion transport (elch, elch_diffcond, elch_scl)",
            "cardiac_monodomain": "Cardiac electrophysiology (FHN, TenTusscher models)",
            "level_set": "Level-set advection + reinitialization",
            "porous_media": "Scalar transport in porous media",
            "growth_remodel": "Growth and remodeling scalar transport",
        },

        "elements": "TRANSP QUAD4/8/9 (2D), TRANSP HEX8/20/27 (3D), TRANSP TRI3/6, TRANSP TET4/10",

        "pitfalls": [
            "Element section: TRANSPORT ELEMENTS (not STRUCTURE or FLUID)",
            "Material: MAT_scatra with DIFFUSIVITY parameter",
            "For Poisson: TIMEINTEGR Stationary, VELOCITYFIELD zero, source via DESIGN SURF NEUMANN",
            "For heat: same as Poisson but T_left/T_right via DESIGN LINE DIRICH",
            "IO/RUNTIME VTK OUTPUT/SCATRA may crash 4C — use post_vtu conversion instead",
            "Field name in VTU output: phi_1 (not temperature or u)",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # THERMAL (thermo)
    # ═══════════════════════════════════════════════════════════════════════
    "thermal": {
        "description": "Thermal analysis (standalone or coupled via TSI/STI/SSTI)",
        "problemtype": "Thermo",
        "yaml_section": "THERMAL DYNAMIC",
        "time_integration": ["Statics", "GenAlpha", "OneStepTheta"],
        "boundary_conditions": {
            "DESIGN SURF THERMO DIRICH CONDITIONS": "Prescribed temperature",
            "DESIGN SURF THERMO NEUMANN CONDITIONS": "Prescribed heat flux",
            "ThermoConvections": "Convective heat transfer BC (h*(T-T_inf))",
            "ThermoRobin": "Robin BC for thermal",
        },
        "pitfalls": [
            "Use THERMO not THERMAL in section names (DESIGN SURF THERMO DIRICH)",
            "For TSI: thermal field is solved by 4C, not prescribed externally",
            "INITIALFIELD: field_by_function + INITFUNCNO for initial temperature",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # MULTI-PHYSICS COUPLING
    # ═══════════════════════════════════════════════════════════════════════
    "tsi": {
        "description": "Thermo-Structure Interaction — the key multi-physics coupling in 4C",
        "problemtype": "Thermo_Structure_Interaction",
        "yaml_sections": ["STRUCTURAL DYNAMIC", "THERMAL DYNAMIC", "TSI DYNAMIC"],

        "coupling_algorithms": {
            "tsi_oneway": "One-way: thermal → structural (no feedback)",
            "tsi_sequstagg": "Sequential staggered (solve thermal, then structural, once per step)",
            "tsi_iterstagg": "Iterative staggered (iterate until convergence)",
            "tsi_iterstaggaitken": "Iterative staggered with Aitken acceleration",
            "tsi_iterstaggaitkenirons": "Aitken-Irons variant",
            "tsi_iterstaggfixedrel": "Fixed relaxation iterative staggered",
            "monolithic": "Simultaneous solve of all fields (TSI DYNAMIC/MONOLITHIC section)",
        },

        "requirements": [
            "SOLIDSCATRA HEX8 elements (NOT plain SOLID — the SCATRA coupling is needed)",
            "MAT_Struct_ThermoStVenantK (structural material with thermal expansion)",
            "MAT_Fourier (thermal material linked via THERMOMAT parameter)",
            "CLONING MATERIAL MAP: SRC_FIELD structure → TAR_FIELD thermo",
            "Two LINEAR_SOLVERs: one for thermal, one for structural",
            "FUNCT for INITIALFIELD: SYMBOLIC_FUNCTION_OF_SPACE_TIME for initial temperature",
        ],

        "pitfalls": [
            "Without CLONING MATERIAL MAP: 4C crashes at initialization",
            "THEXPANS in MAT_Struct_ThermoStVenantK: thermal expansion coefficient (units must match T units)",
            "INITTEMP: reference temperature for zero thermal strain",
            "TSI DYNAMIC controls coupling; STRUCTURAL/THERMAL DYNAMIC control individual solvers",
            "For one-way: ITEMAX 1 (only one coupling iteration needed)",
            "SOLIDSCATRA elements REQUIRE 'TYPE Undefined' in the element definition. "
            "Full format: <id> SOLIDSCATRA HEX8 <n1..n8> MAT <id> KINEM nonlinear TYPE Undefined",
            "For one-way thermal→structural: MUST add TSI DYNAMIC/PARTITIONED section "
            "with COUPVARIABLE: Temperature. Without this, 4C defaults to displacement "
            "coupling (structural→thermal), which is backwards for heating problems "
            "and produces zero displacement.",
            "Monolithic TSI requires Belos iterative solver with block preconditioner "
            "(NOT UMFPACK). For simple one-way problems, use partitioned tsi_oneway "
            "instead — it works with UMFPACK and is simpler to set up.",
            "Volume-level thermal Dirichlet: use DESIGN VOL THERMO DIRICH CONDITIONS "
            "with DVOL-NODE TOPOLOGY to prescribe temperature on all nodes.",
        ],
    },

    "fsi": {
        "description": "Fluid-Structure Interaction — partitioned and monolithic coupling",
        "problemtype": "Fluid_Structure_Interaction",

        "partitioned_algorithms": {
            "Dirichlet-Neumann": "Standard: displacement/velocity/force coupling at interface",
            "DirichletNeumannSlideALE": "Sliding interface variant",
            "relaxation": ["Fixed", "Aitken", "Steepest descent", "Chebyshev", "NLCG"],
            "MFNK": "Matrix-free Newton-Krylov (advanced, robust)",
        },

        "monolithic_algorithms": {
            "fluid_split": "Monolithic with fluid-based splitting",
            "structure_split": "Monolithic with structure-based splitting",
            "mortar": "Mortar-based monolithic (non-matching meshes)",
            "xfem": "XFEM-based monolithic (no mesh conformity needed)",
        },

        "required_sections": [
            "PROBLEM TYPE", "PROBLEM SIZE",
            "STRUCTURAL DYNAMIC", "STRUCTURAL DYNAMIC/GENALPHA",
            "FLUID DYNAMIC", "ALE DYNAMIC",
            "FSI DYNAMIC", "FSI DYNAMIC/MONOLITHIC SOLVER",
            "MATERIALS", "CLONING MATERIAL MAP",
            "STRUCTURE GEOMETRY", "FLUID GEOMETRY",
            "DESIGN FSI COUPLING LINE CONDITIONS (2D) or SURF CONDITIONS (3D)",
        ],

        "ale_boundary_conditions": {
            "rules": [
                "ALL walls with no-slip fluid BC: apply ALE Dirichlet (fix mesh)",
                "Inflow boundary: apply ALE Dirichlet (fix mesh)",
                "Outflow boundary: apply ALE Dirichlet (fix mesh)",
                "Cylinder/obstacle surfaces: apply ALE Dirichlet (fix mesh)",
                "FSI interface: do NOT apply ALE Dirichlet (mesh moves with structure)",
            ],
            "common_mistake": (
                "Forgetting ALE Dirichlet on some outer boundary causes the "
                "ALE mesh to distort freely, leading to inverted elements."
            ),
        },

        "valid_2d_elements": {
            "FLUID": ["QUAD4", "QUAD9", "TRI3", "TRI6"],
            "WALL (structure)": ["QUAD4", "QUAD9", "TRI3", "TRI6"],
            "notes": "QUAD4 most validated. TRI3 less accurate for pressure.",
        },

        "pitfalls": [
            "FSI is the most complex problem type in 4C.",
            "Fluid elements MUST use NA: ALE (not Euler!).",
            "ALE Dirichlet BCs on all outer fluid boundaries except FSI interface.",
            "CLONING MATERIAL MAP required: maps fluid -> ALE material.",
            "SHAPEDERIVATIVES: true required in FSI DYNAMIC/MONOLITHIC SOLVER.",
            "Each field (structure, fluid, ALE) needs its own SOLVER N entry.",
            "2D: DESIGN FSI COUPLING LINE CONDITIONS. 3D: SURF CONDITIONS.",
            "Structure NUMDOF matches dimension (2 or 3), fluid NUMDOF = dim+1.",
            "DESIGN LINE DIRICH CONDITIONS applies to ALL discretizations "
            "containing a node. Shared nodes between structure (NUMDOF=2) and "
            "fluid (NUMDOF=3) cause NUMDOF mismatch errors. Offset meshes or "
            "use mortar coupling to avoid shared nodes at Dirichlet boundaries.",
            "DESIGN FLUID LINE LIFT&DRAG does NOT exist in 2D. Use LIFTDRAG: "
            "true in FLUID DYNAMIC instead.",
            "IO section has no EVERY_ITERATION parameter.",
            "FUNCT with SYMBOLIC_FUNCTION_OF_SPACE_TIME + VARIABLE requires "
            "COMPONENT: 0 in the same list item. Without it, the variable "
            "is silently ignored and the function returns wrong values.",
            "Monolithic FSI requires SEPARATE nodes at the FSI interface — "
            "structure and fluid must not share nodes. If using a single Gmsh "
            "mesh, post-process to duplicate interface nodes and remap fluid "
            "connectivity. Or use mortar coupling which handles non-matching meshes.",
            "WALL TRI3 does NOT exist in 4C. The structural domain must use "
            "all-QUAD elements (QUAD4, QUAD8, QUAD9). Use transfinite meshing "
            "or recombination in Gmsh to avoid triangles in the structure.",
            "IO/RUNTIME VTK OUTPUT/STRUCTURE may conflict with FSI (INT_STRATEGY "
            "override). If it causes errors, remove it and use post_vtu instead.",
            "2D fluid VTK output may show NaN pressure and garbage vz — this is "
            "a VTK artifact, not divergence. Check vx/vy and convergence logs.",
            "For complex FSI geometries (e.g. flag attached to cylinder): offset "
            "the flag slightly (e.g. 0.1mm gap) to avoid Gmsh fragment operations "
            "that create non-quad-meshable surfaces. This is a negligible geometric "
            "approximation that vastly simplifies meshing.",
            "FSI SLAVE interface CANNOT carry Dirichlet BCs. With "
            "iter_monolithicstructuresplit (structure=slave), structural Dirichlet "
            "nodes must NOT overlap FSI coupling nodes. If they do, switch to "
            "iter_monolithicfluidsplit (structure=master) or exclude overlapping "
            "nodes from the FSI interface.",
            "IO/RUNTIME VTK OUTPUT/ALE does NOT exist — it crashes 4C. Only "
            "/STRUCTURE and /FLUID subsections are valid for FSI VTK output. "
            "For ALE fields, use post_processor --filter=vtu on native output.",
            "Valid COUPALGO values for monolithic FSI: "
            "iter_monolithicfluidsplit (structure=master, recommended), "
            "iter_monolithicstructuresplit (structure=slave), "
            "iter_mortar_monolithicfluidsplit (non-matching meshes), "
            "iter_sliding_monolithicfluidsplit (sliding interface). "
            "For partitioned: iter_stagg_AITKEN_rel_force (default), "
            "iter_stagg_fixed_rel_force.",
            "Inflow ramp rate affects FSI stability. For initial testing, use "
            "a slow ramp (5-10s period, e.g. cos(pi*t/5)) rather than the "
            "standard 2s Turek-Hron ramp. Fast ramps cause Newton divergence "
            "even with laminar flow.",
        ],
    },

    "ssi": {
        "description": "Structure-Scalar Interaction (e.g., battery electrode mechanics)",
        "problemtype": "Structure_Scalar_Interaction",
        "coupling_types": ["OneWay_ScatraToSolid", "OneWay_SolidToScatra",
                          "IterStagg", "IterStaggFixedRel", "IterStaggAitken", "Monolithic"],
    },

    "ssti": {
        "description": "Structure-Scalar-Thermo Interaction (three-field coupling)",
        "problemtype": "Structure_Scalar_Thermo_Interaction",
        "coupling": "Monolithic (all three fields simultaneously)",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # CONTACT MECHANICS
    # ═══════════════════════════════════════════════════════════════════════
    "contact": {
        "description": "Contact mechanics with multiple enforcement methods",
        "methods": {
            "penalty": "Penalty method (simple, parameter-dependent)",
            "lagrange": "Lagrange multiplier (exact enforcement, saddle-point)",
            "nitsche": "Nitsche method (consistent, no extra DOFs)",
            "mortar": "Mortar method (surface integration, non-matching meshes)",
        },
        "variants": ["Standard contact", "Self-contact (binary tree search)",
                     "Wear contact", "Friction (Coulomb)",
                     "TSI contact", "Poro contact", "FSI contact", "SSI contact"],
        "constitutive_laws": ["Linear", "Cubic", "Power law", "Broken rational",
                              "MIRCO (microscale)", "Python surrogate"],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # PARTICLE METHODS
    # ═══════════════════════════════════════════════════════════════════════
    "particles": {
        "description": "Particle methods: SPH, DEM, Peridynamics",
        "problemtype": "Particle",

        "sph": {
            "kernels": ["CubicSpline (default)", "QuinticSpline"],
            "eos": ["GenTait (generalized Tait)", "IdealGas"],
            "momentum": ["Adami formulation", "Monaghan formulation"],
            "density": ["Summation", "Integration", "Predict-Correct"],
            "boundary": ["Adami boundary particles", "Virtual wall particles"],
            "extra_physics": ["Surface tension (CSF)", "Phase change", "Temperature"],
        },

        "dem": {
            "contact_normal": ["LinearSpring", "LinearSpringDamp", "Hertz",
                               "LeeHerrmann", "KuwabaraKono", "Tsuji"],
            "contact_tangential": ["None", "LinearSpringDamp"],
            "rolling": ["None", "Viscous", "Coulomb"],
            "adhesion": ["None", "VdWDMT", "RegDMT"],
        },

        "peridynamics": {
            "dimensions": ["3D (Peridynamic_3D)", "2D Plane Stress (Peridynamic_2DPlaneStress)",
                          "2D Plane Strain (Peridynamic_2DPlaneStrain)"],
            "features": ["Bond-based PD", "Damage via critical stretch criterion",
                        "Volume correction factor", "Pre-crack definition via line segments"],
            "material": "MAT_ParticlePD: INITRADIUS, INITDENSITY, YOUNG, CRITICAL_STRETCH",
            "input_section": "PARTICLE DYNAMIC/PD",
            "key_params": {
                "INTERACTION_HORIZON": "delta = m * dx (typically m=3, so horizon = 3*particle_spacing)",
                "PERIDYNAMIC_GRID_SPACING": "dx (particle spacing, must match actual particle grid)",
                "PD_DIMENSION": "Peridynamic_2DPlaneStrain / Peridynamic_2DPlaneStress / Peridynamic_3D",
                "PRE_CRACKS": "Line segments: 'x1 y1 x2 y2 ; x3 y3 x4 y4' — bonds crossing these are pre-broken",
                "NORMALCONTACTLAW": "NormalLinearSpring (for impactor-body contact)",
                "NORMAL_STIFF": "Contact stiffness (e.g., 1.0e4)",
            },
            "particle_grid_generation": {
                "description": "PD requires a REGULAR GRID of particles with sufficient resolution",
                "pattern": "Loop over nx*ny (2D) or nx*ny*nz (3D) with uniform spacing dx",
                "spacing": "dx should be chosen based on the problem scale; horizon = m*dx (m=3 typical)",
                "notches_cracks": "Skip particles inside notch gaps OR use PRE_CRACKS line segments",
                "example": "for iy in range(ny): for ix in range(nx): particles.append((ix*dx, iy*dx, 0.0))",
                "convergence": "PD converges as dx→0 AND m→∞ (delta-convergence AND m-convergence)",
            },
        },

        "time_integration": ["Semi-implicit Euler (SemiImplicitEuler)", "Velocity Verlet (VelocityVerlet)"],

        "vtk_output": {
            "description": "CRITICAL: Particle VTK output must be explicitly configured",
            "yaml_section": """
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 10
IO/RUNTIME VTK OUTPUT/PARTICLES:
  PARTICLE_OUTPUT: true
  DISPLACEMENT: true
  VELOCITY: true
  ACCELERATION: false
  OWNER: true""",
            "output_format": "VTP (VTK PolyData) files, one per time step, with PVD time series",
            "pitfall": "Without IO/RUNTIME VTK OUTPUT/PARTICLES section, 4C produces NO particle output files!",
        },

        "mandatory_sph_section": {
            "description": "Even for PURE peridynamics, the SPH section is MANDATORY in 4C",
            "reason": "The PD implementation lives inside the SPH interaction framework. Without SPH section, pd_neighbor_pairs=0 → no PD forces computed",
            "yaml": """
PARTICLE DYNAMIC/SPH:
  KERNEL: QuinticSpline
  KERNEL_SPACE_DIM: Kernel2D
  INITIALPARTICLESPACING: 1.0
  BOUNDARYPARTICLEFORMULATION: AdamiBoundaryFormulation
  TRANSPORTVELOCITYFORMULATION: StandardTransportVelocity""",
        },

        "impactor_setup": {
            "description": "Rigid impactor as boundary phase particles",
            "material": "MAT_ParticleSPHBoundary: INITRADIUS, INITDENSITY",
            "phase_mapping": "PHASE_TO_MATERIAL_ID: 'boundaryphase 1 pdphase 2'",
            "velocity": "Applied via FUNCT + DIRICHLET_BOUNDARY_CONDITION on boundaryphase",
        },

        "pitfalls": [
            "SPH section is MANDATORY even for pure PD — without it, no PD bonds are found",
            "IO/RUNTIME VTK OUTPUT/PARTICLES must be added for ParaView output (VTP files)",
            "Particle grid must be REGULAR for PD (uniform spacing in all directions)",
            "INTERACTION_HORIZON must equal m*dx where m is the horizon ratio (typically 3)",
            "PERIDYNAMIC_GRID_SPACING must match the actual particle spacing exactly",
            "PRE_CRACKS uses semicolon-separated line segments: 'x1 y1 x2 y2 ; x3 y3 x4 y4'",
            "PDBODYID must be specified for PD phase particles (e.g., PDBODYID 0)",
            "Boundary phase particles (impactor) need TYPE boundaryphase, PD particles need TYPE pdphase",
            "CFL condition: dt < 0.5 * dx / c_wave where c_wave = sqrt(E/rho)",
            "BINNING STRATEGY BIN_SIZE_LOWER_BOUND must be > horizon for neighbor search",
            "DOMAINBOUNDINGBOX must enclose all particles INCLUDING impactor motion range",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # POROUS MEDIA
    # ═══════════════════════════════════════════════════════════════════════
    "porous_media": {
        "description": "Biot poroelasticity and porous flow",
        "problem_types": {
            "Poroelasticity": "Biot consolidation (structure + fluid in pores)",
            "Poroelastic_scalar_transport": "Poro + scalar transport",
            "porofluid_pressure_based": "Pressure-based porous flow (standalone)",
        },
        "coupling": ["Monolithic", "Partitioned", "1-way", "2-way"],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # CARDIOVASCULAR / BIOMEDICAL
    # ═══════════════════════════════════════════════════════════════════════
    "cardiovascular": {
        "description": "Cardiovascular and biomedical simulation capabilities",
        "models": {
            "0D_windkessel": "4-element Windkessel for arterial pressure",
            "arterial_network": "1D arterial blood flow network (artery elements)",
            "reduced_airways": "Reduced lung airways with acinus elements",
            "cardiac_monodomain": "Cardiac electrophysiology (FHN, TenTusscher, etc.)",
        },
        "applications": "Arterial hemodynamics, cardiac mechanics, lung ventilation",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # BEAM INTERACTION
    # ═══════════════════════════════════════════════════════════════════════
    "beam_interaction": {
        "description": "Beam-to-beam, beam-to-solid, beam-to-sphere contact and meshtying",
        "contact_pairs": [
            "Beam-to-beam (point coupling, tangent smoothing)",
            "Beam-to-solid volume meshtying (Gauss point, mortar)",
            "Beam-to-solid surface meshtying",
            "Beam-to-solid surface contact",
            "Beam-to-sphere contact",
        ],
        "cross_linking": "Pin-jointed, rigid-jointed, truss links (biopolymer networks)",
        "brownian_dynamics": "Stochastic dynamics of beam networks",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # LINEAR SOLVERS
    # ═══════════════════════════════════════════════════════════════════════
    "solvers": {
        "direct": {
            "UMFPACK": "Serial direct solver (recommended for small problems)",
            "SuperLU": "Parallel direct solver (SuperLU_Dist)",
            "MUMPS": "Parallel direct solver (MPI, recommended for large problems)",
            "KLU2": "Serial direct solver (alternative to UMFPACK)",
        },
        "iterative": {
            "CG": "Conjugate gradient (symmetric positive definite systems only)",
            "GMRES": "Generalized minimal residual (non-symmetric systems)",
            "BiCGSTAB": "Bi-conjugate gradient stabilized (non-symmetric, lower memory)",
        },
        "preconditioners": {
            "ILU": "Incomplete LU factorization (Ifpack package)",
            "MueLu": "Algebraic multigrid (MueLu, recommended for large problems)",
            "Block_Teko": "Block preconditioning for multi-field problems (Teko package)",
        },
        "nonlinear": {
            "NOX": "Trilinos NOX framework (Newton + line search + PTC + convergence tests)",
        },
        "yaml_example": """
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct_solver"
SOLVER 2:
  SOLVER: "Belos"
  SOLVER_XML_FILE: "iterative_gmres_template.xml"
  AZPREC: "MueLu"
  MUELU_XML_FILE: "elasticity_template.xml"
  NAME: "iterative_solver"
""",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # INPUT FILE FORMAT
    # ═══════════════════════════════════════════════════════════════════════
    "input_format": {
        "description": "YAML-based input files (.4C.yaml) — can use inline mesh or Exodus file",

        "mandatory_sections": [
            "PROBLEM SIZE (DIM: 2 or 3)",
            "PROBLEM TYPE (PROBLEMTYPE: Structure/Scalar_Transport/Fluid/...)",
            "Dynamics section matching problem type (STRUCTURAL DYNAMIC, etc.)",
            "At least one SOLVER",
            "MATERIALS",
            "Mesh (NODE COORDS + ELEMENTS, or STRUCTURE GEOMETRY with FILE)",
        ],

        "boundary_conditions": {
            "structural": {
                "DESIGN POINT/LINE/SURF/VOL DIRICH CONDITIONS": "Prescribed displacement",
                "DESIGN POINT/LINE/SURF/VOL NEUMANN CONDITIONS": "Applied force/traction/body force",
            },
            "thermal": {
                "DESIGN SURF/VOL THERMO DIRICH CONDITIONS": "Prescribed temperature",
                "DESIGN SURF/VOL THERMO NEUMANN CONDITIONS": "Applied heat flux",
            },
            "bc_format": """
DESIGN SURF DIRICH CONDITIONS:
  - E: 1            # Design entity ID
    NUMDOF: 3       # Number of DOFs per node
    ONOFF: [1, 1, 0] # Which DOFs are constrained (1=yes, 0=no)
    VAL: [0.0, 0.0, 0.0]  # Prescribed values
    FUNCT: [0, 0, 0]       # Time function IDs (0=constant)
""",
        },

        "topology_sections": {
            "DNODE-NODE TOPOLOGY": "Map single nodes to design nodes (for point BCs)",
            "DLINE-NODE TOPOLOGY": "Map nodes to design lines (for line BCs in 2D)",
            "DSURF-NODE TOPOLOGY": "Map nodes to design surfaces (for surface BCs in 3D)",
            "DVOL-NODE TOPOLOGY": "Map nodes to design volumes (for volume BCs)",
        },

        "functions": """
# --- Simple space-time function (no time-varying sub-variables) ---
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "sin(2*pi*x)*cos(pi*t)"
  # Supports: x, y, z, t as variables

# --- Function with VARIABLE (e.g. ramp-up) ---
# IMPORTANT: COMPONENT: 0 is REQUIRED when using VARIABLE/multifunction.
# Without COMPONENT, the VARIABLE definition is NOT parsed correctly
# and the function silently returns wrong values.
FUNCT2:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "6*U_bar*y*(H-y)/(H*H)*a"
  - VARIABLE: 0
    NAME: "a"
    TYPE: "multifunction"
    NUMPOINTS: 3
    TIMES: [0, 2, 10000]
    DESCRIPTION: ["0.5*(1-cos(pi*t/2))", "1.0"]

# --- Pure time function (no COMPONENT needed) ---
FUNCT3:
  - SYMBOLIC_FUNCTION_OF_TIME: "a"
  - VARIABLE: 0
    NAME: "a"
    TYPE: "multifunction"
    NUMPOINTS: 3
    TIMES: [0, 1, 10000]
    DESCRIPTION: ["0.5*(1.0-cos((t*pi)/1.0))", "1.0"]

# --- Linear interpolation (piecewise linear in time) ---
FUNCT4:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "1*a"
  - VARIABLE: 0
    NAME: "a"
    TYPE: "linearinterpolation"
    NUMPOINTS: 3
    TIMES: [0, 1, 101]
    VALUES: [0, 1, 100]
""",

        "inline_mesh_example": """
NODE COORDS:
  - "NODE 1 COORD 0.000000 0.000000 0.0"
  - "NODE 2 COORD 1.000000 0.000000 0.0"
TRANSPORT ELEMENTS:
  - "1 TRANSP QUAD4 1 2 3 4 MAT 1 TYPE Std"
""",

        "general_pitfalls": [
            # ExodusII block IDs
            "CRITICAL: meshio (Python) writes ExodusII element block IDs starting "
            "at 0 (0-indexed), but 4C YAML ELEMENT_BLOCKS use 1-indexed IDs.  "
            "Getting this wrong produces cryptic errors like 'Pressure map empty'.  "
            "Fix: after writing with meshio, patch with netCDF4: "
            "import netCDF4; ds=netCDF4.Dataset('mesh.e','r+'); "
            "ds.variables['eb_prop1'][:] += 1; ds.close()  "
            "Or verify IDs: python3 -c \"import meshio; m=meshio.read('mesh.e'); "
            "print([c.type for c in m.cells])\"",

            # FUNCT COMPONENT requirement
            "SYMBOLIC_FUNCTION_OF_SPACE_TIME with VARIABLE/multifunction "
            "REQUIRES 'COMPONENT: 0' in the same list item.  Without COMPONENT, "
            "the VARIABLE definition is silently ignored and the function returns "
            "wrong values.  SYMBOLIC_FUNCTION_OF_TIME (pure time functions) do "
            "NOT need COMPONENT.",

            # Shared-node NUMDOF conflict
            "In multi-physics problems (FSI, TSI, SSI), DESIGN ... DIRICH "
            "CONDITIONS apply to ALL discretizations containing a node.  If a "
            "node is shared between structure (NUMDOF=2 in 2D) and fluid "
            "(NUMDOF=3), applying a structural Dirichlet with NUMDOF=2 on that "
            "node will fail because the fluid discretization expects NUMDOF=3.  "
            "Solutions: (a) use separate node sets, (b) offset meshes to avoid "
            "shared nodes at Dirichlet boundaries, (c) use mortar coupling.",

            # Invalid section names
            "4C is strict about section names.  Common invalid sections: "
            "EVERY_ITERATION (not a valid IO parameter), "
            "DESIGN FLUID LINE LIFT&DRAG (does not exist for 2D), "
            "DESIGN THERMO LINE DIRICH CONDITIONS (wrong — must be "
            "DESIGN LINE THERMO DIRICH CONDITIONS).  "
            "Check valid names with: 4C --parameters | grep DESIGN",

            # Output
            "4C writes native .control/.mesh/.result files.  To get VTU, either: "
            "(a) add IO/RUNTIME VTK OUTPUT sections (recommended), or "
            "(b) run post_vtu --file=output_prefix after the simulation.",

            # 2026-06-01: .dat extension rejected
            "[Syntax] 4C 2026.3.0-dev accepts ONLY .yaml / .yml / .json input "
            "files. Passing a legacy .dat-format file (or any other extension) "
            "is rejected at file-open time with 'Cannot infer format of input "
            "file ... Only .yaml, .yml, and .json are supported.' from "
            "core/io/src/4C_io_input_file.cpp. Note: the section-name "
            "vocabulary (PROBLEM TYPE, STRUCTURAL DYNAMIC, DESIGN SURF NEUMANN, "
            "MAT_scatra, etc.) is unchanged — those are still valid as YAML "
            "keys; only the overall file format moved from dat-style "
            "section-header text to YAML mapping syntax. Signal: 4C ERROR "
            "from 4C_io_input_file.cpp with 'Cannot infer format' and "
            "'Only .yaml, .yml, and .json' substrings when an unsupported "
            "extension is passed on the CLI. (Verified empirically 2026-06-01.)",

            "[Syntax] 4C validates enum-like keys against an allowed set at "
            "input-parse time. Mis-spelling a PROBLEMTYPE value (e.g. "
            "'Hyperelasticity' instead of 'Structure', or a typo like "
            "'Scalar_Tranzport') triggers 'PROC 0 ERROR ... Could not match "
            "this input' from core/io/src/4C_io_input_spec_builders.cpp, "
            "with the offending YAML block echoed in the message. Signal: "
            "the substrings 'Could not match this input', 'PROBLEMTYPE', and "
            "'input_spec_builders' all appear in 4C stderr when the value is "
            "not in the allowed enum set. (Verified empirically 2026-06-01.)",

            # THICK parameter for 2D plane strain
            "For 2D plane strain WALL elements, THICK is the out-of-plane depth "
            "(unit thickness), NOT the element width or column width. Almost always "
            "THICK: 1.0. Getting this wrong silently scales all forces/stresses.",

            # 2D VTK output artifacts — applies to fluid AND porofluid
            "In 2D simulations, fluid AND porofluid VTK output may show NaN for "
            "pressure and garbage for the z-velocity component. This is a VTK "
            "output artifact, NOT divergence. Read pressure from native HDF5 result "
            "files instead. This affects fluid, poro, and FSI problems in 2D.",

            # Poro-specific
            "4C poro uses a DYNAMIC formulation (with inertia) even for quasi-static "
            "problems. Use slow load ramps (>10x wave traversal time) to avoid "
            "elastic waves. For 1D consolidation, ramp time >> H/sqrt(E/rho).",


            # WALL element types
            "WALL TRI3 and WALL TRI6 do NOT exist in 4C.  For 2D structural "
            "elements, only QUAD variants are available (QUAD4, QUAD8, QUAD9).  "
            "If your mesh generator produces triangles in the structural domain, "
            "use transfinite meshing or recombination to get all-QUAD elements.",

            # FSI mesh requirements
            "For monolithic FSI: the structure and fluid meshes must have SEPARATE "
            "nodes at the FSI interface (not shared conforming nodes).  A single "
            "Gmsh mesh shares nodes — you must post-process to duplicate interface "
            "nodes, remapping fluid element connectivity and boundary node sets.  "
            "Alternative: use mortar coupling (iter_mortar_monolithicfluidsplit) "
            "which handles non-matching meshes natively.",

            # Large inline YAML performance
            "For meshes with >200 nodes, use an ExodusII mesh file (.e) instead "
            "of inline NODE COORDS + ELEMENTS sections. Inline YAML with 1000+ "
            "lines is slow to generate and can cause MCP transport timeouts. "
            "Use meshio to write the mesh to .e format, then reference it with "
            "STRUCTURE GEOMETRY: FILE: mesh.e",

            # FSI + runtime VTK
            "IO/RUNTIME VTK OUTPUT/STRUCTURE may be incompatible with FSI — "
            "FSI overrides INT_STRATEGY internally.  If structure VTK output "
            "causes errors, remove it and use post_vtu for post-processing instead.",

            # GPU / hardware acceleration
            "4C linear algebra is CPU-only (Epetra-based, Trilinos 16.2.0). "
            "Epetra does NOT support GPU execution. Tpetra (GPU-capable via "
            "Kokkos CUDA/HIP/SYCL backends) is not yet integrated. Do NOT "
            "expect GPU speedup for assembly or linear solves.",

            # ArborX optional GPU component
            "The only GPU-accelerated component in 4C is ArborX (optional, "
            "OFF by default), used for geometric search (bounding volume "
            "hierarchy queries in contact/particle problems). Enable with "
            "-DFOUR_C_WITH_ARBORX=ON and a Kokkos GPU backend in Trilinos. "
            "This does NOT accelerate the solver itself.",

            # MPI parallelism
            "4C supports MPI parallelism for domain decomposition. Use "
            "mpirun -np N 4C input.4C.yaml for parallel runs. MPI is the "
            "primary parallelism mechanism for large-scale problems. "
            "Thread-level parallelism uses OpenMP (set OMP_NUM_THREADS).",
        ],

        "element_type_per_physics": {
            "FLUID (2D)": ["QUAD4", "QUAD9", "TRI3", "TRI6"],
            "FLUID (3D)": ["HEX8", "HEX20", "HEX27", "TET4", "TET10", "NURBS27"],
            "WALL (2D structure)": ["QUAD4", "QUAD8", "QUAD9"],
            # NOTE: WALL TRI3 and WALL TRI6 do NOT exist in 4C.
            # For 2D structural elements, only QUAD variants are supported.
            "SOLID (3D structure)": ["HEX8", "HEX20", "HEX27", "TET4", "TET10",
                                     "WEDGE6", "PYRAMID5"],
            "TRANSP (scalar transport)": ["QUAD4", "QUAD9", "HEX8", "HEX27",
                                          "TRI3", "TRI6", "TET4", "TET10"],
            "SOLIDSCATRA (TSI/SSI)": ["HEX8", "TET4", "TET10", "HEX27"],
            "ALE (2D)": ["QUAD4", "TRI3"],
            "ALE (3D)": ["HEX8", "TET4"],
            "PORO (2D)": ["WALLQ4PORO", "WALLQ9PORO"],
            "PORO (3D)": ["SOLIDH8PORO", "SOLIDT4PORO", "SOLIDH27PORO"],
            "BEAM": ["BEAM3R LINE2", "BEAM3EB LINE2", "BEAM3R LINE3"],
            "ARTERY": ["ARTERY LINE2"],
            "notes": (
                "QUAD4 is the workhorse element for most 2D problems.  "
                "HEX8 for 3D.  Higher-order elements (QUAD9, HEX27) give "
                "better accuracy but are slower.  TRI3/TET4 are available "
                "but less accurate for pressure in fluid problems."
            ),
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # CELL TYPES
    # ═══════════════════════════════════════════════════════════════════════
    "cell_types": {
        "1D": ["line2", "line3", "line4", "line5", "line6", "point1"],
        "2D": ["quad4", "quad6", "quad8", "quad9", "tri3", "tri6"],
        "3D": ["hex8", "hex16", "hex18", "hex20", "hex27", "tet4", "tet10",
               "wedge6", "wedge15", "pyramid5"],
        "NURBS": ["nurbs2", "nurbs3 (1D)", "nurbs4", "nurbs9 (2D)",
                  "nurbs8", "nurbs27 (3D)"],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # XFEM
    # ═══════════════════════════════════════════════════════════════════════
    "xfem": {
        "description": "Extended Finite Element Method for interface problems",
        "capabilities": [
            "Level-set based interfaces (weak Dirichlet, Neumann, Navier slip, two-phase)",
            "Surface-based interfaces (displacement, FSI, FPI)",
            "Robin conditions (Dirichlet/Neumann)",
            "Edge stabilization",
            "Semi-Lagrangean time integration",
        ],
        "applications": "Fluid-XFEM, FSI-XFEM (no mesh conformity at interface)",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # ALL 40 PROBLEM TYPES
    # ═══════════════════════════════════════════════════════════════════════
    "all_problem_types": {
        "Structure": "Structural mechanics",
        "Scalar_Transport": "Convection-diffusion / scalar transport",
        "Thermo": "Pure thermal analysis",
        "Fluid": "Incompressible Navier-Stokes",
        "Fluid_Ale": "Fluid on ALE mesh",
        "Ale": "Pure ALE mesh movement",
        "Fluid_Structure_Interaction": "FSI (standard)",
        "Fluid_Structure_Interaction_XFEM": "FSI with XFEM",
        "Thermo_Structure_Interaction": "TSI",
        "Structure_Scalar_Interaction": "SSI (electrode mechanics, etc.)",
        "Structure_Scalar_Thermo_Interaction": "SSTI (three-field)",
        "Scalar_Thermo_Interaction": "STI",
        "Fluid_Beam_Interaction": "3D fluid + 1D beam",
        "Fluid_Porous_Structure_Interaction": "FPSI",
        "Particle": "SPH / DEM / Peridynamics",
        "Particle_Structure_Interaction": "PASI",
        "Poroelasticity": "Biot poroelasticity",
        "Poroelastic_scalar_transport": "Poro + scalar",
        "Level_Set": "Level-set interface tracking",
        "Low_Mach_Number_Flow": "Variable-density flow",
        "Lubrication": "Thin film lubrication",
        "Elastohydrodynamic_Lubrication": "EHL coupling",
        "Electrochemistry": "Nernst-Planck electrochemistry",
        "ArterialNetwork": "1D arterial blood flow",
        "ReducedDimensionalAirWays": "Lung airways",
        "Cardiac_Monodomain": "Cardiac electrophysiology",
        "Biofilm_Fluid_Structure_Interaction": "Biofilm FSI",
        "Gas_Fluid_Structure_Interaction": "Gas + FSI",
        "Polymer_Network": "Polymer network",
    },
}
