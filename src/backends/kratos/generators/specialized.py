"""Kratos specialized application generators and knowledge.

Covers: PoroMechanics, ShallowWater, WindEngineering, Dam, ConstitutiveLaws,
ThermalDEM, SwimmingDEM, DEM-Structures, FEM-DEM, CableNet, Chimera,
Droplet, FreeSurface, FluidBiomedical, FluidHydraulics, Optimization.
"""


def _generic_kratos_template(app_name: str, pip_name: str, capabilities: list) -> str:
    """Generate a generic Kratos application check template."""
    caps_str = str(capabilities)
    return f'''\
"""{app_name} — Kratos"""
import json
try:
    import KratosMultiphysics as KM
    import KratosMultiphysics.{app_name}
    print("{app_name} available")
    summary = {{"note": "{app_name} available", "capabilities": {caps_str}}}
except ImportError:
    print("{app_name} not installed")
    print("Install: pip install {pip_name}")
    summary = {{"note": "not installed"}}
with open("results_summary.json", "w") as f: json.dump(summary, f, indent=2)
'''


def _poromechanics_2d(params: dict) -> str:
    return _generic_kratos_template("PoromechanicsApplication",
        "KratosPoromechanicsApplication",
        ["consolidation", "fracture_propagation", "dam_engineering", "tunneling"])

def _shallow_water_2d(params: dict) -> str:
    return _generic_kratos_template("ShallowWaterApplication",
        "KratosShallowWaterApplication",
        ["shallow_water_equations", "saint_venant", "dam_break_2d", "flood_simulation"])

def _wind_engineering_2d(params: dict) -> str:
    return _generic_kratos_template("WindEngineeringApplication",
        "KratosWindEngineeringApplication",
        ["wind_loading", "atmospheric_boundary_layer", "vortex_shedding_wind"])

def _dam_2d(params: dict) -> str:
    return _generic_kratos_template("DamApplication",
        "KratosDamApplication",
        ["thermal_dam", "mechanical_dam", "thermo_mechanical_dam", "seepage"])

def _constitutive_laws_2d(params: dict) -> str:
    return _generic_kratos_template("ConstitutiveLawsApplication",
        "KratosConstitutiveLawsApplication",
        ["hyperelastic_models", "plasticity_models", "damage_models",
         "viscoplasticity", "small_strain_isotropic_plasticity"])

def _thermal_dem_2d(params: dict) -> str:
    return _generic_kratos_template("ThermalDEMApplication",
        "KratosThermalDEMApplication",
        ["heat_conduction_particles", "convection_radiation_particles",
         "sintering", "thermal_granular_flow"])

def _swimming_dem_2d(params: dict) -> str:
    return _generic_kratos_template("SwimmingDEMApplication",
        "KratosSwimmingDEMApplication",
        ["particle_laden_flow", "fluidized_bed", "sedimentation",
         "drag_models", "CFD_DEM_coupling"])

def _dem_structures_2d(params: dict) -> str:
    return _generic_kratos_template("DemStructuresCouplingApplication",
        "KratosDemStructuresCouplingApplication",
        ["DEM_FEM_coupling", "impact_on_structures", "blast_loading"])

def _fem_to_dem_2d(params: dict) -> str:
    return _generic_kratos_template("FemToDemApplication",
        "KratosFemToDemApplication",
        ["fracture_FEM_to_DEM", "progressive_fracture", "concrete_fracture"])

def _cable_net_2d(params: dict) -> str:
    return _generic_kratos_template("CableNetApplication",
        "KratosCableNetApplication",
        ["cable_elements", "net_structures", "membrane_cable_coupling", "form_finding"])

def _chimera_2d(params: dict) -> str:
    return _generic_kratos_template("ChimeraApplication",
        "KratosChimeraApplication",
        ["overset_grids", "chimera_method", "moving_bodies_in_flow"])

def _droplet_2d(params: dict) -> str:
    return _generic_kratos_template("DropletDynamicsApplication",
        "KratosDropletDynamicsApplication",
        ["droplet_impact", "spreading", "contact_angle", "two_phase_droplet"])

def _free_surface_2d(params: dict) -> str:
    return _generic_kratos_template("FreeSurfaceApplication",
        "KratosFreeSurfaceApplication",
        ["free_surface_flow", "wave_propagation", "sloshing_Eulerian"])

def _fluid_biomedical_2d(params: dict) -> str:
    return _generic_kratos_template("FluidDynamicsBiomedicalApplication",
        "KratosFluidDynamicsBiomedicalApplication",
        ["blood_flow", "hemodynamics", "wall_shear_stress", "aneurysm_flow"])

def _fluid_hydraulics_2d(params: dict) -> str:
    return _generic_kratos_template("FluidDynamicsHydraulicsApplication",
        "KratosFluidDynamicsHydraulicsApplication",
        ["open_channel_flow", "pipe_flow", "hydraulic_structures", "spillway_flow"])

def _optimization_2d(params: dict) -> str:
    return _generic_kratos_template("OptimizationApplication",
        "KratosOptimizationApplication",
        ["gradient_based_optimization", "response_functions", "constraint_handling",
         "multi_objective", "adjoint_sensitivity"])


KNOWLEDGE = {
    "poromechanics": {
        "description": "Poromechanics: consolidation, fracture in porous media, dam/tunnel engineering",
        "application": "PoromechanicsApplication",
        "elements": ["SmallStrainUPwDiffOrderElement2D6N", "SmallStrainUPwDiffOrderElement3D10N"],
        "capabilities": ["u-pw coupling", "fracture_propagation", "interface_elements"],
        "pitfalls": [
                        '[Physics] Different from GeoMechanicsApplication — this focuses on fracture in porous media '
                        'Signal: post-processed quantity (max displacement, integrated flux, pressure) disagrees with analytic / textbook reference by 10-100%.',
                    ]    },
    "shallow_water": {
        "description": "Shallow water equations (Saint-Venant) for flood/dam-break/coastal simulation",
        "application": "ShallowWaterApplication",
        # Real registered names in KratosShallowWaterApplication
        # are BoussinesqElement2D{3,4}N — NOT ShallowWaterElement.
        # Catalog drift caught 2026-06-01 by kratos_eletype_
        # scanner; corrected per the rans_shallowwater_element_
        # naming Tier-2 fixture.
        "elements": [
            "BoussinesqElement2D3N",
            "BoussinesqElement2D4N",
        ],
        "solver_types": ["explicit", "semi-implicit"],
        "pitfalls": [
                        '[API] The KratosShallowWaterApplication '
                        'element name is BoussinesqElement2D3N / '
                        'BoussinesqElement2D4N, NOT '
                        'ShallowWaterElement2D3N. The Application '
                        'class is named "ShallowWater" but the '
                        'underlying element registration uses the '
                        '"Boussinesq" stem (after the Boussinesq '
                        'equations underlying the depth-averaged '
                        'formulation). '
                        "Signal: model_part.CreateNewElement("
                        "\"ShallowWaterElement2D3N\", ...) raises "
                        "'is not registered' from kratos/python/"
                        "add_model_part_to_python.cpp:173; the same "
                        "call with 'BoussinesqElement2D3N' succeeds. "
                        "(Verified empirically 2026-06-01 — same "
                        "Tier-2 fixture as the RANS naming entry, "
                        "rans_shallowwater_element_naming in "
                        "scripts/tier2_fixtures/kratos/.)",
                        '[Numerical] 2D only (depth-averaged) '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Wetting/drying needs special treatment '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Friction: Manning formula with roughness coefficient '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                    ],
    },
    "wind_engineering": {
        "description": "Wind engineering: atmospheric boundary layer, wind loading on structures",
        "application": "WindEngineeringApplication",
        "capabilities": ["ABL_inlet_generation", "wind_pressure_coefficients", "vortex_shedding"],
        "pitfalls": [
                        '[Numerical] Requires FluidDynamicsApplication + RANSApplication '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                    ],
    },
    "dam": {
        "description": "Dam engineering: thermal-mechanical analysis, seepage, cracking",
        "application": "DamApplication",
        "capabilities": ["thermal_analysis", "mechanical_analysis", "thermo_mechanical_coupled",
                         "seepage_analysis", "joint_elements"],
    },
    "constitutive_laws": {
        "description": "Extended constitutive law library: hyperelastic, plasticity, damage, viscoplastic",
        "application": "ConstitutiveLawsApplication",
        "laws": {
            "hyperelastic": ["Ogden", "Yeoh", "Arruda-Boyce", "Blatz-Ko"],
            "plasticity": ["VonMises", "Tresca", "DruckerPrager", "MohrCoulomb",
                           "ModifiedCamClay", "CriticalStateLine"],
            "damage": ["Mazars", "SimoJu", "RankineFragile", "ModifiedMohrCoulomb"],
            "viscoplastic": ["Perzyna", "DruckerPragerViscoplastic"],
        },
        "pitfalls": [
                        '[Numerical] These laws extend StructuralMechanicsApplication '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[API] Must be registered via constitutive_law.name in MaterialsDEM.json '
                        "Signal: RuntimeError or TypeError from the Kratos binding (e.g. 'not registered', 'incompatible function arguments') when the API call is made.",
                    ],
    },
    "thermal_dem": {
        "description": "Thermal DEM: heat transfer between particles (conduction, convection, radiation)",
        "application": "ThermalDEMApplication",
        "capabilities": ["particle_heat_conduction", "convection", "radiation", "sintering"],
        "pitfalls": [
                        '[Numerical] Requires DEMApplication as base '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Physics] Temperature DOF per particle '
                        'Signal: post-processed quantity (max displacement, integrated flux, pressure) disagrees with analytic / textbook reference by 10-100%.',
                    ],
    },
    "swimming_dem": {
        "description": "Swimming DEM: particles in fluid flow (CFD-DEM coupling)",
        "application": "SwimmingDEMApplication",
        "capabilities": ["particle_laden_flow", "fluidized_bed", "sedimentation",
                         "Schiller-Naumann_drag", "virtual_mass", "Basset_history"],
        "pitfalls": [
                        '[Numerical] Requires FluidDynamicsApplication + DEMApplication '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Two-way coupling: particles affect fluid momentum '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                    ],
    },
    "dem_structures_coupling": {
        "description": "DEM-FEM coupling: particle impact on deformable structures",
        "application": "DemStructuresCouplingApplication",
        "capabilities": ["impact_loading", "blast_on_structures", "wear"],
        "pitfalls": [
                        '[Numerical] Requires DEMApplication + StructuralMechanicsApplication '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                    ],
    },
    "fem_to_dem": {
        "description": "FEM-to-DEM transition: continuum fracture → discrete particles",
        "application": "FemToDemApplication",
        "capabilities": ["progressive_fracture", "concrete_cracking", "rock_fragmentation"],
        "pitfalls": [
                        '[Numerical] Mesh-dependent fracture — requires damage regularization '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                    ]    },
    "cable_net": {
        "description": "Cable and net structures: cables, membranes, form-finding",
        "application": "CableNetApplication",
        "elements": ["CableElement3D2N", "MembraneElement3D3N", "MembraneElement3D4N"],
        "capabilities": ["form_finding", "prestress", "wind_loading_on_cables"],
    },
    "chimera": {
        "description": "Chimera/overset grid method for moving bodies in flow",
        "application": "ChimeraApplication",
        "capabilities": ["overset_grids", "moving_bodies", "interpolation_at_interfaces"],
        "pitfalls": [
                        '[Numerical] Requires FluidDynamicsApplication '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Hole-cutting algorithm needed '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Conservation at chimera boundaries is approximate '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                    ],
    },
    "droplet_dynamics": {
        "description": "Droplet dynamics: impact, spreading, contact angles",
        "application": "DropletDynamicsApplication",
        "capabilities": ["droplet_impact", "contact_angle", "surface_tension", "two_phase"],
    },
    "free_surface": {
        "description": "Free-surface flow (Eulerian approach)",
        "application": "FreeSurfaceApplication",
        "capabilities": ["free_surface_tracking", "wave_propagation", "sloshing"],
    },
    "fluid_biomedical": {
        "description": "Biomedical fluid dynamics: blood flow, hemodynamics",
        "application": "FluidDynamicsBiomedicalApplication",
        "capabilities": ["blood_flow", "WSS_computation", "aneurysm_risk", "stent_flow"],
        "pitfalls": [
                        '[Numerical] Non-Newtonian blood models (Carreau-Yasuda) '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Patient-specific geometry from CT/MRI '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                    ],
    },
    "fluid_hydraulics": {
        "description": "Hydraulic fluid dynamics: open channels, pipes, spillways",
        "application": "FluidDynamicsHydraulicsApplication",
        "capabilities": ["open_channel", "pipe_network", "spillway", "hydraulic_jump"],
    },
    "optimization": {
        "description": "General optimization framework: gradient-based, adjoint, multi-objective",
        "application": "OptimizationApplication",
        "capabilities": ["gradient_based", "adjoint_sensitivity", "constraint_handling",
                         "multi_objective", "response_function_library"],
        "pitfalls": [
                        '[Numerical] Adjoint requires application-specific adjoint solver support '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                    ],
    },
}

GENERATORS = {
    "poromechanics_2d": _poromechanics_2d,
    "shallow_water_2d": _shallow_water_2d,
    "wind_engineering_2d": _wind_engineering_2d,
    "dam_2d": _dam_2d,
    "constitutive_laws_2d": _constitutive_laws_2d,
    "thermal_dem_2d": _thermal_dem_2d,
    "swimming_dem_2d": _swimming_dem_2d,
    "dem_structures_2d": _dem_structures_2d,
    "fem_to_dem_2d": _fem_to_dem_2d,
    "cable_net_2d": _cable_net_2d,
    "chimera_2d": _chimera_2d,
    "droplet_dynamics_2d": _droplet_2d,
    "free_surface_2d": _free_surface_2d,
    "fluid_biomedical_2d": _fluid_biomedical_2d,
    "fluid_hydraulics_2d": _fluid_hydraulics_2d,
    "optimization_2d": _optimization_2d,
}
