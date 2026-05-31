"""Kratos fluid dynamics generators and knowledge."""


def _fluid_cavity_kratos(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Navier-Stokes via Kratos FluidDynamicsApplication."""
    return f'''\
"""Navier-Stokes — Kratos FluidDynamicsApplication"""
import json
try:
    import KratosMultiphysics as KM
    import KratosMultiphysics.FluidDynamicsApplication as FDA
    print("FluidDynamicsApplication available")
    # Full Kratos fluid analysis uses:
    # from KratosMultiphysics.FluidDynamicsApplication.fluid_dynamics_analysis import FluidDynamicsAnalysis
    # Requires: ProjectParameters.json + FluidMaterials.json + mesh.mdpa
    summary = {{"note": "Kratos FDA available — use FluidDynamicsAnalysis for full simulation"}}
except ImportError:
    print("FluidDynamicsApplication not installed")
    print("Install: pip install KratosFluidDynamicsApplication")
    summary = {{"note": "KratosFluidDynamicsApplication not installed"}}
with open("results_summary.json", "w") as _f: json.dump(summary, _f, indent=2)
'''


KNOWLEDGE = {
    "fluid": {
        "description": "Incompressible Navier-Stokes via FluidDynamicsApplication (FDA)",
        "application": "FluidDynamicsApplication (pip install KratosFluidDynamicsApplication)",
        "elements": {
            "stabilized": ["VMS2D3N/3D4N (Variational Multiscale)",
                          "QSVMS2D3N/3D4N (Quasi-static VMS, default)"],
            "fractional_step": ["FractionalStep2D3N/3D4N"],
            "two_fluid": ["TwoFluidNavierStokes2D3N/3D4N (level-set free surface)"],
        },
        "solver_types": ["monolithic (navier_stokes_solver_vmsmonolithic)",
                        "fractional_step (navier_stokes_solver_fractionalstep)"],
        "stabilization": {
            "ASGS": "oss_switch=0 (Algebraic SubGrid Scale)",
            "OSS": "oss_switch=1 (Orthogonal SubScale)",
            "dynamic_tau": "Automatic stabilization parameter",
        },
        "turbulence": "k-epsilon, k-omega SST via RANSApplication",
        "pitfalls": [
                        '[API] VELOCITY (vector) and PRESSURE (scalar) are the primary fluid variables — both must be added to the ModelPart via AddNodalSolutionStepVariable BEFORE any Node is created. '
                        "Signal: RuntimeError 'variable VELOCITY/PRESSURE not found in variables list of ModelPart' from the VMS element InitializeSolutionStep.",
                        '[Integration] Materials defined in FluidMaterials.json with DENSITY and DYNAMIC_VISCOSITY. Forgetting either key leaves the constitutive call returning zero stress. '
                        'Signal: solver converges to a zero pressure field with uniform velocity equal to BC (no momentum balance enforced).',
                        '[Physics] Wall BCs are mutually exclusive: no-slip (VELOCITY = 0), Navier slip (tangential traction), or wall-law (log-law for high Re). Mixing them on the same boundary applies the LAST one written in the JSON. '
                        'Signal: integrated wall shear stress disagrees with analytic Couette/Poiseuille by an order of magnitude.',
                        '[Physics] Inlet: impose VELOCITY vector; Outlet: impose PRESSURE = 0. Reversing them (pressure inlet, velocity outlet) over-determines pressure and gives the wrong mass flux. '
                        'Signal: integrated outlet flow rate is 0 or oscillates; convergence stalls near the inlet face.',
                        '[Numerical] For free-surface (TwoFluid solver): the DISTANCE variable (level-set signed distance) must be initialised before the first solve and re-distanced every step. Skipping the re-initialisation drifts the interface. '
                        'Signal: DISTANCE field develops spurious zero-crossings inside the bulk fluid; post-processed phase indicator shows artefacts.',
                    ],
    },
}

GENERATORS = {
    "fluid_2d_cavity": _fluid_cavity_kratos,
}
