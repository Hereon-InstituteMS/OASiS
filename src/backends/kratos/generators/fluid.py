"""Kratos fluid dynamics generators and knowledge."""


# NOTE (2026-06-26 honesty audit): the previous _fluid_cavity_kratos
# generator was an availability-probe stub (import-check + {"note": ...},
# no solver run). FluidDynamicsApplication is NOT importable in the installed
# Kratos stack, so 'fluid' has been removed from the generator registry and
# from KratosBackend.supported_physics(). KNOWLEDGE retained for reference.


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
                        "Signal: RuntimeError 'This container only can store the variables specified in its variables list. The variables list doesn't have this variable: VELOCITY' from kratos/containers/variables_list_data_value_container at the first GetSolutionStepValue / SetSolutionStepValue on the node. (Verified empirically 2026-06-01 — the prior wording 'not found in variables list of ModelPart' is rearranged; the real text says 'variables list doesn't have this variable' and originates in the container code, not the VMS InitializeSolutionStep.)",
                        '[Integration] Materials defined in FluidMaterials.json with DENSITY and DYNAMIC_VISCOSITY. Forgetting either key leaves the constitutive call returning zero stress. '
                        'Signal: the FluidDynamicsApplication solver converges to a zero PRESSURE field with uniform VELOCITY equal to the inlet BC (no momentum balance enforced) — VMS / QSVMS reports residual < tol despite trivial flow.',
                        '[Physics] Wall BCs are mutually exclusive: no-slip (VELOCITY = 0), Navier slip (tangential traction), or wall-law (log-law for high Re). Mixing them on the same boundary applies the LAST one written in the JSON. '
                        'Signal: integrated wall shear stress disagrees with analytic Couette/Poiseuille by an order of magnitude.',
                        '[Physics] Inlet: impose VELOCITY vector; Outlet: impose PRESSURE = 0. Reversing them (pressure inlet, velocity outlet) over-determines pressure and gives the wrong mass flux. '
                        "Signal: integrated mass flow rate computed from VELOCITY on the outlet SubModelPart is 0 or oscillates around 0; navier_stokes_solver_vmsmonolithic ResidualBasedBlockBuilderAndSolver reports the pressure residual stalling near the inlet face.",
                        '[Numerical] For free-surface (TwoFluid solver): the DISTANCE variable (level-set signed distance) must be initialised before the first solve and re-distanced every step. Skipping the re-initialisation drifts the interface. '
                        'Signal: DISTANCE field develops spurious zero-crossings inside the bulk fluid; post-processed phase indicator shows artefacts.',
                    ],
    },
}

# Empty: FluidDynamicsApplication not installable in this Kratos stack; the
# prior generator was a no-solve probe stub (removed).
GENERATORS = {}
