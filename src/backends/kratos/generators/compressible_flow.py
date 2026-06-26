"""Kratos compressible flow generators and knowledge.

Covers compressible potential flow and compressible Euler/Navier-Stokes.
Applications: CompressiblePotentialFlowApplication.
"""


# NOTE (2026-06-26 honesty audit): the previous _compressible_potential_2d
# generator was an availability-probe stub (import-check +
# {"note": "not installed"}, no solver run). CompressiblePotentialFlow-
# Application is NOT importable in the installed Kratos stack, so
# 'compressible_potential' has been removed from the generator registry and
# from KratosBackend.supported_physics(). KNOWLEDGE retained for reference.


KNOWLEDGE = {
    "compressible_potential": {
        "description": "Compressible potential flow (subsonic/transonic) around aerodynamic bodies",
        "application": "CompressiblePotentialFlowApplication",
        "elements": {
            "2D": ["IncompressiblePotentialFlowElement2D3N", "CompressiblePotentialFlowElement2D3N",
                   "TransonicPerturbationPotentialFlowElement2D3N"],
            "3D": ["IncompressiblePotentialFlowElement3D4N", "CompressiblePotentialFlowElement3D4N"],
        },
        "solver_types": ["potential_flow_solver (linear/nonlinear)"],
        "pitfalls": [
                        '[Numerical] Far-field BC: use PotentialWallCondition for solid walls '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Physics] Freestream: set FREESTREAM_VELOCITY and MACH_INFINITY '
                        'Signal: the post-processed VtkOutput .post.bin shows the integrated_flux / max_displacement / PRESSURE channels disagreeing with analytic / textbook reference by 10-100%.',
                        '[Numerical] Transonic: requires shock-capturing stabilization '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Physics] Lift/drag computed from pressure integration on body surface '
                        'Signal: the post-processed VtkOutput .post.bin shows the integrated_flux / max_displacement / PRESSURE channels disagreeing with analytic / textbook reference by 10-100%.',
                    ],
    },
}

# Empty: CompressiblePotentialFlowApplication not installable in this Kratos
# stack; the prior generator was a no-solve probe stub (removed).
GENERATORS = {}
