"""Cardiac Monodomain generator for 4C.

Covers cardiac electrophysiology using the monodomain equation.  The
monodomain model is a reaction-diffusion PDE that describes the propagation
of the transmembrane potential through cardiac tissue.  It couples with
ionic cell models (e.g. Aliev-Panfilov, ten Tusscher) that represent the
local ionic currents at each point in the myocardium.  Applications include
simulation of action potential propagation, arrhythmia modeling, and
cardiac resynchronization therapy planning.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class CardiacMonodomainGenerator(BaseGenerator):
    """Generator for cardiac monodomain electrophysiology problems in 4C."""

    module_key = "cardiac_monodomain"
    display_name = "Cardiac Monodomain (Electrophysiology)"
    problem_type = "Cardiac_Monodomain"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The cardiac monodomain module solves a reaction-diffusion "
                "equation for the transmembrane potential in cardiac tissue.  "
                "The monodomain equation is a simplified model derived from "
                "the bidomain equations under the assumption of equal "
                "anisotropy ratios in intra- and extracellular conductivity "
                "tensors.  The reaction term is provided by an ionic cell "
                "model (e.g. Aliev-Panfilov, FitzHugh-Nagumo, ten Tusscher) "
                "that describes local ionic currents.  The diffusion tensor "
                "is anisotropic, aligned with the myocardial fiber "
                "orientation.  The PROBLEM TYPE is 'Cardiac_Monodomain'.  "
                "The dynamics section is 'CARDIAC MONODOMAIN DYNAMIC' or "
                "uses SCALAR TRANSPORT DYNAMIC with an appropriate "
                "reaction material.  The mesh represents the myocardial "
                "tissue domain with fiber orientation data."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "SCALAR TRANSPORT DYNAMIC",
                "SCALAR TRANSPORT DYNAMIC/STABILIZATION",
                "SOLVER 1",
                "MATERIALS",
            ],
            "optional_sections": [
                "SCALAR TRANSPORT DYNAMIC/NONLINEAR",
                "IO",
                "IO/RUNTIME VTK OUTPUT",
                "RESULT DESCRIPTION",
            ],
            "materials": {
                "MAT_myocard": {
                    "description": (
                        "Cardiac tissue material for the monodomain "
                        "equation.  Defines tissue conductivity (diffusion "
                        "coefficient), membrane capacitance, and the ionic "
                        "cell model for the reaction term."
                    ),
                    "parameters": {
                        "DIFF1": {
                            "description": (
                                "Diffusion coefficient along fiber "
                                "direction [mm^2/ms]"
                            ),
                            "range": "> 0",
                        },
                        "DIFF2": {
                            "description": (
                                "Diffusion coefficient transverse to "
                                "fiber (sheet direction) [mm^2/ms]"
                            ),
                            "range": "> 0",
                        },
                        "DIFF3": {
                            "description": (
                                "Diffusion coefficient in sheet-normal "
                                "direction [mm^2/ms]"
                            ),
                            "range": "> 0 (often = DIFF2)",
                        },
                        "PERTURBATION_DERIV": {
                            "description": (
                                "Perturbation for numerical derivatives "
                                "of the ionic model"
                            ),
                            "range": "> 0 (typically 1e-6)",
                        },
                        "MODEL": {
                            "description": (
                                "Ionic cell model name (e.g. "
                                "'AlievPanfilov', 'FitzhughNagumo', "
                                "'tenTusscher')"
                            ),
                            "range": "string",
                        },
                    },
                },
                "MAT_scatra_reaction": {
                    "description": (
                        "Alternative: generic scalar transport reaction "
                        "material that can be configured for cardiac "
                        "monodomain-like problems."
                    ),
                },
            },
            "solver": {
                "direct": {
                    "type": "UMFPACK",
                    "notes": (
                        "Suitable for small cardiac meshes.  The "
                        "monodomain system is symmetric positive definite."
                    ),
                },
                "iterative": {
                    "type": "Belos with MueLu AMG",
                    "notes": (
                        "Required for large 3-D cardiac meshes (millions "
                        "of unknowns).  AMG is very efficient for the "
                        "diffusion operator."
                    ),
                },
            },
            "time_integration": {
                "TIMESTEP": (
                    "Time step size [ms].  Must resolve the action "
                    "potential upstroke (typically 0.01-0.1 ms for "
                    "detailed ionic models)."
                ),
                "NUMSTEP": "Total number of time steps.",
                "MAXTIME": (
                    "Maximum simulation time [ms].  Typical single beat: "
                    "500-1000 ms."
                ),
                "THETA": (
                    "Implicit time integration parameter.  theta=0.5 "
                    "(Crank-Nicolson) or theta=1.0 (backward Euler).  "
                    "CN is second-order but may oscillate; BE is stable."
                ),
            },
            "fiber_orientation": {
                "description": (
                    "Cardiac tissue has a complex fiber architecture.  "
                    "Fiber orientations can be provided via element-wise "
                    "coordinate systems (LOCSYS) or through a fiber "
                    "field read from the mesh file."
                ),
            },
            "pitfalls": [
                (
                    "[Numerical] Time step must resolve the "
                    "action-potential UPSTROKE. For the ten "
                    "Tusscher model, dt <= 0.05 ms is "
                    "typical. Signal: dt too large gives "
                    "MISSED ACTIVATION (a stimulus that "
                    "should fire produces no AP) or "
                    "spurious oscillations during the "
                    "depolarisation phase; visualising the "
                    "v(t) at a single node shows the "
                    "missing fast upstroke. Halve dt and "
                    "re-run to verify. (Audit 2026-06-02.)"
                ),
                (
                    "[Mesh] Mesh resolution must resolve the "
                    "AP wavefront. Typical element size is "
                    "0.2-0.5 mm for cardiac tissue. Signal: "
                    "h > 1 mm gives visibly stepped "
                    "propagation (wavefront discretised "
                    "across too few cells); conduction "
                    "velocity is also wrong by 10-30% — "
                    "compare against a reference 1D fiber "
                    "with known CV (~ 0.5 m/s). (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] Fiber orientation strongly "
                    "affects propagation velocity and "
                    "pattern. Ensure fiber data is "
                    "correctly mapped to the mesh. Isotropic "
                    "diffusion (DIFF1 = DIFF2 = DIFF3) gives "
                    "SPHERICAL propagation (no fiber "
                    "alignment). Signal: anisotropic "
                    "physiology (DIFF1 ~ 4*DIFF2 for "
                    "myocardium) but isotropic DIFF in the "
                    "MAT_myocard MATERIALS block gives "
                    "spherical instead of elliptical "
                    "wavefronts in the SCATRA DYNAMIC "
                    "post_vtu output — visible in ParaView; "
                    "the long axis aligns with fiber. "
                    "(Audit 2026-06-02.)"
                ),
                (
                    "[Input] Ionic cell model parameters "
                    "(MODEL in MAT_myocard) must match the "
                    "desired physiology. Different models "
                    "(FHN, ten Tusscher, O'Hara-Rudy) "
                    "produce different AP morphologies and "
                    "conduction velocities. Signal: a FHN "
                    "model on a problem expecting human "
                    "ventricular myocyte response gives "
                    "wrong APD (action-potential duration "
                    "~ 50 ms FHN vs ~ 300 ms human "
                    "ventricular). Select model by target "
                    "tissue and validate APD. (Audit "
                    "2026-06-02.)"
                ),
                (
                    "[Input] Cardiac EP units are typically "
                    "mm, ms, mV, uA/cm^2. Signal: mixing "
                    "in SI (m, s, V, A/m^2) produces "
                    "silent unit conversion errors — "
                    "conduction velocity off by 10x or "
                    "100x. Verify by running a planar-wave "
                    "test and comparing CV against the "
                    "published value for the chosen "
                    "ionic model. (Audit 2026-06-02.)"
                ),
                (
                    "[Input] Stimulation is applied via "
                    "Neumann BC (DESIGN SURF TRANSPORT "
                    "NEUMANN CONDITIONS) or point stimuli. "
                    "The stimulus must be SUPRATHRESHOLD "
                    "to trigger an action potential. "
                    "Signal: too-small stimulus produces "
                    "NO AP at all (subthreshold) — visible "
                    "as a brief perturbation that decays "
                    "back to rest. Typical threshold for "
                    "ten Tusscher: ~ 50 uA/cm^2 for ~1 ms "
                    "duration. (Audit 2026-06-02.)"
                ),
            ],
            "typical_experiments": [
                {
                    "name": "planar_wave_3d",
                    "description": (
                        "Planar action potential wave propagating through "
                        "a 3-D slab of cardiac tissue.  One face is "
                        "stimulated, the wave propagates to the opposite "
                        "face.  Tests conduction velocity, wavefront "
                        "shape, and ionic model integration."
                    ),
                    "template_variant": "monodomain_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "monodomain_3d",
                "description": (
                    "3-D cardiac monodomain: action potential propagation "
                    "in a tissue slab.  MAT_myocard with Aliev-Panfilov "
                    "ionic model, anisotropic diffusion, UMFPACK solver."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "monodomain_3d") -> str:
        templates = {
            "monodomain_3d": self._template_monodomain_3d,
        }
        if variant == "default":
            variant = "monodomain_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_monodomain_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D Cardiac Monodomain (Electrophysiology)
            #
            # Action potential propagation through a cardiac tissue slab.
            # The monodomain equation is a reaction-diffusion PDE with
            # anisotropic diffusion (fiber-aligned) and an ionic cell
            # model providing the reaction term.
            #
            # Mesh: exodus file with:
            #   element_block 1 = cardiac tissue (HEX8 or TET4)
            #   node_set 1 = stimulation face
            #   node_set 2 = recording face (optional)
            # ---------------------------------------------------------------
            TITLE:
              - "3-D cardiac monodomain -- generated template"
            PROBLEM SIZE:
              DIM: 3
            PROBLEM TYPE:
              PROBLEMTYPE: "Cardiac_Monodomain"
            IO:
              STDOUTEVERY: <stdout_interval>
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>

            # == Scalar Transport (monodomain solver) ==========================
            SCALAR TRANSPORT DYNAMIC:
              SOLVERTYPE: "nonlinear"
              TIMESTEP: <timestep>
              NUMSTEP: <number_of_steps>
              MAXTIME: <end_time>
              THETA: <time_integration_theta>
              MATID: <tissue_material_id>
              INITIALFIELD: "field_by_function"
              INITFUNCNO: <initial_potential_function_id>
              LINEAR_SOLVER: 1
              RESULTSEVERY: <results_output_interval>
              RESTARTEVERY: <restart_interval>
            SCALAR TRANSPORT DYNAMIC/STABILIZATION:
              STABTYPE: "no_stabilization"
            SCALAR TRANSPORT DYNAMIC/NONLINEAR:
              ITEMAX: <max_nonlinear_iterations>
              CONVTOL: <nonlinear_convergence_tolerance>

            # == Solver ========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "monodomain_solver"

            # == Materials =====================================================
            MATERIALS:
              - MAT: <tissue_material_id>
                MAT_myocard:
                  DIFF1: <diffusion_fiber>
                  DIFF2: <diffusion_sheet>
                  DIFF3: <diffusion_normal>
                  PERTURBATION_DERIV: <perturbation_derivative>
                  MODEL: "<ionic_cell_model>"

            # == Initial condition: resting potential ==========================
            FUNCT<initial_potential_function_id>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<resting_potential_expression>"

            # == Stimulation (Neumann BC on stimulus face) =====================
            DESIGN SURF TRANSPORT NEUMANN CONDITIONS:
              - E: <stimulus_face_id>
                NUMDOF: 1
                ONOFF: [1]
                VAL: [<stimulus_current_density>]
                FUNCT: [<stimulus_time_function>]

            # Stimulus time function (square pulse)
            FUNCT<stimulus_time_function>:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<stimulus_pulse_expression>"

            # == Geometry ======================================================
            TRANSPORT GEOMETRY:
              FILE: "<mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  TRANSP:
                    HEX8:
                      MAT: <tissue_material_id>
                      TYPE: Cardiac

            RESULT DESCRIPTION:
              - SCATRA:
                  DIS: "scatra"
                  NODE: <result_node_id>
                  QUANTITY: "phi"
                  VALUE: <expected_potential>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check diffusion coefficients
        for key in ("DIFF1", "DIFF2", "DIFF3"):
            val = params.get(key)
            if val is not None:
                try:
                    d = float(val)
                    if d <= 0:
                        issues.append(f"{key} must be > 0, got {d}.")
                except (TypeError, ValueError):
                    issues.append(
                        f"{key} must be a positive number, got {val!r}."
                    )

        # Check diffusion anisotropy ratio
        d1 = params.get("DIFF1")
        d2 = params.get("DIFF2")
        if d1 is not None and d2 is not None:
            try:
                ratio = float(d1) / float(d2)
                if ratio > 100:
                    issues.append(
                        f"Very high anisotropy ratio DIFF1/DIFF2 = {ratio:.1f}.  "
                        f"This may require very fine mesh resolution "
                        f"transverse to fibers."
                    )
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        # Check timestep (cardiac-specific)
        timestep = params.get("TIMESTEP")
        if timestep is not None:
            try:
                dt = float(timestep)
                if dt <= 0:
                    issues.append(f"TIMESTEP must be > 0, got {dt}.")
                elif dt > 0.1:
                    issues.append(
                        f"TIMESTEP = {dt} ms may be too large for cardiac "
                        f"monodomain.  Typical range: 0.01-0.1 ms."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"TIMESTEP must be a positive number, "
                    f"got {timestep!r}."
                )

        # Check THETA
        theta = params.get("THETA")
        if theta is not None:
            try:
                th = float(theta)
                if th < 0.5 or th > 1.0:
                    issues.append(
                        f"THETA should be in [0.5, 1.0], got {th}.  "
                        f"0.5 = Crank-Nicolson, 1.0 = backward Euler."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"THETA must be a number in [0.5, 1.0], "
                    f"got {theta!r}."
                )

        # Check ionic model
        model = params.get("MODEL")
        known_models = {
            "AlievPanfilov", "FitzhughNagumo", "tenTusscher",
            "SanGarciaBueno", "Minimal",
        }
        if model is not None and model not in known_models:
            issues.append(
                f"MODEL '{model}' is not a recognized ionic model.  "
                f"Known models: {sorted(known_models)}."
            )

        return issues
