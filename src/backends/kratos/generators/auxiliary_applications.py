"""Kratos auxiliary applications — utility, infrastructure, and legacy apps.

These do not own their own physics problem types, so they have no
GENERATORS — but the agent needs to know about them because other
generators reference them (e.g. FSI pulls in MappingApplication;
parallel runs need TrilinosApplication + MetisApplication; HDF5
output needs HDF5Application).

Source: upstream Kratos `applications/` directory listing.  Every name
in `KNOWLEDGE` below must correspond to a real sub-application.
"""


KNOWLEDGE = {
    "_auxiliary_overview": {
        "description": (
            "Kratos applications that the agent must know about even though "
            "they do not provide their own physics problem type: parallel "
            "infrastructure, I/O, meshing, mapping, statistics, and legacy "
            "predecessors of currently-active applications.  Pulled in as "
            "dependencies of physics applications rather than driven directly."
        ),
        "infrastructure_apps": {
            "TrilinosApplication": (
                "Trilinos linear-solver wrappers (Epetra, AztecOO, Amesos, ML, "
                "MueLu) for distributed-memory MPI runs.  Required by any "
                "parallel Kratos analysis using iterative or AMG-preconditioned "
                "solvers.  Not published as a PyPI wheel — obtain it by "
                "building Kratos from source with `-DTRILINOS_APPLICATION=ON` "
                "against an MPI-built Trilinos."
            ),
            "MetisApplication": (
                "Metis-based mesh partitioner for MPI runs.  Used by the model-"
                "part-IO splitter to produce per-rank .mdpa files.  Pulled in "
                "by every MPI workflow; no Python-facing physics on its own.  "
                "Pip install hint: KratosMetisApplication."
            ),
            "LinearSolversApplication": (
                "Linear-solver wrappers beyond Kratos core (Eigen-based "
                "sparse_qr/sparse_lu/sparse_cg, PARDISO, complex-valued solvers).  "
                "Used by structural and electromagnetic analyses needing direct "
                "factorisation or complex arithmetic.  Pip install hint: "
                "KratosLinearSolversApplication."
            ),
            "HDF5Application": (
                "Parallel HDF5 I/O.  Provides HDF5OutputProcess and HDF5IO for "
                "checkpointing and restart, plus XDMF/ParaView/VisIt-friendly "
                "result storage.  Used by long simulations and FSI restart "
                "workflows.  Pip install hint: KratosHDF5Application."
            ),
            "MedApplication": (
                "MED file format I/O (Salome ecosystem).  Provides import/"
                "export of Salome-generated meshes and post-processing into "
                "MED format.  Pip install hint: KratosMedApplication."
            ),
        },
        "meshing_and_mapping": {
            "MeshingApplication": (
                "Adaptive mesh refinement (h-refinement) and remeshing.  "
                "Provides MMG / PMMG bindings for tetrahedral and triangular "
                "remeshing driven by error indicators.  Used by large-"
                "deformation solid mechanics and adaptive CFD.  Pip install "
                "hint: KratosMeshingApplication."
            ),
            "MeshMovingApplication": (
                "ALE (arbitrary Lagrangian-Eulerian) mesh-moving strategies: "
                "Laplacian smoothing, structural-similarity, and rigid-body "
                "displacement of inner boundaries.  Required by FSI (the fluid "
                "mesh follows the structural interface) and by any moving-"
                "boundary CFD.  Pip install hint: KratosMeshMovingApplication."
            ),
            "MappingApplication": (
                "Inter-mesh field mapping for non-conforming or non-matching "
                "discretisations.  Provides nearest-neighbour, nearest-element, "
                "barycentric, RBF, and beam mappers used by FSI partitioned "
                "solvers, thermo-mechanical coupling, and CoSimulation.  A "
                "mortar mapper is upstream-flagged as under development and "
                "should not be relied on yet.  Pip install hint: "
                "KratosMappingApplication."
            ),
        },
        "analysis_utilities": {
            "StatisticsApplication": (
                "Statistical post-processing of time-series and ensemble "
                "fields: mean, variance, RMS, time-averaged Reynolds stresses, "
                "spatially-averaged quantities.  Used in turbulence post-"
                "processing and uncertainty quantification.  Pip install "
                "hint: KratosStatisticsApplication."
            ),
            "SystemIdentificationApplication": (
                "Sensor-based system identification.  Provides sensor types "
                "(displacement, strain, sensor_view) and a "
                "measurement_residual_response_function used for sensor "
                "placement, damage detection, and parameter calibration "
                "against measured response data.  Upstream README is empty "
                "as of writing; see `custom_sensors/` and `custom_responses/` "
                "for the actually-implemented surface area.  Pip install "
                "hint: KratosSystemIdentificationApplication."
            ),
        },
        "older_solid_and_contact_apps": {
            "SolidMechanicsApplication": (
                "Older solid-mechanics application focused on FEM for solids, "
                "shells and beams (per its upstream README).  "
                "StructuralMechanicsApplication is the modern path that the "
                "current Kratos generators target — prefer it for new work; "
                "use SolidMechanicsApplication only if you are reading an "
                "existing input deck that already targets it.  Pip install "
                "hint: KratosSolidMechanicsApplication."
            ),
            "ContactMechanicsApplication": (
                "Older contact-mechanics application.  For new structural-"
                "contact workflows the established path is "
                "ContactStructuralMechanicsApplication, which is what the "
                "agent's contact generator already targets.  Upstream README "
                "is empty; surface area is whatever lives under the app's "
                "`custom_*/` directories.  Pip install hint: "
                "KratosContactMechanicsApplication."
            ),
        },
        "pitfalls": [
                        '[Integration] TrilinosApplication is not published on PyPI; to get it you must build Kratos from source with `-DTRILINOS_APPLICATION=ON` against an MPI-built Trilinos.  MetisApplication does have a PyPI wheel (KratosMetisApplication), but it is only useful in an MPI-parallel run. '
                        "Signal: RuntimeError 'KeyError' from JSON parsing OR 'SubModelPart not found' / 'Property ID ... missing' during AnalysisStage.Initialize.",
                        "[Integration] MappingApplication and MeshMovingApplication are *required* (not optional) for partitioned FSI even if not named explicitly in the user-facing JSON — `FSIApplication`'s `partitioned_fsi_base_solver.py` imports both at runtime. "
                        "Signal: RuntimeError 'KeyError' from JSON parsing OR 'SubModelPart not found' / 'Property ID ... missing' during AnalysisStage.Initialize.",
                        '[Numerical] For new analyses prefer StructuralMechanicsApplication and ContactStructuralMechanicsApplication over the older SolidMechanicsApplication / ContactMechanicsApplication paths; the current Kratos generators target the modern apps. '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[API] MappingApplication.MapperFactory is DEPRECATED (moved to KratosMultiphysics core in Kratos 10.x). Calling MAP.MapperFactory.CreateMapper still works in 10.4 but prints a deprecation warning: "[WARNING] DEPRECATION-Warning; MappingApplication: The \\"MapperFactory\\" was moved to the Core! (used for \\"CreateMapper\\")". New code should use `KratosMultiphysics.MapperFactory.CreateMapper(origin, destination, params)` directly. '
                        "Signal: KratosMultiphysics emits the literal string 'DEPRECATION-Warning' + 'MapperFactory' + 'moved to the Core' on stderr at the call site. (Verified empirically 2026-06-01.)",
                        '[API] DEM is always 3D internally: the MDPA must reference SphericParticle3D even for problems posed in a 2D plane. There is NO SphericParticle2D element registered; attempting ModelPart.CreateNewElement("SphericParticle2D", ...) raises RuntimeError "Element \\"SphericParticle2D\\" is not registered!" listing the available DEM element types. '
                        "Signal: RuntimeError 'Element \"SphericParticle2D\" is not registered!' or 'Maybe you need to import the application where it is defined?' when an agent generates a 2D DEM template with the literal 2D element name. (Verified empirically 2026-06-01.)",
                        '[Integration] BOTH KratosContactStructuralMechanicsApplication (modern) AND the legacy KratosContactMechanicsApplication / KratosSolidMechanicsApplication are NOT published on PyPI. There is no pip-only path to Contact physics in Kratos 10.x. Building Kratos from source with `-DCONTACT_STRUCTURAL_MECHANICS_APPLICATION=ON` is the only option. '
                        "Signal: pip install reports 'ERROR: Could not find a version that satisfies the requirement KratosContactStructuralMechanicsApplication'; python import raises ModuleNotFoundError 'No module named KratosMultiphysics.ContactStructuralMechanicsApplication' (or '...ContactMechanicsApplication' on the legacy fallback path). (Verified empirically 2026-06-01 — both modern and legacy contact apps absent from PyPI; the legacy-fallback hint in prior catalog text was misleading.)",
                    ],
    },
}


GENERATORS: dict = {}
