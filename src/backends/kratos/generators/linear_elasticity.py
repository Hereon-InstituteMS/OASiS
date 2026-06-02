"""Kratos linear elasticity generators and knowledge."""


def _elasticity_2d_kratos(params: dict) -> str:
    """Real Kratos plane-strain linear elasticity on a rectangular beam.

    Uses StructuralMechanicsApplication with `SmallDisplacementElement2D4N`
    on a structured quad grid, `LinearElasticPlaneStrain2DLaw`, and a
    Newton-Raphson static strategy.  The left edge is clamped; the
    mid-tip node is given a prescribed y-displacement so the cell
    produces a finite, deterministic displacement field without
    needing a separate Condition for the point load.  Output is
    written via `KM.VtkOutput` (legacy `.vtk` — Kratos's VtkOutput
    does not write `.vtu`; the sweep harness accepts both).
    """
    nx = params.get("nx", 32)
    ny = params.get("ny", 4)
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    lx = params.get("lx", 10.0)
    ly = params.get("ly", 1.0)
    tip_uy = params.get("tip_uy", -0.5)  # prescribed tip displacement
    return f'''\
"""Linear elastic 2D cantilever (plane strain) — Kratos StructuralMechanicsApplication.

Clamped left edge, prescribed y-displacement at the mid-tip node.
Writes the converged DISPLACEMENT and REACTION fields as `Structure_0_1.vtk`.
"""
import json
import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA

# Problem geometry / material (template parameters at file generation time)
nx, ny = {nx}, {ny}
L,  h  = {lx}, {ly}
E,  nu = {E}, {nu}
tip_uy = {tip_uy}

# Linear interpolation in x and y produces a structured quad grid.
def node_id(i, j):
    return 1 + j * (nx + 1) + i

model = KM.Model()
mp = model.CreateModelPart("Structure")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
mp.SetBufferSize(2)
for v in (KM.DISPLACEMENT, KM.REACTION, KM.VOLUME_ACCELERATION):
    mp.AddNodalSolutionStepVariable(v)

# Nodes
for j in range(ny + 1):
    yj = -h / 2.0 + j * h / ny
    for i in range(nx + 1):
        xi = i * L / nx
        mp.CreateNewNode(node_id(i, j), xi, yj, 0.0)

# Properties: plane-strain linear elastic
prop = mp.CreateNewProperties(1)
prop.SetValue(KM.YOUNG_MODULUS, E)
prop.SetValue(KM.POISSON_RATIO, nu)
prop.SetValue(KM.DENSITY, 0.0)
prop.SetValue(KM.CONSTITUTIVE_LAW, SMA.LinearElasticPlaneStrain2DLaw())

# Quad elements (CCW orientation)
eid = 1
for j in range(ny):
    for i in range(nx):
        mp.CreateNewElement(
            "SmallDisplacementElement2D4N", eid,
            [node_id(i, j), node_id(i + 1, j),
             node_id(i + 1, j + 1), node_id(i, j + 1)],
            prop,
        )
        eid += 1

# Add DOFs.  SmallDisplacementElement2D4N inherits SolidElementCheck which
# requires the Z dof in every node (Kratos uses 3-component vectors
# internally); we add it everywhere and Dirichlet-pin Z = 0.
for node in mp.Nodes:
    node.AddDof(KM.DISPLACEMENT_X, KM.REACTION_X)
    node.AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y)
    node.AddDof(KM.DISPLACEMENT_Z, KM.REACTION_Z)
    node.Fix(KM.DISPLACEMENT_Z)
    node.SetSolutionStepValue(KM.DISPLACEMENT_Z, 0.0)

# Clamp left edge (i = 0)
for j in range(ny + 1):
    n = mp.Nodes[node_id(0, j)]
    n.Fix(KM.DISPLACEMENT_X)
    n.Fix(KM.DISPLACEMENT_Y)
    n.SetSolutionStepValue(KM.DISPLACEMENT_X, 0.0)
    n.SetSolutionStepValue(KM.DISPLACEMENT_Y, 0.0)

# Prescribe tip mid-node y-displacement
j_mid = ny // 2
tip_node = mp.Nodes[node_id(nx, j_mid)]
tip_node.Fix(KM.DISPLACEMENT_Y)
tip_node.SetSolutionStepValue(KM.DISPLACEMENT_Y, tip_uy)

# Newton-Raphson static solver
scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
builder_and_solver = KM.ResidualBasedBlockBuilderAndSolver(
    KM.SkylineLUFactorizationSolver()
)
conv = KM.ResidualCriteria(1.0e-8, 1.0e-12)
strat = KM.ResidualBasedNewtonRaphsonStrategy(
    mp, scheme, conv, builder_and_solver,
    20, True, False, True,
)
strat.SetEchoLevel(0)
strat.Check()
mp.CloneTimeStep(1.0)
mp.ProcessInfo[KM.STEP] = 1
strat.Solve()

# VTK output
vtk_params = KM.Parameters(json.dumps({{
    "model_part_name": "Structure",
    "output_control_type": "step",
    "output_interval": 1,
    "file_format": "ascii",
    "output_path": ".",
    "output_sub_model_parts": False,
    "save_output_files_in_folder": False,
    "nodal_solution_step_data_variables": ["DISPLACEMENT", "REACTION"],
}}))
KM.VtkOutput(mp, vtk_params).PrintOutput()

# Scalar summary for the layer-3 sweep
tip = mp.Nodes[node_id(nx, j_mid)]
summary = {{
    "tip_ux": float(tip.GetSolutionStepValue(KM.DISPLACEMENT_X)),
    "tip_uy": float(tip.GetSolutionStepValue(KM.DISPLACEMENT_Y)),
    "n_nodes": mp.NumberOfNodes(),
    "n_elements": mp.NumberOfElements(),
}}
print(f"tip displacement: ux={{summary['tip_ux']:.6f}}  uy={{summary['tip_uy']:.6f}}")
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


def _elasticity_nonlinear_kratos(params: dict) -> str:
    """Geometrically-nonlinear plane-strain elasticity — Kratos SMA.

    Uses `TotalLagrangianElement2D4N` (large-rotation kinematics) with
    `LinearElasticPlaneStrain2DLaw` as the small-strain constitutive
    law.  The pip-installed Kratos wheel does NOT ship a Neo-Hookean
    2D law (those live in `ConstitutiveLawsApplication`, not in the
    base StructuralMechanicsApplication on PyPI), so "nonlinear" here
    means geometric nonlinearity only.  Switching to a hyperelastic
    material is straightforward once `ConstitutiveLawsApplication`
    is available — replace the SetValue on KM.CONSTITUTIVE_LAW with
    e.g. `CLA.HyperElasticIsotropicNeoHookeanPlaneStrain2DLaw()`.
    """
    nx = params.get("nx", 16)
    ny = params.get("ny", 4)
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    lx = params.get("lx", 4.0)
    ly = params.get("ly", 1.0)
    tip_uy = params.get("tip_uy", -0.5)
    n_substeps = params.get("n_substeps", 5)
    return f'''\
"""Geometrically-nonlinear plane-strain elasticity — Kratos SMA.

TotalLagrangianElement2D4N + LinearElasticPlaneStrain2DLaw on a
structured quad grid.  Clamped left edge; tip y-displacement applied
in `n_substeps` Newton-Raphson load steps.  Writes the converged
DISPLACEMENT and REACTION fields as `Structure_0_*.vtk`.
"""
import json
import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA

nx, ny = {nx}, {ny}
L,  h  = {lx}, {ly}
E,  nu = {E}, {nu}
tip_uy = {tip_uy}
n_substeps = {n_substeps}


def node_id(i, j):
    return 1 + j * (nx + 1) + i


model = KM.Model()
mp = model.CreateModelPart("Structure")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
mp.SetBufferSize(2)
for v in (KM.DISPLACEMENT, KM.REACTION, KM.VOLUME_ACCELERATION):
    mp.AddNodalSolutionStepVariable(v)

for j in range(ny + 1):
    yj = -h / 2.0 + j * h / ny
    for i in range(nx + 1):
        mp.CreateNewNode(node_id(i, j), i * L / nx, yj, 0.0)

prop = mp.CreateNewProperties(1)
prop.SetValue(KM.YOUNG_MODULUS, E)
prop.SetValue(KM.POISSON_RATIO, nu)
prop.SetValue(KM.DENSITY, 0.0)
prop.SetValue(KM.CONSTITUTIVE_LAW, SMA.LinearElasticPlaneStrain2DLaw())

eid = 1
for j in range(ny):
    for i in range(nx):
        mp.CreateNewElement(
            "TotalLagrangianElement2D4N", eid,
            [node_id(i, j), node_id(i + 1, j),
             node_id(i + 1, j + 1), node_id(i, j + 1)],
            prop,
        )
        eid += 1

for node in mp.Nodes:
    node.AddDof(KM.DISPLACEMENT_X, KM.REACTION_X)
    node.AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y)
    node.AddDof(KM.DISPLACEMENT_Z, KM.REACTION_Z)
    node.Fix(KM.DISPLACEMENT_Z)
    node.SetSolutionStepValue(KM.DISPLACEMENT_Z, 0.0)

for j in range(ny + 1):
    n = mp.Nodes[node_id(0, j)]
    n.Fix(KM.DISPLACEMENT_X)
    n.Fix(KM.DISPLACEMENT_Y)
    # Set the Dirichlet value explicitly to 0.0 alongside the Fix()
    # call.  Kratos's `Fix(...)` flags the DOF as constrained but
    # does not by itself prescribe the value (the solution-step
    # value defaults to 0.0 for fresh nodes, but for nodes that have
    # already participated in an earlier solve step on the same
    # ModelPart the prior value persists — a real bug when the
    # template is re-used inside a larger workflow).
    n.SetSolutionStepValue(KM.DISPLACEMENT_X, 0.0)
    n.SetSolutionStepValue(KM.DISPLACEMENT_Y, 0.0)

j_mid = ny // 2
tip_node = mp.Nodes[node_id(nx, j_mid)]
tip_node.Fix(KM.DISPLACEMENT_Y)

scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
builder_and_solver = KM.ResidualBasedBlockBuilderAndSolver(
    KM.SkylineLUFactorizationSolver()
)
conv = KM.ResidualCriteria(1.0e-8, 1.0e-12)
strat = KM.ResidualBasedNewtonRaphsonStrategy(
    mp, scheme, conv, builder_and_solver,
    50, True, False, True,
)
strat.SetEchoLevel(0)

vtk_params = KM.Parameters(json.dumps({{
    "model_part_name": "Structure",
    "output_control_type": "step",
    "output_interval": 1,
    "file_format": "ascii",
    "output_path": ".",
    "output_sub_model_parts": False,
    "save_output_files_in_folder": False,
    "nodal_solution_step_data_variables": ["DISPLACEMENT", "REACTION"],
}}))
vtk = KM.VtkOutput(mp, vtk_params)

# Ramp the tip displacement in n_substeps load steps.  TotalLagrangian
# requires sufficiently small increments for Newton convergence at
# large rotations.
strat.Check()
for step in range(1, n_substeps + 1):
    mp.CloneTimeStep(float(step))
    mp.ProcessInfo[KM.STEP] = step
    tip_node.SetSolutionStepValue(KM.DISPLACEMENT_Y, tip_uy * step / n_substeps)
    strat.Solve()
vtk.PrintOutput()

tip = mp.Nodes[node_id(nx, j_mid)]
summary = {{
    "tip_ux": float(tip.GetSolutionStepValue(KM.DISPLACEMENT_X)),
    "tip_uy": float(tip.GetSolutionStepValue(KM.DISPLACEMENT_Y)),
    "n_steps": n_substeps,
    "n_nodes": mp.NumberOfNodes(),
    "n_elements": mp.NumberOfElements(),
}}
print(f"tip ux={{summary['tip_ux']:.6f}}  uy={{summary['tip_uy']:.6f}}  (target uy={{tip_uy}})")
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


KNOWLEDGE = {
    "linear_elasticity": {
        "description": "Structural mechanics via StructuralMechanicsApplication (SMA)",
        "application": "StructuralMechanicsApplication (pip install KratosStructuralMechanicsApplication)",
        "elements": {
            "2D": ["SmallDisplacementElement2D3N/4N/6N/8N/9N (linear, small strain)",
                   "TotalLagrangianElement2D3N/4N (nonlinear, large deformation)",
                   "UpdatedLagrangianElement2D3N/4N"],
            "3D": ["SmallDisplacementElement3D4N/8N/10N/20N/27N",
                   "TotalLagrangianElement3D4N/8N"],
            "shells": ["ShellThinElement3D3N (MITC, Kirchhoff-Love)",
                      "ShellThickElement3D4N (Reissner-Mindlin)"],
            "beams": ["CrBeamElement3D2N (co-rotational)", "CrLinearBeamElement3D2N"],
            "trusses": ["TrussElement3D2N", "TrussLinearElement3D2N"],
            "cables": ["CableElement3D2N"],
            "springs": ["SpringDamperElement3D2N", "NodalConcentratedElement2D1N/3D1N"],
        },
        "constitutive_laws": {
            "linear": ["LinearElastic3DLaw", "LinearElasticPlaneStrain2DLaw",
                       "LinearElasticPlaneStress2DLaw", "LinearElasticAxisymmetric2DLaw",
                       "TrussConstitutiveLaw", "BeamConstitutiveLaw"],
            "hyperelastic": ["HyperElastic3DLaw (Saint Venant-Kirchhoff)",
                            "HyperElasticIsotropicNeoHookean2D/3DLaw"],
            "plasticity": "Factory: 7 yield surfaces x 5 plastic potentials x 6 hardening curves",
            "damage": ["IsotropicDamage (factory)", "DplusDminusDamage (tension/compression split)"],
            "viscoelastic": ["GeneralizedMaxwell (relaxation)", "GeneralizedKelvin (creep)"],
        },
        "solver_types": ["static (Newton-Raphson)", "dynamic (Newmark, Bossak, GenAlpha)",
                        "explicit (central differences)", "formfinding"],
        # Pitfalls are tagged with Table-1 category prefix
        # ([Syntax]/[Physics]/[Numerical]/[API]/[Integration]) so the
        # agent and any downstream tooling can filter on category.
        # New entries SHOULD include a `Signal:` clause stating the
        # observable symptom — silent no-ops (the FSI velocity-30x
        # case in the Open-FEM-Agent paper, Section 3.2) are the
        # category that has hurt users most, so making the symptom
        # explicit lets the post-exec critic match against it.
        "pitfalls": [
            "[Syntax] Element names MUST include node count: "
            "SmallDisplacementElement2D3N, not SmallDisplacement2D. "
            "Signal: `KratosMultiphysics.Exception: Element name not found`.",
            "[API] Materials defined in StructuralMaterials.json, referenced "
            "by Properties ID. The .json + mdpa + ProjectParameters.json "
            "trio must agree on the Properties ID or the law silently "
            "defaults to a zero constitutive response. "
            "Signal: displacements are linear in load but unrealistically large.",
            "[Syntax] SubModelParts must match between .mdpa and "
            "ProjectParameters.json exactly. A typo binds the process "
            "to an empty SubModelPart and the BC silently no-ops. "
            "Signal: BC nodes show non-zero residuals at convergence.",
            "[Numerical] For nonlinear: increase max_iterations beyond "
            "the default 10. With contact, plasticity, or large rotation "
            "10 iterations is usually insufficient. "
            "Signal: solver reports 'max iterations reached' but exits 0.",
            "[Physics] DISPLACEMENT is the primary DOF for solid elements; "
            "ROTATION is also required on beams/shells. Adding only "
            "DISPLACEMENT for a beam element silently drops the rotational "
            "DOF and gives a hinge-like response. "
            "Signal: beam tip deflects but does not rotate under moment load.",
            "[Numerical] SHEAR LOCKING — Linear hex8 (3D8N) and quad4 "
            "(2D4N) elements lock in bending-dominated problems, "
            "producing overly stiff results and wrong frequencies. Use "
            "quadratic elements (3D20N, 3D27N, 2D8N, 2D9N) for any "
            "problem with significant bending. "
            "Signal: tip deflection is order-of-magnitude smaller than "
            "Euler-Bernoulli prediction.",
            "[API] For POINT_LOAD application: use assign_vector_variable_process "
            "with constrained: [false, false, false]. Do NOT use "
            "assign_vector_by_direction_process — it expects a kinematic "
            "variable (DISPLACEMENT class) and silently no-ops for load "
            "variables, leaving the model unloaded. "
            "Signal: nodal reactions sum to zero and tip displacement is zero "
            "despite the process being present in ProjectParameters.json.",
            "[Syntax] problem_data section MUST include 'echo_level' "
            "field. Missing it raises a confusing 'Parameters' KeyError "
            "inside the analysis stage rather than at parameter validation.",
            # --- Retroactive entries from PR #24 Kratos LE stub-replacement ---
            "[Syntax] SmallDisplacementElement2D{3,4,6,8,9}N inherits the "
            "SolidElementCheck path from BaseSolidElement, which queries "
            "the Z DOF on every node even in plane analyses (Kratos uses "
            "3-component vectors internally). Add DISPLACEMENT_Z and "
            "REACTION_Z DOFs to every node and Fix(DISPLACEMENT_Z)=0. "
            "Signal: `Check failed for DISPLACEMENT_Z` at strat.Check(), "
            "long before the first solver step.",
            "[API] CONSTITUTIVE_LAW Properties binding takes an *instance*, "
            "not a class: `prop.SetValue(KM.CONSTITUTIVE_LAW, "
            "SMA.LinearElasticPlaneStrain2DLaw())`. Passing the class "
            "itself silently keeps the default null law and the element "
            "responds with zero stiffness. "
            "Signal: K matrix assembles but is singular; spsolve reports "
            "extreme condition number or NaN displacements.",
            "[API] Nodal solution-step variables (DISPLACEMENT, REACTION, "
            "VOLUME_ACCELERATION, etc.) MUST be added via "
            "`mp.AddNodalSolutionStepVariable(v)` BEFORE the first node "
            "is created and BEFORE `mp.SetBufferSize(2)`. Adding after "
            "leaves the variable unallocated on existing nodes. "
            "Signal: `RuntimeError: trying to access non-existing "
            "Kratos component` when reading DISPLACEMENT post-solve.",
        ],
    },
}

GENERATORS = {
    "linear_elasticity_2d": _elasticity_2d_kratos,
    "linear_elasticity_2d_nonlinear": _elasticity_nonlinear_kratos,
}
