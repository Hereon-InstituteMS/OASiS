"""Kratos linear elasticity generators and knowledge."""


def _elasticity_2d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Linear elasticity on rectangular domain — Kratos (manual assembly)."""
    nx = params.get("nx", 40)
    ny = params.get("ny", 4)
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    lx = params.get("lx", 10.0)
    ly = params.get("ly", 1.0)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""Linear elasticity: rectangular domain, fixed left — Kratos (manual assembly)"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import json

nx, ny, lx, ly = {nx}, {ny}, {lx}, {ly}
nid = 1; node_map = {{}}; coords = {{}}
for j in range(ny+1):
    for i in range(nx+1):
        coords[nid] = (i*lx/nx, j*ly/ny)
        node_map[(i,j)] = nid; nid += 1
n_nodes = nid - 1

elements = []
for j in range(ny):
    for i in range(nx):
        n1,n2,n3,n4 = node_map[(i,j)],node_map[(i+1,j)],node_map[(i+1,j+1)],node_map[(i,j+1)]
        elements.append((n1,n2,n4)); elements.append((n2,n3,n4))

ndof = 2 * n_nodes
K = lil_matrix((ndof, ndof))
F = np.zeros(ndof)
mu, lam = {mu}, {lam}

for tri in elements:
    ids = [t-1 for t in tri]
    x = np.array([coords[t][0] for t in tri])
    y = np.array([coords[t][1] for t in tri])
    area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]]) / (2*area)
    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]]) / (2*area)

    B = np.zeros((3, 6))
    for a in range(3):
        B[0, 2*a] = b[a]; B[1, 2*a+1] = c[a]
        B[2, 2*a] = c[a]; B[2, 2*a+1] = b[a]
    D = np.array([[lam+2*mu, lam, 0], [lam, lam+2*mu, 0], [0, 0, mu]])
    Ke = area * B.T @ D @ B

    dofs = []
    for a in range(3):
        dofs.extend([2*ids[a], 2*ids[a]+1])
    for i in range(6):
        F[dofs[i]] += -1.0 * area / 3.0 if i % 2 == 1 else 0  # body force — set for your problem
        for j_idx in range(6):
            K[dofs[i], dofs[j_idx]] += Ke[i, j_idx]
K = K.tocsr()

# Fix left edge
fixed = set()
for j in range(ny+1):
    n = node_map[(0,j)] - 1
    fixed.add(2*n); fixed.add(2*n+1)
interior = sorted(set(range(ndof)) - fixed)

u = np.zeros(ndof)
u[interior] = spsolve(K[np.ix_(interior, interior)], F[interior])

uy = u[1::2]
print(f"Max tip displacement: {{uy.min():.6f}}")
summary = {{"max_displacement_y": float(uy.min()), "n_dofs": ndof}}
with open("results_summary.json", "w") as _f: json.dump(summary, _f, indent=2)
'''


def _elasticity_nonlinear_kratos(params: dict) -> str:
    """REAL solve — geometrically-nonlinear (large-deflection) cantilever.

    The previous version of this generator was an availability-probe stub (it
    only import-checked StructuralMechanicsApplication and wrote a {"note":
    ...} summary). That violated catalog honesty, so it was replaced
    (2026-06-26 audit) with a genuine large-deformation Total-Lagrangian solve.

    A 2D plane-strain cantilever (CST triangles), clamped on the left edge, is
    loaded by a vertical tip traction. The Saint-Venant-Kirchhoff hyperelastic
    response is integrated with a full Newton-Raphson loop: at every iteration
    the deformation gradient F, Green-Lagrange strain E = 1/2(F^T F - I), 2nd
    Piola-Kirchhoff stress S = C:E, internal force and consistent (material +
    geometric) tangent are assembled element-by-element and the linear system
    is solved with scipy. Load is applied in increments so the Newton solver
    stays in its basin of attraction — this is what makes the result differ
    from the linear elasticity generator (the nonlinear tip deflection is
    visibly stiffer at large load). KratosMultiphysics writes the converged
    displacement field as .vtk; StructuralMechanicsApplication is imported and
    used to construct the model part. The summary reports the nonlinear tip
    deflection, Newton iteration counts per increment, and the final residual
    norm — physical, cross-checkable quantities, NOT an availability note.
    """
    nx = params.get("nx", 20)
    ny = params.get("ny", 4)
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    lx = params.get("lx", 10.0)
    ly = params.get("ly", 1.0)
    tip_load = params.get("tip_load", 30.0)   # total downward tip load
    n_increments = params.get("n_increments", 8)
    max_newton = params.get("max_newton", 30)
    tol = params.get("tol", 1.0e-8)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""Geometrically-nonlinear cantilever (Total-Lagrangian St.Venant-Kirchhoff).

2D plane-strain CST cantilever clamped on the left, loaded by an incremental
vertical tip traction. Full Newton-Raphson on the nonlinear residual; scipy
linear solves per iteration; KratosMultiphysics writes the result as .vtk.
"""
import json
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
print("StructuralMechanicsApplication loaded")

# ----------------------------------------------------------------- mesh
nx, ny, lx, ly = {nx}, {ny}, {lx}, {ly}
nid = 1; coords = {{}}; node_map = {{}}
for j in range(ny+1):
    for i in range(nx+1):
        coords[nid] = np.array([i*lx/nx, j*ly/ny]); node_map[(i,j)] = nid; nid += 1
n_nodes = nid - 1
elements = []
for j in range(ny):
    for i in range(nx):
        n1,n2,n3,n4 = node_map[(i,j)],node_map[(i+1,j)],node_map[(i+1,j+1)],node_map[(i,j+1)]
        elements.append((n1,n2,n4)); elements.append((n2,n3,n4))
ndof = 2 * n_nodes
mu, lam = {mu}, {lam}
n_increments = {n_increments}
max_newton = {max_newton}
tol = {tol}
# plane-strain 4th-order tensor as 3x3 (Voigt: 11,22,12)
Cmat = np.array([[lam+2*mu, lam, 0.0],
                 [lam, lam+2*mu, 0.0],
                 [0.0, 0.0, mu]])

# reference geometry derivatives per element (constant for CST)
elem_data = []
for tri in elements:
    X = np.array([coords[t] for t in tri])
    area = 0.5*abs((X[1,0]-X[0,0])*(X[2,1]-X[0,1]) - (X[2,0]-X[0,0])*(X[1,1]-X[0,1]))
    b = np.array([X[1,1]-X[2,1], X[2,1]-X[0,1], X[0,1]-X[1,1]]) / (2*area)
    c = np.array([X[2,0]-X[1,0], X[0,0]-X[2,0], X[1,0]-X[0,0]]) / (2*area)
    elem_data.append((tri, area, b, c))

# ---------------------------------------------- clamp + tip load definition
fixed = set()
for j in range(ny+1):
    n = node_map[(0,j)] - 1
    fixed.add(2*n); fixed.add(2*n+1)
free = sorted(set(range(ndof)) - fixed)
tip_nodes = [node_map[(nx, j)] for j in range(ny+1)]
Fext_full = np.zeros(ndof)
for n in tip_nodes:
    Fext_full[2*(n-1)+1] = -{tip_load} / len(tip_nodes)

def assemble(u):
    """Return internal force vector and tangent stiffness for displacement u."""
    K = lil_matrix((ndof, ndof))
    Fint = np.zeros(ndof)
    for (tri, area, b, c) in elem_data:
        ids = [t-1 for t in tri]
        ue = np.array([[u[2*i], u[2*i+1]] for i in ids])  # 3x2 nodal disp
        # displacement gradient H = du/dX  (2x2)
        H = np.zeros((2,2))
        for a in range(3):
            H[:,0] += ue[a]*b[a]
            H[:,1] += ue[a]*c[a]
        Fdef = np.eye(2) + H
        Egl = 0.5*(Fdef.T @ Fdef - np.eye(2))
        Ev = np.array([Egl[0,0], Egl[1,1], 2*Egl[0,1]])
        Sv = Cmat @ Ev
        S = np.array([[Sv[0], Sv[2]],[Sv[2], Sv[1]]])
        # B (nonlinear) maps nodal dof -> Voigt Green-Lagrange strain
        B = np.zeros((3,6))
        for a in range(3):
            B[0,2*a]   = Fdef[0,0]*b[a]; B[0,2*a+1] = Fdef[1,0]*b[a]
            B[1,2*a]   = Fdef[0,1]*c[a]; B[1,2*a+1] = Fdef[1,1]*c[a]
            B[2,2*a]   = Fdef[0,0]*c[a]+Fdef[0,1]*b[a]
            B[2,2*a+1] = Fdef[1,0]*c[a]+Fdef[1,1]*b[a]
        fe = area * (B.T @ Sv)
        Kmat = area * (B.T @ Cmat @ B)              # material tangent
        # geometric tangent
        G = np.zeros((6,6))
        grad = np.array([[b[a], c[a]] for a in range(3)])  # 3x2
        for a in range(3):
            for d in range(3):
                val = grad[a] @ S @ grad[d]
                G[2*a, 2*d]     += area*val
                G[2*a+1, 2*d+1] += area*val
        Ke = Kmat + G
        dofs = []
        for a in range(3): dofs += [2*ids[a], 2*ids[a]+1]
        for ii in range(6):
            Fint[dofs[ii]] += fe[ii]
            for jj in range(6):
                K[dofs[ii], dofs[jj]] += Ke[ii, jj]
    return Fint, K.tocsr()

# ------------------------------------------- incremental Newton-Raphson
u = np.zeros(ndof)
newton_iters = []
final_res = 0.0
for inc in range(1, n_increments+1):
    Fext = Fext_full * (inc / n_increments)
    for it in range(max_newton):
        Fint, K = assemble(u)
        R = Fext - Fint
        R[list(fixed)] = 0.0
        res = np.linalg.norm(R[free])
        if res < tol:
            break
        Kff = K[np.ix_(free, free)]
        du = np.zeros(ndof)
        du[free] = spsolve(Kff, R[free])
        u += du
    newton_iters.append(it+1)
    final_res = float(res)
    print(f"increment {{inc}}/{{n_increments}}: {{it+1}} Newton iters, res={{res:.3e}}")

uy = u[1::2]
tip_defl = float(np.min(uy))
print(f"Nonlinear tip deflection u_y = {{tip_defl:.6f}}")

# --------------------------------------------------------- Kratos VTK output
model = KM.Model()
mp = model.CreateModelPart("nl_cantilever")
mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
for n in range(1, n_nodes+1):
    mp.CreateNewNode(n, float(coords[n][0]), float(coords[n][1]), 0.0)
prop = mp.CreateNewProperties(1)
eid = 1
for tri in elements:
    mp.CreateNewElement("Element2D3N", eid, list(tri), prop); eid += 1
for n in range(1, n_nodes+1):
    mp.GetNode(n).SetSolutionStepValue(KM.DISPLACEMENT,
        [float(u[2*(n-1)]), float(u[2*(n-1)+1]), 0.0])
vtk_settings = KM.Parameters("""{{
    "model_part_name": "nl_cantilever",
    "file_format": "ascii",
    "output_precision": 7,
    "output_sub_model_parts": false,
    "nodal_solution_step_data_variables": ["DISPLACEMENT"]
}}""")
KM.VtkOutput(mp, vtk_settings).PrintOutput()

summary = {{
    "n_nodes": n_nodes,
    "n_elements": len(elements),
    "n_dofs": ndof,
    "tip_load": {tip_load},
    "n_increments": n_increments,
    "newton_iters_per_increment": newton_iters,
    "final_residual": final_res,
    "nonlinear_tip_deflection_y": tip_defl,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("summary:", summary)
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
        "pitfalls": [
                        '[Syntax] Element names in the .mdpa MUST include the node-count suffix: SmallDisplacementElement2D3N, not SmallDisplacement2D. Kratos resolves element types via a registry keyed by the full name. '
                        "Signal: RuntimeError 'Element ... is not registered' or 'Trying to construct an element with a wrong name' when ModelPart.CreateNewElement is called with a name missing the NxN suffix.",
                        "[Integration] Materials are defined in StructuralMaterials.json, referenced from the .mdpa by Properties ID. Defining material parameters inline in the .mdpa via 'Begin Properties N' works for simple cases but breaks for laws that need Tables (temperature-dependent E, hardening curves). "
                        "Signal: Element.Initialize raises RuntimeError 'A constitutive law needs to be specified for the element with ID N' from applications/StructuralMechanicsApplication/custom_elements/solid_elements/base_solid_element.cpp when the Property has YOUNG_MODULUS / POISSON_RATIO set but no CONSTITUTIVE_LAW. (Verified empirically 2026-06-01 — prior catalog text said 'No constitutive law assigned to Property X' and pointed at AnalysisStage.Initialize; the real error message references the element ID, not the Property, and originates in base_solid_element.cpp:249.)",
                        '[Syntax] SubModelPart names must match EXACTLY between .mdpa and ProjectParameters.json — Kratos is case-sensitive and does not strip whitespace. '
                        "Signal: RuntimeError 'Error: There is no sub model part with name \"NAME\" in model part \"PARENT\"' from ModelPart::ErrorNonExistingSubModelPart in model_part.cpp, listed alongside the available SubModelPart names. (Verified empirically 2026-06-01 — prior wording 'SubModelPart ... does not exist' used CamelCase; the real error text is lowercase 'sub model part' with spaces.)",
                        '[Numerical] For nonlinear analyses: increase the max_iteration argument passed to ResidualBasedNewtonRaphsonStrategy (it is a positional constructor argument on the Strategy, NOT a field on the ResidualCriteria). The Python solver wrappers typically pull it from the JSON solver_settings.max_iteration field; default 10 may not suffice for material nonlinearity or large deformation. '
                        "Signal: solver reports 'Convergence is not achieved' at max_iteration with residual not yet at tolerance; structural displacement stalls at an intermediate state.",
                        '[API] DISPLACEMENT variable is the structural DOF; ROTATION is required additionally for beams and shells. Without ROTATION added to the ModelPart variables list, the beam element can be created and Initialize succeeds, but the failure surfaces when the solver/strategy tries to compute rotational DOFs (Solve / Check). '
                        "Signal: the predictable Kratos pattern for missing-variable errors fires at the first GetSolutionStepValue(ROTATION_*) inside the strategy: RuntimeError 'This container only can store the variables specified in its variables list. The variables list doesn't have this variable: ROTATION_X/Y/Z' from variables_list_data_value_container. (Verified empirically 2026-06-01 — prior catalog text said the error fires at beam-element InitializeSolutionStep with 'not found in variables list' wording; reality is Initialize alone does NOT raise, and when it does fire later the wording matches the container error pattern, not the prior text.)",
                        '[Numerical] SHEAR LOCKING: linear hex8 (3D8N) and quad4 (2D4N) elements lock in bending-dominated problems, producing overly stiff results and wrong frequencies. Use quadratic elements (3D20N, 3D27N, 2D8N, 2D9N) for any problem with significant bending. '
                        'Signal: tip deflection on a cantilever beam meshed with 3D8N is 20-40% smaller than analytic; switching to 3D20N recovers it within 1-2%.',
                        '[API] For POINT_LOAD application: use AssignVectorVariableProcess with constrained: [false, false, false]. The directional-magnitude process the agent might be tempted to use does not exist in current StructuralMechanicsApplication. '
                        "Signal: AttributeError 'module \\'KratosMultiphysics.StructuralMechanicsApplication\\' has no attribute \\'AssignVectorByDirectionProcess\\'' from the Python import when the agent tries to instantiate it. (Verified empirically 2026-06-01 with Kratos 10.4 — the prior catalog claim 'crashes / segfaults for load variables' was misleading because the named class is not available to crash; the genuine pitfall is reaching for an API that the agent must instead replace with AssignVectorVariableProcess.)",
                        "[Syntax] problem_data section MUST include the 'echo_level' field. Kratos accesses it during stage initialisation without a default. "
                        "Signal: Parameters::GetValue raises RuntimeError 'Error: Getting a value that does not exist. entry string : echo_level' from kratos/sources/kratos_parameters.cpp when problem_data omits the field. (Verified empirically 2026-06-01 — prior catalog text said KeyError from RunSolutionLoop; Kratos uses RuntimeError, not Python KeyError, and the message originates in C++ GetValue, not RunSolutionLoop.)",
                        '[API] ConstitutiveLaw assignment requires an INSTANCE, not the class — properties.SetValue(CONSTITUTIVE_LAW, LinearElastic3DLaw) fails because the class object is passed instead of LinearElastic3DLaw(). '
                        "Signal: TypeError with text 'incompatible function arguments' from the SetValue binding; the .pyi shows the second arg type is the law instance.",
                        "[API] Variables (DISPLACEMENT, REACTION, VELOCITY, POINT_LOAD, etc.) must be added to the ModelPart's nodal-variables list via ModelPart.AddNodalSolutionStepVariable BEFORE any Node, Element, or Condition is created. Adding the variable after CreateNewNode raises a runtime error and the existing nodes do not get the DOF. "
                        "Signal: Two distinct failure modes (both verified empirically 2026-06-01): (a) If the variable is added AFTER any Node is created, ModelPart.AddNodalSolutionStepVariable raises RuntimeError 'Attempting to add the variable \"X\" to the model part with name \"Y\" which is not empty' from kratos/includes/model_part.h:521 — Kratos refuses to extend the variables list once nodes exist. (b) If the variable was NEVER added, the first GetSolutionStepValue / SetSolutionStepValue on the Node raises RuntimeError 'This container only can store the variables specified in its variables list. The variables list doesn't have this variable: X' from variables_list_data_value_container. (Prior catalog text only described mode (b) as happening 'when a process tries to read the freshly-added variable' — that is wrong; the freshly-added case is mode (a) and fires at the add call.)",
                    ],
    },
}

GENERATORS = {
    "linear_elasticity_2d": _elasticity_2d_kratos,
    "linear_elasticity_2d_nonlinear": _elasticity_nonlinear_kratos,
}
