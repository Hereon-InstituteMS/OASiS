"""Kratos MPM (Material Point Method) generators and knowledge.

THE GENERATOR IN THIS FILE DOES NOT USE MPMApplication.

`_mpm_2d_kratos` emits a standalone explicit MPM written in numpy: it builds its
own background grid, its own shape functions and its own USL update loop. The
string "KratosMultiphysics" does not occur anywhere in what it writes. It runs
and it computes something — 256 material points, 5000 steps, ~3 minutes, a
result.vtu and a results_summary.json reporting max|v| = 3.04 m/s for a body
falling under gravity — but nothing in that path is Kratos.

The KNOWLEDGE below IS about MPMApplication, and every claim in it was verified
by execution against MPMApplication 10.4.3. So an agent reading knowledge('mpm')
and an agent running the template are being told about two different codes.

MEASURED 2026-08-07, on why nothing caught this:

  * KratosBackend.validate_input carries an "honesty guard" whose closing check
    is 'neither uses KratosMultiphysics nor runs a solve'. This template passes
    it on the strength of ONE line: `from scipy.sparse.linalg import spsolve`.
    spsolve is never called, and neither is lil_matrix. Delete that dead import
    and the guard rejects the template outright. The guard is being held green
    by an unused import.

  * Seven of the twenty-one Kratos templates are standalone in this same sense
    (cosimulation, heat, heat_transient, linear_elasticity, mpm,
    shape_optimization, structural_dynamics). For six of them the guard's own
    comment says this is deliberate — the "scipy/numpy assemble-and-solve
    pattern" — and those six do call a real solve (spsolve or factorized).
    mpm is the only one of the seven that calls neither.

  * That comment also justifies the pattern by saying such generators "use
    KratosMultiphysics for I/O". None of the seven contains the string
    KratosMultiphysics, so the justification no longer describes them.

This file has NOT been rewritten to drive MPMApplication. Doing so is a real
piece of work — two mdpa files, a background grid that must enclose the whole
trajectory, MATERIAL_POINTS_PER_ELEMENT from the geometry's allowed set, an
opt-in gravity process, Dirichlet conditions on the grid rather than the body —
and every one of those requirements is already documented, with its failure
signature, in KNOWLEDGE["mpm"]["pitfalls"] below.
"""


def _mpm_2d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable program. All parameter defaults are placeholders.

    Material Point Method for large-deformation solid mechanics."""
    n_cells_x = params.get("n_cells_x", 20)
    n_cells_y = params.get("n_cells_y", 20)
    ppc = params.get("particles_per_cell", 4)
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    density = params.get("density", 1000.0)
    gravity = params.get("gravity", -9.81)
    dt = params.get("dt", 1e-4)
    T_end = params.get("T_end", 0.5)
    domain_x = params.get("domain_x", 1.0)
    domain_y = params.get("domain_y", 1.0)
    body_x0 = params.get("body_x0", 0.3)
    body_x1 = params.get("body_x1", 0.7)
    body_y0 = params.get("body_y0", 0.5)
    body_y1 = params.get("body_y1", 0.9)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""Material Point Method — large-deformation solid — Kratos (standalone)"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import json

# Grid parameters — set for your problem
n_cells_x, n_cells_y = {n_cells_x}, {n_cells_y}
domain_x, domain_y = {domain_x}, {domain_y}
dx = domain_x / n_cells_x
dy = domain_y / n_cells_y
n_nodes_x = n_cells_x + 1
n_nodes_y = n_cells_y + 1
n_nodes = n_nodes_x * n_nodes_y

# Material parameters — set for your problem
mu_val, lam_val = {mu}, {lam}
density = {density}
gravity = np.array([0.0, {gravity}])
dt = {dt}
T_end = {T_end}

def node_id(i, j):
    return j * n_nodes_x + i

def grid_coords(nid):
    j, i = divmod(nid, n_nodes_x)
    return np.array([i * dx, j * dy])

# Generate material points inside body region
ppc = {ppc}
particles_x = []
particles_v = []
particles_vol = []
particles_mass = []
particles_stress = []
particles_F = []

cell_vol = dx * dy / ppc
for cy in range(n_cells_y):
    for cx in range(n_cells_x):
        cell_x0 = cx * dx
        cell_y0 = cy * dy
        cell_cx = cell_x0 + dx / 2
        cell_cy = cell_y0 + dy / 2
        # Check if cell center is inside initial body
        if {body_x0} <= cell_cx <= {body_x1} and {body_y0} <= cell_cy <= {body_y1}:
            # Place particles in a 2x2 grid per cell
            sp = int(np.sqrt(ppc))
            for py in range(sp):
                for px in range(sp):
                    x_p = cell_x0 + (px + 0.5) * dx / sp
                    y_p = cell_y0 + (py + 0.5) * dy / sp
                    particles_x.append(np.array([x_p, y_p]))
                    particles_v.append(np.zeros(2))
                    particles_vol.append(cell_vol)
                    particles_mass.append(density * cell_vol)
                    particles_stress.append(np.zeros((2, 2)))
                    particles_F.append(np.eye(2))

n_particles = len(particles_x)
print(f"MPM: {{n_particles}} particles on {{n_cells_x}}x{{n_cells_y}} grid")

particles_x = np.array(particles_x)
particles_v = np.array(particles_v)
particles_vol = np.array(particles_vol)
particles_mass = np.array(particles_mass)

# Bilinear shape functions
def shape_functions(x_p, cell_i, cell_j):
    x0 = cell_i * dx
    y0 = cell_j * dy
    xi = (x_p[0] - x0) / dx
    eta = (x_p[1] - y0) / dy
    N = np.array([(1 - xi) * (1 - eta), xi * (1 - eta),
                  xi * eta, (1 - xi) * eta])
    dNdx = np.array([[-(1 - eta) / dx, -(1 - xi) / dy],
                      [(1 - eta) / dx, -xi / dy],
                      [eta / dx, xi / dy],
                      [-eta / dx, (1 - xi) / dy]])
    nodes = [node_id(cell_i, cell_j), node_id(cell_i + 1, cell_j),
             node_id(cell_i + 1, cell_j + 1), node_id(cell_i, cell_j + 1)]
    return N, dNdx, nodes

# Time integration — USL (Update Stress Last)
n_steps = int(T_end / dt)
output_interval = max(1, n_steps // 20)
max_disp = 0.0

for step in range(n_steps):
    ndof = 2 * n_nodes
    grid_mass = np.zeros(n_nodes)
    grid_momentum = np.zeros(ndof)
    grid_force = np.zeros(ndof)

    # Particle to grid transfer
    for p in range(n_particles):
        ci = min(int(particles_x[p, 0] / dx), n_cells_x - 1)
        cj = min(int(particles_x[p, 1] / dy), n_cells_y - 1)
        ci = max(0, ci)
        cj = max(0, cj)
        N, dNdx, nodes = shape_functions(particles_x[p], ci, cj)

        for a in range(4):
            nid = nodes[a]
            grid_mass[nid] += N[a] * particles_mass[p]
            grid_momentum[2 * nid] += N[a] * particles_mass[p] * particles_v[p, 0]
            grid_momentum[2 * nid + 1] += N[a] * particles_mass[p] * particles_v[p, 1]
            # Internal force: -sigma * grad(N) * vol
            sigma = particles_stress[p]
            grid_force[2 * nid] -= (sigma[0, 0] * dNdx[a, 0] + sigma[0, 1] * dNdx[a, 1]) * particles_vol[p]
            grid_force[2 * nid + 1] -= (sigma[1, 0] * dNdx[a, 0] + sigma[1, 1] * dNdx[a, 1]) * particles_vol[p]
            # Body force (gravity)
            grid_force[2 * nid + 1] += N[a] * particles_mass[p] * gravity[1]

    # Grid update with boundary conditions
    for nid in range(n_nodes):
        if grid_mass[nid] > 1e-14:
            # Update momentum
            grid_momentum[2 * nid] += dt * grid_force[2 * nid]
            grid_momentum[2 * nid + 1] += dt * grid_force[2 * nid + 1]
        # Floor BC (y=0): zero y-velocity
        j_idx = nid // n_nodes_x
        if j_idx == 0 and grid_mass[nid] > 1e-14:
            grid_momentum[2 * nid + 1] = 0.0

    # Grid to particle transfer and stress update
    for p in range(n_particles):
        ci = min(int(particles_x[p, 0] / dx), n_cells_x - 1)
        cj = min(int(particles_x[p, 1] / dy), n_cells_y - 1)
        ci = max(0, ci)
        cj = max(0, cj)
        N, dNdx, nodes = shape_functions(particles_x[p], ci, cj)

        v_new = np.zeros(2)
        grad_v = np.zeros((2, 2))
        for a in range(4):
            nid = nodes[a]
            if grid_mass[nid] > 1e-14:
                v_node = grid_momentum[2 * nid:2 * nid + 2] / grid_mass[nid]
                v_new += N[a] * v_node
                grad_v += np.outer(v_node, dNdx[a])

        particles_v[p] = v_new
        particles_x[p] += v_new * dt

        # Update deformation gradient
        F_old = particles_F[p]
        particles_F[p] = (np.eye(2) + dt * grad_v) @ F_old

        # Update volume
        J = np.linalg.det(particles_F[p])
        particles_vol[p] = abs(J) * particles_mass[p] / density

        # Neo-Hookean stress update (Cauchy)
        F = particles_F[p]
        J = np.linalg.det(F)
        if J > 0.01:
            b = F @ F.T
            particles_stress[p] = (mu_val / J) * (b - np.eye(2)) + (lam_val * np.log(J) / J) * np.eye(2)

    disp = np.linalg.norm(particles_x - np.array([
        [ci * dx + dx / 2 for ci in range(n_cells_x) for _ in range(ppc)]
        for _ in range(1)
    ]).flatten()[:n_particles] if False else 0.0)
    cur_max = np.max(np.abs(particles_v))
    max_disp = max(max_disp, cur_max)

    if step % output_interval == 0:
        print(f"Step {{step}}/{{n_steps}}, t={{step*dt:.6f}}, max|v|={{cur_max:.6e}}")

print(f"MPM complete: {{n_steps}} steps, {{n_particles}} particles")

# Write VTU output
import meshio
points = np.column_stack([particles_x, np.zeros(n_particles)])
cells = [("vertex", np.arange(n_particles).reshape(-1, 1))]
stress_xx = np.array([particles_stress[p][0, 0] for p in range(n_particles)])
stress_yy = np.array([particles_stress[p][1, 1] for p in range(n_particles)])
point_data = {{
    "velocity_x": particles_v[:, 0],
    "velocity_y": particles_v[:, 1],
    "stress_xx": stress_xx,
    "stress_yy": stress_yy,
    "volume": particles_vol,
}}
meshio.Mesh(points, cells, point_data=point_data).write("result.vtu")

summary = {{
    "n_particles": n_particles,
    "n_steps": n_steps,
    "dt": dt,
    "max_velocity": float(max_disp),
    "grid": f"{{n_cells_x}}x{{n_cells_y}}",
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("MPM simulation complete.")
'''


KNOWLEDGE = {
    "mpm": {
        "description": "Material Point Method via Kratos MPMApplication",
        "application": "MPMApplication (pip install KratosMPMApplication)",
        "elements": {
            "2D": [
                "MPMUpdatedLagrangian2D3N",
                "MPMUpdatedLagrangian2D4N",
            ],
            "3D": [
                "MPMUpdatedLagrangian3D4N",
                "MPMUpdatedLagrangian3D8N",
            ],
            "axisymmetric": [
                "MPMUpdatedLagrangianAxisymmetry2D3N",
                "MPMUpdatedLagrangianAxisymmetry2D4N",
            ],
            "PQ_variant": [
                "MPMUpdatedLagrangianPQ",
            ],
            "UP_variant": [
                "MPMUpdatedLagrangianUP",
                "MPMUpdatedLagrangianUP2D3N",
            ],
        },
        # Registered names, checked with KratosGlobals.HasConstitutiveLaw against the
        # installed MPMApplication 10.4.0 wheel (2026-08-03). The previous entries were
        # family labels ("LinearElastic", "NeoHookean", "HenckyMC", "HenckyBorjaCamClay",
        # "Johnson-Cook") — none of those five strings resolves to a law, so a
        # Materials.json built from them fails with "not registered".
        "constitutive_laws": [
            "LinearElasticIsotropic3DLaw / LinearElasticIsotropicPlaneStrain2DLaw / "
            "LinearElasticIsotropicPlaneStress2DLaw / LinearElasticIsotropicAxisym2DLaw "
            "(small strain)",
            "HyperElasticNeoHookean3DLaw / HyperElasticNeoHookeanPlaneStrain2DLaw / "
            "HyperElasticNeoHookeanAxisym2DLaw / HyperElasticNeoHookeanUP3DLaw "
            "(finite strain)",
            "HenckyMCPlastic3DLaw / HenckyMCPlasticPlaneStrain2DLaw / "
            "HenckyMCPlasticAxisym2DLaw (Mohr-Coulomb with Hencky strain; "
            "HenckyMCStrainSofteningPlastic* for softening)",
            "HenckyBorjaCamClayPlastic3DLaw / HenckyBorjaCamClayPlasticPlaneStrain2DLaw "
            "(critical state)",
            "JohnsonCookThermalPlastic3DLaw / JohnsonCookThermalPlastic2DPlaneStrainLaw / "
            "JohnsonCookThermalPlastic2DAxisymLaw (rate-dependent plasticity)",
        ],
        # solver_type is a SHORT LABEL, not a class name. The accepted strings are
        # "static"/"Static", "quasi_static"/"Quasi-static" and "dynamic"/"Dynamic";
        # for "dynamic" a second key "time_integration_method" selects
        # "implicit" or "explicit". The class names MPMStaticSolver /
        # MPMImplicitDynamicSolver / MPMExplicitSolver are NOT accepted values.
        "solver_types": [
            "static (or Static)",
            "quasi_static (or Quasi-static)",
            "dynamic (or Dynamic) + time_integration_method: implicit | explicit",
        ],
        "scheme_types": {
            "implicit dynamic / quasi-static": ["newmark", "bossak (default)"],
            "explicit dynamic": ["central_difference (default)", "forward_euler"],
            "static": "no scheme_type key exists; supplying one fails validation",
        },
        # USL / USF / MUSL are values of the "stress_update" key and apply ONLY to
        # an explicit solver running scheme_type "forward_euler"; with
        # "central_difference" the option is forced to 0 and stress_update is
        # ignored entirely.
        "stress_update": ["usf (default)", "usl", "musl"],
        "model_parts": (
            "MPM needs TWO mdpa files and creates three model parts. "
            "solver_settings.grid_model_import_settings.input_filename names the "
            "background GRID mdpa (meshed with plain FEM element names such as "
            "Element2D4N); solver_settings.model_import_settings.input_filename names "
            "the BODY mdpa (meshed with MPM* element names). The model parts are "
            "'Background_Grid', 'MPM_Material' and 'Initial_MPM_Material' \u2014 those names "
            "are fixed, there is no key to rename the grid. Boundary conditions attach "
            "to sub model parts of Background_Grid; materials attach to sub model parts "
            "of Initial_MPM_Material."
        ),
        "pitfalls": [
            "[API] Kratos MPM element names ALL start with the literal prefix \"MPM\": MPMUpdatedLagrangian2D4N, MPMUpdatedLagrangian3D8N, MPMUpdatedLagrangianAxisymmetry2D4N, MPMUpdatedLagrangianPQ, MPMUpdatedLagrangianUP, etc. The prior catalog listed UpdatedLagrangianPQ2D / UpdatedLagrangianAxisym (without the MPM prefix) \u2014 none of those are registered. Signal: model_part.CreateNewElement(\"UpdatedLagrangian2D3N\", ...) raises 'Error: The Element \"UpdatedLagrangian2D3N\" is not registered!' and lists the registered elements; prepending MPM makes the identical call succeed. Beware grepping for the bare name \u2014 it matches as a substring of the MPM-prefixed one, so only element creation settles it. (Verified by execution 2026-08-07.)",
            "[Input] MPM reads TWO mdpa files, and the background grid has its own key: solver_settings.grid_model_import_settings.input_filename. Omitting that block does not raise a missing-key error \u2014 the default filename is used and the failure surfaces as a missing file. Signal: RuntimeError 'Error: Error opening mdpa file : \"unknown_name_Grid.mdpa\"' \u2014 the literal string unknown_name_Grid is the giveaway that the grid import block is absent rather than the file being misnamed. (Verified by execution 2026-08-07.)",
            "[Input] MATERIAL_POINTS_PER_ELEMENT is mandatory and lives in the MATERIALS json, under properties[i].Material.Variables \u2014 not in ProjectParameters. On the installed 10.4.3 build its absence is a hard error, not a defaulted warning. Signal: RuntimeError 'Error: \"MATERIAL_POINTS_PER_ELEMENT\" is not specified in Properties' raised from MaterialPointGeneratorUtility during solver Initialize. (Verified by execution 2026-08-07.)",
            "[Input] The number of material points per element is drawn from a fixed set that depends on the GRID element geometry, and the sets are NOT the same across geometries: Triangular 1/3/4/6/12, Quadrilateral 1/4/9/16/25, Tetrahedral 1/4/8/14/24, Hexahedral 1/8/27/64/125. Anything else is rejected on 10.4.3. Signal: RuntimeError 'Error: The input number of MATERIAL_POINTS_PER_ELEMENT (5) is not available for Quadrilateral elements' followed by 'Available options are: 1, 4, 9, 16 and 25.' \u2014 the message names the GRID geometry, so it is also how you discover your background grid is quads when you assumed triangles. (Verified by execution 2026-08-07; the allowed sets were read back from the installed libKratosMPMCore. Kratos master after this release downgrades this to a warning that silently clamps to the geometry default, so on a newer build the same mistake yields a different material-point count instead of an error.)",
            "[Input] The legacy spelling PARTICLES_PER_ELEMENT still works, and the solver REWRITES YOUR MATERIALS FILE ON DISK to the new name as a side effect of running. Signal: the run prints '[DEPRECATED INPUT PARAMETERS] \\'PARTICLES_PER_ELEMENT\\' is deprecated; use \\'MATERIAL_POINTS_PER_ELEMENT\\' instead.' and completes normally, after which the materials json in the working directory no longer contains the string PARTICLES_PER_ELEMENT \u2014 a version-controlled input file is modified by a simulation run. (Verified by execution 2026-08-07.)",
            "[Input] Materials entries address the BODY through 'Initial_MPM_Material.<SubModelPart>'. Using the MPM_Material root instead fails, because at materials-reading time the body sub model parts only exist under Initial_. Signal: RuntimeError 'Error: There is no sub model part with name \"Parts_Parts_Auto1\" in model part \"MPM_Material\"' followed by the list of sub model parts that DO exist. (Verified by execution 2026-08-07.)",
            "[BC] Boundary conditions attach to sub model parts of Background_Grid, never of MPM_Material \u2014 the material points move, the grid does not, so the constrained set has to be a grid region. Signal: pointing a constraints_process_list entry at 'MPM_Material.<name>' raises RuntimeError 'Error: There is no sub model part with name \"DISPLACEMENT_Displacement_Auto1\" in model part \"MPM_Material\"'; the same block with 'Background_Grid.<name>' runs. (Verified by execution 2026-08-07.)",
            "[API] solver_settings.solver_type takes a short label, not a solver class name. Accepted: 'static'/'Static', 'quasi_static'/'Quasi-static', 'dynamic'/'Dynamic' (which then requires time_integration_method 'implicit' or 'explicit'). Signal: 'MPMImplicitDynamicSolver' raises Exception 'The requested solver type \"MPMImplicitDynamicSolver\" is not in the python solvers wrapper' + 'Available options are: \"static\", \"dynamic\", \"quasi_static\"'. (Verified by execution 2026-08-07 \u2014 the class-name spellings were previously served as the solver_types list.)",
            "[API] scheme_type is partitioned by solver: an implicit run takes only newmark or bossak, an explicit run only central_difference or forward_euler, and a static run has no scheme_type key at all. Signal: scheme_type 'central_difference' on an implicit dynamic solver raises Exception 'The requested scheme type \"central_difference\" is not available!' + 'Available options are: \"newmark\", \"bossak\"'. (Verified by execution 2026-08-07.)",
            "[Input] Constitutive law names are the fully-qualified registered strings; the short family label is the single most common MPM setup error. Signal: 'LinearElasticPlaneStrain2DLaw' raises RuntimeError 'Error: Kratos components missing \"LinearElasticPlaneStrain2DLaw\"' \u2014 the fix is LinearElasticIsotropicPlaneStrain2DLaw, i.e. the 'Isotropic' the short name drops. (Verified by execution 2026-08-07.)",
            "[Numerical] Material points that leave the background grid are DELETED, and the mass they carry leaves the simulation with them. The receipt is two log lines, not an error, so a partially-escaped body silently loses mass; only when the last point is gone does the run stop. Signal: 'MPMSearchElementUtility: WARNING: Search Element for Material Point: 26 is failed. Geometry is cleared.' then '[WARNING] MaterialPointEraseProcess: 1 particle elements have been erased.', and once the body is entirely outside, RuntimeError 'Error: No degrees of freedom in model part: MPM_Material'. (Verified by execution 2026-08-07 by letting a body free-fall out of its grid.)",
            "[Physics] Gravity is opt-in. Without an assign_gravity_to_material_point_process block, MP_VOLUME_ACCELERATION stays zero and the body simply does not fall \u2014 the run is successful, converged and wrong. Signal: an otherwise identical deck with the gravity process removed completes with exit code 0, unchanged total material-point mass, and MP_DISPLACEMENT exactly 0.0 where the reference gives -0.04905; no warning is emitted. (Verified by execution 2026-08-07.)",
            "[Input] The BODY mdpa carries MPM* element names but the runtime ignores them: the material-point element type is chosen from the GRID geometry plus the ProjectParameters flags pressure_dofs and is_pqmpm. Writing MPMUpdatedLagrangianUP2D3N in the body mdpa without \"pressure_dofs\": true silently yields plain displacement elements. Signal: no message at all \u2014 the mixed formulation is simply absent, so a nearly-incompressible run shows volumetric locking rather than reporting a configuration error. (Verified from Kratos source 10.4.3, MaterialPointGeneratorUtility hard-codes the element stem; not separately executed.)",
        ],
        "guidance": [
            "[Numerical] The background grid must enclose the whole TRAJECTORY of the body, not just its initial position \u2014 points that exit are erased (see pitfalls).",
            "[Numerical] Penalty Dirichlet conditions take penalty_coefficient (the older name penalty_factor is auto-renamed). It defaults to 0, which silently disables the constraint; shipped tests use 1e10 to 1e12, i.e. two to three orders above YOUNG_MODULUS.",
            "[Numerical] Cell-crossing instability: Kratos MPM does NOT implement GIMP or CPDI — both strings appear in zero files of MPMApplication, so advice to 'use GIMP or CPDI shape functions' cannot be acted on here. The mitigation Kratos does provide is PQMPM (partitioned-quadrature MPM), switched on with \"is_pqmpm\": true in solver_settings, which makes the generator build MPMUpdatedLagrangianPQ elements instead.",
            "[Numerical] Material points per cell: 4-16 typical, but only from the geometry's allowed set (see pitfalls).",
            "[Numerical] Time step: dt < h/c where h=cell size, c=wave speed.",
            "[Numerical] Zero-energy modes possible with linear elements: use stabilization.",
            "[Numerical] An inverted deformation gradient is the canonical MPM blow-up and it does raise: 'MPM UPDATED LAGRANGIAN DISPLACEMENT ELEMENT INVERTED: |F|<0 detF = <value>'. It usually means the time step is too large or there are too few material points per element.",
        ]
    },
}

GENERATORS = {
    "mpm_2d": _mpm_2d_kratos,
}
