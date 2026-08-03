"""Tier-2: ConvectionDiffusionApplication — LaplacianElement2D3N and
EulerianConvDiff2D3N are NOT interchangeable, and swapping them is SILENT.

Catalog claim that this fixture falsified (KNOWLEDGE['poisson'] pitfall,
pre-2026-08-03 wording):

    "with CONVECTION_VELOCITY=0 and a stationary solver, LaplacianElement2D3N
     and EulerianConvDiff2D3N produce solutions that differ by less than 1e-12
     relative norm"

Measured on the installed Kratos 10.4.0: they differ by a relative norm of
exactly 1.0, because EulerianConvDiff2D3N assembles NO diffusion term in this
configuration — its LHS is the consistent mass matrix — so the stationary solve
returns TEMPERATURE == 0 everywhere with rc=0 and no warning.

Two levels of evidence:
  (a) element level: CalculateLocalSystem on a unit right triangle
      * LaplacianElement2D3N  LHS = P1 stiffness [[1,-.5,-.5],[-.5,.5,0],[-.5,0,.5]]
      * EulerianConvDiff2D3N  LHS = consistent mass [[A/6,A/12,A/12],...] , A=0.5
      * both RHS = source*A/3 = 1.6667  (so the SOURCE really is shared)
  (b) system level: 12x12 P1 unit square, f = 2*pi^2*sin(pi x)*sin(pi y),
      u = 0 on the boundary, ResidualBasedLinearStrategy
      * Laplacian  -> max T = 0.9755  (exact 1.0)
      * Eulerian   -> max T = 0.0
"""
from __future__ import annotations

import math
import sys

import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication  # noqa: F401
import KratosMultiphysics.LinearSolversApplication  # noqa: F401


def _settings():
    s = KM.ConvectionDiffusionSettings()
    s.SetUnknownVariable(KM.TEMPERATURE)
    s.SetDiffusionVariable(KM.CONDUCTIVITY)
    s.SetVolumeSourceVariable(KM.HEAT_FLUX)
    s.SetDensityVariable(KM.DENSITY)
    s.SetSpecificHeatVariable(KM.SPECIFIC_HEAT)
    s.SetVelocityVariable(KM.VELOCITY)
    s.SetMeshVelocityVariable(KM.MESH_VELOCITY)
    s.SetReactionVariable(KM.REACTION_FLUX)
    return s


_VARS = (KM.TEMPERATURE, KM.HEAT_FLUX, KM.CONDUCTIVITY, KM.REACTION_FLUX,
         KM.DENSITY, KM.SPECIFIC_HEAT, KM.VELOCITY, KM.MESH_VELOCITY)


def local_system(elem_name):
    model = KM.Model()
    mp = model.CreateModelPart("m")
    mp.ProcessInfo.SetValue(KM.DOMAIN_SIZE, 2)
    for v in _VARS:
        mp.AddNodalSolutionStepVariable(v)
    mp.SetBufferSize(2)
    mp.CreateNewNode(1, 0.0, 0.0, 0.0)
    mp.CreateNewNode(2, 1.0, 0.0, 0.0)
    mp.CreateNewNode(3, 0.0, 1.0, 0.0)
    props = mp.Properties[1]
    props.SetValue(KM.CONDUCTIVITY, 1.0)
    props.SetValue(KM.DENSITY, 1.0)
    props.SetValue(KM.SPECIFIC_HEAT, 1.0)
    mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, _settings())
    mp.ProcessInfo.SetValue(KM.DELTA_TIME, 1.0)
    for n in mp.Nodes:
        n.SetSolutionStepValue(KM.CONDUCTIVITY, 1.0)
        n.SetSolutionStepValue(KM.DENSITY, 1.0)
        n.SetSolutionStepValue(KM.SPECIFIC_HEAT, 1.0)
        n.SetSolutionStepValue(KM.HEAT_FLUX, 10.0)
        n.SetSolutionStepValue(KM.VELOCITY, KM.Array3([0.0, 0.0, 0.0]))
        n.SetSolutionStepValue(KM.MESH_VELOCITY, KM.Array3([0.0, 0.0, 0.0]))
        n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
    el = mp.CreateNewElement(elem_name, 1, [1, 2, 3], props)
    el.Initialize(mp.ProcessInfo)
    lhs = KM.Matrix(3, 3)
    rhs = KM.Vector(3)
    el.CalculateLocalSystem(lhs, rhs, mp.ProcessInfo)
    return [[lhs[i, j] for j in range(3)] for i in range(3)], [rhs[i] for i in range(3)]


def solve(elem_name, n=12):
    model = KM.Model()
    mp = model.CreateModelPart("m")
    mp.ProcessInfo.SetValue(KM.DOMAIN_SIZE, 2)
    for v in _VARS:
        mp.AddNodalSolutionStepVariable(v)
    mp.SetBufferSize(2)
    nid = {}
    k = 1
    for j in range(n + 1):
        for i in range(n + 1):
            mp.CreateNewNode(k, i / n, j / n, 0.0)
            nid[(i, j)] = k
            k += 1
    props = mp.Properties[1]
    props.SetValue(KM.CONDUCTIVITY, 1.0)
    props.SetValue(KM.DENSITY, 1.0)
    props.SetValue(KM.SPECIFIC_HEAT, 1.0)
    eid = 1
    for j in range(n):
        for i in range(n):
            a, b, c, d = nid[(i, j)], nid[(i + 1, j)], nid[(i + 1, j + 1)], nid[(i, j + 1)]
            mp.CreateNewElement(elem_name, eid, [a, b, d], props); eid += 1
            mp.CreateNewElement(elem_name, eid, [b, c, d], props); eid += 1
    mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, _settings())
    for nd in mp.Nodes:
        nd.AddDof(KM.TEMPERATURE, KM.REACTION_FLUX)
        nd.SetSolutionStepValue(KM.CONDUCTIVITY, 1.0)
        nd.SetSolutionStepValue(KM.DENSITY, 1.0)
        nd.SetSolutionStepValue(KM.SPECIFIC_HEAT, 1.0)
        nd.SetSolutionStepValue(KM.VELOCITY, KM.Array3([0.0, 0.0, 0.0]))
        nd.SetSolutionStepValue(KM.MESH_VELOCITY, KM.Array3([0.0, 0.0, 0.0]))
        nd.SetSolutionStepValue(KM.HEAT_FLUX, 2 * math.pi ** 2
                                * math.sin(math.pi * nd.X) * math.sin(math.pi * nd.Y))
        nd.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
        if (abs(nd.X) < 1e-12 or abs(nd.X - 1) < 1e-12
                or abs(nd.Y) < 1e-12 or abs(nd.Y - 1) < 1e-12):
            nd.Fix(KM.TEMPERATURE)
    lin = KM.LinearSolverFactory().Create(
        KM.Parameters('{"solver_type":"LinearSolversApplication.sparse_lu"}'))
    strat = KM.ResidualBasedLinearStrategy(
        mp, KM.ResidualBasedIncrementalUpdateStaticScheme(),
        KM.ResidualBasedBlockBuilderAndSolver(lin), False, False, False, False)
    strat.SetEchoLevel(0)
    mp.ProcessInfo.SetValue(KM.TIME, 1.0)
    mp.ProcessInfo.SetValue(KM.DELTA_TIME, 1.0)
    mp.CloneTimeStep(1.0)
    strat.Solve()
    return [nd.GetSolutionStepValue(KM.TEMPERATURE) for nd in mp.Nodes]


lap_lhs, lap_rhs = local_system("LaplacianElement2D3N")
eul_lhs, eul_rhs = local_system("EulerianConvDiff2D3N")

print(f"laplacian_lhs00={lap_lhs[0][0]:.6f}")
print(f"eulerian_lhs00={eul_lhs[0][0]:.6f}")
print(f"laplacian_rhs0={lap_rhs[0]:.6f}")
print(f"eulerian_rhs0={eul_rhs[0]:.6f}")

# LaplacianElement LHS is the P1 stiffness; EulerianConvDiff LHS is the
# consistent mass matrix of the same triangle (area 0.5 -> A/6, A/12).
stiffness_ok = (abs(lap_lhs[0][0] - 1.0) < 1e-12
                and abs(lap_lhs[0][1] + 0.5) < 1e-12)
mass_ok = (abs(eul_lhs[0][0] - 0.5 / 6.0) < 1e-12
           and abs(eul_lhs[0][1] - 0.5 / 12.0) < 1e-12)
same_source = abs(lap_rhs[0] - eul_rhs[0]) < 1e-9
print(f"laplacian_lhs_is_stiffness={stiffness_ok}")
print(f"eulerian_lhs_is_mass_no_diffusion={mass_ok}")
print(f"both_assemble_same_source={same_source}")

lap = solve("LaplacianElement2D3N")
eul = solve("EulerianConvDiff2D3N")
nrm = math.sqrt(sum(v * v for v in lap))
rel = math.sqrt(sum((a - b) ** 2 for a, b in zip(lap, eul))) / nrm
print(f"laplacian_maxT={max(abs(v) for v in lap):.6f}")
print(f"eulerian_maxT={max(abs(v) for v in eul):.6f}")
print(f"relative_difference={rel:.6f}")
print(f"eulerian_field_is_identically_zero={max(abs(v) for v in eul) == 0.0}")
print(f"catalog_1e-12_equivalence_claim_holds={rel < 1e-12}")

if not (stiffness_ok and mass_ok and same_source and rel > 0.5):
    print("FAIL: fixture expectations not met")
    sys.exit(1)
