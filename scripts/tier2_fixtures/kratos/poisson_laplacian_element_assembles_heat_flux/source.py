"""Tier-2: LaplacianElement2D3N DOES assemble HEAT_FLUX volume source.

Empirical resolution of a long-standing contradiction in the
Kratos catalog:

  data/kratos_knowledge.py L933:
    'LaplacianElement DOES assemble HEAT_FLUX volumetric source
     terms (verified against laplacian_element.cpp).'

  data/kratos_knowledge.py L1908 (in same file):
    'CRITICAL: LaplacianElement does NOT assemble source terms
     (HEAT_FLUX)!'

  src/backends/kratos/generators/poisson.py pitfall #1:
    '[Numerical] LaplacianElement does NOT assemble source terms
     (HEAT_FLUX) — only -div(k*grad(T))=0'

Test setup:
  * Single P1 triangle with vertices (0,0), (1,0), (0,1) — area
    = 0.5.
  * LaplacianElement2D3N via mp.CreateNewElement(...).
  * ConvectionDiffusionSettings with VolumeSource = HEAT_FLUX
    set on ProcessInfo before Element.Initialize.
  * Call Element.CalculateRightHandSide once with HEAT_FLUX=0
    on all nodes, then again with HEAT_FLUX=10 on all nodes.

Expected result (linear P1 shape functions, constant source):
  RHS_node_i = source * triangle_area / num_nodes
             = 10 * 0.5 / 3
             = 1.66667
  on EVERY node, for the second call. RHS = 0 for the first
  call.

This conclusively shows: LaplacianElement DOES assemble the
HEAT_FLUX volumetric source term. The 'does NOT' claims in the
catalog are wrong and must be removed.
"""
from __future__ import annotations

import sys

import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication  # noqa: F401


def main() -> int:
    model = KM.Model()
    mp = model.CreateModelPart("Main")

    mp.AddNodalSolutionStepVariable(KM.TEMPERATURE)
    mp.AddNodalSolutionStepVariable(KM.HEAT_FLUX)
    mp.AddNodalSolutionStepVariable(KM.FACE_HEAT_FLUX)
    mp.AddNodalSolutionStepVariable(KM.NORMAL)
    mp.AddNodalSolutionStepVariable(KM.CONDUCTIVITY)
    mp.AddNodalSolutionStepVariable(KM.DENSITY)
    mp.AddNodalSolutionStepVariable(KM.SPECIFIC_HEAT)

    mp.CreateNewNode(1, 0.0, 0.0, 0.0)
    mp.CreateNewNode(2, 1.0, 0.0, 0.0)
    mp.CreateNewNode(3, 0.0, 1.0, 0.0)

    props = mp.CreateNewProperties(0)
    props.SetValue(KM.CONDUCTIVITY, 1.0)
    props.SetValue(KM.DENSITY, 1.0)
    props.SetValue(KM.SPECIFIC_HEAT, 1.0)

    settings = KM.ConvectionDiffusionSettings()
    settings.SetUnknownVariable(KM.TEMPERATURE)
    settings.SetVolumeSourceVariable(KM.HEAT_FLUX)
    settings.SetDiffusionVariable(KM.CONDUCTIVITY)
    settings.SetDensityVariable(KM.DENSITY)
    settings.SetSpecificHeatVariable(KM.SPECIFIC_HEAT)
    mp.ProcessInfo[KM.CONVECTION_DIFFUSION_SETTINGS] = settings

    element = mp.CreateNewElement(
        "LaplacianElement2D3N", 1, [1, 2, 3], props)
    element.Initialize(mp.ProcessInfo)

    # Zero source
    for node in mp.Nodes:
        node.SetSolutionStepValue(KM.HEAT_FLUX, 0, 0.0)
    rhs_zero = KM.Vector(3)
    element.CalculateRightHandSide(rhs_zero, mp.ProcessInfo)
    rhs_zero_max = max(abs(rhs_zero[i]) for i in range(3))
    print(f"rhs_zero_norm_inf={rhs_zero_max:.6e}")

    # Source = 10 on every node
    for node in mp.Nodes:
        node.SetSolutionStepValue(KM.HEAT_FLUX, 0, 10.0)
    rhs_10 = KM.Vector(3)
    element.CalculateRightHandSide(rhs_10, mp.ProcessInfo)
    print(f"rhs10_node0={rhs_10[0]:.6f}")
    print(f"rhs10_node1={rhs_10[1]:.6f}")
    print(f"rhs10_node2={rhs_10[2]:.6f}")

    expected = 10.0 * 0.5 / 3.0
    print(f"expected_per_node={expected:.6f}")
    tol = 1e-6
    matches = all(abs(rhs_10[i] - expected) < tol for i in range(3))
    print(f"per_node_matches_expected={matches}")
    print(f"laplacian_assembles_heat_flux={rhs_zero_max < 1e-12 and matches}")

    if rhs_zero_max < 1e-12 and matches:
        return 0
    print("ERROR: empirical claim falsified differently than "
          "expected", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
