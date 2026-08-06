"""Tier-2: LaplacianElement2D3N reads CONDUCTIVITY from the NODES, not from
the Properties object.

Catalog claim falsified (KNOWLEDGE['poisson'], pre-2026-08-03 wording):

    "Properties (CONDUCTIVITY, DENSITY, SPECIFIC_HEAT) go on Properties object,
     NOT on nodes"

That is backwards for this element, and it contradicted
KNOWLEDGE['curved_mms'] pitfall #1, which had it right.

Swap test on a unit right triangle, reading LHS[0][0] of the element system:
  nodal k=1   , Properties k=1     -> 1.0
  nodal k=1   , Properties k=999   -> 1.0     (Properties IGNORED)
  nodal k=999 , Properties k=1     -> 999.0   (nodal value drives it)

Mutation control: T2_MUTATE=1 sets the NODAL conductivity to 1.0 in the probe that is supposed to raise it to 999, so the nodal and property values no longer disagree. The stiffness entry that reads 999.0 collapses to the base value and the nodal-drives-diffusion contrast disappears.
"""
from __future__ import annotations

import os
import sys

import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication  # noqa: F401

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=nodal_conductivity_left_at_the_property_value")

_VARS = (KM.TEMPERATURE, KM.HEAT_FLUX, KM.CONDUCTIVITY, KM.REACTION_FLUX,
         KM.DENSITY, KM.SPECIFIC_HEAT, KM.VELOCITY, KM.MESH_VELOCITY)


def lhs00(prop_k: float, nodal_k: float) -> float:
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
    props.SetValue(KM.CONDUCTIVITY, prop_k)
    props.SetValue(KM.DENSITY, 1.0)
    props.SetValue(KM.SPECIFIC_HEAT, 1.0)
    s = KM.ConvectionDiffusionSettings()
    s.SetUnknownVariable(KM.TEMPERATURE)
    s.SetDiffusionVariable(KM.CONDUCTIVITY)
    s.SetVolumeSourceVariable(KM.HEAT_FLUX)
    s.SetDensityVariable(KM.DENSITY)
    s.SetSpecificHeatVariable(KM.SPECIFIC_HEAT)
    s.SetVelocityVariable(KM.VELOCITY)
    s.SetMeshVelocityVariable(KM.MESH_VELOCITY)
    s.SetReactionVariable(KM.REACTION_FLUX)
    mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, s)
    mp.ProcessInfo.SetValue(KM.DELTA_TIME, 1.0)
    for n in mp.Nodes:
        n.SetSolutionStepValue(KM.CONDUCTIVITY, nodal_k)
        n.SetSolutionStepValue(KM.DENSITY, 1.0)
        n.SetSolutionStepValue(KM.SPECIFIC_HEAT, 1.0)
        n.SetSolutionStepValue(KM.HEAT_FLUX, 0.0)
        n.SetSolutionStepValue(KM.VELOCITY, KM.Array3([0.0, 0.0, 0.0]))
        n.SetSolutionStepValue(KM.MESH_VELOCITY, KM.Array3([0.0, 0.0, 0.0]))
        n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
    el = mp.CreateNewElement("LaplacianElement2D3N", 1, [1, 2, 3], props)
    el.Initialize(mp.ProcessInfo)
    lhs = KM.Matrix(3, 3)
    rhs = KM.Vector(3)
    el.CalculateLocalSystem(lhs, rhs, mp.ProcessInfo)
    return lhs[0, 0]


base = lhs00(1.0, 1.0)
prop_only = lhs00(999.0, 1.0)
nodal_only = lhs00(1.0, 1.0 if MUTATE else 999.0)

print(f"lhs00_prop1_nodal1={base:.6f}")
print(f"lhs00_prop999_nodal1={prop_only:.6f}")
print(f"lhs00_prop1_nodal999={nodal_only:.6f}")

properties_ignored = abs(prop_only - base) < 1e-9
nodal_drives = abs(nodal_only - 999.0 * base) < 1e-6
print(f"properties_conductivity_ignored={properties_ignored}")
print(f"nodal_conductivity_drives_diffusion={nodal_drives}")

if not (properties_ignored and nodal_drives):
    print("FAIL: fixture expectations not met")
    sys.exit(1)
