"""Falsify/confirm the OASiS structural_dynamics pitfall:
  'Linear quad4 shear-locks in bending — use quad8/quad9. Cantilever tip
   deflection 20-40% smaller than analytic; quadratic recovers it.'

Real Kratos StructuralMechanicsAnalysis, 2D plane-stress cantilever with a tip
shear load, quad4 (SmallDisplacementElement2D4N) vs quad8
(SmallDisplacementElement2D8N) on the SAME element grid.
"""
import json, os, sys, shutil

ETYPE = sys.argv[1]           # quad4 | quad8
NX = int(sys.argv[2])
NY = int(sys.argv[3])
WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lock_%s_%dx%d" % (ETYPE, NX, NY))
shutil.rmtree(WORK, ignore_errors=True); os.makedirs(WORK); os.chdir(WORK)

LX, LY, TH = 10.0, 1.0, 1.0
E, NU, P = 2.0e11, 0.0, -1000.0

coords = {}
if ETYPE == "quad4":
    gx, gy = NX, NY
else:
    gx, gy = 2 * NX, 2 * NY       # quad8 needs mid-side nodes
nid = {}
k = 1
for j in range(gy + 1):
    for i in range(gx + 1):
        if ETYPE == "quad8" and (i % 2 == 1 and j % 2 == 1):
            continue              # serendipity: drop centre node
        nid[(i, j)] = k
        coords[k] = (i * LX / gx, j * LY / gy)
        k += 1

elems = []
eid = 1
for ej in range(NY):
    for ei in range(NX):
        if ETYPE == "quad4":
            i, j = ei, ej
            n = [nid[(i, j)], nid[(i + 1, j)], nid[(i + 1, j + 1)], nid[(i, j + 1)]]
        else:
            i, j = 2 * ei, 2 * ej
            n = [nid[(i, j)], nid[(i + 2, j)], nid[(i + 2, j + 2)], nid[(i, j + 2)],
                 nid[(i + 1, j)], nid[(i + 2, j + 1)], nid[(i + 1, j + 2)], nid[(i, j + 1)]]
        elems.append("  %d 1 %s" % (eid, " ".join(str(x) for x in n))); eid += 1

left = [v for (i, j), v in nid.items() if i == 0]
tip = [v for (i, j), v in nid.items() if i == gx]

ename = "SmallDisplacementElement2D4N" if ETYPE == "quad4" else "SmallDisplacementElement2D8N"
m = []
m.append("Begin Properties 1\nEnd Properties\n")
m.append("Begin Nodes\n" + "\n".join("  %d %.12f %.12f 0.0" % (i, c[0], c[1])
                                     for i, c in sorted(coords.items())) + "\nEnd Nodes\n")
m.append("Begin Elements %s\n" % ename + "\n".join(elems) + "\nEnd Elements\n")
m.append("Begin Conditions PointLoadCondition2D1N\n"
         + "\n".join("  %d 1 %d" % (a + 1, n) for a, n in enumerate(tip)) + "\nEnd Conditions\n")
m.append("Begin SubModelPart Parts_Body\nBegin SubModelPartNodes\n"
         + "\n".join("  %d" % n for n in sorted(coords)) + "\nEnd SubModelPartNodes\n"
         + "Begin SubModelPartElements\n" + "\n".join("  %d" % e for e in range(1, eid))
         + "\nEnd SubModelPartElements\nEnd SubModelPart\n")
m.append("Begin SubModelPart DISPLACEMENT_Fixed\nBegin SubModelPartNodes\n"
         + "\n".join("  %d" % n for n in sorted(left)) + "\nEnd SubModelPartNodes\nEnd SubModelPart\n")
m.append("Begin SubModelPart PointLoad_Tip\nBegin SubModelPartNodes\n"
         + "\n".join("  %d" % n for n in sorted(tip)) + "\nEnd SubModelPartNodes\n"
         + "Begin SubModelPartConditions\n"
         + "\n".join("  %d" % (a + 1) for a in range(len(tip))) + "\nEnd SubModelPartConditions\n"
         + "End SubModelPart\n")
open("beam.mdpa", "w").write("\n".join(m))

open("StructuralMaterials.json", "w").write(json.dumps({"properties": [{
    "model_part_name": "Structure.Parts_Body", "properties_id": 1,
    "Material": {"constitutive_law": {"name": "LinearElasticPlaneStress2DLaw"},
                 "Variables": {"DENSITY": 7850.0, "YOUNG_MODULUS": E,
                               "POISSON_RATIO": NU, "THICKNESS": TH}, "Tables": {}}}]}))

pp = {
    "problem_data": {"problem_name": "beam", "parallel_type": "OpenMP", "echo_level": 0,
                     "start_time": 0.0, "end_time": 1.0},
    "solver_settings": {"solver_type": "Static", "model_part_name": "Structure",
                        "domain_size": 2, "echo_level": 0, "analysis_type": "linear",
                        "model_import_settings": {"input_type": "mdpa", "input_filename": "beam"},
                        "material_import_settings": {"materials_filename": "StructuralMaterials.json"},
                        "time_stepping": {"time_step": 1.0}, "rotation_dofs": False,
                        "linear_solver_settings": {"solver_type": "LinearSolversApplication.sparse_lu"}},
    "processes": {"constraints_process_list": [{
        "python_module": "assign_vector_variable_process", "kratos_module": "KratosMultiphysics",
        "process_name": "AssignVectorVariableProcess",
        "Parameters": {"model_part_name": "Structure.DISPLACEMENT_Fixed",
                       "variable_name": "DISPLACEMENT", "constrained": [True, True, True],
                       "value": [0.0, 0.0, 0.0], "interval": [0.0, "End"]}}],
        "loads_process_list": [{
            "python_module": "assign_vector_by_direction_to_condition_process",
            "kratos_module": "KratosMultiphysics",
            "process_name": "AssignVectorByDirectionToConditionProcess",
            "Parameters": {"model_part_name": "Structure.PointLoad_Tip",
                           "variable_name": "POINT_LOAD", "modulus": abs(P) / len(tip),
                           "direction": [0.0, -1.0, 0.0], "interval": [0.0, "End"]}}],
        "list_other_processes": []},
    "output_processes": {},
}
open("ProjectParameters.json", "w").write(json.dumps(pp, indent=2))

import KratosMultiphysics as KM
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import (
    StructuralMechanicsAnalysis)
params = KM.Parameters(open("ProjectParameters.json").read())
model = KM.Model()
sim = StructuralMechanicsAnalysis(model, params); sim.Run()
mp = model.GetModelPart("Structure")
uy = min(n.GetSolutionStepValue(KM.DISPLACEMENT_Y) for n in mp.Nodes)
I = TH * LY ** 3 / 12.0
G = E / (2 * (1 + NU))
eb = P * LX ** 3 / (3.0 * E * I)
tim = eb + P * LX / (5.0 / 6.0 * G * LY * TH)
print("JSONRESULT " + json.dumps({"etype": ETYPE, "nx": NX, "ny": NY,
                                  "n_nodes": len(coords), "n_elems": eid - 1,
                                  "uy": uy, "euler_bernoulli": eb, "timoshenko": tim,
                                  "ratio_to_timoshenko": uy / tim}))
