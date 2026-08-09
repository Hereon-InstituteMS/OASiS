"""Tier-2 (kratos.dem::9): FinalTime decides when a DEM run stops, and its
default silently overwrites problem_data.end_time.

A user coming from StructuralMechanics writes problem_data.end_time and expects
it to be honoured. DEM validates the deck, fills FinalTime from its own default
(0.05) and then OVERWRITES problem_data.end_time with it in
FixParametersInconsistencies, so the run stops 200x short with no warning.

    problem_data.end_time = 10.0, no FinalTime  ->  stops at TIME ~= 0.05

MUTATION CONTROL (T2_MUTATE=1): FinalTime is written explicitly, as 0.002. The
run then stops there instead of at the 0.05 default, so "the default won" is no
longer what happened. Only the deck changes; the deck still really runs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

MUTATE = os.environ.get("T2_MUTATE") == "1"


_RUNNER = r'''
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import KratosMultiphysics as KM
import KratosMultiphysics.DEMApplication  # noqa: F401
from KratosMultiphysics.DEMApplication.DEM_analysis_stage import DEMAnalysisStage

observed = {"reached_finalize": False}


class Probe(DEMAnalysisStage):
    # DEMAnalysisStage.Finalize() deletes its model parts, and a python
    # reference held across that call segfaults, so read the numbers here.
    def Finalize(self):
        s, w = self.spheres_model_part, self.rigid_face_model_part
        observed.update({
            "reached_finalize": True,
            "spheres_elements": s.NumberOfElements(),
            "spheres_nodes": s.NumberOfNodes(),
            "wall_conditions": w.NumberOfConditions(),
            "time": s.ProcessInfo[KM.TIME],
            "delta_time": s.ProcessInfo[KM.DELTA_TIME],
            "particle_density": None,
            "young_modulus": None,
            "density": None,
        })
        # PARTICLE_DENSITY is registered by DEMApplication, DENSITY and
        # YOUNG_MODULUS by core; resolve all three through the kernel so the
        # lookup cannot silently pick the wrong module.
        for key in ("PARTICLE_DENSITY", "YOUNG_MODULUS", "DENSITY"):
            var = KM.KratosGlobals.GetVariable(key)
            for props in s.Properties:
                observed[key.lower()] = props[var] if props.Has(var) else None
                break
        super().Finalize()


try:
    with open("ProjectParametersDEM.json") as f:
        pars = KM.Parameters(f.read())
    Probe(KM.Model(), pars).Run()
    observed["exception"] = None
except Exception as exc:                     # noqa: BLE001 - classifying
    observed["exception"] = type(exc).__name__
    observed["message"] = str(exc).replace("\n", " ")[:400]
with open("observed.json", "w") as f:
    json.dump(observed, f)
'''


def _write_deck(d, *, problem="case", n=4, radius=0.01, density=2650.0,
                young=1.0e7, poisson=0.25, dt=2.0e-5, final_time=2.0e-3,
                spheres_stem=None, walls=True, materials=None, params=None,
                assignation=None, particle_density_key="PARTICLE_DENSITY",
                omit_young=False):
    """Write a minimal, WORKING 2D DEM deck into directory ``d``.

    Deliberately tiny (4 CylinderParticle2D discs, 100 steps) so a fixture can
    afford several runs; the reference deck completes in well under a second.
    Returns the ProjectParametersDEM dict that was written.
    """
    stem = spheres_stem if spheres_stem is not None else problem + "DEM"
    pos = [(0.05 + i * 2.2 * radius, 0.03 + radius, 0.0) for i in range(n)]
    L = ["Begin ModelPartData", "End ModelPartData", "",
         "Begin Properties 1", "End Properties", "", "Begin Nodes"]
    L += ["  %d %.10f %.10f %.10f" % (i, x, y, z)
          for i, (x, y, z) in enumerate(pos, 1)]
    L += ["End Nodes", "", "Begin Elements CylinderParticle2D"]
    L += ["  %d 1 %d" % (i, i) for i in range(1, n + 1)]
    L += ["End Elements", "", "Begin NodalData RADIUS"]
    L += ["  %d 0 %.10f" % (i, radius) for i in range(1, n + 1)]
    L += ["End NodalData", ""]
    with open(os.path.join(d, stem + ".mdpa"), "w") as f:
        f.write("\n".join(L))

    if walls:
        zl, zh = -5.0 * radius, 5.0 * radius
        W = ["Begin ModelPartData", "End ModelPartData", "",
             "Begin Properties 2", "End Properties", "", "Begin Nodes",
             "  1 0.0 0.0 %.6f" % zl, "  2 0.5 0.0 %.6f" % zl,
             "  3 0.5 0.0 %.6f" % zh, "  4 0.0 0.0 %.6f" % zh,
             "End Nodes", "", "Begin Conditions RigidFace3D3N",
             "  1 2 1 2 3", "  2 2 1 3 4", "End Conditions", "",
             "Begin SubModelPart DEM-FEM-Wall_floor", "  Begin SubModelPartData",
             "  End SubModelPartData", "  Begin SubModelPartNodes",
             "    1", "    2", "    3", "    4", "  End SubModelPartNodes",
             "  Begin SubModelPartConditions", "    1", "    2",
             "  End SubModelPartConditions", "End SubModelPart", ""]
        with open(os.path.join(d, problem + "DEM_FEM_boundary.mdpa"), "w") as f:
            f.write("\n".join(W))

    pair = {"COEFFICIENT_OF_RESTITUTION": 0.5, "STATIC_FRICTION": 0.4,
            "DYNAMIC_FRICTION": 0.4, "FRICTION_DECAY": 500.0,
            "DEM_DISCONTINUUM_CONSTITUTIVE_LAW_NAME":
                "DEM_D_Hertz_viscous_Coulomb2D"}
    if materials is None:
        grain = {particle_density_key: density, "POISSON_RATIO": poisson}
        if not omit_young:
            grain["YOUNG_MODULUS"] = young
        if assignation is None:
            # Address the ROOT SpheresPart, not a sub model part: a sub model
            # part lives inside the spheres mdpa, so naming one here would make
            # a missing mdpa raise on the assignation table instead of
            # exhibiting the silent-empty-model behaviour under test.
            assignation = ([["SpheresPart", "g"]]
                           + ([["RigidFacePart.DEM-FEM-Wall_floor", "w"]]
                              if walls else []))
        materials = {
            "materials": [
                {"material_name": "g", "material_id": 1, "Variables": grain},
                {"material_name": "w", "material_id": 2,
                 "Variables": {"PARTICLE_DENSITY": 7850.0,
                               "YOUNG_MODULUS": 1.0e9,
                               "POISSON_RATIO": 0.3}}],
            "material_relations": [
                {"material_names_list": ["g", "g"], "material_ids_list": [1, 1],
                 "Variables": dict(pair)},
                {"material_names_list": ["g", "w"], "material_ids_list": [1, 2],
                 "Variables": dict(pair)}],
            "material_assignation_table": assignation,
        }
    with open(os.path.join(d, "MaterialsDEM.json"), "w") as f:
        json.dump(materials, f, indent=2)

    pp = {"problem_name": problem, "Dimension": 2, "FinalTime": final_time,
          "MaxTimeStep": dt, "OutputTimeStep": final_time,
          "GravityX": 0.0, "GravityY": -9.81, "GravityZ": 0.0,
          "ElementType": "CylinderPartDEMElement2D",
          "TranslationalIntegrationScheme": "Symplectic_Euler",
          "RotationalIntegrationScheme": "Direct_Integration",
          "BoundingBoxOption": True,
          "BoundingBoxMinX": -1.0, "BoundingBoxMinY": -1.0, "BoundingBoxMinZ": -1.0,
          "BoundingBoxMaxX": 1.0, "BoundingBoxMaxY": 1.0, "BoundingBoxMaxZ": 1.0,
          "NeighbourSearchFrequency": 10, "dem_inlet_option": False,
          "do_print_results_option": False, "post_gid_option": False,
          "post_vtk_option": False,
          "solver_settings": {
              "strategy": "sphere_strategy",
              "material_import_settings": {
                  "materials_filename": "MaterialsDEM.json"}}}
    if params:
        pp.update(params)
    with open(os.path.join(d, "ProjectParametersDEM.json"), "w") as f:
        json.dump(pp, f, indent=2)
    return pp


def _run(d, timeout=300):
    """Run the deck in ``d`` in a CHILD process; return (observed, rc, log).

    A child, not this process: Kratos logs from C++ and buffers it, so an
    in-process capture would silently miss every DEM line, and a DEM stage that
    has been Finalize()d cannot be probed from python at all without
    segfaulting. The child writes what it measured to observed.json.
    """
    runner = os.path.join(d, "_runner.py")
    with open(runner, "w") as f:
        f.write(_RUNNER)
    proc = subprocess.run([sys.executable, "_runner.py"], cwd=d, timeout=timeout,
                          capture_output=True, text=True,
                          stdin=subprocess.DEVNULL)
    log = (proc.stdout or "") + (proc.stderr or "")
    path = os.path.join(d, "observed.json")
    observed = {}
    if os.path.exists(path):
        with open(path) as f:
            observed = json.load(f)
    return observed, proc.returncode, log



DEM_FINAL_TIME_DEFAULT = 0.05


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="dem_endtime_") as d:
        _write_deck(d, walls=False, assignation=[["SpheresPart", "g"]])
        pp_path = os.path.join(d, "ProjectParametersDEM.json")
        with open(pp_path) as f:
            pp = json.load(f)
        del pp["FinalTime"]
        pp["problem_data"] = {"problem_name": "case", "parallel_type": "OpenMP",
                              "echo_level": 0, "start_time": 0.0,
                              "end_time": 10.0}
        if MUTATE:
            print("mutation=FinalTime_written_explicitly_as_0.002")
            pp["FinalTime"] = 0.002
        with open(pp_path, "w") as f:
            json.dump(pp, f, indent=2)
        obs, rc, log = _run(d)

    t = obs.get("time")
    checks = [
        ("stopped_at_the_FinalTime_default",
         t is not None and abs(t - DEM_FINAL_TIME_DEFAULT) < 1e-3, True),
        ("did_not_stop_at_problem_data_end_time",
         t is not None and abs(t - 10.0) > 1.0, True),
        ("raises_nothing", obs.get("exception"), None),
        ("completes", "ANALYSIS COMPLETED" in log, True),
        ("exit_zero", rc, 0),
    ]
    return _report(checks, [("stop_time_rounded", None if t is None
                             else round(t, 4))],
                   "dem_finaltime_check", "the FinalTime-wins claim")


def _report(checks, extras, ok_line, what):
    mismatches = 0
    for label, got, must in checks:
        if got != must:
            mismatches += 1
        print("probe[%s]=%s_expected=%s" % (label, got, must))
    for k, v in extras:
        print("%s=%s" % (k, v))
    print("probe_mismatches=%d" % mismatches)
    if mismatches:
        print("FIXTURE_FAILED: %s does not hold on this build" % what,
              file=sys.stderr)
        return 1
    print("%s=ok" % ok_line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
