"""Tier-2 (kratos.dem::16): the core vtk_output_process cannot be attached to a
DEM SpheresPart, and DEM's own VTK path needs a package Kratos does not install.

Adding the core VtkOutputProcess to output_processes -- which is what a user
does after reading that DEM output goes through the core KM.VtkOutput class --
fails during Initialize, before a single step:

    output_control_type "step"  -> RuntimeError: Error: STEP not found in
                                   process info of SpheresPart.
    output_control_type "time"  -> RuntimeError: Error: TIME not found in
                                   process info of SpheresPart.

Both come out of OutputController::Check() in
kratos/controllers/output_controller.cpp. The cause is ordering:
DEM_analysis_stage.Initialize does its own work and only then calls
AnalysisStage.Initialize, whose Check() runs before DEM has populated that
model part's ProcessInfo. Probed directly, the same ProcessInfo has TIME once
Initialize has returned -- so this is not "TIME does not exist", it is "TIME
does not exist yet when the process is checked".

The DEM-native alternative, post_vtk_option, is not a workaround on a stock
install: KratosMultiphysics.DEMApplication.dem_vtk_output opens with
`from pyevtk import hl`, and pyevtk is neither a Kratos dependency nor shipped
with it. The third probe states that as the environment-independent biconditional
it is -- dem_vtk_output imports if and only if pyevtk does -- so the fixture is
correct on a machine that happens to have pyevtk as well as on one that does not.

What works instead is post_gid_option, which is what the OASiS DEM generator now
emits.

MUTATION CONTROL (T2_MUTATE=1): the vtk_output block is removed from both decks
that must fail, so they run cleanly. Only the decks change.
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



def _vtk_block(control, interval):
    return {"vtk_output": [{"python_module": "vtk_output_process",
                            "kratos_module": "KratosMultiphysics",
                            "process_name": "VtkOutputProcess",
                            "Parameters": {"model_part_name": "SpheresPart",
                                           "output_control_type": control,
                                           "output_interval": interval,
                                           "file_format": "ascii"}}]}


def main() -> int:
    if MUTATE:
        print("mutation=vtk_output_block_removed_from_both_decks")

    results = {}
    for label, control, interval in (("step", "step", 50),
                                     ("time", "time", 1.0e-3)):
        with tempfile.TemporaryDirectory(prefix="dem_vtk_") as d:
            params = {} if MUTATE else {
                "output_processes": _vtk_block(control, interval)}
            _write_deck(d, params=params)
            obs, rc, log = _run(d)
            results[label] = obs

    step_msg = results["step"].get("message") or ""
    time_msg = results["time"].get("message") or ""

    # Environment-independent: dem_vtk_output is importable exactly when pyevtk
    # is, because it imports it at module scope.
    import importlib.util
    have_pyevtk = importlib.util.find_spec("pyevtk") is not None
    try:
        importlib.import_module(
            "KratosMultiphysics.DEMApplication.dem_vtk_output")
        dem_vtk_imports = True
    except ModuleNotFoundError:
        dem_vtk_imports = False

    # The failure UNDERNEATH the ProcessInfo one: even handed a DEM model part
    # directly, the core writer has no cell type for a DEM particle. Driven
    # here without any analysis stage, so nothing shields it.
    #
    # This half was briefly deleted from the catalog on 2026-08-07 as a
    # suspected fabrication, because the message cannot be grepped: it lives in
    # libKratosCore.so as two adjacent literals with a runtime slot between
    # them, so a search for the whole sentence returns nothing whichever way
    # you space it. It is real, and this probe prints it.
    import KratosMultiphysics as _KM

    def _vtk_geometry_verdict(element_name):
        model = _KM.Model()
        part = model.CreateModelPart("Probe")
        part.AddNodalSolutionStepVariable(_KM.DISPLACEMENT)
        part.CreateNewNode(1, 0.0, 0.0, 0.0)
        props = part.CreateNewProperties(1)
        name = element_name if not MUTATE else "Element2D3N"
        if MUTATE:
            part.CreateNewNode(2, 1.0, 0.0, 0.0)
            part.CreateNewNode(3, 0.0, 1.0, 0.0)
            nodes = [1, 2, 3]
        else:
            nodes = [1]
        part.CreateNewElement(name, 1, nodes, props)
        settings = _KM.Parameters(
            '{"model_part_name": "Probe", "file_format": "ascii",'
            ' "output_control_type": "step", "output_interval": 1,'
            ' "output_sub_model_parts": false, "output_path": "vtk_probe"}')
        try:
            _KM.VtkOutput(part, settings).PrintOutput()
            return ""
        except Exception as exc:                  # noqa: BLE001 - classifying
            return str(exc).replace("\n", " ")

    geom_2d = _vtk_geometry_verdict("CylinderParticle2D")
    geom_3d = _vtk_geometry_verdict("SphericParticle3D")
    NO_WRITER = ("Modelpart contains elements or conditions with geometries "
                 "for which no VTK-output is implemented!")

    checks = [
        ("dem_2d_particle_has_no_vtk_cell_type", NO_WRITER in geom_2d, True),
        ("dem_3d_particle_has_no_vtk_cell_type", NO_WRITER in geom_3d, True),
        ("step_control_raises", results["step"].get("exception"), "RuntimeError"),
        ("step_control_names_STEP_on_spherespart",
         "STEP not found in process info of SpheresPart" in step_msg, True),
        ("time_control_raises", results["time"].get("exception"), "RuntimeError"),
        ("time_control_names_TIME_on_spherespart",
         "TIME not found in process info of SpheresPart" in time_msg, True),
        ("neither_reached_finalize",
         (results["step"].get("reached_finalize"),
          results["time"].get("reached_finalize")), (False, False)),
        ("dem_vtk_output_imports_iff_pyevtk_does",
         dem_vtk_imports == have_pyevtk, True),
    ]
    return _report(checks, [("pyevtk_present", have_pyevtk)],
                   "dem_vtk_output_check", "the DEM VTK-output claim")


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
