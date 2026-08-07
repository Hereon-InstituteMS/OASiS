"""Tier-2 (kratos.dem::8): MaxTimeStep is used verbatim, and the keys that look
like safety nets are dead.

DEM performs no stability check at any value. Ask for a 1 s step on a case whose
stable step is ~2e-5 and it obliges:

    MaxTimeStep 1.0, FinalTime 0.05
        -> 'DEM: Total number of time steps expected in the calculation: 0'
        -> DELTA_TIME == 1.0, TIME == 1.0, exit 0, no warning of any kind

DeltaTimeSafetyFactor is a real key -- it is in the DEM defaults, so it
validates -- and it is stored and never read: adding
DeltaTimeSafetyFactor = 0.001 to that same deck changes nothing, not DELTA_TIME,
not the step count. (DEM_timestep_safety_factor, by contrast, is not a Kratos
key at all; it does not appear anywhere in the distribution and would be
rejected by ValidateAndAssignDefaults, which dem::11 covers.)

MUTATION CONTROL (T2_MUTATE=1): MaxTimeStep is set to a sane 2e-5, so the step
that must be taken verbatim is a reasonable one and the zero-step signature
disappears. Only the deck changes.
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



def main() -> int:
    dt = 2.0e-5 if MUTATE else 1.0
    if MUTATE:
        print("mutation=MaxTimeStep_lowered_to_a_stable_value")

    with tempfile.TemporaryDirectory(prefix="dem_dt_") as d:
        _write_deck(d, params={"MaxTimeStep": dt, "FinalTime": 0.05})
        plain, plain_rc, plain_log = _run(d)

    with tempfile.TemporaryDirectory(prefix="dem_dtsf_") as d:
        _write_deck(d, params={"MaxTimeStep": dt, "FinalTime": 0.05,
                               "DeltaTimeSafetyFactor": 0.001})
        safe, safe_rc, safe_log = _run(d)

    checks = [
        ("delta_time_is_the_value_asked_for", plain.get("delta_time"), 1.0),
        ("zero_steps_expected",
         "Total number of time steps expected in the calculation: 0"
         in plain_log, True),
        ("no_stability_warning",
         any(w in plain_log for w in ("critical time", "Critical time",
                                      "unstable", "too large")), False),
        ("raises_nothing", plain.get("exception"), None),
        ("completes", "ANALYSIS COMPLETED" in plain_log, True),
        ("safety_factor_changes_delta_time_not_at_all",
         safe.get("delta_time"), plain.get("delta_time")),
        ("safety_factor_changes_the_step_count_not_at_all",
         ("Total number of time steps expected in the calculation: 0"
          in safe_log),
         ("Total number of time steps expected in the calculation: 0"
          in plain_log)),
        ("both_exit_zero", (plain_rc, safe_rc), (0, 0)),
    ]
    return _report(checks, [("delta_time", plain.get("delta_time"))],
                   "dem_maxtimestep_verbatim_check",
                   "the verbatim-MaxTimeStep claim")


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
