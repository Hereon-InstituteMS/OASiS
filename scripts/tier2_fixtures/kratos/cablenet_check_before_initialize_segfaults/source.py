"""Tier-2: strategy.Check() before Initialize() SEGFAULTS on SlidingCable.

Pitfall (kratos.cable_net #0). The element's Check dereferences
the constitutive-law pointer, which is only cloned in
Initialize(). A segfault is observable: the child process dies
with SIGSEGV and prints no Kratos exception.

Mutation control: T2_MUTATE=1 calls strat.Initialize() before strat.Check() in the run that is supposed to crash, i.e. it uses the documented order and removes the ordering pathology. That subprocess then exits 0 and survives instead of dying on SIGSEGV.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

os.environ.setdefault("OMP_NUM_THREADS", "1")

# Imported here too, so this fixture cannot report anything at all on a
# machine without Kratos.
import KratosMultiphysics as KM

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=check_called_after_initialize_the_documented_order")

print(f"kratos_version_present={bool(KM.__file__)}")

_MODEL = textwrap.dedent("""
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    import KratosMultiphysics as KM
    import KratosMultiphysics.StructuralMechanicsApplication as SMA
    import KratosMultiphysics.CableNetApplication  # noqa: F401

    model = KM.Model()
    mp = model.CreateModelPart("c")
    for v in (KM.DISPLACEMENT, KM.REACTION, KM.VELOCITY, KM.ACCELERATION):
        mp.AddNodalSolutionStepVariable(v)
    mp.SetBufferSize(2)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
    for i, (x, y, z) in enumerate([(0, 0, 0), (1, 0.2, 0), (2, 0, 0)]):
        mp.CreateNewNode(i + 1, float(x), float(y), float(z))
    prop = mp.CreateNewProperties(1)
    prop.SetValue(KM.YOUNG_MODULUS, 2.1e11)
    prop.SetValue(KM.DENSITY, 7850.0)
    prop.SetValue(SMA.CROSS_AREA, 1e-4)
    prop.SetValue(KM.CONSTITUTIVE_LAW, SMA.TrussConstitutiveLaw())
    mp.CreateNewElement("SlidingCableElement3D3N", 1, [1, 2, 3], prop)
    for n in mp.Nodes:
        n.AddDof(KM.DISPLACEMENT_X, KM.REACTION_X)
        n.AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y)
        n.AddDof(KM.DISPLACEMENT_Z, KM.REACTION_Z)

    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    lin = KM.LinearSolverFactory().Create(
        KM.Parameters('{"solver_type":"skyline_lu_factorization"}'))
    bs = KM.ResidualBasedBlockBuilderAndSolver(lin)
    conv = KM.DisplacementCriteria(1e-6, 1e-9)
    strat = KM.ResidualBasedNewtonRaphsonStrategy(
        mp, scheme, conv, bs, 10, False, False, False)
    print("REACHED_STRATEGY", flush=True)
    ORDER
    print("SURVIVED", flush=True)
""")


def run(order: str):
    code = _MODEL.replace("ORDER", order)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=600)
    return r.returncode, (r.stdout or "")


def main() -> int:
    bad = 0

    rc, out = run("strat.Initialize()\nstrat.Check()" if MUTATE
                  else "strat.Check()")
    reached = "REACHED_STRATEGY" in out
    survived = "SURVIVED" in out
    print(f"check_first_reached_strategy={reached}")
    print(f"check_first_returncode={rc}")
    print(f"check_first_survived={survived}")
    if not reached:
        print("FAIL: the model did not even build; the fixture is not "
              "exercising Check()", file=sys.stderr)
        bad += 1
    if rc != -11:
        print(f"FAIL: expected SIGSEGV (-11) calling Check() before "
              f"Initialize(), got returncode {rc}", file=sys.stderr)
        bad += 1
    if survived:
        print("FAIL: the process survived Check() before Initialize()",
              file=sys.stderr)
        bad += 1

    # Initialize() first is the documented fix and must NOT crash,
    # otherwise the fixture pins a crash with no remedy.
    rc2, out2 = run("strat.Initialize()\nstrat.Check()")
    print(f"initialize_first_returncode={rc2}")
    print(f"initialize_first_survived={'SURVIVED' in out2}")
    if rc2 != 0 or "SURVIVED" not in out2:
        print(f"FAIL: Initialize()-then-Check() did not survive "
              f"(rc={rc2})", file=sys.stderr)
        bad += 1

    print(f"cable_check_order_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
