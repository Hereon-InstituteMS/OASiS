"""Tier-2: which application registers the cable/membrane element names.

CableNetApplication's own Register() exposes a
Sliding/Ring/EmpiricalSpring vocabulary. The simpler
CableElement3D2N / MembraneElement3D[34]N come from
StructuralMechanicsApplication, which CableNet imports
transitively. Two subprocesses, one per import set, are the
only way to see the split from inside one process.

Mutation control: T2_MUTATE=1 runs the 'StructuralMechanics alone' probe with CableNetApplication imported as well, removing the isolation that the whole registry-split claim rests on. The Sliding/Ring/EmpiricalSpring names then register in the supposedly SMA-only probe.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import json
import subprocess
import textwrap

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=sma_only_probe_also_loads_cablenet")


_PROBE = textwrap.dedent("""
    import json
    import KratosMultiphysics as KM
    IMPORTLINE
    m = KM.Model().CreateModelPart('x')
    m.SetBufferSize(1)
    m.ProcessInfo[KM.DOMAIN_SIZE] = 3
    for i in range(4):
        m.CreateNewNode(i + 1, float(i), 0.0, 0.0)
    p = m.CreateNewProperties(1)
    res = {}
    for nm, n in (('CableElement3D2N', 2), ('MembraneElement3D3N', 3),
                  ('MembraneElement3D4N', 4), ('SlidingCableElement3D3N', 3),
                  ('RingElement3D3N', 3), ('EmpiricalSpringElement3D2N', 2)):
        try:
            m.CreateNewElement(nm, abs(hash(nm)) % 9000 + 10,
                               list(range(1, n + 1)), p)
            res[nm] = True
        except Exception:
            res[nm] = False
    print('RESULT' + json.dumps(res))
""")


def probe(importline: str) -> dict:
    code = _PROBE.replace("IMPORTLINE", importline)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=600)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[len("RESULT"):])
    raise RuntimeError(f"probe produced no result (rc={r.returncode})")


def main() -> int:
    bad = 0
    sma = probe("import KratosMultiphysics.CableNetApplication" if MUTATE
                else "import KratosMultiphysics.StructuralMechanicsApplication")
    cn = probe("import KratosMultiphysics.CableNetApplication")
    for name in sorted(sma):
        print(f"sma_only[{name}]={sma[name]}")
    for name in sorted(cn):
        print(f"cablenet[{name}]={cn[name]}")

    # The simpler names come from StructuralMechanics.
    for name in ("CableElement3D2N", "MembraneElement3D3N",
                 "MembraneElement3D4N"):
        if not sma[name]:
            print(f"FAIL: {name} absent with SMA alone", file=sys.stderr)
            bad += 1
    # The Sliding/Ring/EmpiricalSpring vocabulary does NOT.
    for name in ("SlidingCableElement3D3N", "RingElement3D3N",
                 "EmpiricalSpringElement3D2N"):
        if sma[name]:
            print(f"FAIL: {name} was registered by SMA alone, so it is not "
                  f"CableNet's own", file=sys.stderr)
            bad += 1
        if not cn[name]:
            print(f"FAIL: {name} absent even with CableNet loaded",
                  file=sys.stderr)
            bad += 1
    # CableNet pulls SMA transitively, so it has everything.
    for name in sorted(cn):
        if not cn[name]:
            print(f"FAIL: {name} absent with CableNet loaded", file=sys.stderr)
            bad += 1
    print(f"cablenet_registry_split_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
