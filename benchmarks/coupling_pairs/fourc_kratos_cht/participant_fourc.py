"""Coupling participant: slab A solved by the REAL 4C binary (PROBLEMTYPE Thermo).

Contract (see src/core/coupling_driver.py): runs in its work_dir, reads
imports.json (partner exports), writes exports.json (InterfaceData dict).

Physics: steady heat conduction on [0, L1] x [0, H], conductivity k1.
  x = 0   : Dirichlet T = TL (hot side)
  x = L1  : Dirichlet T = T_if  <- imported from the Kratos slab-B export
            (mean of partner interface temperatures; the interface of this
            flat two-slab problem is isothermal, so the mean is exact)
  top/bot : insulated (natural)

Each iteration this wrapper generates a self-contained inline-mesh 4C deck
(THERMO QUAD4 + MAT_Fourier, THERMAL DYNAMIC Statics — the same deck pattern
as src/backends/fourc/inline_mesh.matched_thermo_2d_input), runs the real 4C
binary, and parses the ascii runtime-VTK output.

Export: InterfaceData on the interface nodes x = L1 with
  values        = nodal temperatures at the interface (from the 4C solve)
  normal_fluxes = outward (+x) normal flux q = -k1 * dT/dx at the interface.

Flux extraction: 4C's pure-thermo runtime VTK output has no native heat-flux
field (checked 4C source: THERMAL DYNAMIC/RUNTIME VTK OUTPUT only offers
TEMPERATURE / TEMPERATURE_RATE / CONDUCTIVITY), so the flux is computed by a
one-sided finite difference between the interface node column and the adjacent
column, times k1. On this structured mesh with a (piecewise-)linear steady
solution the one-sided difference is exact.
"""
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import meshio
import numpy as np

PARAMS = json.loads(Path("params.json").read_text())
PARTNER = "KratosSlabB"

L1, H = PARAMS["L1"], PARAMS["H"]
k1, TL = PARAMS["k1"], PARAMS["TL"]
nx, ny = PARAMS["nx"], PARAMS["ny"]


def interface_temperature_from_imports() -> float:
    imp = json.loads(Path("imports.json").read_text() or "{}")
    if PARTNER in imp:
        return float(np.mean(imp[PARTNER]["values"]))
    return float(PARAMS["T_if_initial_guess"])


def write_deck(T_if: float, path: Path) -> None:
    nid = {}
    nodes = []
    cnt = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            nid[(i, j)] = cnt
            nodes.append((cnt, L1 * i / nx, H * j / ny))
            cnt += 1
    y = f'''TITLE:
  - "fourc_kratos_cht slab A: steady conduction, interface Dirichlet from partner"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo"
IO:
  VERBOSITY: "Standard"
IO/RUNTIME VTK OUTPUT:
  OUTPUT_DATA_FORMAT: ascii
THERMAL DYNAMIC:
  DYNAMICTYPE: Statics
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
THERMAL DYNAMIC/RUNTIME VTK OUTPUT:
  OUTPUT_THERMO: true
  TEMPERATURE: true
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Thermal_Solver"
MATERIALS:
  - MAT: 1
    MAT_Fourier:
      CAPA: 1.0
      CONDUCT:
        constant: [{k1}]
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{TL}]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_if:.14g}]
    FUNCT: [0]
DLINE-NODE TOPOLOGY:
'''
    for j in range(ny + 1):
        y += f'  - "NODE {nid[(0, j)]} DLINE 1"\n'
    for j in range(ny + 1):
        y += f'  - "NODE {nid[(nx, j)]} DLINE 2"\n'
    y += 'NODE COORDS:\n'
    for n, xx, yy in nodes:
        y += f'  - "NODE {n} COORD {xx:.14g} {yy:.14g} 0.0"\n'
    y += 'THERMO ELEMENTS:\n'
    eid = 1
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = nid[(i, j)], nid[(i + 1, j)], nid[(i + 1, j + 1)], nid[(i, j + 1)]
            y += f'  - "{eid} THERMO QUAD4 {a} {b} {c} {d} MAT 1"\n'
            eid += 1
    path.write_text(y)


def run_4c() -> None:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (PARAMS["fourc_ld_library_path"] + ":"
                              + env.get("LD_LIBRARY_PATH", "")).rstrip(":")
    r = subprocess.run([PARAMS["fourc_binary"], "slabA.4C.yaml", "out"],
                       capture_output=True, text=True, env=env, timeout=300)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:] + "\n" + r.stderr[-2000:])
        raise RuntimeError(f"4C exited with rc={r.returncode}")


def extract_interface() -> dict:
    vtus = sorted(glob.glob("out-vtk-files/thermo-*-0.vtu"))
    if not vtus:
        raise RuntimeError("no 4C VTU output found under out-vtk-files/")
    m = meshio.read(vtus[-1])                      # last (= converged steady) step
    pts = np.asarray(m.points)
    T = np.asarray(m.point_data["temperature"]).ravel()

    # deduplicate the per-element VTK nodes onto unique (x, y) grid points
    key = np.round(pts[:, :2], 10)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    Tu = np.zeros(len(uniq))
    for k in range(len(uniq)):
        Tu[k] = T[inv == k].mean()

    dx = L1 / nx
    on_if = np.abs(uniq[:, 0] - L1) < 1e-9
    near = np.abs(uniq[:, 0] - (L1 - dx)) < 1e-9
    if not on_if.any() or not near.any():
        raise RuntimeError("interface / near-interface node columns not found in VTU")
    order = np.argsort(uniq[on_if, 1])
    if_coords = uniq[on_if][order]
    if_T = Tu[on_if][order]
    near_order = np.argsort(uniq[near, 1])
    near_T = Tu[near][near_order]
    # one-sided FD, exact for the linear steady profile:  q_x = -k1 * dT/dx
    q = -k1 * (if_T - near_T) / dx
    return {
        "field_name": "temperature",
        "n_points": int(on_if.sum()),
        "coordinates": [[float(x), float(yv)] for x, yv in if_coords],
        "values": [float(v) for v in if_T],
        "normal_fluxes": [float(v) for v in q],   # outward (+x) flux leaving slab A
    }


def main() -> None:
    T_if = interface_temperature_from_imports()
    write_deck(T_if, Path("slabA.4C.yaml"))
    run_4c()
    export = extract_interface()
    Path("exports.json").write_text(json.dumps(export, indent=2))
    print(f"slab A (4C): T_if(applied)={T_if:.6f}, "
          f"q_out={np.mean(export['normal_fluxes']):.6f}")


if __name__ == "__main__":
    main()
