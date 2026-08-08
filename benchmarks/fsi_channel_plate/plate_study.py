#!/usr/bin/env python3
"""
Structure-only study that isolates the 4C WALL-QUAD4 plate discretisation from the FSI.

Both 4C and scikit-fem are given the IDENTICAL analytic load
    traction on the bottom edge y=0.2:   t = (0, P0*(1-x)),  P0 = 268.739
(a linear pressure, exactly representable in both codes), the same clamped ends and the
same plane-strain material.  Any remaining difference is purely element technology.
"""
import json
import os
import subprocess
import sys

import numpy as np

SD = "/tmp/claude-1001/-home-alexander-4C/b1c8e459-ec06-467a-bad7-474c74f9d0f3/scratchpad/fourc_fsi"
FOURC = "/home/alexander/4C/build/4C"
ENV = dict(os.environ, LD_LIBRARY_PATH="/opt/4C-dependencies/lib")

L, Y0, Y1 = 1.0, 0.2, 0.25
E_S, NU_S = 3.0e6, 0.3
P0 = 268.739


def deck(nx, ny, kinem, eas, path):
    s_id, nodes, nid = {}, [], 0
    for j in range(ny + 1):
        y = Y0 + (Y1 - Y0) * j / ny
        for i in range(nx + 1):
            nid += 1
            s_id[(i, j)] = nid
            nodes.append((nid, L * i / nx, y))
    ele = []
    for j in range(ny):
        for i in range(nx):
            n = (s_id[(i, j)], s_id[(i + 1, j)], s_id[(i + 1, j + 1)], s_id[(i, j + 1)])
            ele.append(f"{len(ele)+1} WALL QUAD4 {n[0]} {n[1]} {n[2]} {n[3]} MAT 1 "
                       f"KINEM {kinem} EAS {eas} THICK 1.0 STRESS_STRAIN plane_strain GP 2 2")
    dl1 = [s_id[(0, j)] for j in range(ny + 1)] + [s_id[(nx, j)] for j in range(ny + 1)]
    dl2 = [s_id[(i, 0)] for i in range(nx + 1)]
    top = "\n".join(f'  - "NODE {n} DLINE 1"' for n in sorted(set(dl1))) + "\n" + \
          "\n".join(f'  - "NODE {n} DLINE 2"' for n in sorted(set(dl2)))
    txt = f"""TITLE:
  - "plate only: clamped-clamped plane-strain strip under linear pressure"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
IO:
  VERBOSITY: "Standard"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: ascii
  COMPRESSION_LEVEL: no_compression
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
STRUCTURAL DYNAMIC:
  INT_STRATEGY: "Standard"
  DYNAMICTYPE: "Statics"
  LINEAR_SOLVER: 1
  TIMESTEP: 1
  NUMSTEP: 1
  MAXTIME: 1
  RESULTSEVERY: 1
  TOLRES: 1e-06
  TOLDISP: 1e-12
  MAXITER: 50
SOLVER 1:
  SOLVER: "UMFPACK"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: {E_S}
      NUE: {NU_S}
      DENS: 1
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{P0}*(1.0-x)"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0, 0]
    FUNCT: [0, 0]
DESIGN LINE NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 2
    ONOFF: [0, 1]
    VAL: [0, 1]
    FUNCT: [0, 1]
    TYPE: "Live"
DLINE-NODE TOPOLOGY:
{top}
NODE COORDS:
""" + "\n".join(f'  - "NODE {n} COORD {x:.16e} {y:.16e} 0.0"' for n, x, y in nodes) + \
        "\nSTRUCTURE ELEMENTS:\n" + "\n".join(f'  - "{e}"' for e in ele) + "\n"
    open(path, "w").write(txt)


def run4c(tag, nx, ny, kinem, eas):
    d = os.path.join(SD, "plate_runs")
    os.makedirs(d, exist_ok=True)
    y = os.path.join(d, f"{tag}.4C.yaml")
    deck(nx, ny, kinem, eas, y)
    r = subprocess.run([FOURC, y, os.path.join(d, tag)], env=ENV,
                       capture_output=True, text=True, cwd=d)
    log = os.path.join(d, f"{tag}.log")
    open(log, "w").write(r.stdout + r.stderr)
    if "processor 0 finished normally" not in r.stdout:
        return None, r.stdout[-1500:] + r.stderr[-1500:]
    import meshio
    f = os.path.join(d, f"{tag}-vtk-files", "structure-00001-0.vtu")
    m = meshio.read(f)
    pts, dsp = m.points, m.point_data["displacement"]
    key = np.round(pts[:, :2], 10)
    _, idx = np.unique(key, axis=0, return_index=True)
    pts, dsp = pts[idx], dsp[idx]
    sel = np.abs(pts[:, 1] - Y0) < 1e-9
    x, dy = pts[sel, 0], dsp[sel, 1]
    o = np.argsort(x)
    return (float(np.abs(dy).max()), float(x[o][int(np.argmax(np.abs(dy[o])))])), None


def skfem_ref(nx, ny):
    import skfem
    from skfem import (MeshQuad, Basis, FacetBasis, ElementQuad2, ElementVector,
                       BilinearForm, LinearForm, asm, condense, solve)
    from skfem.helpers import ddot, sym_grad, eye, trace
    lam = E_S * NU_S / ((1 + NU_S) * (1 - 2 * NU_S))
    mu = E_S / (2 * (1 + NU_S))

    @BilinearForm
    def K(u, v, w):
        T = sym_grad(u)
        return ddot(2. * mu * T + lam * eye(trace(T), T.shape[0]), sym_grad(v))

    @LinearForm
    def f(v, w):
        return P0 * (1.0 - w.x[0]) * v[1]

    m = (MeshQuad.init_tensor(np.linspace(0, L, nx + 1), np.linspace(Y0, Y1, ny + 1))
         .with_boundaries({"clamped": lambda x: (np.abs(x[0]) < 1e-12) | (np.abs(x[0] - L) < 1e-12),
                           "wet": lambda x: np.abs(x[1] - Y0) < 1e-12}))
    b = Basis(m, ElementVector(ElementQuad2()))
    u = solve(*condense(asm(K, b), asm(f, FacetBasis(m, b.elem, facets=m.boundaries["wet"])),
                        D=b.get_dofs("clamped")))
    sel = np.abs(m.p[1] - Y0) < 1e-12
    x, dy = m.p[0, sel], u[b.nodal_dofs[1, sel]]
    o = np.argsort(x)
    return float(np.abs(dy).max()), float(x[o][int(np.argmax(np.abs(dy[o])))])


if __name__ == "__main__":
    print(f"load: t_y(x) = {P0}*(1-x)   plane strain E={E_S} nu={NU_S}\n")
    print("scikit-fem biquadratic reference (locking-free):")
    for nx, ny in [(48, 4), (96, 8), (192, 16)]:
        v, xs = skfem_ref(nx, ny)
        print(f"   Quad2 {nx:3d}x{ny:2d}: max|dy| = {v:.6e} at x={xs:.4f}")
    ref = skfem_ref(192, 16)[0]
    print()
    print("4C WALL QUAD4:")
    rows = []
    for nx, ny, kin, eas in [
            (48, 4, "linear", "none"), (48, 4, "nonlinear", "none"),
            (48, 4, "nonlinear", "full"),
            (48, 8, "linear", "none"), (96, 4, "linear", "none"),
            (96, 8, "linear", "none"), (192, 16, "linear", "none"),
            (96, 8, "nonlinear", "full"), (192, 16, "nonlinear", "full")]:
        tag = f"p_{nx}x{ny}_{kin}_{eas}"
        res, err = run4c(tag, nx, ny, kin, eas)
        if res is None:
            print(f"   {nx:3d}x{ny:2d} KINEM {kin:9s} EAS {eas:4s}: FAILED\n{err}")
            continue
        v, xs = res
        print(f"   {nx:3d}x{ny:2d} KINEM {kin:9s} EAS {eas:4s}: max|dy| = {v:.6e} "
              f"at x={xs:.4f}   ({100*(v/ref-1):+.2f}% vs skfem)")
        rows.append({"nx": nx, "ny": ny, "kinem": kin, "eas": eas,
                     "max_abs_dy": v, "x": xs, "pct_vs_skfem": 100 * (v / ref - 1)})
    json.dump({"skfem_reference": ref, "P0": P0, "rows": rows},
              open(os.path.join(SD, "plate_study.json"), "w"), indent=1)
    print(f"\nskfem reference = {ref:.6e}")
