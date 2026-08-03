"""Kratos Multiphysics participant for the OASiS `couple` driver (DIRICHLET side).

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST. Needs KratosMultiphysics + ConvectionDiffusionApplication
importable in the interpreter named in `command`.

This is the DIRICHLET side: it imports the partner's `values` (interface temperature), applies them as fixed nodal
TEMPERATURE on the interface, exports temperature + its own outward normal flux.

Slab [X0, X1] x [0, H], conductivity K.
  x = X0 : Dirichlet T = T_OUTER (hot side)
  x = X1 : INTERFACE, Dirichlet T = imported partner values (interpolated over y)
  top/bot: insulated (natural)

Outward normal at the interface is +x (s = +1), so per SPEC.md
  q_out = -K * s * dT/dx = -K * (T_if - T_near) / dx   (one-sided, backward)
"""
import json
from pathlib import Path

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication  # noqa: F401

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
PARTNER = "right"
X0, X1 = 0.0, 0.5
H = 1.0
K = 1.0
T_OUTER = 100.0
T_INIT = 50.0            # iteration-1 fallback interface temperature
nx, ny = 32, 32
# ─────────────────────────────────────────────────────────────────────────


def imported_T(y_coords: np.ndarray) -> np.ndarray:
    p = Path("imports.json")
    imp = json.loads(p.read_text() or "{}") if p.is_file() else {}
    if PARTNER not in imp or "values" not in imp[PARTNER]:
        return np.full_like(y_coords, float(T_INIT))
    d = imp[PARTNER]
    src_y = np.asarray(d["coordinates"], float)[:, 1]
    src_v = np.asarray(d["values"], float).ravel()
    o = np.argsort(src_y)
    return np.interp(y_coords, src_y[o], src_v[o])


def solve(T_if_in: np.ndarray):
    model = KM.Model()
    mp = model.CreateModelPart("thermal")
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
    settings = KM.ConvectionDiffusionSettings()
    settings.SetUnknownVariable(KM.TEMPERATURE)
    settings.SetDiffusionVariable(KM.CONDUCTIVITY)
    settings.SetVolumeSourceVariable(KM.HEAT_FLUX)
    settings.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
    mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, settings)
    for v in (KM.TEMPERATURE, KM.CONDUCTIVITY, KM.HEAT_FLUX, KM.FACE_HEAT_FLUX):
        mp.AddNodalSolutionStepVariable(v)
    mp.SetBufferSize(1)

    props = mp.CreateNewProperties(1)
    nid, cnt = {}, 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            mp.CreateNewNode(cnt, X0 + (X1 - X0) * i / nx, H * j / ny, 0.0)
            nid[(i, j)] = cnt
            cnt += 1
    eid = 1
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = nid[(i, j)], nid[(i+1, j)], nid[(i+1, j+1)], nid[(i, j+1)]
            mp.CreateNewElement("LaplacianElement2D3N", eid, [a, b, d], props); eid += 1
            mp.CreateNewElement("LaplacianElement2D3N", eid, [b, c, d], props); eid += 1

    for node in mp.Nodes:
        node.SetSolutionStepValue(KM.CONDUCTIVITY, K)
        node.SetSolutionStepValue(KM.HEAT_FLUX, 0.0)
        node.SetSolutionStepValue(KM.FACE_HEAT_FLUX, 0.0)
    for j in range(ny + 1):                       # outer hot side
        n = mp.Nodes[nid[(0, j)]]
        n.SetSolutionStepValue(KM.TEMPERATURE, T_OUTER)
        n.Fix(KM.TEMPERATURE)
    for j in range(ny + 1):                       # INTERFACE Dirichlet = imported
        n = mp.Nodes[nid[(nx, j)]]
        n.SetSolutionStepValue(KM.TEMPERATURE, float(T_if_in[j]))
        n.Fix(KM.TEMPERATURE)

    KM.VariableUtils().AddDof(KM.TEMPERATURE, mp)
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
    strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder,
                                              False, False, False, False)
    strategy.Initialize()
    strategy.Solve()
    return mp, nid


def main() -> None:
    y_if = np.array([H * j / ny for j in range(ny + 1)])
    T_in = imported_T(y_if)
    mp, nid = solve(T_in)

    dx = (X1 - X0) / nx
    T_if = np.array([mp.Nodes[nid[(nx, j)]].GetSolutionStepValue(KM.TEMPERATURE)
                     for j in range(ny + 1)])
    T_near = np.array([mp.Nodes[nid[(nx - 1, j)]].GetSolutionStepValue(KM.TEMPERATURE)
                       for j in range(ny + 1)])
    q_out = -K * (T_if - T_near) / dx          # s = +1 at this interface
    Path("exports.json").write_text(json.dumps({
        "field_name": "temperature",
        "n_points": len(T_if),
        "coordinates": [[X1, float(y)] for y in y_if],
        "values": [float(v) for v in T_if],
        "normal_fluxes": [float(v) for v in q_out],
    }, indent=2))
    print("kratos DIRICHLET side: T_if(applied)=%.6f q_out(computed)=%.6f"
          % (float(T_if.mean()), float(q_out.mean())))


if __name__ == "__main__":
    main()
