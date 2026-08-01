"""Coupling participant: slab B solved by REAL Kratos Multiphysics
(ConvectionDiffusionApplication) — must run under /usr/bin/python3 on this
machine (the .venv pip Kratos is GLIBC-broken).

Contract (see src/core/coupling_driver.py): runs in its work_dir, reads
imports.json (partner exports), writes exports.json (InterfaceData dict).

Physics: steady heat conduction on [L1, L1+L2] x [0, H], conductivity k2.
  x = L1     : Neumann, incoming normal flux q  <- imported from the 4C
               slab-A export (normal_fluxes; +x flux leaving A = flux
               entering B), applied as nodal FACE_HEAT_FLUX on
               FluxCondition2D2N line conditions
  x = L1+L2  : Dirichlet T = TR (cold side)
  top/bot    : insulated (natural)

Kratos model built programmatically (ModelPart + LaplacianElement2D3N +
FluxCondition2D2N + ResidualBasedLinearStrategy) in the style of the repo's
Kratos backend templates — the solve itself is genuine Kratos, not scipy.

Export: InterfaceData on the interface nodes x = L1 with
  values        = nodal temperatures at the interface (from the Kratos solve)
  normal_fluxes = outward (-x) normal flux of slab B at the interface,
                  computed one-sided as +k2 * dT/dx (equals -q at convergence,
                  so the driver's interface-balance check applies).
"""
import json
from pathlib import Path

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication  # noqa: F401  (registers elements/conditions)

PARAMS = json.loads(Path("params.json").read_text())
PARTNER = "FourCSlabA"

L1, L2, H = PARAMS["L1"], PARAMS["L2"], PARAMS["H"]
k2, TR = PARAMS["k2"], PARAMS["TR"]
nx, ny = PARAMS["nx"], PARAMS["ny"]


def imported_flux(y_coords: np.ndarray) -> np.ndarray:
    """Incoming interface flux at slab B's left-edge nodes (interp over y)."""
    imp = json.loads(Path("imports.json").read_text() or "{}")
    if PARTNER not in imp or "normal_fluxes" not in imp[PARTNER]:
        return np.zeros_like(y_coords)          # iteration-0 fallback: adiabatic
    d = imp[PARTNER]
    src_y = np.asarray(d["coordinates"], float)[:, 1]
    src_q = np.asarray(d["normal_fluxes"], float)
    order = np.argsort(src_y)
    return np.interp(y_coords, src_y[order], src_q[order])


def solve(q_in: np.ndarray):
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
    nid = {}
    cnt = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            mp.CreateNewNode(cnt, L1 + L2 * i / nx, H * j / ny, 0.0)
            nid[(i, j)] = cnt
            cnt += 1
    eid = 1
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = nid[(i, j)], nid[(i + 1, j)], nid[(i + 1, j + 1)], nid[(i, j + 1)]
            mp.CreateNewElement("LaplacianElement2D3N", eid, [a, b, d], props); eid += 1
            mp.CreateNewElement("LaplacianElement2D3N", eid, [b, c, d], props); eid += 1
    cid = 1
    for j in range(ny):
        mp.CreateNewCondition("FluxCondition2D2N", cid,
                              [nid[(0, j)], nid[(0, j + 1)]], props)
        cid += 1

    for node in mp.Nodes:
        node.SetSolutionStepValue(KM.CONDUCTIVITY, k2)
        node.SetSolutionStepValue(KM.HEAT_FLUX, 0.0)
        node.SetSolutionStepValue(KM.FACE_HEAT_FLUX, 0.0)
    for j in range(ny + 1):                      # Neumann: imported incoming flux
        mp.Nodes[nid[(0, j)]].SetSolutionStepValue(KM.FACE_HEAT_FLUX, float(q_in[j]))
    for j in range(ny + 1):                      # Dirichlet: cold side
        n = mp.Nodes[nid[(nx, j)]]
        n.SetSolutionStepValue(KM.TEMPERATURE, TR)
        n.Fix(KM.TEMPERATURE)

    KM.VariableUtils().AddDof(KM.TEMPERATURE, mp)
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    linear_solver = KM.SkylineLUFactorizationSolver()
    builder = KM.ResidualBasedBlockBuilderAndSolver(linear_solver)
    strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder,
                                              False, False, False, False)
    strategy.Initialize()
    strategy.Solve()
    return mp, nid


def main() -> None:
    y_if = np.array([H * j / ny for j in range(ny + 1)])
    q_in = imported_flux(y_if)
    mp, nid = solve(q_in)

    dx = L2 / nx
    T_if = np.array([mp.Nodes[nid[(0, j)]].GetSolutionStepValue(KM.TEMPERATURE)
                     for j in range(ny + 1)])
    T_near = np.array([mp.Nodes[nid[(1, j)]].GetSolutionStepValue(KM.TEMPERATURE)
                       for j in range(ny + 1)])
    # outward (-x) normal flux of slab B: q_out = -k2 * dT/d(-x) = +k2 * dT/dx
    q_out = k2 * (T_near - T_if) / dx
    export = {
        "field_name": "temperature",
        "n_points": len(T_if),
        "coordinates": [[L1, float(y)] for y in y_if],
        "values": [float(v) for v in T_if],
        "normal_fluxes": [float(v) for v in q_out],
    }
    Path("exports.json").write_text(json.dumps(export, indent=2))
    print("slab B (Kratos): T_if(computed)=%.6f, q_in(applied)=%.6f"
          % (float(T_if.mean()), float(q_in.mean())))


if __name__ == "__main__":
    main()
