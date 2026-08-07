#!/usr/bin/env python3
"""
Extract the FSI reference numbers from a 4C run of the channel/elastic-plate case.

Two independent sources for the interface vertical displacement dy(x) at y = 0.2:

  (a) the STRUCTURE field, from 4C's legacy binary output post-processed with
      `post_processor --filter=vtu` (monolithic FSI cannot use runtime VTK output for
      the structure -- 4C aborts with "Runtime output is not available in the old
      structure time integration").  Nodes are identified by their REFERENCE
      coordinates (the post-processor writes the undeformed mesh + a displacement field).

  (b) the FLUID/ALE field, from the runtime VTK output, using `node_gid` to map back to
      the global node ids written by gen_deck.py.  At the interface the ALE displacement
      equals the structure displacement by the FSI kinematic coupling, so (a) and (b)
      must agree to solver tolerance.  Used as a cross-check.

Also reports the inflow-surface mean pressure and the settling history of max|dy|.
"""
import argparse
import glob
import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import meshio

TOL = 1e-9
_trapz = getattr(np, "trapezoid", None) or np.trapz


def pvd_series(pvd):
    """return [(timestep_value, abs_path), ...] sorted by time"""
    root = ET.parse(pvd).getroot()
    base = os.path.dirname(os.path.abspath(pvd))
    out = []
    for ds in root.iter("DataSet"):
        f = os.path.join(base, ds.attrib["file"])
        if f.endswith(".pvtu"):          # single-rank run -> the actual piece
            cand = f[:-5] + "-0.vtu"
            if os.path.exists(cand):
                f = cand
        out.append((float(ds.attrib["timestep"]), f))
    return sorted(out)


def dedup_nodes(points, data, decimals=10):
    """collapse the discontinuous (per-cell) point list onto unique coordinates"""
    key = np.round(points[:, :2], decimals)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return points[idx], {k: v[idx] for k, v in data.items()}


def struct_interface(vtu, y_iface):
    m = meshio.read(vtu)
    pts, pd = dedup_nodes(m.points, m.point_data)
    sel = np.abs(pts[:, 1] - y_iface) < 1e-9
    x = pts[sel, 0]
    d = pd["displacement"][sel]
    o = np.argsort(x)
    return x[o], d[o, 0], d[o, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--prefix", required=True, help="4C output prefix basename")
    ap.add_argument("--pp", required=True, help="post_processor output basename")
    ap.add_argument("--json", required=True)
    a = ap.parse_args()

    meta = json.load(open(a.meta))
    H, L, U, MU = meta["H"], meta["L"], meta["U_MEAN"], meta["MU_F"]

    # ---------------------------------------------------- (a) structure, all steps
    series = pvd_series(os.path.join(a.rundir, f"{a.pp}-structure.pvd"))
    hist = []
    for t, f in series:
        x, dx, dy = struct_interface(f, H)
        hist.append((t, x, dx, dy))
    t_last, x, dx, dy = hist[-1]

    i = int(np.argmax(np.abs(dy)))
    maxdy, x_maxdy = float(dy[i]), float(x[i])

    # settling: relative change of max|dy| over the last steps
    settle = []
    prev = None
    for t, _, _, d in hist:
        m = float(np.max(np.abs(d)))
        rel = None if prev in (None, 0.0) else abs(m - prev) / abs(prev)
        settle.append({"t": t, "max_abs_dy": m, "rel_change": rel})
        prev = m

    # ---------------------------------------------------- (b) fluid/ALE cross-check
    # runtime VTK: node_gid lets us map exactly onto the deck's global node ids.
    # NOTE the runtime-VTK 'pressure' field is written as all-NaN by this 4C build in the
    # monolithic FSI path, so pressure is taken from the post_processor output instead.
    fl = pvd_series(os.path.join(a.rundir, f"{a.prefix}-fluid.pvd"))
    mf = meshio.read(fl[-1][1])
    gid = mf.point_data["node_gid"]           # 0-based
    disp = mf.point_data["displacement"]

    def pick(nodes, arr):
        out = []
        for n in nodes:
            k = np.nonzero(gid == n - 1)[0]
            out.append(arr[k[0]])
        return np.array(out)

    dy_ale = pick(meta["iface_fluid_nodes"], disp)[:, 1]
    ale_mismatch = float(np.max(np.abs(dy_ale - dy)))

    # ---------------------------------------------------- fluid pressure / interface force
    mp = meshio.read(pvd_series(os.path.join(a.rundir, f"{a.pp}-fluid.pvd"))[-1][1])
    fpts, fpd = dedup_nodes(mp.points, mp.point_data)
    fp = fpd["pressure"]
    # the FSI Lagrange multiplier is written on the SLAVE field only: it is on the fluid
    # for iter_monolithicfluidsplit, on the structure for iter_monolithicstructuresplit
    lam = fpd.get("fsilambda")
    if lam is None:
        ms = meshio.read(pvd_series(os.path.join(a.rundir, f"{a.pp}-structure.pvd"))[-1][1])
        spts, spd = dedup_nodes(ms.points, ms.point_data)
        lam = spd.get("fsilambda")
        if lam is not None:
            fpts_lam, lam_src = spts, lam
        else:
            fpts_lam, lam_src = None, None
    else:
        fpts_lam, lam_src = fpts, lam

    def line_mean(mask, coord):
        s = np.argsort(coord[mask])
        c, v = coord[mask][s], fp[mask][s]
        return float(_trapz(v, c) / (c[-1] - c[0])), c, v

    inlet = np.abs(fpts[:, 0] - 0.0) < 1e-9
    outlet = np.abs(fpts[:, 0] - L) < 1e-9
    p_in_mean, y_in, p_in = line_mean(inlet, fpts[:, 1])
    p_out_mean, _, _ = line_mean(outlet, fpts[:, 1])

    # net force transferred across the FSI interface (sum of the interface Lagrange
    # multiplier over the interface nodes) -- compare against the rigid-wall value dp*L/2
    if lam_src is None:
        fsi_force = [float("nan"), float("nan")]
    else:
        ifa = np.abs(fpts_lam[:, 1] - H) < 1e-9
        fsi_force = [float(lam_src[ifa][:, 0].sum()), float(lam_src[ifa][:, 1].sum())]

    dp_poiseuille = 12.0 * MU * U * L / (H * H)

    out = {
        "x": [float(v) for v in x],
        "dy": [float(v) for v in dy],
        "dx": [float(v) for v in dx],
        "deck": meta["deck"],
        "steps": meta["nstep"],
        "dt": meta["dt"],
        "t_end": meta["t_end"],
        # ---- extras
        "max_abs_dy": abs(maxdy),
        "max_dy_signed": maxdy,
        "x_at_max_abs_dy": x_maxdy,
        "max_abs_dx": float(np.max(np.abs(dx))),
        "p_inflow_mean": p_in_mean,
        "p_outflow_mean": p_out_mean,
        "dp_inlet_minus_outlet": p_in_mean - p_out_mean,
        "poiseuille_rigid_wall_dp": dp_poiseuille,
        "poiseuille_rigid_wall_net_vertical_force": dp_poiseuille / 2.0,
        "p_inflow_profile_y": [float(v) for v in y_in],
        "p_inflow_profile_p": [float(v) for v in p_in],
        "fsi_interface_net_force_xy": fsi_force,
        "settling_history": settle,
        "ale_vs_structure_max_dy_mismatch": ale_mismatch,
        "n_interface_nodes": int(len(x)),
        "mesh": {"nx_f": meta["nx_f"], "ny_f": meta["ny_f"],
                 "nx_s": meta["nx_s"], "ny_s": meta["ny_s"]},
        "coupalgo": meta["coupalgo"], "ale_type": meta["ale_type"],
        "kinem": meta["kinem"],
        "material": {"mu_f": MU, "rho_f": meta["RHO_F"], "E_s": meta["E_S"],
                     "nu_s": meta["NU_S"], "rho_s": meta["RHO_S"]},
        "inflow_function": meta["inflow_function"],
    }
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1)

    print(f"interface nodes           : {len(x)}")
    print(f"final time                : {meta['t_end']}  ({meta['nstep']} steps, dt={meta['dt']})")
    print(f"max |dy|                  : {abs(maxdy):.6e}  at x = {x_maxdy:.4f}")
    print(f"max |dx|                  : {np.max(np.abs(dx)):.6e}")
    print(f"ALE-vs-structure mismatch : {ale_mismatch:.3e}")
    print(f"mean inflow  pressure     : {p_in_mean:.4f}")
    print(f"mean outflow pressure     : {p_out_mean:.4f}")
    print(f"  -> dp                   : {p_in_mean - p_out_mean:.4f}"
          f"   (rigid-wall Poiseuille 12*mu*U*L/H^2 = {dp_poiseuille:.1f})")
    print(f"FSI interface net force   : Fx={fsi_force[0]:.4f}  Fy={fsi_force[1]:.4f}"
          f"   (rigid-wall dp*L/2 = {dp_poiseuille/2:.1f})")
    print("settling of max|dy| (last 8 steps):")
    for s in settle[-8:]:
        rc = "     -    " if s["rel_change"] is None else f"{s['rel_change']:.3e}"
        print(f"   t={s['t']:.5f}  max|dy|={s['max_abs_dy']:.10e}  rel.change={rc}")
    print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
