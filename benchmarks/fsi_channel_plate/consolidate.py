#!/usr/bin/env python3
"""Collect all 4C FSI variants into one comparison table + the final reference JSON."""
import json
import os
import sys

SD = "/tmp/claude-1001/-home-alexander-4C/b1c8e459-ec06-467a-bad7-474c74f9d0f3/scratchpad/fourc_fsi"

CASES = [
    ("fourc_reference.json", "fsi_channel_plate", "48x10 / 48x4", "none", "nonlinear",
     "fluidsplit", 0.005, 400),
    ("ref_v_dtchk.json", "v_dtchk", "48x10 / 48x4", "none", "nonlinear",
     "fluidsplit", 0.01, 200),
    ("ref_v_ss.json", "v_ss", "48x10 / 48x4", "none", "nonlinear",
     "structuresplit", 0.005, 400),
    ("ref_v_lin.json", "v_lin", "48x10 / 48x4", "none", "linear",
     "fluidsplit", 0.005, 400),
    ("ref_v_fine.json", "v_fine", "96x20 / 96x8", "none", "nonlinear",
     "fluidsplit", 0.01, 200),
    ("ref_v_eas.json", "v_eas", "48x10 / 48x4", "full(Q1E4)", "nonlinear",
     "fluidsplit", 0.005, 400),
    ("ref_v_easfine.json", "v_easfine", "48x10 / 48x8", "full(Q1E4)", "nonlinear",
     "fluidsplit", 0.005, 400),
    ("ref_v_rigid.json", "v_rigid (E=3e10)", "48x10 / 48x4", "none", "nonlinear",
     "fluidsplit", 0.005, 400),
]

rows = []
print(f"{'case':<20}{'mesh f/s':<15}{'EAS':<12}{'KINEM':<11}{'split':<15}"
      f"{'max|dy|':>13}{'x*':>9}{'dp':>9}{'Fy':>9}{'settle':>10}")
print("-" * 123)
for fn, tag, mesh, eas, kin, split, dt, ns in CASES:
    p = os.path.join(SD, fn)
    if not os.path.exists(p):
        print(f"{tag:<20}{mesh:<15}{eas:<12}{kin:<11}{split:<15}{'--- not finished ---':>13}")
        continue
    d = json.load(open(p))
    last = [s for s in d["settling_history"] if s["rel_change"] is not None][-1]
    print(f"{tag:<20}{mesh:<15}{eas:<12}{kin:<11}{split:<15}"
          f"{d['max_abs_dy']:13.6e}{d['x_at_max_abs_dy']:9.4f}"
          f"{d['dp_inlet_minus_outlet']:9.2f}{d['fsi_interface_net_force_xy'][1]:9.2f}"
          f"{last['rel_change']:10.1e}")
    rows.append({"case": tag, "mesh_fluid_structure": mesh, "EAS": eas, "KINEM": kin,
                 "split": split, "dt": dt, "steps": ns,
                 "max_abs_dy": d["max_abs_dy"], "x_at_max_abs_dy": d["x_at_max_abs_dy"],
                 "dp_inlet_minus_outlet": d["dp_inlet_minus_outlet"],
                 "p_inflow_mean": d["p_inflow_mean"],
                 "fsi_interface_net_Fy": d["fsi_interface_net_force_xy"][1],
                 "last_step_rel_change": last["rel_change"],
                 "json": p})

with open(os.path.join(SD, "variant_study.json"), "w") as f:
    json.dump(rows, f, indent=1)
print(f"\nwrote {SD}/variant_study.json")
