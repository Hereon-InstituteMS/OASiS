#!/usr/bin/env python3
"""Assemble the final fourc_reference.json: primary deck result + full variant study."""
import json
import os

SD = "/tmp/claude-1001/-home-alexander-4C/b1c8e459-ec06-467a-bad7-474c74f9d0f3/scratchpad/fourc_fsi"

VARIANTS = [
    ("fourc_reference.json", "PRIMARY (deck as specified)", "48x10", "48x4", "none", "nonlinear",
     "fluidsplit", 3.0e6),
    ("ref_v_dtchk.json", "dt check: dt=0.01 x 200 instead of 0.005 x 400", "48x10", "48x4",
     "none", "nonlinear", "fluidsplit", 3.0e6),
    ("ref_v_ss.json", "coupling variant: iter_monolithicstructuresplit", "48x10", "48x4",
     "none", "nonlinear", "structuresplit", 3.0e6),
    ("ref_v_lin.json", "geometrically linear plate", "48x10", "48x4", "none", "linear",
     "fluidsplit", 3.0e6),
    ("ref_v_eas.json", "EAS Q1E4 plate (locking-free)", "48x10", "48x4", "full", "nonlinear",
     "fluidsplit", 3.0e6),
    ("ref_v_easfine.json", "EAS Q1E4, plate refined 2x through thickness", "48x10", "48x8",
     "full", "nonlinear", "fluidsplit", 3.0e6),
    ("ref_v_fine.json", "uniformly refined mesh", "96x20", "96x8", "none", "nonlinear",
     "fluidsplit", 3.0e6),
    ("ref_v_finlin.json", "uniformly refined + geometrically linear plate", "96x20", "96x8",
     "none", "linear", "fluidsplit", 3.0e6),
    ("ref_v_rigid.json", "SANITY: quasi-rigid top wall (E x1000)", "48x10", "48x4", "none",
     "nonlinear", "fluidsplit", 3.0e9),
    ("ref_v_rigidfine.json", "SANITY: quasi-rigid top wall, refined fluid", "96x20", "96x4",
     "none", "nonlinear", "fluidsplit", 3.0e9),
    ("ref_v_rigidy40.json", "SANITY: quasi-rigid top wall, 40 cells across channel", "48x40",
     "48x4", "none", "nonlinear", "fluidsplit", 3.0e9),
]

study = []
for fn, note, mf, ms, eas, kin, split, E in VARIANTS:
    p = os.path.join(SD, fn)
    if not os.path.exists(p):
        study.append({"file": fn, "note": note, "status": "NOT AVAILABLE"})
        continue
    d = json.load(open(p))
    last = [s for s in d["settling_history"] if s["rel_change"] is not None][-1]
    study.append({
        "note": note, "deck": d["deck"],
        "mesh_fluid": mf, "mesh_structure": ms, "EAS": eas, "KINEM": kin,
        "coupalgo": d["coupalgo"], "E_structure": E,
        "dt": d["dt"], "steps": d["steps"], "t_end": d["t_end"],
        "max_abs_dy": d["max_abs_dy"], "x_at_max_abs_dy": d["x_at_max_abs_dy"],
        "p_inflow_mean": d["p_inflow_mean"],
        "dp_inlet_minus_outlet": d["dp_inlet_minus_outlet"],
        "fsi_interface_net_Fy": d["fsi_interface_net_force_xy"][1],
        "last_step_rel_change_of_max_dy": last["rel_change"],
        "ale_vs_structure_max_dy_mismatch": d["ale_vs_structure_max_dy_mismatch"],
        "json": p,
    })

ref = json.load(open(os.path.join(SD, "fourc_reference.json")))
ref["variant_study"] = study

ps = os.path.join(SD, "plate_study.json")
if os.path.exists(ps):
    ref["structure_only_discretisation_study"] = json.load(open(ps))
    ref["structure_only_discretisation_study"]["what"] = (
        "4C WALL QUAD4 vs an independent locking-free scikit-fem biquadratic solve of the "
        "SAME clamped plate under the SAME analytic linear traction t_y = 268.739*(1-x). "
        "Isolates element technology from the FSI coupling.")

ref["notes"] = {
    "geometry": "fluid [0,1]x[0,0.2]; plate [0,1]x[0.2,0.25]; interface y=0.2, node matching",
    "dy_sign": "positive dy = plate bulges upward (channel widens)",
    "coupalgo_choice": (
        "iter_monolithicfluidsplit is REQUIRED here. 4C forbids Dirichlet BCs on the SLAVE "
        "side of the FSI interface. With structuresplit the structure is the slave, so the "
        "clamp at the interface corner x=1 must be released and the plate is then NOT fully "
        "clamped there (measured dx=-1.64e-4 at x=1). With fluidsplit the structure is the "
        "master and stays exactly clamped; only the fluid velocity DBC at the single inlet-top "
        "corner node (0,0.2) is released, where the coupling imposes the same value (zero)."),
    "known_bias": (
        "The primary deck uses plain WALL QUAD4 (EAS none) with 4 elements through the plate "
        "thickness. That element shear-locks in bending and under-predicts max|dy| by ~9%. "
        "See structure_only_discretisation_study and variant_study."),
    "recommended_comparison_value": {
        "for_a_geometrically_LINEAR_partitioned_reference": (
            "use the converged linear 4C value; the primary-deck number is locking-biased"),
        "primary_deck_max_abs_dy": ref["max_abs_dy"],
    },
    "fsi_interface_net_Fy_caveat": (
        "fsi_interface_net_Fy is the nodal sum of 4C's FSI Lagrange multiplier, which lives on "
        "the SLAVE field. For fluidsplit runs (slave = fluid) it is the fluid->structure load "
        "and is directly comparable to the pressure integral. For the structuresplit run the "
        "multiplier lives on the structure and the raw nodal sum is NOT the same quantity -- "
        "ignore that row's Fy."),
    "runtime_vtk_caveats": (
        "monolithic FSI cannot use IO/RUNTIME VTK OUTPUT/STRUCTURE (4C aborts: 'Runtime output "
        "is not available in the old structure time integration'), and the runtime-VTK fluid "
        "'pressure' array is written as all-NaN in this build; structure displacement and fluid "
        "pressure therefore come from post_processor --filter=vtu on the legacy binary output."),
}

with open(os.path.join(SD, "fourc_reference.json"), "w") as f:
    json.dump(ref, f, indent=1)

print(f"{'variant':<52}{'mesh f / s':<16}{'EAS':<6}{'KINEM':<11}"
      f"{'max|dy|':>13}{'dp':>9}{'Fy':>9}")
print("-" * 116)
for s in study:
    if s.get("status"):
        print(f"{s['note']:<52}{'--- not available ---'}")
        continue
    print(f"{s['note']:<52}{s['mesh_fluid']+' / '+s['mesh_structure']:<16}{s['EAS']:<6}"
          f"{s['KINEM']:<11}{s['max_abs_dy']:13.6e}{s['dp_inlet_minus_outlet']:9.2f}"
          f"{s['fsi_interface_net_Fy']:9.2f}")
print(f"\nupdated {SD}/fourc_reference.json")
