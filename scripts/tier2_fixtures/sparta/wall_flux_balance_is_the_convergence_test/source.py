"""Tier-2: in a driven run the two opposed wall fluxes must balance, and that
balance — not the step count — is the convergence test.

  rarefied_flow:8  wall and boundary fluxes need several flow-through times;
                   the first stats block is not an answer and the run looks
                   perfectly healthy throughout the transient.

A 2d argon slab between a hot diffuse wall (ylo) and a cold one (yhi), started
from a uniform gas at the cold temperature. 'compute boundary all etot' is read
every step and block-averaged here rather than in SPARTA, so the averaging
window is explicit and under this fixture's control.

HOW THE THRESHOLD IS SET, which matters because this is a Monte-Carlo quantity
and a fixed tolerance would be a guess. The noise floor is MEASURED, from the
late-time windows of ONE seed. The assertions are then made against the OTHER
seed, so nothing is tested against a floor derived from itself:

  * seed B's late windows must sit inside a small multiple of seed A's floor —
    cross-seed consistency, which is what "converged" can honestly mean for a
    DSMC tally;
  * both seeds' FIRST window must be far outside it. That is the claim: the
    first block is not an answer.

No imbalance value is pinned, and no wall-clock is read. The run length is fixed
so the comparison is between windows of the same run, not between runs of
different length.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import col, errors, run, skip_if_unavailable, stats_rows  # noqa: E402

DATA = skip_if_unavailable("ar.species", "ar.vss")

DECK = """seed {seed}
dimension 2
boundary p s p
global gridcut 0.0
create_box 0 1e-3 0 1e-3 -0.5 0.5
create_grid 20 20 1
global nrho 1e22 fnum 5e11
species ar.species Ar
mixture gas Ar temp 300.0
create_particles gas n 0
collide vss gas ar.vss
surf_collide hot diffuse 800.0 1.0
surf_collide cold diffuse 300.0 1.0
bound_modify ylo collide hot
bound_modify yhi collide cold
compute b boundary all etot
timestep 1e-8
stats 1
stats_style step np c_b[3][1] c_b[4][1]
run 3000
"""

NWINDOWS = 10


def probe(seed: int) -> dict:
    rc, txt = run(DECK.format(seed=seed), DATA, timeout=900)
    header, rows = stats_rows(txt)
    hot = (col(header, rows, "c_b[3][1]")
           if header and rows and "c_b[3][1]" in header else [])
    cold = (col(header, rows, "c_b[4][1]")
            if header and rows and "c_b[4][1]" in header else [])
    windows = []
    if hot and cold:
        w = len(hot) // NWINDOWS
        for k in range(NWINDOWS):
            h = hot[k * w:(k + 1) * w]
            c = cold[k * w:(k + 1) * w]
            if not h or not c:
                continue
            mh, mc = sum(h) / len(h), sum(c) / len(c)
            denom = abs(mh) + abs(mc)
            windows.append({"hot": mh, "cold": mc,
                            "imbalance": abs(mh + mc) / denom if denom else 1.0})
    return {"rc": rc, "errors": errors(txt),
            "warnings": [l for l in txt.splitlines() if l.startswith("WARNING")],
            "n_rows": len(rows), "windows": windows}


a = probe(12345)
b = probe(54321)

for tag, r in (("seed_A", a), ("seed_B", b)):
    print(f"{tag}_rc={r['rc']} n_errors={len(r['errors'])} "
          f"n_warnings={len(r['warnings'])} n_stats_rows={r['n_rows']}")
    print(f"{tag}_imbalance_by_window="
          f"{[round(w['imbalance'], 4) for w in r['windows']]}")

both_clean = (a["rc"] == 0 and b["rc"] == 0 and not a["errors"]
              and not b["errors"] and not a["warnings"] and not b["warnings"])
have_windows = len(a["windows"]) == NWINDOWS == len(b["windows"])

# Opposed walls: energy in at the hot face, out at the cold face, every window.
opposed_signs = all(w["hot"] * w["cold"] < 0
                    for r in (a, b) for w in r["windows"])

LATE = slice(NWINDOWS // 2, NWINDOWS)
# The floor is MEASURED on seed A and applied to seed B.
floor = max(w["imbalance"] for w in a["windows"][LATE]) if have_windows else 0.0
b_late = [w["imbalance"] for w in b["windows"][LATE]] if have_windows else []
a_first = a["windows"][0]["imbalance"] if have_windows else 0.0
b_first = b["windows"][0]["imbalance"] if have_windows else 0.0

print(f"measured_noise_floor_from_seed_A_late_windows={floor:.4g}")
print(f"seed_B_late_window_imbalances={[round(v, 4) for v in b_late]}")
print(f"seed_A_first_window_imbalance={a_first:.4g}")
print(f"seed_B_first_window_imbalance={b_first:.4g}")

cross_seed_consistent = bool(b_late) and max(b_late) < 3.0 * floor
first_block_is_far_outside = (floor > 0 and a_first > 10.0 * floor
                              and b_first > 10.0 * floor)
imbalance_decays = all(r["windows"][0]["imbalance"]
                       > r["windows"][2]["imbalance"]
                       > r["windows"][-1]["imbalance"] * 2
                       for r in (a, b)) if have_windows else False

print(f"both_runs_exit_zero_with_no_error_and_no_warning={both_clean}")
print(f"both_runs_produced_all_windows={have_windows}")
print(f"the_two_opposed_walls_carry_opposite_signed_flux={opposed_signs}")
print(f"late_windows_of_the_other_seed_sit_inside_the_measured_floor="
      f"{cross_seed_consistent}")
print(f"the_FIRST_window_is_more_than_10x_the_floor={first_block_is_far_outside}")
print(f"the_imbalance_decays_through_the_transient={imbalance_decays}")

ok = (both_clean and have_windows and opposed_signs and cross_seed_consistent
      and first_block_is_far_outside and imbalance_decays)
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
