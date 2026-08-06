"""Tier-2: emitting on ONE face of an outflow box does not hold the domain at the
freestream — and the entries were wrong about how you can tell.

  particle_emission:2  emitting on one face is an initial condition, not an
                       inflow boundary; particles leave faster than one face
                       injects.
  hypersonic_flow:1    the same failure stated for the hypersonic row.
  particle_emission:3  a mixture with vstream 0 still emits, but at the thermal
                       flux only, which is several times lower than the drift
                       flux you meant.

WHAT THIS FALSIFIED. Both :2 and hypersonic_flow:1 said the Np column "decreases
monotonically over the run instead of levelling off", and :2's advice was to
"drive the case to steady state and check Np has flattened". Executed on a seeded
box emitting on xlo with the other faces outflow, Np falls from its seeded value
and then FLATTENS, at a bit over half of it — the plateau is stable to a fraction
of a percent over the last 600 steps. So the recommended test PASSES on the
broken deck, which is the worst possible property for a check to have.

The discriminating comparison, which this fixture makes and the entries now
carry, is against a run that emits on every inflow face: that one holds within a
couple of percent of the seeded count. One number, three runs, no stored values —
every assertion is a ratio between decks that differ in exactly one line.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import col, errors, run, skip_if_unavailable, stats_rows  # noqa: E402

DATA = skip_if_unavailable("ar.species", "ar.vss")

DECK = """seed 12345
dimension 2
global gridcut 0.0 comm/sort yes
boundary o o p
create_box 0 1 0 1 -0.5 0.5
create_grid 20 20 1
global nrho 1e20 fnum 1e15
species ar.species Ar
mixture gas Ar vstream {vs} 0 0 temp 273.15
create_particles gas n 0
{emit}collide vss gas ar.vss
timestep 1e-5
stats 200
stats_style step np nexit
run 1000
"""


def probe(vs: int, emit: str) -> dict:
    rc, txt = run(DECK.format(vs=vs, emit=emit), DATA)
    header, rows = stats_rows(txt)
    def column(name):
        return (col(header, rows, name)
                if (header and rows and name in header) else [])
    return {"rc": rc, "errors": errors(txt),
            "warnings": [l for l in txt.splitlines() if l.startswith("WARNING")],
            "np": column("Np"), "nexit": column("Nexit")}


one_face = probe(300, "fix in emit/face gas xlo\n")
all_faces = probe(300, "fix in emit/face gas xlo ylo yhi\n")
no_emit = probe(300, "")
vstream_zero = probe(0, "fix in emit/face gas xlo\n")

for tag, r in (("emit_on_one_face", one_face),
               ("emit_on_every_inflow_face", all_faces),
               ("no_emit_at_all", no_emit),
               ("emit_on_one_face_vstream_zero", vstream_zero)):
    print(f"{tag}_rc={r['rc']} n_errors={len(r['errors'])} "
          f"n_warnings={len(r['warnings'])}")
    print(f"{tag}_np_column={[int(v) for v in r['np']]}")


def flat_over_tail(np_col, tol=0.02):
    """Is the last half of the column flat to within tol of its mean?"""
    tail = np_col[len(np_col) // 2:]
    if len(tail) < 3:
        return False
    mean = sum(tail) / len(tail)
    return mean > 0 and all(abs(v - mean) / mean < tol for v in tail)


def last_step_change(np_col):
    """Relative change over the final stats interval."""
    if len(np_col) < 2 or not np_col[-2]:
        return 1.0
    return abs(np_col[-1] - np_col[-2]) / np_col[-2]


def first_step_change(np_col):
    if len(np_col) < 2 or not np_col[0]:
        return 0.0
    return abs(np_col[1] - np_col[0]) / np_col[0]


seeded = one_face["np"][0] if one_face["np"] else 0.0
one_plateau = one_face["np"][-1] if one_face["np"] else 0.0
all_plateau = all_faces["np"][-1] if all_faces["np"] else 0.0
zero_plateau = vstream_zero["np"][-1] if vstream_zero["np"] else 0.0

held_fraction = one_plateau / seeded if seeded else 0.0
all_faces_fraction = all_plateau / seeded if seeded else 0.0
drift_over_thermal = one_plateau / zero_plateau if zero_plateau else 0.0

print(f"seeded_np={int(seeded)}")
print(f"one_face_plateau_as_fraction_of_seeded={held_fraction:.4g}")
print(f"all_faces_plateau_as_fraction_of_seeded={all_faces_fraction:.4g}")
print(f"drift_over_thermal_plateau_ratio={drift_over_thermal:.4g}")

# The falsification, asserted so it cannot silently revert. The column IS still
# creeping downward at the end — but by a fraction of a percent, against a
# ~28 % drop over the first interval. That is a plateau by any test an agent
# would actually apply, and the entry said there would not be one.
first_change = first_step_change(one_face["np"])
last_change = last_step_change(one_face["np"])
print(f"one_face_first_interval_relative_change={first_change:.4g}")
print(f"one_face_last_interval_relative_change={last_change:.4g}")
print(f"one_face_np_DOES_flatten_so_flatness_is_not_the_test="
      f"{flat_over_tail(one_face['np']) and last_change < 0.01}")
print(f"the_flattening_is_not_just_slow_decay="
      f"{first_change > 20 * last_change}")
print(f"one_face_plateau_is_far_below_the_seeded_density={held_fraction < 0.75}")
print(f"emitting_on_every_inflow_face_holds_the_density="
      f"{all_faces_fraction > 0.9}")
print(f"no_emit_at_all_drains_the_box="
      f"{(no_emit['np'][-1] / seeded if seeded else 1.0) < 0.05}")
print(f"vstream_zero_still_emits={zero_plateau > 0}")
print(f"vstream_zero_plateau_is_several_times_lower={drift_over_thermal > 2.0}")
print(f"nexit_stays_nonzero_on_the_one_face_run="
      f"{all(v > 0 for v in one_face['nexit'][1:])}")

ok = (flat_over_tail(one_face["np"]) and last_change < 0.01
      and first_change > 20 * last_change
      and held_fraction < 0.75
      and all_faces_fraction > 0.9
      and (no_emit["np"][-1] / seeded if seeded else 1.0) < 0.05
      and zero_plateau > 0 and drift_over_thermal > 2.0
      and all(v > 0 for v in one_face["nexit"][1:]))
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
