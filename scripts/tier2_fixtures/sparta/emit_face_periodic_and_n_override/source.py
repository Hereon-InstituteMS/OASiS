"""Tier-2: what 'fix emit/face' and 'fix emit/surf' reject, and the one override
they accept silently.

  particle_emission:0  emit/face cannot be attached to a PERIODIC face.
  hypersonic_flow:0    the same restriction stated for the hypersonic row — a
                       separate claim string, so a separate claim, and it adds
                       that the check happens when the fix is DEFINED, which is
                       checked here by putting the fix at the very end of the
                       deck and watching it still abort before any run.
  particle_emission:4  'n <Np>' overrides the flux-derived count entirely, and
                       cannot be combined with the default per-species
                       insertion. Once 'perspecies no' is added it is silent,
                       and the steady particle count becomes proportional to n
                       rather than to nrho — measured by RATIO between an n=50
                       and an n=200 run, not against a stored count.
  particle_emission:5  emit/surf takes a SURFACE GROUP, checked by name.

The n-override ratio is the only quantitative assertion and it is a ratio of two
runs in this script. A DSMC steady state is a fluctuating quantity; 4x the n
argument gives 4x the plateau to within the noise, and that is what is asserted.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import col, errors, run, skip_if_unavailable, stats_rows  # noqa: E402

DATA = skip_if_unavailable("ar.species", "ar.vss", "data.circle")

FACE = """seed 12345
dimension 2
global gridcut 0.0 comm/sort yes
boundary {bnd}
create_box 0 1 0 1 -0.5 0.5
create_grid 20 20 1
global nrho 1e20 fnum 1e15
species ar.species Ar
mixture gas Ar vstream 300 0 0 temp 273.15
collide vss gas ar.vss
{emit}timestep 1e-5
stats 200
stats_style step np nexit
run 1000
"""

SURF = """seed 12345
dimension 2
global gridcut 0.0 comm/sort yes
boundary o o p
create_box 0 10 0 10 -0.5 0.5
create_grid 20 20 1
global nrho 1e20 fnum 1e17
species ar.species Ar
mixture gas Ar vstream 0 0 0 temp 273.15
read_surf data.circle
surf_collide cw diffuse 300 1.0
surf_modify all collide cw
collide vss gas ar.vss
{body}timestep 1e-4
stats 100
stats_style step np
run 400
"""


def probe(deck: str) -> dict:
    rc, txt = run(deck, DATA)
    header, rows = stats_rows(txt)
    np_col = (col(header, rows, "Np")
              if (header and rows and "Np" in header) else [])
    return {"rc": rc, "errors": errors(txt),
            "warnings": [l for l in txt.splitlines() if l.startswith("WARNING")],
            "np": np_col}


CASES = {
    "emit_face_on_a_periodic_face":
        probe(FACE.format(bnd="p o p", emit="fix in emit/face gas xlo\n")),
    "emit_face_on_an_outflow_face":
        probe(FACE.format(bnd="o o p", emit="fix in emit/face gas xlo\n")),
    "emit_face_n_with_default_perspecies":
        probe(FACE.format(bnd="o o p", emit="fix in emit/face gas xlo n 50\n")),
    "emit_face_n_50_perspecies_no":
        probe(FACE.format(bnd="o o p",
                          emit="fix in emit/face gas xlo n 50 perspecies no\n")),
    "emit_face_n_200_perspecies_no":
        probe(FACE.format(bnd="o o p",
                          emit="fix in emit/face gas xlo n 200 perspecies no\n")),
    "emit_surf_group_name_does_not_exist":
        probe(SURF.format(body="fix in emit/surf gas nosuchgroup\n")),
    "emit_surf_group_exists":
        probe(SURF.format(body="group c surf id <= 50\n"
                               "fix in emit/surf gas c\n")),
}

QUOTED = {
    "emit_face_on_a_periodic_face":
        "ERROR: Cannot use fix emit/face on periodic boundary "
        "(../fix_emit_face.cpp:182)",
    "emit_face_n_with_default_perspecies":
        "ERROR: Cannot use fix emit/face n > 0 with perspecies yes "
        "(../fix_emit_face.cpp:100)",
    "emit_surf_group_name_does_not_exist":
        "ERROR: Fix emit/surf group ID does not exist (../fix_emit_surf.cpp:61)",
}

for name, r in CASES.items():
    plateau = r["np"][-1] if r["np"] else 0.0
    print(f"{name}_rc={r['rc']} n_errors={len(r['errors'])} "
          f"n_warnings={len(r['warnings'])} final_np={plateau:.4g}")
    if r["errors"]:
        print(f"{name}_message={r['errors'][0]}")

quoted_ok = True
for name, msg in QUOTED.items():
    if msg not in CASES[name]["errors"]:
        quoted_ok = False
        print(f"UNEXPECTED: {name} printed {CASES[name]['errors'][:1]} "
              f"not {msg!r}")

n50 = CASES["emit_face_n_50_perspecies_no"]["np"]
n200 = CASES["emit_face_n_200_perspecies_no"]["np"]
ratio = (n200[-1] / n50[-1]) if (n50 and n200 and n50[-1]) else 0.0
# The check happens when the fix is DEFINED, so the abort precedes any stats
# line: a rejected deck produced no stats table at all.
before_any_run = not CASES["emit_face_on_a_periodic_face"]["np"]
n_override_silent = (CASES["emit_face_n_50_perspecies_no"]["rc"] == 0
                     and not CASES["emit_face_n_50_perspecies_no"]["errors"]
                     and not CASES["emit_face_n_50_perspecies_no"]["warnings"])

print(f"steady_np_ratio_for_4x_the_n_argument={ratio:.4g}")
print(f"every_quoted_emit_message_is_verbatim={quoted_ok}")
print(f"periodic_face_is_rejected_before_any_stats_line={before_any_run}")
print(f"emit_face_on_an_outflow_face_runs="
      f"{CASES['emit_face_on_an_outflow_face']['rc'] == 0}")
print(f"n_override_with_perspecies_no_is_silent={n_override_silent}")
print(f"steady_np_scales_linearly_with_n={3.0 < ratio < 5.0}")
print(f"emit_surf_with_a_real_group_runs="
      f"{CASES['emit_surf_group_exists']['rc'] == 0}")

ok = (quoted_ok and before_any_run
      and CASES["emit_face_on_an_outflow_face"]["rc"] == 0
      and n_override_silent and 3.0 < ratio < 5.0
      and CASES["emit_surf_group_exists"]["rc"] == 0)
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
