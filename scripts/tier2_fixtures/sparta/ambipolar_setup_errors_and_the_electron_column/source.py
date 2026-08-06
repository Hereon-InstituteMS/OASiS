"""Tier-2: what 'fix ambipolar' rejects, and what the electron column actually
means — which is the OPPOSITE of what the catalog said.

  ambipolar_plasma:0  every species named on the fix line is checked by name
                      and by CHARGE, and the electron argument and the ion
                      arguments are checked from different source lines.
  ambipolar_plasma:1  'mixture <old> copy <new>' — the copy TARGET is the
                      second argument; the other order aborts. And leaving the
                      electron in the create_particles mixture shows up as a
                      nonzero electron count at step 0.
  ambipolar_plasma:2  ionisation is a high-threshold channel: the ion columns
                      are identically zero for the first stats blocks of a
                      correctly configured run.
  ambipolar_plasma:3  'fix ambipolar' without 'collide_modify ambipolar yes'
                      is silent — and the reverse is NOT.

WHAT THIS FALSIFIED, and it is the whole point of the fixture.

The entry for :3 said the two commands "are accepted independently and NEITHER
warns about the other's absence", and told the reader to detect a missing
collide_modify line like this: "once ions appear, the electron count must track
the total ion count. If ions accumulate and the electron column stays at 0, the
collide_modify line is missing."

Both halves are wrong, and the second is wrong in the direction that matters.

  * 'collide_modify ambipolar yes' WITHOUT 'fix ambipolar' does not run: it
    aborts with 'Collision ambipolar without fix ambipolar (../collide.cpp:293)'.
    Only the other direction is silent.

  * Run upstream's own examples/ambi/in.ambi unmodified — collide_modify
    present, the reference configuration — and the electron column is
    IDENTICALLY ZERO on every stats line while the ion columns climb into the
    hundreds. Delete the collide_modify line and the electron column becomes
    nonzero and grows monotonically. That is the ambipolar approximation
    working as designed: the electron is carried as a per-particle attribute of
    its ion, not as a particle, so 'compute count species' never sees one.

So the published diagnostic reports "collide_modify is missing" exactly when it
is PRESENT, and reports healthy exactly when it is missing. This script runs
both variants of the upstream deck and asserts the inversion, and the entry was
rewritten in the same commit.

No number from these runs is pinned. Every assertion is a comparison between two
decks that differ in one line, or a verbatim error string.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import (  # noqa: E402
    col, errors, find_data, require, run, skip_if_example_unavailable,
    stats_rows,
)

# The upstream deck is reproduced from the directory it belongs to: air.surf
# and data.circle both exist elsewhere in the distribution meaning different
# things (see _spahelp.find_example).
AMBI = skip_if_example_unavailable(
    "ambi", "in.ambi", "air.species", "air.vss", "air.tce", "air.surf",
    "data.circle")

SPECIES_ONLY = require("air.species", "air.vss")
if SPECIES_ONLY is None:                                    # pragma: no cover
    print("SKIP: SPARTA distribution data files not found (set SPARTA_ROOT)")
    sys.exit(0)

# ---------------------------------------------------------------- setup errors
HEAD = """seed 12345
dimension 2
boundary p p p
global gridcut 0.0
global nrho 2.6404e20 fnum 2.6404e17
create_box 0 1 0 1 -0.5 0.5
create_grid 10 10 1
species air.species N2 O2 N O NO N2+ O2+ N+ O+ NO+ e
mixture hot N2 O2 N O NO N2+ O2+ N+ O+ NO+ e
mixture hot temp 180000.0
mixture hot copy noelec
mixture noelec delete e
compute 10 count species
stats_style step np c_10[11]
stats 1
"""

SETUP = {
    # ambipolar_plasma:0 — name checks, from two different source lines
    "electron_argument_is_a_typo":
        "fix ambi ambipolar elec N+ N2+\n",
    "ion_argument_is_a_typo":
        "fix ambi ambipolar e Nplus N2+\n",
    # ambipolar_plasma:0 — charge checks
    "electron_argument_is_a_neutral":
        "fix ambi ambipolar N2 N+\n",
    "ion_argument_is_a_neutral":
        "fix ambi ambipolar e N2\n",
    # ambipolar_plasma:1 — the copy argument order
    "mixture_copy_with_the_arguments_swapped":
        "mixture noelec2 copy hot\n",
    # ambipolar_plasma:3 — the direction that is NOT silent
    "collide_modify_ambipolar_without_the_fix":
        ("collide vss species air.vss\n"
         "collide_modify ambipolar yes\n"
         "create_particles noelec n 0\n"
         "timestep 1e-8\nrun 2\n"),
    # ambipolar_plasma:3 — the direction that IS silent
    "fix_ambipolar_without_collide_modify":
        ("fix ambi ambipolar e N+ N2+ NO+ O+ O2+\n"
         "collide vss species air.vss\n"
         "create_particles noelec n 0\n"
         "timestep 1e-8\nrun 2\n"),
    # ambipolar_plasma:1 — electron left in the create_particles mixture
    "electron_left_in_the_create_particles_mixture":
        ("fix ambi ambipolar e N+ N2+ NO+ O+ O2+\n"
         "collide vss species air.vss\n"
         "collide_modify ambipolar yes\n"
         "create_particles hot n 0\n"
         "timestep 1e-8\nrun 2\n"),
    "electron_deleted_from_the_create_particles_mixture":
        ("fix ambi ambipolar e N+ N2+ NO+ O+ O2+\n"
         "collide vss species air.vss\n"
         "collide_modify ambipolar yes\n"
         "create_particles noelec n 0\n"
         "timestep 1e-8\nrun 2\n"),
}


def probe(body: str) -> dict:
    rc, txt = run(HEAD + body, SPECIES_ONLY)
    header, rows = stats_rows(txt)
    e_col = (col(header, rows, "c_10[11]")
             if (header and rows and "c_10[11]" in header) else [])
    return {"rc": rc, "errors": errors(txt), "electrons": e_col,
            "warnings": [l for l in txt.splitlines() if l.startswith("WARNING")]}


CASES = {name: probe(body) for name, body in SETUP.items()}

QUOTED = {
    "electron_argument_is_a_typo":
        "ERROR: Fix ambipolar species does not exist (../fix_ambipolar.cpp:44)",
    "ion_argument_is_a_typo":
        "ERROR: Fix ambipolar species does not exist (../fix_ambipolar.cpp:57)",
    "electron_argument_is_a_neutral":
        "ERROR: Fix ambipolar electron species has charge >= 0.0 "
        "(../fix_ambipolar.cpp:46)",
    "ion_argument_is_a_neutral":
        "ERROR: Fix ambipolar ion species has charge <= 0.0 "
        "(../fix_ambipolar.cpp:59)",
    "mixture_copy_with_the_arguments_swapped":
        "ERROR: New mixture copy mixture already exists (../mixture.cpp:439)",
    "collide_modify_ambipolar_without_the_fix":
        "ERROR: Collision ambipolar without fix ambipolar (../collide.cpp:293)",
}

for name, r in CASES.items():
    print(f"{name}_rc={r['rc']} n_errors={len(r['errors'])} "
          f"n_warnings={len(r['warnings'])}")
    if r["errors"]:
        print(f"{name}_message={r['errors'][0]}")

quoted_ok = True
for name, msg in QUOTED.items():
    if msg not in CASES[name]["errors"]:
        quoted_ok = False
        print(f"UNEXPECTED: {name} printed {CASES[name]['errors'][:1]} "
              f"not {msg!r}")

# The two name checks share a message and differ ONLY in the source line, which
# is exactly why a guard built on the line number misses the likelier mistake.
def first_error(name: str) -> str:
    errs = CASES[name]["errors"]
    return errs[0] if errs else ""


same_text_two_lines = (
    "Fix ambipolar species does not exist"
    in first_error("electron_argument_is_a_typo")
    and "Fix ambipolar species does not exist"
    in first_error("ion_argument_is_a_typo")
    and first_error("electron_argument_is_a_typo")
    != first_error("ion_argument_is_a_typo"))

silent = CASES["fix_ambipolar_without_collide_modify"]
loud = CASES["collide_modify_ambipolar_without_the_fix"]
with_e = CASES["electron_left_in_the_create_particles_mixture"]
without_e = CASES["electron_deleted_from_the_create_particles_mixture"]

print(f"the_two_name_checks_share_a_message_and_differ_only_in_the_line="
      f"{same_text_two_lines}")
print(f"fix_without_collide_modify_is_silent="
      f"{silent['rc'] == 0 and not silent['errors'] and not silent['warnings']}")
print(f"collide_modify_without_fix_ABORTS={loud['rc'] != 0}")
print(f"electron_left_in_mixture_is_nonzero_at_step_zero="
      f"{bool(with_e['electrons']) and with_e['electrons'][0] > 0}")
print(f"electron_deleted_is_zero_at_step_zero="
      f"{bool(without_e['electrons']) and without_e['electrons'][0] == 0}")

# ------------------------------------------------- the upstream deck, two ways
DECK = AMBI["in.ambi"].read_text()
if "collide_modify      ambipolar yes" not in DECK:          # pragma: no cover
    print("UNEXPECTED: examples/ambi/in.ambi no longer carries the "
          "'collide_modify ambipolar yes' line this fixture removes")
    sys.exit(1)
NO_AMBI = DECK.replace("collide_modify      ambipolar yes\n", "")

AMBI_DATA = {k: v for k, v in AMBI.items() if k != "in.ambi"}


def upstream(deck: str) -> dict:
    rc, txt = run(deck, AMBI_DATA, timeout=900)
    header, rows = stats_rows(txt)

    def column(name):
        return (col(header, rows, name)
                if (header and rows and name in header) else [])
    ions = [column(f"c_10[{i}]") for i in (6, 7, 8, 9, 10)]
    return {"rc": rc, "errors": errors(txt),
            "electrons": column("c_10[11]"),
            "ion_total": [sum(c[k] for c in ions if k < len(c))
                          for k in range(len(ions[0]))] if ions[0] else [],
            "ncoll": column("Ncoll")}


ref = upstream(DECK)
nomod = upstream(NO_AMBI)

print(f"upstream_reference_rc={ref['rc']} n_errors={len(ref['errors'])}")
print(f"upstream_without_collide_modify_rc={nomod['rc']} "
      f"n_errors={len(nomod['errors'])}")
print(f"reference_electron_column={[int(v) for v in ref['electrons']]}")
print(f"no_collide_modify_electron_column={[int(v) for v in nomod['electrons']]}")
print(f"reference_ion_total_column={[int(v) for v in ref['ion_total']]}")

# ambipolar_plasma:3, inverted. The reference configuration reports NO
# electrons; the broken one reports plenty.
ref_e_all_zero = bool(ref["electrons"]) and all(v == 0 for v in ref["electrons"])
nomod_e_grows = (len(nomod["electrons"]) > 2
                 and nomod["electrons"][-1] > 0
                 and nomod["electrons"][-1] > nomod["electrons"][len(nomod["electrons"]) // 2] > 0)
ref_ions_climb = bool(ref["ion_total"]) and ref["ion_total"][-1] > 0

# ambipolar_plasma:2 — the ion columns are zero for the first stats blocks of a
# correct run, so a short run looks broken. Asserted as "the first block is zero
# and the last is not", which is a shape, not a step number.
first_ion_block = ref["ion_total"][0] if ref["ion_total"] else -1
zero_ion_blocks = 0
for v in ref["ion_total"]:
    if v:
        break
    zero_ion_blocks += 1
print(f"n_leading_stats_blocks_with_zero_ions={zero_ion_blocks}")

print(f"reference_deck_reports_ZERO_electrons_throughout={ref_e_all_zero}")
print(f"deleting_collide_modify_MAKES_the_electron_column_nonzero="
      f"{nomod_e_grows}")
print(f"the_published_diagnostic_is_inverted="
      f"{ref_e_all_zero and nomod_e_grows and ref_ions_climb}")
print(f"reference_ions_do_accumulate={ref_ions_climb}")
print(f"ions_are_zero_for_the_first_stats_blocks="
      f"{first_ion_block == 0 and zero_ion_blocks >= 2}")

ok = (quoted_ok and same_text_two_lines
      and silent["rc"] == 0 and not silent["errors"] and not silent["warnings"]
      and loud["rc"] != 0
      and with_e["electrons"] and with_e["electrons"][0] > 0
      and without_e["electrons"] and without_e["electrons"][0] == 0
      and ref["rc"] == 0 and nomod["rc"] == 0
      and ref_e_all_zero and nomod_e_grows and ref_ions_climb
      and first_ion_block == 0 and zero_ion_blocks >= 2)
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
