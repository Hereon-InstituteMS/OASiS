"""Tier-2: `react` never validates the file TYPE, only its line count parity.

ReactBird reads non-blank, non-comment lines in PAIRS (formula, then
coefficients). A complete pair of junk aborts loudly; a file whose data lines
run out mid-pair ends silently with the dangling line discarded and ZERO
reactions loaded. That is why a one-line file such as ar.vss is accepted in
silence while an eleven-line file such as air.vss is rejected: the
discriminator is parity, not purpose.

The only place SPARTA states how many reactions it parsed is the line
'  style <style> #-of-reactions <N>' under 'Gas reaction tallies:', printed
after a run. It counts entries PARSED, not entries ACTIVE: reactions naming
undeclared species are silently deactivated without changing it.

Mutation control: T2_MUTATE=1 hands the odd-line case the REAL reaction file
(air.tce) instead of ar.vss, i.e. it removes the wrong-file-type mistake and
nothing else. That run then parses 45 reactions instead of zero, so
`odd_line_file_loads_silently_with_zero_reactions` goes False. It is the only
check here that any such mutation can move. `qk_style_on_a_tce_file_is_accepted`
and `undeclared_species_do_not_change_the_parsed_count` assert that SPARTA fails
to NOTICE something, so removing the mistake leaves them true by construction;
`real_reaction_file_parses` and `tally_block_only_appears_with_a_react_command`
are healthy-case controls; and `even_line_junk_file_aborts_loudly` would need
its own mutation, on a different deck.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import errors, run, skip_if_unavailable  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

DATA = skip_if_unavailable("air.species", "air.vss", "air.tce", "ar.vss")

BASE = """seed 12345
dimension 3
boundary rr rr rr
create_box 0 1e-4 0 1e-4 0 1e-4
create_grid 5 5 5
species air.species {species}
mixture air {species} vstream 0 0 0 temp 20000
global nrho 7.07043e22 fnum 7.07043e7
collide vss air air.vss
{react}create_particles air n 0
timestep 1e-9
stats 20
stats_style step np ncoll nreact
run 40
"""


def probe(react_line: str, species: str = "N O N2 O2 NO", extra=None):
    data = dict(DATA)
    if extra:
        data.update(extra)
    rc, txt = run(BASE.format(react=react_line, species=species), data)
    nreact = None
    for line in txt.splitlines():
        if "#-of-reactions" in line:
            nreact = int(line.split()[-1])
    tally_block = "Gas reaction tallies:" in txt
    return rc, nreact, tally_block, errors(txt)


# junk files with an ODD and an EVEN number of data lines
tmp = Path(tempfile.mkdtemp(prefix="spa_react_"))
odd = tmp / "odd.txt"
odd.write_text("# comment\n\nthis is not a reaction\n")
even = tmp / "even.txt"
even.write_text("# comment\n\nthis is not a reaction\nnor is this\n")

if MUTATE:
    print("mutation=odd_line_case_given_the_real_air_tce_reaction_file")

# Under mutation the pathology is REMOVED: 'react' is pointed at the real tce
# reaction file instead of a vss file that happens to have an odd line count.
rc_ar, n_ar, blk_ar, e_ar = probe("react tce air.tce\n" if MUTATE
                                  else "react tce ar.vss\n")
rc_air, n_air, blk_air, e_air = probe("react tce air.vss\n")
rc_odd, n_odd, blk_odd, e_odd = probe("react tce odd.txt\n",
                                      extra={"odd.txt": odd})
rc_even, n_even, blk_even, e_even = probe("react tce even.txt\n",
                                          extra={"even.txt": even})
rc_ok, n_ok, blk_ok, _ = probe("react tce air.tce\n")
rc_qk, n_qk, blk_qk, e_qk = probe("react qk air.tce\n")
rc_few, n_few, blk_few, _ = probe("react tce air.tce\n", species="N2 O2")
rc_none, n_none, blk_none, _ = probe("")

print(f"ar_vss_one_data_line: rc={rc_ar} n_reactions={n_ar} err={e_ar[:1]}")
print(f"air_vss_many_lines:   rc={rc_air} err={e_air[:1]}")
print(f"junk_odd_lines:       rc={rc_odd} n_reactions={n_odd} err={e_odd[:1]}")
print(f"junk_even_lines:      rc={rc_even} err={e_even[:1]}")
print(f"real_tce_file:        rc={rc_ok} n_reactions={n_ok}")
print(f"qk_style_on_tce_file: rc={rc_qk} n_reactions={n_qk} err={e_qk[:1]}")
print(f"few_species_declared: rc={rc_few} n_reactions={n_few}")
print(f"no_react_command:     rc={rc_none} tally_block={blk_none}")

checks = {
    "odd_line_file_loads_silently_with_zero_reactions":
        rc_ar == 0 and n_ar == 0 and rc_odd == 0 and n_odd == 0,
    "even_line_junk_file_aborts_loudly":
        rc_air != 0 and rc_even != 0
        and any("Invalid reaction formula in file" in e for e in e_air + e_even),
    "real_reaction_file_parses": rc_ok == 0 and (n_ok or 0) > 0,
    "qk_style_on_a_tce_file_is_accepted": rc_qk == 0 and (n_qk or 0) > 0,
    "undeclared_species_do_not_change_the_parsed_count":
        rc_few == 0 and n_few == n_ok,
    "tally_block_only_appears_with_a_react_command":
        blk_ok and not blk_none,
}
for k, v in checks.items():
    print(f"{k}={v}")
if not all(checks.values()):
    print("FAIL: fixture expectations not met")
    sys.exit(1)
