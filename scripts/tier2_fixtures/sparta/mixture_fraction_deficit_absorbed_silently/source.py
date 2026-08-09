"""Tier-2: SPARTA checks mixture fractions only for EXCEEDING 1.0, and the
deficit when they sum to less is absorbed silently — by the LAST species in the
mixture, whether or not you set it.

This fixture falsified half of chemistry:2 as it was written. The entry said the
remainder is 'distributed over the species you left unset', and offered as its
example a deck that means 80/20 but writes 0.1/0.1 — a deck in which NOTHING is
unset. Executed, the 0.1/0.1 deck gives roughly 9 % of the first species and
91 % of the second, so the second absorbed the whole deficit despite carrying an
explicit 0.1.

mixture.cpp:278 init_fraction() is why, and it does two separate things:

    for (i...) if (fflag[i]) f[i] = fuser[i];
               else f[i] = (1.0-sum) / nimplicit;     <- unset species share it
    if (nspecies) c[nspecies-1] = 1.0;                <- and the LAST species
                                                         absorbs the rest

Sampling uses the cumulative array c, whose last entry is clamped to 1.0. So the
'distributed over the unset species' rule is real but only covers the case where
there IS an unset species; with every fraction set, the last-listed species takes
the deficit and its own stated fraction is ignored.

Both cases are executed here so the entry can state the rule that actually holds.
Fractions are asserted as PROPORTIONS with a tolerance, not as counts: which
particle gets which species is drawn from the RNG, so the count moves with the
seed while the proportion does not.

Mutation control: T2_MUTATE=1 gives every deck fractions that sum to exactly
1.0, with the DECLARED tuple travelling with the deck as it must — the 0.1/0.1
deck becomes 0.8/0.2, the deck with an unset NO declares 'NO frac 0.8', and the
over-one 0.9/0.6 deck becomes 0.9/0.1. There is then no deficit for anything to
absorb and nothing exceeds 1.0, so
`with_every_fraction_set_the_last_species_absorbs_the_deficit`,
`with_an_unset_species_present_the_unset_one_absorbs_it` and
`quoted_exceed_one_message_is_verbatim` all go False.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import col, errors, run, skip_if_unavailable, stats_rows  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

DATA = skip_if_unavailable("air.species")

DECK = """seed 12345
dimension 2
boundary rr rr p
create_box 0 1e-4 0 1e-4 -0.5 0.5
create_grid 10 10 1
global nrho 1e23 fnum 1e12
species air.species N2 O2 NO
{mix}mixture gas temp 273.15
create_particles gas n 0
compute c count N2 O2 NO
timestep 1e-9
stats 50
stats_style step np c_c[1] c_c[2] c_c[3]
run 100
"""


def probe(mix: str) -> dict:
    rc, txt = run(DECK.format(mix=mix), DATA)
    header, rows = stats_rows(txt)
    out = {"rc": rc, "errors": errors(txt),
           "warnings": [l for l in txt.splitlines() if l.startswith("WARNING")]}
    if header and rows:
        # Step 0 is what the entry's Signal says to read: composition is fixed
        # at creation, so the first stats line already carries it.
        step0 = rows[0]
        n = col(header, rows, "Np")[0]
        out["np"] = n
        out["frac"] = [step0[header.index(f"c_c[{i}]")] / n if n else 0.0
                       for i in (1, 2, 3)]
    else:
        out["np"], out["frac"] = 0, []
    return out


# DECLARED fractions travel with each deck, because every assertion below is
# "realised versus DECLARED" rather than "realised versus a number I wrote down".
# The first version compared the realised second fraction against a bare 0.8, and
# a mutation that changed the deck to a legitimate 0.1/0.9 SURVIVED it: 0.9 is
# also above 0.8, so the fixture could not tell a silently absorbed deficit from
# a composition the user actually asked for. That is the whole claim, so the
# assertion has to be about the DISCREPANCY.
# Under mutation every deck declares fractions that sum to exactly 1.0 — the
# deficit, which is the pathology, is gone — and the declared tuple travels with
# the deck, so nothing is compared against a number the deck did not ask for.
if MUTATE:
    print("mutation=every_deck_declares_fractions_summing_to_exactly_one")
    BOTH_SET = ("mixture gas N2 frac 0.8\nmixture gas O2 frac 0.2\n",
                (0.8, 0.2, None))
    ONE_UNSET = ("mixture gas N2 frac 0.1\nmixture gas O2 frac 0.1\n"
                 "mixture gas NO frac 0.8\n", (0.1, 0.1, 0.8))
    OVER_ONE = ("mixture gas N2 frac 0.9\nmixture gas O2 frac 0.1\n",
                (0.9, 0.1, None))
else:
    BOTH_SET = ("mixture gas N2 frac 0.1\nmixture gas O2 frac 0.1\n",
                (0.1, 0.1, None))
    ONE_UNSET = ("mixture gas N2 frac 0.1\nmixture gas O2 frac 0.1\n"
                 "mixture gas NO\n", (0.1, 0.1, None))
    OVER_ONE = ("mixture gas N2 frac 0.9\nmixture gas O2 frac 0.6\n",
                (0.9, 0.6, None))
INTENDED = ("mixture gas N2 frac 0.8\nmixture gas O2 frac 0.2\n", (0.8, 0.2, None))

both = probe(BOTH_SET[0])
unset = probe(ONE_UNSET[0])
intended = probe(INTENDED[0])
over = probe(OVER_ONE[0])

for tag, r in (("all_fractions_set_summing_to_0.2", both),
               ("one_species_left_unset", unset),
               ("fractions_as_intended_80_20", intended),
               ("fractions_summing_above_one", over)):
    print(f"{tag}_rc={r['rc']} n_errors={len(r['errors'])} "
          f"n_warnings={len(r['warnings'])}")
    if r["frac"]:
        print(f"{tag}_step0_fractions="
              f"{[round(f, 3) for f in r['frac']]}")

quoted = "ERROR: Mixture gas fractions exceed 1.0 (../mixture.cpp:225)"
over_ok = quoted in over["errors"]
if not over_ok:
    print(f"UNEXPECTED: over-1 case printed {over['errors'][:1]} not {quoted!r}")


def close(x, target, tol=0.05):
    return abs(x - target) < tol


under_is_silent = (both["rc"] == 0 and not both["errors"]
                   and not both["warnings"])

# The claim: the LAST species' realised fraction is not the one it declared —
# it absorbed the deficit. Asserted as a RATIO against what the deck asked for,
# so a deck whose fractions already sum to 1 cannot satisfy it.
declared_both = BOTH_SET[1]
deficit_both = 1.0 - sum(f for f in declared_both if f is not None)
last_overshoot = (both["frac"][1] / declared_both[1]) if declared_both[1] else 0.0
first_held = close(both["frac"][0], declared_both[0])
last_absorbs = (deficit_both > 0.5 and first_held and last_overshoot > 5.0
                and close(both["frac"][1], declared_both[1] + deficit_both, 0.06))

# With an unset species present, the unset one takes the deficit and BOTH set
# fractions hold at what they declared.
declared_unset = ONE_UNSET[1]
deficit_unset = 1.0 - sum(f for f in declared_unset if f is not None)
unset_takes_it = (close(unset["frac"][0], declared_unset[0])
                  and close(unset["frac"][1], declared_unset[1])
                  and close(unset["frac"][2], deficit_unset))

# The control: fractions that already sum to 1 leave nothing to absorb, so every
# species is realised at exactly what it declared.
declared_int = INTENDED[1]
intended_ok = (abs(1.0 - sum(f for f in declared_int if f is not None)) < 1e-9
               and close(intended["frac"][0], declared_int[0])
               and close(intended["frac"][1], declared_int[1]))

print(f"declared_fractions_both_set={declared_both[:2]} "
      f"deficit={deficit_both:.2f}")
print(f"last_species_realised_over_declared_ratio={last_overshoot:.3g}")
print(f"quoted_exceed_one_message_is_verbatim={over_ok}")
print(f"under_one_is_accepted_with_no_error_and_no_warning={under_is_silent}")
print(f"with_every_fraction_set_the_last_species_absorbs_the_deficit="
      f"{last_absorbs}")
print(f"with_an_unset_species_present_the_unset_one_absorbs_it="
      f"{unset_takes_it}")
print(f"fractions_that_sum_to_one_are_realised_as_written={intended_ok}")

if not (over_ok and under_is_silent and last_absorbs and unset_takes_it
        and intended_ok):
    print("FAIL: fixture expectations not met")
    sys.exit(1)
