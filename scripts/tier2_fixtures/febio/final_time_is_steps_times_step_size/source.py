"""Tier-2: FINAL TIME = time_steps x step_size, and the completed-step
count cannot tell the two cases apart.

Verifies febio::hyperelasticity#4. Two decks that differ only in
<step_size>: 10 x 1.0 and 10 x 0.1. Both reach normal termination with
exit 0 and both report `Number of time steps completed` equal to 10 — so
the step count is useless as a check. The last `*Time` in the logfile CSV
reads 10 for the first and 1 for the second.

The fixture reads the final *Time out of the CSV and requires it to equal
time_steps x step_size in both cases, which is the check the pitfall
tells you to perform.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    # MUTATION CONTROL. The pathology this fixture reproduces is the
    # EDIT that turns a correct deck into the broken one, so removing
    # the pathology means not making that edit. Neutralising L.swap /
    # L.drop does exactly that: every deck built below is the correct
    # one, the pitfall is never triggered, the diagnostic must not
    # appear and the verdict token flips to not_reproduced.
    print("mutation=the_deck_edits_that_introduce_the_pitfall_are_"
          "neutralised")
    L.swap = lambda deck, old, new, **kw: deck
    L.drop = lambda deck, fragment: deck


def with_log(deck: str) -> str:
    """Add the standard position + stress logfile rows to a deck.

    Merges into the template's existing <Output> block — appending a
    second one leaves a deck that does not run, and the fixture would
    then compare two empty lists and call them identical.
    """
    return L.add_logfile(deck,
                         ("node_data", "z", "p.csv"),
                         ("element_data", "sx;sy;sz;J", "e.csv"))


def series(run, name="e.csv"):
    """Every logged value of one file, flattened in file order."""
    out = []
    for _t, rows in L.parse_log_csv(run.files.get(name) or ""):
        for nid in sorted(rows):
            out.extend(rows[nid])
    return out


def max_rel_dev(a, b) -> float:
    if not a or len(a) != len(b):
        return float("inf")
    return max(abs(x - y) / max(abs(x), abs(y), 1e-12)
               for x, y in zip(a, b))


def noise_floor(deck: str, timeout=900):
    """Max relative deviation between two IDENTICAL runs of one deck.

    FEBio on this build is not bit-reproducible in general — even a
    direct-solver deck moves in the last few significant digits between
    identical runs. An identity assertion therefore has to be made
    against a measured floor; asserting exact equality is either flaky
    or, worse, an apparent "effect" that is only round-off.
    """
    a = L.run(deck, collect=("e.csv", "p.csv"), timeout=timeout)
    b = L.run(deck, collect=("e.csv", "p.csv"), timeout=timeout)
    return max_rel_dev(series(a), series(b)), a


def final_time(run):
    blocks = L.parse_log_csv(run.files.get("p.csv") or "")
    return blocks[-1][0] if blocks else None


def deck(steps: int, dt: float) -> str:
    base = L.template("hyperelasticity_3d_cube")
    d = L.swap(base, "<time_steps>10</time_steps>",
               f"<time_steps>{steps}</time_steps>")
    d = L.swap(d, "<step_size>0.1</step_size>",
               f"<step_size>{dt}</step_size>")
    return with_log(d)


def main() -> int:
    long_run = L.run(deck(10, 1.0), collect=("p.csv",), timeout=900)
    short_run = L.run(deck(10, 0.1), collect=("p.csv",), timeout=900)
    tl, ts = final_time(long_run), final_time(short_run)
    print(f"steps=10 step_size=1.0: rc={long_run.rc} "
          f"normal={int(long_run.normal_termination)} "
          f"completed={long_run.steps_completed} final_Time={tl}")
    print(f"steps=10 step_size=0.1: rc={short_run.rc} "
          f"normal={int(short_run.normal_termination)} "
          f"completed={short_run.steps_completed} final_Time={ts}")
    same_count = (long_run.steps_completed == short_run.steps_completed == 10)
    print(f"completed_step_count_cannot_distinguish={int(same_count)}")
    times_ok = (tl is not None and ts is not None
                and abs(tl - 10.0) < 1e-9 and abs(ts - 1.0) < 1e-9)
    print(f"final_time_equals_steps_times_step_size={int(times_ok)}")
    good = (same_count and times_ok
            and long_run.rc == 0 and short_run.rc == 0
            and long_run.normal_termination and short_run.normal_termination)
    return L.report(good, "final_time_not_step_count", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
