"""Tier-2: <density> is optional, defaults to 1.0, and under DYNAMIC that
is a clean run with the wrong physics.

Verifies febio::hyperelasticity#2. Four runs of the same neo-Hookean cube
under <analysis>DYNAMIC</analysis>:

  * <density>1.0</density> — the reference,
  * the tag omitted entirely — must agree with the reference to within
    the solver's own run-to-run noise, which the fixture MEASURES by
    running the reference deck twice; that is how the default value is
    established rather than assumed,
  * <density>1000.0</density> — must differ,
  * <density>0.0</density> — accepted, and must reproduce the STATIC
    answer to the same tolerance, because zero density removes the
    inertia term.

All four terminate normally with exit 0 and nothing in the log to
distinguish them. The identity assertions are the point: an "it differs"
test would pass on noise, and a "it runs" test would pass on anything.
Note this deck is NOT bit-reproducible — identical runs move in the last
couple of significant digits — so identity is asserted against the
measured floor, never against equality.
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


NEO = ('    <material id="1" name="Material1" type="neo-Hookean">\n'
       "      <density>1.0</density><E>1000.0</E><v>0.3</v>\n"
       "    </material>")


def variant(material_body: str, analysis: str) -> str:
    base = L.template("hyperelasticity_3d_cube")
    deck = L.swap(base, NEO,
                  '    <material id="1" name="Material1" type="neo-Hookean">\n'
                  + material_body + "    </material>")
    return with_log(L.swap(deck, "<analysis>STATIC</analysis>",
                           f"<analysis>{analysis}</analysis>"))


def main() -> int:
    runs = {}
    for tag, body, analysis in (
            ("dynamic_rho1", "      <density>1.0</density>"
                             "<E>1000.0</E><v>0.3</v>\n", "DYNAMIC"),
            ("dynamic_omitted", "      <E>1000.0</E><v>0.3</v>\n", "DYNAMIC"),
            ("dynamic_rho1000", "      <density>1000.0</density>"
                                "<E>1000.0</E><v>0.3</v>\n", "DYNAMIC"),
            ("dynamic_rho0", "      <density>0.0</density>"
                             "<E>1000.0</E><v>0.3</v>\n", "DYNAMIC"),
            ("static_rho1", "      <density>1.0</density>"
                            "<E>1000.0</E><v>0.3</v>\n", "STATIC")):
        r = L.run(variant(body, analysis), collect=("p.csv", "e.csv"),
                  timeout=600)
        runs[tag] = (r, series(r))
        print(f"{tag}: rc={r.rc} normal={int(r.normal_termination)} "
              f"steps={r.steps_completed} "
              f"warnings={int('WARNING' in r.text)}")

    noise, _ref_run = noise_floor(variant(
        "      <density>1.0</density><E>1000.0</E><v>0.3</v>\n", "DYNAMIC"))
    tol = max(1e-9, 100 * noise)
    print(f"solver_noise_floor={noise:.3e} identity_tolerance={tol:.3e}")

    ref = runs["dynamic_rho1"][1]
    omitted = runs["dynamic_omitted"][1]
    heavy = runs["dynamic_rho1000"][1]
    zero = runs["dynamic_rho0"][1]
    static = runs["static_rho1"][1]
    d_omitted = max_rel_dev(ref, omitted)
    d_heavy = max_rel_dev(ref, heavy)
    d_zero_static = max_rel_dev(zero, static)
    print(f"omitted_equals_density_1: max_rel_dev={d_omitted:.3e} "
          f"within_noise={int(d_omitted < tol)}")
    print(f"density_1000_differs: max_rel_dev={d_heavy:.3e} "
          f"far_above_noise={int(d_heavy > 1e-3)}")
    print(f"density_0_equals_STATIC: max_rel_dev={d_zero_static:.3e} "
          f"within_noise={int(d_zero_static < tol)}")
    all_clean = all(r.rc == 0 and r.normal_termination
                    for r, _s in runs.values())
    good = (all_clean and d_omitted < tol and d_zero_static < tol
            and d_heavy > 1e-3)
    return L.report(good, "density_default", "reproduced", "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
