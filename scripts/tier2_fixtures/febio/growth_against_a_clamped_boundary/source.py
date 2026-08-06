"""Tier-2: growth against a fully clamped boundary converges QUIETLY to a
motionless body full of residual stress.

Verifies febio::growth_remodeling#2 — the dangerous case, because there is
no divergence to warn you. Two runs of the same growing block:

  * constrained on three symmetry planes only: J tracks the growth
    multiplier cubed and the stress stays at round-off,
  * every outer node fully clamped: J stays at 1, nothing moves, and the
    residual stress is a large fraction of the elastic modulus.

Both reach normal termination with exit 0 and all steps completed. The
fixture computes the multiplier cubed from the load curve rather than
carrying a number, and asserts the clamped run's stress against the
modulus in the deck.
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


import re

MULTIPLIER_END = 1.2      # the load curve's last point in the shipped deck
E_MODULUS = 1000.0        # the shipped elastic child's E


def elements(run):
    blocks = L.parse_log_csv(run.files.get("e.csv") or "")
    return blocks[-1][1] if blocks else {}


def main() -> int:
    base = L.template("growth_remodeling_3d_isotropic")
    i = base.find("  <Boundary>")
    j = base.find("</Boundary>") + len("</Boundary>\n")
    if i < 0 or j <= i:
        L.die("the growth template has no <Boundary> section to replace")
    symmetry = base[i:j]

    node_ids = sorted({int(v) for v in re.findall(r'<node id="(\d+)"', base)})
    if not node_ids:
        L.die("no nodes found in the growth template")
    clamped_deck = L.swap(
        base, '<NodeSet name="face_x0">',
        f'<NodeSet name="all_outer">{",".join(str(v) for v in node_ids)}'
        f'</NodeSet>\n    <NodeSet name="face_x0">')
    clamped_deck = L.swap(
        clamped_deck, symmetry,
        "  <Boundary>\n"
        '    <bc name="fixall" type="zero displacement" '
        'node_set="all_outer">\n'
        "      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>\n"
        "    </bc>\n  </Boundary>\n")

    free = L.run(with_log(base), collect=("e.csv",), timeout=900)
    clamped = L.run(with_log(clamped_deck), collect=("e.csv",), timeout=900)

    def summarise(run):
        rows = elements(run)
        if not rows:
            return None, None
        js = [v[3] for v in rows.values()]
        stress = [max(abs(v[0]), abs(v[1]), abs(v[2]))
                  for v in rows.values()]
        return (min(js), max(js)), max(stress)

    (jf_lo, jf_hi), sf = summarise(free)
    (jc_lo, jc_hi), sc = summarise(clamped)
    expected_j = MULTIPLIER_END ** 3
    print(f"symmetry_planes: rc={free.rc} "
          f"normal={int(free.normal_termination)} "
          f"steps={free.steps_completed} J=[{jf_lo:.6f},{jf_hi:.6f}] "
          f"expected_multiplier_cubed={expected_j:.6f} max_stress={sf:.6g}")
    print(f"fully_clamped: rc={clamped.rc} "
          f"normal={int(clamped.normal_termination)} "
          f"steps={clamped.steps_completed} J=[{jc_lo:.6f},{jc_hi:.6f}] "
          f"max_stress={sc:.6g} stress_over_E={sc / E_MODULUS:.3f}")
    grew = abs(jf_hi - expected_j) < 1e-6 and sf < 1e-3 * E_MODULUS
    frozen = abs(jc_hi - 1.0) < 1e-9 and abs(jc_lo - 1.0) < 1e-9
    stressed = sc > 0.1 * E_MODULUS
    silent = (free.rc == 0 and clamped.rc == 0
              and free.normal_termination and clamped.normal_termination
              and "ERROR" not in clamped.text
              and "WARNING" not in clamped.text)
    print(f"free_growth_tracks_multiplier_cubed={int(grew)} "
          f"clamped_J_stays_at_one={int(frozen)} "
          f"clamped_residual_stress_exceeds_a_tenth_of_E={int(stressed)} "
          f"both_terminate_silently={int(silent)}")
    good = grew and frozen and stressed and silent
    return L.report(good, "clamped_growth", "reproduced", "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
