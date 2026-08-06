"""Tier-2: THREE properties are named when missing — and <phi0> is not
one of them.

Verifies febio::biphasic_fsi#1, and CORRECTS it. The claim says the
material "needs FOUR things and each missing one is named separately:
<phi0>, a <solid>, a <fluid> and a <permeability>".

Executed: the three PROPERTIES do behave that way, each reported as
`Component "Tissue" needs to have property "<name>" defined (line N)`
with the material's own name. <phi0> does not. It is a plain parameter
with a default: a deck without it READS, RUNS and reaches normal
termination with no message of any kind. So it is three named errors and
one silent default, not four named errors.

Whether the silent default changes the ANSWER cannot be settled on this
deck, and the fixture says so rather than pretending otherwise: with the
interstitial velocity and the dilatation fully constrained and the
problem driven through the solid, the logged fields are the same to a
tolerance far above the solver noise at phi0 = 0.2, omitted, 0.0 and 0.9.
The deck cannot see phi0 at all.

A NOTE ON HOW NOT TO WRITE THIS FIXTURE. The first version compared md5
digests of the logged CSV. That is unsound here: this deck runs on
bicgstab with a non-symmetric matrix, and repeated identical runs agree
only to about eleven significant digits, so their digests differ every
time. An md5 comparison would have "detected" an effect of phi0 that is
pure round-off. Compare with a tolerance, never with a checksum, on any
deck whose linear solver is iterative.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

PROPS = {
    "solid": ('      <solid type="neo-Hookean">\n'
              "        <density>1.0</density>\n"
              "        <E>100.0</E>\n"
              "        <v>0.0</v>\n"
              "      </solid>\n"),
    "fluid": ('      <fluid type="fluid">\n'
              "        <density>1.0</density>\n"
              "        <k>1e3</k>\n"
              '        <viscous type="Newtonian fluid">\n'
              "          <mu>0.01</mu>\n"
              "        </viscous>\n"
              "      </fluid>\n"),
    "permeability": ('      <permeability type="perm-const-iso">\n'
                     "        <perm>0.001</perm>\n"
                     "      </permeability>\n"),
}
PHI = "      <phi0>0.2</phi0>\n"
CSV = "biphasic_fsi_nodes.csv"


def series(run):
    """Every logged value, flattened, in file order."""
    out = []
    for _t, rows in L.parse_log_csv(run.files.get(CSV) or ""):
        for nid in sorted(rows):
            out.extend(rows[nid])
    return out


def max_rel_dev(a, b) -> float:
    if not a or len(a) != len(b):
        return float("inf")
    worst = 0.0
    for x, y in zip(a, b):
        scale = max(abs(x), abs(y), 1e-12)
        worst = max(worst, abs(x - y) / scale)
    return worst


def main() -> int:
    right = L.template("biphasic_fsi_3d_block")
    named = 0
    for prop, block in PROPS.items():
        w = L.run(L.drop(right, block))
        msg = w.has(f'Component "Tissue" needs to have property '
                    f'"{prop}" defined')
        print(f"missing_{prop}: rc={w.rc} read_failed={int(w.read_failed)} "
              f"named_with_material_name={int(msg)}")
        if msg and w.read_failed and w.rc != 0:
            named += 1

    # Solver noise floor: two identical runs of the same deck.
    a = L.run(right, collect=(CSV,))
    b = L.run(right, collect=(CSV,))
    noise = max_rel_dev(series(a), series(b))
    print(f"identical_runs_max_relative_deviation={noise:.3e} "
          f"bit_identical={int(series(a) == series(b))}")

    variants = {
        "omitted": L.drop(right, PHI),
        "zero": L.swap(right, PHI, "      <phi0>0.0</phi0>\n"),
        "high": L.swap(right, PHI, "      <phi0>0.9</phi0>\n"),
    }
    silent = True
    insensitive = True
    tol = max(1e-6, 1000 * noise)
    for tag, deck in variants.items():
        r = L.run(deck, collect=(CSV,))
        dev = max_rel_dev(series(a), series(r))
        named_err = "needs to have property" in r.text
        print(f"phi0_{tag}: rc={r.rc} normal={int(r.normal_termination)} "
              f"steps={r.steps_completed} "
              f"named_error={int(named_err)} "
              f"max_rel_dev_vs_shipped={dev:.3e}")
        silent = silent and r.rc == 0 and r.normal_termination             and not named_err
        insensitive = insensitive and dev < tol
    print(f"properties_named_when_missing={named} of 3")
    print(f"phi0_is_a_silent_default_not_a_named_error={int(silent)}")
    print(f"shipped_deck_cannot_see_phi0={int(insensitive)} "
          f"(tolerance {tol:.1e})")
    good = (named == 3 and silent and insensitive
            and a.rc == 0 and a.normal_termination)
    return L.report(good, "biphasic_fsi_properties",
                    "reproduced_with_correction", "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
