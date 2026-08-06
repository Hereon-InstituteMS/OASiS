"""Tier-2: prove the polar term is ACTIVE before comparing anything.

Verifies febio::polar_fluid#2. On the shipped channel deck, scaling
<tau>, <beta> and <gamma> across four decades leaves the logged element
data BIT-IDENTICAL. The deck's no-slip BC fixes only wy/wz, so wx is free
at every node, there is no wall shear layer, and with no velocity gradient
there is nothing for the micro-rotation to couple to.

The fixture first establishes that this deck IS bit-reproducible run to
run — otherwise "bit-identical" would be meaningless — and then requires
exact equality across the sweep. Reading "polar and classical agree" as
"the polar correction is small here" is the mistake the claim is about.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

OUT = ("  <Output>\n    <logfile>\n"
       '      <element_data data="fJ;fp;fsxx" delim="," file="e.csv"/>\n'
       "    </logfile>\n  </Output>\n")
POLAR = ('      <polar type="polar linear">\n'
         "        <tau>0.001</tau>\n"
         "        <alpha>0.0</alpha>\n"
         "        <beta>0.001</beta>\n"
         "        <gamma>0.001</gamma>\n"
         "      </polar>\n")
SCALES = (0.001, 0.1, 10.0)


def with_log(deck: str) -> str:
    return L.swap(deck, "</febio_spec>", OUT + "</febio_spec>")


def series(run):
    out = []
    for _t, rows in L.parse_log_csv(run.files.get("e.csv") or ""):
        for nid in sorted(rows):
            out.extend(rows[nid])
    return out


def main() -> int:
    base = L.template("polar_fluid_3d_channel")
    a = L.run(with_log(base), collect=("e.csv",), timeout=900)
    b = L.run(with_log(base), collect=("e.csv",), timeout=900)
    sa, sb = series(a), series(b)
    reproducible = bool(sa) and sa == sb
    print(f"identical_runs_bit_identical={int(reproducible)} "
          f"samples={len(sa)}")
    if not reproducible:
        print("FAIL: this deck is not bit-reproducible run to run, so an "
              "exact-invariance assertion would be meaningless")
        return L.report(False, "polar_invariance", "reproduced",
                        "not_reproduced")

    results = {}
    for scale in SCALES:
        polar = ('      <polar type="polar linear">\n'
                 f"        <tau>{scale}</tau>\n"
                 "        <alpha>0.0</alpha>\n"
                 f"        <beta>{scale}</beta>\n"
                 f"        <gamma>{scale}</gamma>\n"
                 "      </polar>\n")
        r = L.run(with_log(L.swap(base, POLAR, polar)),
                  collect=("e.csv",), timeout=900)
        results[scale] = series(r)
        print(f"polar_moduli={scale}: rc={r.rc} "
              f"normal={int(r.normal_termination)} "
              f"steps={r.steps_completed} "
              f"identical_to_reference={int(results[scale] == sa)}")
    span = max(SCALES) / min(SCALES)
    invariant = all(v == sa for v in results.values())
    print(f"invariant_over_a_factor_of_{span:.0f}={int(invariant)}")
    good = invariant and a.rc == 0 and a.normal_termination
    return L.report(good, "polar_invariance", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
