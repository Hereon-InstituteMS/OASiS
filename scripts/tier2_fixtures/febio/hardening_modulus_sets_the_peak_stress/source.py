"""Tier-2: the peak stress rises monotonically with H, and every value
runs clean.

Verifies febio::plasticity#0. A peak stress well above the yield stress Y
is not evidence that yield was missed — with H comparable to E it is the
correct answer. The fixture sweeps H/E over three decades on the shipped
uniaxial deck at a fixed prescribed stretch and requires:

  * every run to reach normal termination with exit 0 (there is no
    diagnostic to look for — that is the point),
  * the peak |sz| to rise monotonically with H,
  * and the highest ratio to exceed the lowest by a large factor, so a
    flat response would fail.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

E_MODULUS = 200000.0
RATIOS = (0.001, 0.01, 0.1, 1.0)


def peak_stress(run):
    blocks = L.parse_log_csv(run.files.get("s.csv") or "")
    values = [rows[1][0] for _t, rows in blocks if 1 in rows]
    return max(abs(v) for v in values) if values else None


def main() -> int:
    base = L.template("plasticity_3d_uniaxial")
    peaks = []
    all_clean = True
    for ratio in RATIOS:
        deck = L.add_logfile(
            L.swap(base, "<H>1000.0</H>", f"<H>{E_MODULUS * ratio}</H>"),
            ("element_data", "sz", "s.csv"))
        r = L.run(deck, collect=("s.csv",), timeout=900)
        p = peak_stress(r)
        peaks.append(p)
        all_clean = all_clean and r.rc == 0 and r.normal_termination
        print(f"H/E={ratio}: rc={r.rc} normal={int(r.normal_termination)} "
              f"steps={r.steps_completed} peak_abs_sz={p} "
              f"errors={int('ERROR' in r.text)}")
    if any(p is None for p in peaks):
        print("FAIL: a run logged no element stress")
        return L.report(False, "hardening_sweep", "reproduced",
                        "not_reproduced")
    monotone = all(a < b for a, b in zip(peaks, peaks[1:]))
    spread = peaks[-1] / peaks[0]
    print(f"peak_rises_monotonically_with_H={int(monotone)} "
          f"highest_over_lowest={spread:.4f}")
    good = all_clean and monotone and spread > 2.0
    return L.report(good, "hardening_sweep", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
