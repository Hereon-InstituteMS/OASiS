#!/bin/bash
# Tier-2 for fourc::lubrication#0 — 4C never checks the thin-film assumption, and
# the size of the error is NOT the "~2-10x" the entry claimed.
#
# Upstream lubrication_sb_2d is a slider bearing 30 long with a film height
# h(x) = 0.045 - 5e-4 x, so h/L ~ 1.5e-3 — deep in the Reynolds regime, and its
# result test pins the pressure at node 10 to 1e-10.
#
# Multiply the height function by 1000 (h ~ 45 over the same L = 30, h/L ~ 1.5 —
# a "film" one and a half times thicker than the bearing is long) and 4C:
#   * emits no warning about the thin-film assumption at all,
#   * converges in the same number of Newton steps,
#   * returns a finite, perfectly plausible-looking pressure,
#   * scaled by EXACTLY 1e-6 = (1/1000)^2.
# That last number is the point: the Reynolds equation is solved verbatim, so the
# pressure follows p ~ mu U L / h^2 whatever the geometry, and the user gets no
# signal that the model has left its regime.  It is not a 2-10x discrepancy; it
# is the lubrication scaling law applied outside its validity, silently.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream lubrication_sb_2d.4C.yaml) || exit 3
grep -q '"(0.045)-(x\*0.5e-3)"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "VALUE: 73.94356207110268" "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_result_value_changed"; exit 3; }

# The pathology, in one place: the film height the second arm is given.
BAD_HEIGHT='(45.0)-(x*0.5)'

cp "$BASE" "$TMP/thin.yaml"
sed "s|(0.045)-(x\*0.5e-3)|$BAD_HEIGHT|" "$BASE" > "$TMP/thick.yaml"
grep -qF "$BAD_HEIGHT" "$TMP/thick.yaml" \
  || { echo "FIXTURE_ABORT=height_substitution_failed"; exit 3; }

probe THIN  "$TMP/thin.yaml"
probe THICK "$TMP/thick.yaml"

grep -m1 -F "is CORRECT, abs(diff)=" "$TMP/THIN.log"
grep -m1 -F "processor 0 finished normally" "$TMP/THIN.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/THICK.log"

# 4C says nothing whatsoever about the film being too thick for Reynolds.
echo "THICK_FILM_WARNINGS=$(grep -ciE 'thin[ -]film|lubrication approximation|reynolds.*(valid|assumption)|aspect ratio' "$TMP/THICK.log")"
# It converged just as happily as the valid case.
echo "THIN_NEWTON_LINES=$(grep -c '\[L_2 \]' "$TMP/THIN.log")"
echo "THICK_NEWTON_LINES=$(grep -c '\[L_2 \]' "$TMP/THICK.log")"
# ...and the number it returns is the pure Reynolds 1/h^2 scaling.
P=$(grep -m1 -oE 'actresult=[ ]*[-0-9.eE+]+' "$TMP/THICK.log" | tr -d ' ' | cut -d= -f2)
echo "THICK_FILM_PRESSURE=$P"
echo "THICK_OVER_THIN_PRESSURE_RATIO=$(python3 -c "print('%.4e' % ($P/73.94356207110268))")"
echo "THICK_FILM_PRESSURE_IS_FINITE=$(python3 -c "
import math;print('yes' if math.isfinite($P) else 'no')")"
exit 0
