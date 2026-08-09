#!/bin/bash
# Tier-2 for fourc::mixture#3 — the effect is real; the remedy 4C offers is not
# the one the entry names.
#
# Claimed: "a near-incompressible mixture with pure-displacement solve LOCKS
#          volumetrically — deflection is wrong by 10-1000x.  Switch to mixed
#          (u, p) or augmented-Lagrangian penalty."
#
# Observed: the locking is exactly as advertised, and the cure in 4C is an
# ELEMENT TECHNOLOGY on the element line — `TECH fbar` — not a mixed (u,p)
# formulation of the mixture material.  Three arms on the upstream mixture cube
# (fixed at z=0, surface traction on top, so the lateral contraction really is
# constrained):
#
#   SOFT       nu = 0.27, plain HEX8   -> the deck's own result tests pass
#   LOCKED     nu = 0.4999, plain HEX8 -> every test fails; node 17 dispz
#                                         collapses
#   FBAR       nu = 0.4999, HEX8 with TECH fbar -> the same tests fail (the
#              reference is the compressible one) but dispz is back in the
#              right range
#
# The fixture computes the ratio FBAR/LOCKED from the two logs and asserts it
# exceeds the order of magnitude the entry claims.  That number is the
# fixture's own measurement; it is deliberately NOT put into the knowledge text.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream mixture_elast_hyper_dynamic.4C.yaml) || exit 3
grep -q "      C2: 0.27" "$BASE"             || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q 'MAT 1 KINEM nonlinear"' "$BASE"     || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/soft.yaml"
sed 's/      C2: 0.27/      C2: 0.4999/' "$BASE" > "$TMP/locked.yaml"
sed 's/MAT 1 KINEM nonlinear"/MAT 1 KINEM nonlinear TECH fbar"/' "$TMP/locked.yaml" > "$TMP/fbar.yaml"

probe SOFT   "$TMP/soft.yaml"
probe LOCKED "$TMP/locked.yaml"
probe FBAR   "$TMP/fbar.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/SOFT.log"
echo "SOFT_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/SOFT.log")"
grep -m1 -F "is WRONG --> actresult=" "$TMP/LOCKED.log"

dispz() {  # $1 = log -> node-17 dispz actual value
  grep -m1 -E 'dispz +at node +17' "$1" \
    | sed -n 's/.*actresult=[[:space:]]*\(-\?[0-9.eE+-]*\),.*/\1/p'
}
L=$(dispz "$TMP/LOCKED.log")
F=$(dispz "$TMP/FBAR.log")
echo "LOCKED_NODE17_DISPZ=$L"
echo "FBAR_NODE17_DISPZ=$F"
python3 - "$L" "$F" <<'PY'
import sys
l, f = float(sys.argv[1]), float(sys.argv[2])
r = abs(f / l) if l else float("inf")
print(f"FBAR_OVER_LOCKED_DISPZ_RATIO={r:.1f}")
print("VERDICT: PURE_DISPLACEMENT_HEX8_LOCKS_BY_MORE_THAN_10X="
      + ("yes" if r > 10.0 else "no"))
PY
exit 0
