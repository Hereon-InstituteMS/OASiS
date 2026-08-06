#!/bin/bash
# Tier-2 for fourc::ale#3 — and a FALSIFICATION of how that pitfall was worded.
#
# The claim was: "using realistic E and nu values (e.g. steel) makes the ALE mesh
# too stiff and prevents geometry tracking".  Scale YOUNG on the upstream 2D ALE
# deck from 250 to 2.1e11 — steel — and the displacement field does not move by
# one part in 1e12: the result test, which asserts node 3 to 1e-12, still passes.
#
# That is not a surprise once stated: stand-alone ALE mesh motion is a
# Dirichlet-driven elasticity problem with no body force and no external load, so
# a UNIFORM scaling of the stiffness cancels out of K u = 0 exactly.  What does
# change the mesh is the stiffness DISTRIBUTION — NUE, or a per-element material
# — and the fixture shows that too: NUE 0.3 -> 0.49 shifts node 3 by 1.2e-1 in x.
#
# So the fixture pins the corrected claim: tune NUE / the material's spatial
# variation, and do not expect YOUNG's absolute value to do anything at all.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ale2d_solid.4C.yaml) || exit 3
grep -q "YOUNG: 250" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/soft.yaml"
sed 's/YOUNG: 250/YOUNG: 210000000000.0/' "$BASE" > "$TMP/steel.yaml"
sed 's/NUE: 0.3/NUE: 0.49/'              "$BASE" > "$TMP/nue.yaml"

probe SOFT  "$TMP/soft.yaml"
probe STEEL "$TMP/steel.yaml"
probe NUE   "$TMP/nue.yaml"

# The result test in the deck pins node 3 to 1e-12.  Passing it under a
# 8.4e8-fold stiffness change is the whole point.
echo "STEEL_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/STEEL.log")"
echo "NUE_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/NUE.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/STEEL.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/NUE.log"
if [ "$(grep -c 'is WRONG' "$TMP/STEEL.log")" = "0" ]; then
  echo "VERDICT: UNIFORM_YOUNG_SCALING_CHANGES_ALE_MOTION=no"
else
  echo "VERDICT: UNIFORM_YOUNG_SCALING_CHANGES_ALE_MOTION=yes"
fi
exit 0
