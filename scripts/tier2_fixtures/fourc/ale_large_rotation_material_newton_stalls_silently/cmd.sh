#!/bin/bash
# Tier-2 for fourc::ale#2 — and a FALSIFICATION of its Signal.
#
# Claimed: a linear StVenantKirchhoff ALE material on a mesh rotating > 30°
#          "produces INVERTED elements (det(J) < 0) — 4C aborts with 'negative
#          Jacobian'".
# Observed: no such abort, and no such string anywhere in 4C. The upstream
#          large-rotation deck rotates the inner boundary through 60° in 720
#          steps. With ELAST_CoupLogNeoHooke every step converges and the run is
#          clean. Swap in MAT_Struct_StVenantKirchhoff at the same YOUNG/NUE and
#          the ALE Newton stalls on 3 steps — where 4C prints
#          "ALE newton not converged in 10 iterations. Continue"
#          and then CARRIES ON. The mesh is left wrong by 54% at the tracked
#          node, and nothing but the deck's own result test notices.
#
# The corrected claim is therefore about a silent stall, not an abort: the run
# reports success per step, keeps going, and the damage is only visible if you
# compare the answer.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ale2d_solid_nln_large_rot.4C.yaml) || exit 3

HYPER='  - MAT: 1
    MAT_ElastHyper:
      NUMMAT: 1
      MATIDS: [5]
      DENS: 1
  - MAT: 5
    ELAST_CoupLogNeoHooke:
      MODE: "YN"
      C1: 1
      C2: 0.4995
'
python3 - "$BASE" "$TMP/svk.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
hyper = """  - MAT: 1
    MAT_ElastHyper:
      NUMMAT: 1
      MATIDS: [5]
      DENS: 1
  - MAT: 5
    ELAST_CoupLogNeoHooke:
      MODE: "YN"
      C1: 1
      C2: 0.4995
"""
assert hyper in t, "upstream large-rotation deck no longer carries CoupLogNeoHooke"
t = t.replace(hyper, """  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1
      NUE: 0.4995
      DENS: 1
""")
open(sys.argv[2], "w").write(t)
PY

cp "$BASE" "$TMP/hyper.yaml"
probe HYPER "$TMP/hyper.yaml"
probe SVK   "$TMP/svk.yaml"

echo "HYPER_NONCONVERGED_STEPS=$(grep -c 'ALE newton not converged in 10 iterations' "$TMP/HYPER.log")"
echo "SVK_NONCONVERGED_STEPS=$(grep -c 'ALE newton not converged in 10 iterations' "$TMP/SVK.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/HYPER.log"
grep -m1 -F "ALE newton not converged in 10 iterations. Continue" "$TMP/SVK.log"
grep -m1 -F "Result check failed with 3 errors out of 4 tests" "$TMP/SVK.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/SVK.log"
# The claimed abort does not exist.
echo "CLAIMED_NEGATIVE_JACOBIAN_TEXT=$(grep -ci 'negative jacobian' "$TMP/SVK.log")"
# It did not abort: it ran the whole schedule and reached the result test. That
# is the point — the stall is reported per step and then ignored.
echo "SVK_REACHED_RESULT_TEST=$(grep -c 'Checking results of 4 tests:' "$TMP/SVK.log")"
grep -m1 -E "STEP = 1[0-9][0-9]/720" "$TMP/SVK.log" > /dev/null && echo "SVK_RAN_PAST_STEP_100=yes" || echo "SVK_RAN_PAST_STEP_100=no"
exit 0
