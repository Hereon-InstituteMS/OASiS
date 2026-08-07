#!/bin/bash
# Tier-2 for fourc::xfem_fluid#4 -- MomentFitting does not warn and does not
# fall back.  It segfaults.
#
# Claimed: runtime warning `MomentFitting did not converge for element X --
#          falling back to Tessellation`.
# Observed: no such warning exists.  On the upstream level-set XFEM deck,
#          VOLUME_GAUSS_POINTS_BY: "MomentFitting" dies with SIGSEGV inside
#          Core::FE::GaussPointsComposite::num_points -- a null dereference, no
#          message, no fallback, exit 139.  "Tessellation" on the same deck runs
#          to completion.  So the advice "Tessellation is more robust" is right,
#          but the failure mode to watch for is a crash, not a warning.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream xfluid_ls_neumann_inflow_stab.4C.yaml) || exit 3
grep -q 'VOLUME_GAUSS_POINTS_BY: "DirectDivergence"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

sed 's/VOLUME_GAUSS_POINTS_BY: "DirectDivergence"/VOLUME_GAUSS_POINTS_BY: "Tessellation"/'  "$BASE" > "$TMP/tess.yaml"
sed 's/VOLUME_GAUSS_POINTS_BY: "DirectDivergence"/VOLUME_GAUSS_POINTS_BY: "MomentFitting"/' "$BASE" > "$TMP/mf.yaml"

probe TESS "$TMP/tess.yaml"
probe MF   "$TMP/mf.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/TESS.log"
echo "TESS_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/TESS.log")"
# MomentFitting: a signal, not a diagnostic.
grep -m1 -F "Signal: Segmentation fault (11)" "$TMP/MF.log"
grep -m1 -F "GaussPointsComposite" "$TMP/MF.log"
echo "MF_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/MF.log")"
echo "CLAIMED_FALLBACK_TEXT=$(grep -ciE 'falling back to Tessellation|MomentFitting did not converge' "$TMP/MF.log")"
exit 0
