#!/bin/bash
# Tier-2 for fourc::fsi_xfem#3 -- the Nitsche penalty in XFSI is NIT_STAB_FAC
# (default 35), and mis-setting it gives a wrong answer, not a solver failure.
#
# Claimed: gamma_N in [10, 100] * mu / h_cut is the typical range; verify with a
#          flow-around-cylinder benchmark, interface velocity jump ~0 at machine
#          precision when gamma_N is right.
# Observed: the parameter is a bare dimensionless factor NIT_STAB_FAC whose
#          default (35) already sits in the quoted range -- 4C does the mu/h
#          scaling itself via VISC_STAB_TRACE_ESTIMATE and VISC_STAB_HK.  Sweep
#          it to 1e-3 or 1e12 on the upstream monolithic XFSI deck and six of
#          seven pinned values go wrong while Newton still converges and the
#          run still reaches the result test.  Nothing warns.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream xfsi_3D_boxes.4C.yaml) || exit 3
grep -q '  VISC_ADJOINT_SYMMETRY: "no"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/default.yaml"
sed 's/  VISC_ADJOINT_SYMMETRY: "no"/  VISC_ADJOINT_SYMMETRY: "no"\n  NIT_STAB_FAC: 0.001/'  "$BASE" > "$TMP/lo.yaml"
sed 's/  VISC_ADJOINT_SYMMETRY: "no"/  VISC_ADJOINT_SYMMETRY: "no"\n  NIT_STAB_FAC: 1.0e12/' "$BASE" > "$TMP/hi.yaml"

probe DEFAULT "$TMP/default.yaml"
probe LO      "$TMP/lo.yaml"
probe HI      "$TMP/hi.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/DEFAULT.log"
echo "DEFAULT_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/DEFAULT.log")"
echo "LO_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/LO.log")"
echo "HI_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/HI.log")"
echo "LO_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/LO.log")"
echo "HI_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/HI.log")"
grep -m1 -F "is WRONG --> actresult=" "$TMP/HI.log"
echo "CLAIMED_JUMP_OR_COND_TEXT=$(grep -ciE 'velocity jump|condition number|ill-conditioned' "$TMP/LO.log" "$TMP/HI.log" | awk -F: '{s+=$2} END {print s+0}')"
exit 0
