#!/bin/bash
# Tier-2 for fourc::ssi#1 — and a FALSIFICATION of how it was worded.
#
# Claimed: a plain MAT_ElastHyper on an electrode "cannot capture swelling —
#          lithium intercalation produces zero deformation even at high
#          concentration", i.e. a quiet wrong answer.
#
# Observed: it is not quiet.  Swapping MAT_MultiplicativeSplitDefgradElastHyper
# for MAT_ElastHyper (same elastic summand, same density) on the upstream
# NMC-622 intercalation deck aborts with
#
#     Your material does not allow to evaluate a monolithic ssi material!
#     src/solid_scatra_3D_ele/4C_solid_scatra_3D_ele_calc.cpp
#
# raised from SolidScatraEleCalc::evaluate_d_stress_d_scalar — the element asks
# the material for dStress/dScalar, the coupling term that IS the swelling, and
# a plain hyperelastic material cannot supply it.  Exit 1; no displacement field
# is ever written, so there is no "zero deformation" result to read.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ssi_mono_3D_1hex8_elch_polyiso_NMC-622_growthlaw.4C.yaml) || exit 3
grep -q "MAT_MultiplicativeSplitDefgradElastHyper" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
cp "$BASE" "$TMP/split.yaml"

python3 - "$BASE" "$TMP/plain.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
old = ("    MAT_MultiplicativeSplitDefgradElastHyper:\n"
       "      NUMMATEL: 1\n"
       "      MATIDSEL: [3]\n"
       "      NUMFACINEL: 1\n"
       "      INELDEFGRADFACIDS: [4]\n"
       "      DENS: 7480\n")
new = ("    MAT_ElastHyper:\n"
       "      NUMMAT: 1\n"
       "      MATIDS: [3]\n"
       "      DENS: 7480\n")
if old not in t:
    print("FIXTURE_ABORT=upstream_deck_changed"); sys.exit(3)
open(sys.argv[2], "w").write(t.replace(old, new, 1))
PY
[ -f "$TMP/plain.yaml" ] || exit 3

probe SPLIT "$TMP/split.yaml"
probe PLAIN "$TMP/plain.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/SPLIT.log"
grep -m1 -F "Your material does not allow to evaluate a monolithic ssi material!" "$TMP/PLAIN.log"
grep -m1 -oF "4C_solid_scatra_3D_ele_calc.cpp" "$TMP/PLAIN.log"
# It fails asking for the concentration derivative of the stress — the swelling
# coupling term itself.
echo "FAILS_IN_DSTRESS_DSCALAR=$(grep -c 'evaluate_d_stress_d_scalar' "$TMP/PLAIN.log")"
# So there is no quiet zero-deformation answer to be misled by.
echo "PLAIN_RESULT_TESTS_PERFORMED=$(grep -c 'is WRONG --> actresult=' "$TMP/PLAIN.log")"
exit 0
