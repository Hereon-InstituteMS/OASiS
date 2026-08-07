#!/bin/bash
# Tier-2 for fourc::particle_sph#9 — DENSITYEVALUATION and DENSITYCORRECTION are
# locked to each other in BOTH directions, with a message per direction from a
# different source file.
#
# T2_MUTATE=1 removes both edits; the deck keeps its default DensitySummation
# with no correction, nothing aborts and the fixture must go red.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

BASE=$(upstream particle_sph_1d_pressurewave_boundary_densitysummation_cubicspline_adami.4C.yaml) || exit 3
grep -q "DENSITYCORRECTION" "$BASE" && { echo "FIXTURE_ABORT=upstream_deck_already_corrects"; exit 3; }
REF=$(upstream particle_sph_1d_hydrostatic_freesurface_densityrandlesreinit_quinticspline_adami.4C.yaml) || exit 3

python3 - "$BASE" "$TMP" "$MUTATE" <<'PY'
import sys
src, tmp, mutate = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3] == "1"
a = '  BOUNDARYPARTICLEFORMULATION: "AdamiBoundaryFormulation"'
assert a in src
def w(n, t): open("%s/%s.yaml" % (tmp, n), "w").write(src if mutate else t)
# a correction on top of the default summation density
w("corrwithout", src.replace(a, '  DENSITYCORRECTION: "RandlesCorrection"\n' + a))
# the predictor-corrector density with no correction at all
w("pcnone",      src.replace(a, '  DENSITYEVALUATION: "DensityPredictCorrect"\n' + a))
PY

probe CORRWITHOUT "$TMP/corrwithout.yaml"
probe PCNONE      "$TMP/pcnone.yaml"
probe LEGAL       "$REF"

grep -m1 -F "the density correction scheme set is not valid with the current density evaluation scheme!" "$TMP/CORRWITHOUT.log"   && echo "CORRECTION_WITHOUT_PC_ABORTS=yes" || echo "CORRECTION_WITHOUT_PC_ABORTS=no"
grep -m1 -F "no density correction scheme set via parameter 'DENSITYCORRECTION'!" "$TMP/PCNONE.log"   && echo "PC_WITHOUT_CORRECTION_ABORTS=yes" || echo "PC_WITHOUT_CORRECTION_ABORTS=no"
# The two halves come from different files, so a search for one message will
# not find the other.
grep -m1 -oE "4C_particle_interaction_sph\.cpp" "$TMP/CORRWITHOUT.log" | head -1 | sed 's/^/FILE_A=/'
grep -m1 -oE "4C_particle_interaction_sph_density\.cpp" "$TMP/PCNONE.log" | head -1 | sed 's/^/FILE_B=/'
echo "BOTH_ABORT_AT_SETUP=$(cat "$TMP/CORRWITHOUT.log" "$TMP/PCNONE.log" | grep -c '^TIME:')"
# The legal pairing runs: PredictCorrect WITH a correction.
grep -m1 -E "^OK \(" "$TMP/LEGAL.log" && echo "LEGAL_PAIRING_PASSES=yes"
echo "LEGAL_DECK_HAS_BOTH=$( { grep -q 'DENSITYEVALUATION: "DensityPredictCorrect"' "$REF" && grep -q 'DENSITYCORRECTION' "$REF"; } && echo yes || echo no)"
exit 0
