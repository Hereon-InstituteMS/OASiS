#!/bin/bash
# Tier-2 for fourc::particle_sph#10 — THERMALCAPACITY is required on EVERY
# phase's material and names the phase when missing; THERMALCONDUCTIVITY is not
# validated at all and its absence is a raw SIGFPE.
#
# T2_MUTATE=1 removes both edits, so both probe decks are the untouched thermal
# deck, nothing aborts and the fixture must go red.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

BASE=$(upstream particle_sph_1d_thermalconduction_boundary_temperatureintegration_quinticspline.4C.yaml) || exit 3
grep -q "THERMALCONDUCTIVITY: 0.001" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$BASE" "$TMP" "$MUTATE" <<'PY'
import sys
src, tmp, mutate = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3] == "1"
i = src.find("MAT_ParticleSPHBoundary")
assert i > 0, "upstream deck no longer has a boundary material"
head, tail = src[:i], src[i:]
# capacity dropped from the BOUNDARY material only -- the easy one to forget
open(tmp + "/nocapbound.yaml", "w").write(
    src if mutate else head + tail.replace("      THERMALCAPACITY: 1\n", "", 1))
# conductivity dropped from the FLUID material only
open(tmp + "/nocondfluid.yaml", "w").write(
    src if mutate else head.replace("      THERMALCONDUCTIVITY: 0.001\n", "") + tail)
PY

probe NOCAPBOUND  "$TMP/nocapbound.yaml"
probe NOCONDFLUID "$TMP/nocondfluid.yaml"
probe BASE        "$BASE"

grep -m1 -E "^OK \(" "$TMP/BASE.log" && echo "BASE_PASSES=yes"
# The capacity check names the PHASE, which is the actionable part.
grep -m1 -F "thermal capacity for particles of type 'boundaryphase' not positive!" "$TMP/NOCAPBOUND.log"   && echo "CAPACITY_ABORT_NAMES_THE_PHASE=yes" || echo "CAPACITY_ABORT_NAMES_THE_PHASE=no"
# ... and it is the BOUNDARY material that is missing it, not the fluid.
echo "CAPACITY_BLAMES_THE_FLUID=$(grep -cF "type 'phase1' not positive" "$TMP/NOCAPBOUND.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
# Conductivity has NO validation: the run is killed by a signal instead.
grep -m1 -F "Signal: Floating point exception (8)" "$TMP/NOCONDFLUID.log"   && echo "CONDUCTIVITY_IS_A_SIGNAL=yes" || echo "CONDUCTIVITY_IS_A_SIGNAL=no"
echo "CONDUCTIVITY_FOURC_ERROR_BLOCKS=$(grep -c 'PROC 0 ERROR' "$TMP/NOCONDFLUID.log")"
echo "CONDUCTIVITY_DIAGNOSTICS=$(grep -cE 'thermal conductivity|THERMALCONDUCTIVITY' "$TMP/NOCONDFLUID.log")"
# It is not a frozen temperature field: the run does not finish at all.
echo "CONDUCTIVITY_REACHED_RESULT_TESTS=$(grep -c 'Checking results of' "$TMP/NOCONDFLUID.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
exit 0
