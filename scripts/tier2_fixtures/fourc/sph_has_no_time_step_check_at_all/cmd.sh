#!/bin/bash
# Tier-2 for fourc::particle_sph#12 — 4C runs NO stability check for SPH.
#
# Claimed: no CFL computation, no critical step, and not one line mentioning the
#          time step at ANY size. The first thing that ever complains is the
#          binning strategy, which names neither the step nor CFL. The DEM half
#          of the same engine DOES print a critical step, which is what makes
#          the absence easy to mistake for a passing check.
#
# T2_MUTATE=1 removes the pathology: every deck keeps the upstream time step, so
# the escalation never happens, the bin-travel abort never fires and
# BIN_ABORT_AT_X200 flips to no.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

BASE=$(upstream particle_sph_1d_pressurewave_boundary_densitysummation_cubicspline_adami.4C.yaml) || exit 3
DEM=$(upstream particle_dem_1d_normalcontact_linspring_stiffset.4C.yaml) || exit 3
grep -q "TIMESTEP: 0.001" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$BASE" "$DEM" "$TMP" "$MUTATE" <<'PY'
import sys
sph, dem = open(sys.argv[1]).read(), open(sys.argv[2]).read()
tmp, mutate = sys.argv[3], sys.argv[4] == "1"
for tag, dt in (("x10", "0.01"), ("x50", "0.05"), ("x200", "0.2")):
    t = sph.replace("  TIMESTEP: 0.001", "  TIMESTEP: " + dt)
    open("%s/%s.yaml" % (tmp, tag), "w").write(sph if mutate else t)
# the DEM contrast, at 20x its own step
open(tmp + "/demx20.yaml", "w").write(
    dem.replace("  TIMESTEP: 0.001", "  TIMESTEP: 0.02").replace("  NUMSTEP: 1000", "  NUMSTEP: 50"))
PY

probe X10    "$TMP/x10.yaml"
probe X50    "$TMP/x50.yaml"
probe X200   "$TMP/x200.yaml"
probe DEMX20 "$TMP/demx20.yaml"

# Not one line about the time step at ANY of the three SPH sizes.
echo "SPH_TIMESTEP_LINES=$(cat "$TMP/X10.log" "$TMP/X50.log" "$TMP/X200.log" | grep -ciE 'cfl|courant|critical time step|stable time step')"
# At 10x and 50x the time loop runs to the end and the result-test manager is
# reached -- so nothing in the solver objected. ("processor 0 finished normally"
# is NOT the token to use here: a failing result test aborts through MPI and
# that line is never printed, which would make a healthy-but-wrong run look
# like a crash.)
echo "X10_REACHED_RESULT_TESTS=$(grep -c 'Checking results of' "$TMP/X10.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
echo "X50_REACHED_RESULT_TESTS=$(grep -c 'Checking results of' "$TMP/X50.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
echo "X10_ABORTED_IN_THE_TIME_LOOP=$(grep -c 'traveled more than one bin' "$TMP/X10.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
echo "X50_ABORTED_IN_THE_TIME_LOOP=$(grep -c 'traveled more than one bin' "$TMP/X50.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
echo "X10_VERDICTS_WRONG=$(grep -c 'is WRONG' "$TMP/X10.log")"
echo "X50_VERDICTS_WRONG=$(grep -c 'is WRONG' "$TMP/X50.log")"
# Push far enough and the BINNING strategy complains -- about bins, not steps.
grep -m1 -F "traveled more than one bin on this processor!" "$TMP/X200.log"   && echo "BIN_ABORT_AT_X200=yes" || echo "BIN_ABORT_AT_X200=no"
echo "BIN_ABORT_MENTIONS_THE_STEP=$(grep -A1 'traveled more than one bin' "$TMP/X200.log" | grep -ciE 'time step|cfl' | sed 's/^0$/no/;s/^[1-9].*/yes/')"
# The contrast that makes the absence legible: DEM, same engine, does warn.
echo "DEM_WARNS_ABOUT_THE_STEP=$(grep -c 'larger than critical time step' "$TMP/DEMX20.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
exit 0
