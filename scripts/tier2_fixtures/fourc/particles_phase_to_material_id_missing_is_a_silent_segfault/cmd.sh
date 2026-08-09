#!/bin/bash
# Tier-2 for fourc::particles#11 — the two phase-mapping strings fail in wildly
# different ways, and the dangerous one is the one that looks redundant.
#
# Claimed: PHASE_TO_DYNLOADBALFAC declares the phases and a mistake there is a
#          clean abort naming the offending phase; PHASE_TO_MATERIAL_ID gives no
#          message at all -- a raw segfault, exit 139, before the first step.
#          Engine-level, so it is identical in DEM and in SPH.
#
# T2_MUTATE=1 removes every edit; both mappings stay correct, nothing crashes
# and the fixture must go red.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

DEM=$(upstream particle_dem_1d_normalcontact_linspring_stiffset.4C.yaml) || exit 3
SPH=$(upstream particle_sph_1d_pressurewave_boundary_densitysummation_cubicspline_adami.4C.yaml) || exit 3
grep -q 'PHASE_TO_MATERIAL_ID: "phase1 1"' "$DEM" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$DEM" "$SPH" "$TMP" "$MUTATE" <<'PY'
import re, sys
dem, sph = open(sys.argv[1]).read(), open(sys.argv[2]).read()
tmp, mutate = sys.argv[3], sys.argv[4] == "1"
def w(n, src, t): open("%s/%s.yaml" % (tmp, n), "w").write(src if mutate else t)
# the material map dropped entirely
w("nomatid",    dem, re.sub(r'\n  PHASE_TO_MATERIAL_ID: "[^"]*"', "", dem))
# the material map present but naming a phase no particle uses
w("wrongmatid", dem, dem.replace('PHASE_TO_MATERIAL_ID: "phase1 1"',
                                 'PHASE_TO_MATERIAL_ID: "phase2 1"'))
# the load-balance map dropped -- the twin, for contrast
w("nodlb",      dem, re.sub(r'\n  PHASE_TO_DYNLOADBALFAC: "[^"]*"', "", dem))
# the same material-map omission on the SPH side: engine-level, not DEM-specific
w("sphnomatid", sph, re.sub(r'\n  PHASE_TO_MATERIAL_ID: "[^"]*"', "", sph))
PY

probe NOMATID    "$TMP/nomatid.yaml"
probe WRONGMATID "$TMP/wrongmatid.yaml"
probe NODLB      "$TMP/nodlb.yaml"
probe SPHNOMATID "$TMP/sphnomatid.yaml"

# The silent half: a segfault with no 4C diagnostic of any kind.
grep -m1 -F "Signal: Segmentation fault (11)" "$TMP/NOMATID.log"   && echo "MISSING_MATID_SEGFAULTS=yes" || echo "MISSING_MATID_SEGFAULTS=no"
echo "NOMATID_FOURC_ERROR_BLOCKS=$(grep -c 'PROC 0 ERROR' "$TMP/NOMATID.log")"
echo "NOMATID_PHASE_DIAGNOSTICS=$(grep -cE 'PHASE_TO_MATERIAL_ID|particle type|not defined|cast to specific particle material' "$TMP/NOMATID.log")"
echo "NOMATID_STEPS_PRINTED=$(grep -c '^TIME:' "$TMP/NOMATID.log")"
# A material map naming the wrong phase crashes the same way.
echo "WRONG_MATID_ALSO_SEGFAULTS=$(grep -c 'Segmentation fault' "$TMP/WRONGMATID.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
# The loud half: the twin key gives a clean abort that NAMES the phase.
grep -m1 -F "particle type 'phase1' of initial particle not defined!" "$TMP/NODLB.log"   && echo "MISSING_DYNLOADBALFAC_NAMES_THE_PHASE=yes" || echo "MISSING_DYNLOADBALFAC_NAMES_THE_PHASE=no"
# Engine-level: the SPH deck behaves identically.
echo "SPH_SIDE_SEGFAULTS_TOO=$(grep -c 'Segmentation fault' "$TMP/SPHNOMATID.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
exit 0
