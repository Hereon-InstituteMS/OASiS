#!/bin/bash
# Tier-2 for fourc::particle_sph#8 — writing boundaryphase particles does not
# give you a wall, and forgetting BOUNDARYPARTICLEFORMULATION is SILENT.
#
# Claimed: with the default NoBoundaryFormulation the boundary states are
#          allocated but never filled, the momentum equation reads them as
#          zeros, and the fluid sees a wall at zero pressure and zero velocity.
#          Nothing is printed. The MIRROR case does abort -- and what matters
#          there is the PHASE MAP, not the PARTICLES list.
#
# T2_MUTATE=1 removes the pathology: the formulation line stays in, the wall
# works, VERDICTS_WRONG goes to 0 and the fixture must go red.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

BASE=$(upstream particle_sph_1d_pressurewave_boundary_densitysummation_cubicspline_adami.4C.yaml) || exit 3
grep -q 'BOUNDARYPARTICLEFORMULATION: "AdamiBoundaryFormulation"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$BASE" "$TMP" "$MUTATE" <<'PY'
import re, sys
src, tmp, mutate = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3] == "1"
line = '  BOUNDARYPARTICLEFORMULATION: "AdamiBoundaryFormulation"\n'
open(tmp + "/noform.yaml", "w").write(src if mutate else src.replace(line, ""))
# formulation ON, every boundary PARTICLE deleted, phase still in the maps
noparts = re.sub(r'\n  - "TYPE boundaryphase[^"]*"', "", src)
open(tmp + "/noparts.yaml", "w").write(noparts)
# formulation ON, boundary phase removed from the PHASE MAPS as well
nomap = noparts.replace(" boundaryphase 1.0", "").replace(" boundaryphase 2", "")
open(tmp + "/nomap.yaml", "w").write(nomap)
PY

probe NOFORM  "$TMP/noform.yaml"
probe NOPARTS "$TMP/noparts.yaml"
probe NOMAP   "$TMP/nomap.yaml"
probe BASE    "$BASE"

# Untouched: passes.
grep -m1 -E "^OK \(" "$TMP/BASE.log" && echo "BASE_PASSES=yes"
# Without the formulation: no abort from the particle module, no warning
# naming the formulation, the boundary phase or a wall, and a wrong answer.
echo "NOFORM_PARTICLE_ABORTS=$(grep -c 'PROC 0 ERROR in /.*particle' "$TMP/NOFORM.log")"
# Not a keyword grep: 'wall' also matches an unrelated timing line. Count 4C's
# own boundary-particle diagnostics by name instead.
echo "NOFORM_BOUNDARY_DIAGNOSTICS=$(grep -cE 'no boundary or rigid particles defined|unknown boundary particle formulation|not found in container' "$TMP/NOFORM.log")"
grep -m1 -F "Checking results of" "$TMP/NOFORM.log" && echo "NOFORM_REACHED_RESULT_TESTS=yes"
echo "NOFORM_VERDICTS_WRONG=$(grep -c 'is WRONG' "$TMP/NOFORM.log")"
echo "NOFORM_IS_SILENT_AND_WRONG=$([ "$(grep -c 'is WRONG' "$TMP/NOFORM.log")" -gt 0 ] && [ "$(grep -c 'PROC 0 ERROR in /.*particle' "$TMP/NOFORM.log")" = 0 ] && echo yes || echo no)"
# The mirror case, and the distinction the entry turns on: deleting every
# boundary PARTICLE while the phase stays mapped does NOT trigger the guard ...
echo "NOPARTS_TRIGGERS_THE_GUARD=$(grep -c 'no boundary or rigid particles defined' "$TMP/NOPARTS.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
# ... removing the phase from the maps does.
grep -m1 -F "no boundary or rigid particles defined but a boundary particle formulation is set!" "$TMP/NOMAP.log"   && echo "NOMAP_TRIGGERS_THE_GUARD=yes" || echo "NOMAP_TRIGGERS_THE_GUARD=no"
exit 0
