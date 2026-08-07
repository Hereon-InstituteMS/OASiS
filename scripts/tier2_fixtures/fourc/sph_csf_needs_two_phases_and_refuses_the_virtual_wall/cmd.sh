#!/bin/bash
# Tier-2 for fourc::particle_sph#13 (and #14, the virtual-wall source) —
# ContinuumSurfaceForce is a two-phase formulation, needs a positive
# coefficient, and is not implemented with the virtual wall; the virtual wall in
# turn needs PARTICLE_WALL_SOURCE from a different section.
#
# T2_MUTATE=1 removes every edit; all probe decks are their untouched upstream
# originals, nothing aborts and the fixture must go red.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

CSF=$(upstream particle_sph_2d_continuumsurfaceforce_bubble_twophase_equaldensity.4C.yaml) || exit 3
VW=$(upstream particle_sph_3d_hydrostatic_virtualwall_densitysummation.4C.yaml) || exit 3
grep -q "SURFACETENSIONCOEFFICIENT: 1" "$CSF" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q 'WALLFORMULATION: "VirtualParticleWallFormulation"' "$VW" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$CSF" "$VW" "$TMP" "$MUTATE" <<'PY'
import re, sys
csf, vw = open(sys.argv[1]).read(), open(sys.argv[2]).read()
tmp, mutate = sys.argv[3], sys.argv[4] == "1"
def w(n, src, t): open("%s/%s.yaml" % (tmp, n), "w").write(src if mutate else t)
w("nocoef", csf, csf.replace("  SURFACETENSIONCOEFFICIENT: 1\n", ""))
t = re.sub(r'\n  - "TYPE phase2[^"]*"', "", csf).replace(" phase2 1.0", "").replace(" phase2 3", "")
w("nophase2", csf, t)
w("csfwall", vw, vw.replace('  WALLFORMULATION: "VirtualParticleWallFormulation"',
    '  WALLFORMULATION: "VirtualParticleWallFormulation"\n'
    '  SURFACETENSIONFORMULATION: "ContinuumSurfaceForce"\n  SURFACETENSIONCOEFFICIENT: 1'))
w("nowallsrc", vw, re.sub(r'\n  PARTICLE_WALL_SOURCE: "[^"]*"', "", vw))
PY

probe NOCOEF    "$TMP/nocoef.yaml"
probe NOPHASE2  "$TMP/nophase2.yaml"
probe CSFWALL   "$TMP/csfwall.yaml"
probe NOWALLSRC "$TMP/nowallsrc.yaml"

grep -m1 -F "constant factor of surface tension coefficient not positive!" "$TMP/NOCOEF.log"   && echo "COEFFICIENT_IS_MANDATORY=yes" || echo "COEFFICIENT_IS_MANDATORY=no"
grep -m1 -F "no particle container for particle type 'phase2' found!" "$TMP/NOPHASE2.log"   && echo "CSF_NEEDS_PHASE2=yes" || echo "CSF_NEEDS_PHASE2=no"
grep -m1 -F "surface tension formulation with wall interaction not implemented!" "$TMP/CSFWALL.log"   && echo "CSF_REFUSES_THE_VIRTUAL_WALL=yes" || echo "CSF_REFUSES_THE_VIRTUAL_WALL=no"
grep -m1 -F "interface to particle wall handler required in virtual wall particle handler!" "$TMP/NOWALLSRC.log"   && echo "VIRTUAL_WALL_NEEDS_A_SOURCE=yes" || echo "VIRTUAL_WALL_NEEDS_A_SOURCE=no"
# The wall-source message names an internal interface, not the missing key.
echo "WALLSRC_MESSAGE_NAMES_THE_KEY=$(grep -c 'PARTICLE_WALL_SOURCE' "$TMP/NOWALLSRC.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
# All four abort at setup.
echo "STEPS_BEFORE_ABORT=$(cat "$TMP/NOCOEF.log" "$TMP/NOPHASE2.log" "$TMP/CSFWALL.log" "$TMP/NOWALLSRC.log" | grep -c '^TIME:')"
# STATICCONTACTANGLE, by contrast, is not validated at all -- no upstream CSF
# deck is refused for omitting it.
echo "CSF_DECKS_WITHOUT_CONTACTANGLE=$(for f in "$DECKS"/particle_sph_*continuumsurfaceforce*.4C.yaml; do grep -q 'STATICCONTACTANGLE' "$f" || echo x; done | wc -l)"
exit 0
