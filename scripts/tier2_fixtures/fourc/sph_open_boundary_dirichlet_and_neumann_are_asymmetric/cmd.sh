#!/bin/bash
# Tier-2 for fourc::particle_sph#15 — the two open-boundary types are not
# symmetric, and only the Dirichlet one tells you when it is under-specified.
#
# T2_MUTATE=1 removes both edits; the open-boundary block is complete, nothing
# aborts and the fixture must go red.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

OB=$(upstream particle_sph_2d_openboundary_straight_channel.4C.yaml) || exit 3
grep -q "DIRICHLET_FUNCT: 1" "$OB" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$OB" "$TMP" "$MUTATE" <<'PY'
import sys
src, tmp, mutate = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3] == "1"
def w(n, t): open("%s/%s.yaml" % (tmp, n), "w").write(src if mutate else t)
w("nofunct",  src.replace("  DIRICHLET_FUNCT: 1\n", ""))
w("nonormal", src.replace('  DIRICHLET_OUTWARD_NORMAL: "-1.0 0.0 0.0"\n', ""))
PY

probe NOFUNCT  "$TMP/nofunct.yaml"
probe NONORMAL "$TMP/nonormal.yaml"

grep -m1 -F "no function id of prescribed state set!" "$TMP/NOFUNCT.log"   && echo "DIRICHLET_FUNCT_IS_MANDATORY=yes" || echo "DIRICHLET_FUNCT_IS_MANDATORY=no"
grep -m1 -F "no outward normal set!" "$TMP/NONORMAL.log"   && echo "DIRICHLET_NORMAL_IS_MANDATORY=yes" || echo "DIRICHLET_NORMAL_IS_MANDATORY=no"
echo "STEPS_BEFORE_ABORT=$(cat "$TMP/NOFUNCT.log" "$TMP/NONORMAL.log" | grep -c '^TIME:')"
# The asymmetry, read off the upstream deck rather than asserted: 4C's own
# open-boundary decks set a Neumann boundary and never a NEUMANN_FUNCT, so the
# Neumann side has no equivalent requirement.
echo "UPSTREAM_SETS_NEUMANN_TYPE=$(grep -c 'NEUMANNBOUNDARYTYPE' "$OB" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
echo "UPSTREAM_SETS_NEUMANN_FUNCT=$(grep -c 'NEUMANN_FUNCT' "$OB" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
echo "OPEN_BOUNDARY_DECKS_WITH_NEUMANN_FUNCT=$(grep -l 'NEUMANN_FUNCT' "$DECKS"/particle_sph_*openboundary*.4C.yaml 2>/dev/null | wc -l)"
exit 0
