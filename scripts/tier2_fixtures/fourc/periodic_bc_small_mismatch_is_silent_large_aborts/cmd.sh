#!/bin/bash
# Tier-2 for fourc::constraint#3 — periodic master and slave surfaces do have to
# match geometrically, and the failure comes in two very different flavours.
#
# Upstream deck: tsi_heatflux_monolithic_pbc — a DESIGN SURF PERIODIC BOUNDARY
# CONDITIONS pair on the z=0 (Slave) and z=8 (Master) faces, PLANE xy, ANGLE 0,
# ABSTREETOL 1e-09. When it matches, 4C reports
#
#     The layout is generated: 15 masters are coupled to at least 1 and up to 1 slaves,
#
# Three ways of breaking it:
#
#   NUDGED   one slave node moved 1e-3 in x — a MILLION times ABSTREETOL — and
#            4C pairs it anyway. Same "15 masters ... up to 1 slaves" line, and
#            not one word about tolerance or geometry. ABSTREETOL guides the
#            search tree; it does not reject a bad partner.
#   MOVED    the same node moved 0.15, off the face footprint entirely, so one
#            partner really is lost:
#              have 14 masters in midtosid list, 15 expected
#              .../core/fem/src/condition/4C_fem_condition_periodic.cpp
#   ROTATED  ANGLE 5 on an xy slave — the entry's "slave is rotated" case:
#              Rotation of slave plane only implemented for xz and yz planes
#
# Only the last two say anything. The first is the dangerous one, and it is what
# NUDGED_PAIRING_WARNINGS=0 pins.
#
# Deliberately NOT asserted: how far the NUDGED answer drifts. Moving any node
# changes a finite-element answer, so the result-test delta on that arm does not
# by itself isolate the mis-pairing; the silence does.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream tsi_heatflux_monolithic_pbc.4C.yaml) || exit 3
SLAVE='"NODE 13 COORD 1.4500000000000000e+00 1.5000000000000000e+00 0.0000000000000000e+00"'
grep -q -F "$SLAVE" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "DESIGN SURF PERIODIC BOUNDARY CONDITIONS" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

# The deck names its solver XMLs by relative path, resolved against the working
# directory, so those two files have to travel with it.
DECKDIR=$(dirname "$BASE")
for rel in linear_solver/iterative_gmres_template.xml block_preconditioner/thermo_solid.xml; do
  [ -f "$DECKDIR/xml/$rel" ] || { echo "FIXTURE_ABORT=missing_upstream_aux"; exit 3; }
  mkdir -p "$TMP/xml/$(dirname "$rel")"
  cp "$DECKDIR/xml/$rel" "$TMP/xml/$rel"
done

cp "$BASE" "$TMP/matched.yaml"
sed 's|1.4500000000000000e+00 1.5000000000000000e+00 0.0000000000000000e+00|1.4510000000000000e+00 1.5000000000000000e+00 0.0000000000000000e+00|' "$BASE" > "$TMP/nudged.yaml"
sed 's|1.4500000000000000e+00 1.5000000000000000e+00 0.0000000000000000e+00|1.6000000000000000e+00 1.5000000000000000e+00 0.0000000000000000e+00|' "$BASE" > "$TMP/moved.yaml"

python3 - "$BASE" "$TMP/rotated.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
slave = """  - E: 3
    ID: 1
    MASTER_OR_SLAVE: "Slave"
    PLANE: "xy"
    LAYER: 1
    ANGLE: 0
"""
assert slave in t, "upstream deck no longer carries the xy slave with ANGLE 0"
open(sys.argv[2], "w").write(t.replace(slave, slave.replace("ANGLE: 0", "ANGLE: 5"), 1))
PY

cd "$TMP" || exit 3
probe MATCHED "$TMP/matched.yaml"
probe NUDGED  "$TMP/nudged.yaml"
probe MOVED   "$TMP/moved.yaml"
probe ROTATED "$TMP/rotated.yaml"

grep -m1 -F "The layout is generated: 15 masters are coupled to at least 1 and up to 1 slaves," "$TMP/MATCHED.log"
grep -m1 -F "processor 0 finished normally" "$TMP/MATCHED.log"
echo "MATCHED_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/MATCHED.log")"

# A 1e-3 offset — six orders above ABSTREETOL — is paired without comment.
grep -m1 -F "The layout is generated: 15 masters are coupled to at least 1 and up to 1 slaves," "$TMP/NUDGED.log"
python3 - "$TMP/NUDGED.log" <<'PY'
import sys
log = open(sys.argv[1], "rb").read().decode("utf-8", "replace").lower()
n = (log.count("abstreetol") + log.count("no matching") + log.count("could not find a partner")
     + log.count("tolerance") + log.count("midtosid list"))
print("NUDGED_PAIRING_WARNINGS=%d" % n)
PY

# Off the footprint: a partner really is lost, and this one is reported.
grep -m1 -F "have 14 masters in midtosid list, 15 expected" "$TMP/MOVED.log"
grep -m1 -F "4C_fem_condition_periodic.cpp" "$TMP/MOVED.log"
# Rotating the slave plane of an xy pair is simply not implemented.
grep -m1 -F "Rotation of slave plane only implemented for xz and yz planes" "$TMP/ROTATED.log"
exit 0
