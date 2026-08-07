#!/bin/bash
# Tier-2 for fourc::particles#4 — PERIDYNAMIC_GRID_SPACING is a free parameter
# that 4C never compares against the particles you actually gave it.
#
# Upstream particle_sph_2d_pdbody_gravity.4C.yaml packs its pdphase body on a
# lattice of pitch 1.0 and declares PERIDYNAMIC_GRID_SPACING: 1.0.  The mutants
# change ONLY the declared number; every particle position stays where it was.
#
#   MATCHED : the reference, 1512 bonds, all ten result tests pass.
#   OFF5PCT : declare 1.05 instead of 1.00.  Identical bond list — the neighbour
#             search uses the horizon, not this key — no warning of any kind,
#             all 3001 steps run, and 8 of the 10 result values are wrong.  A
#             five per cent slip is already fatal to the answer and completely
#             invisible.
#   HALF    : declare 0.5.  Again 1512 bonds and 3001 silent steps, and the body
#             lands somewhere else entirely.  In 2-D the bond force carries
#             dx^4, so this is a sixteen-fold error in every bond stiffness.
#
# There is no check anywhere: 4C reads the key, cubes and squares it into the
# micromodulus and the volume correction, and never looks at a single pairwise
# distance.  The entry's advice to verify dx yourself from the particle
# coordinates is the only thing that catches it.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_gravity.4C.yaml) || exit 3
grep -q "  PERIDYNAMIC_GRID_SPACING: 1.0" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "  INITIALPARTICLESPACING: 1.0" "$BASE"   || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

# Measure the lattice pitch the deck really has, the way the entry tells you to.
python3 - "$BASE" <<'PY'
import re, sys
p = []
for l in open(sys.argv[1]):
    m = re.match(r'\s*- "TYPE pdphase POS (\S+) (\S+) (\S+)', l)
    if m:
        p.append((float(m.group(1)), float(m.group(2))))
    if len(p) == 10:
        break
d = min(((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5
        for i, a in enumerate(p) for b in p[i+1:])
print("MEASURED_LATTICE_PITCH=%.2f" % d)
PY

cp "$BASE" "$TMP/matched.yaml"
sed 's/  PERIDYNAMIC_GRID_SPACING: 1.0/  PERIDYNAMIC_GRID_SPACING: 1.05/' "$BASE" > "$TMP/off5pct.yaml"
sed 's/  PERIDYNAMIC_GRID_SPACING: 1.0/  PERIDYNAMIC_GRID_SPACING: 0.5/'  "$BASE" > "$TMP/half.yaml"

probe MATCHED "$TMP/matched.yaml"
probe OFF5PCT "$TMP/off5pct.yaml"
probe HALF    "$TMP/half.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/MATCHED.log"
echo "MATCHED_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/MATCHED.log")"
grep -m1 -F "Number of initialized peridynamic bonds on this proc: 1512" "$TMP/MATCHED.log"

# The bond topology is untouched, so nothing about the run looks different.
echo "BONDS_OFF5PCT=$(grep -m1 -oE 'peridynamic bonds on this proc: [0-9]+' "$TMP/OFF5PCT.log" | grep -oE '[0-9]+$')"
echo "BONDS_HALF=$(grep -m1 -oE 'peridynamic bonds on this proc: [0-9]+' "$TMP/HALF.log" | grep -oE '[0-9]+$')"
echo "OFF5PCT_STEPS_RUN=$(grep -c 'Number of pd_neighbor_pairs in peridynamic evaluation' "$TMP/OFF5PCT.log")"
echo "HALF_STEPS_RUN=$(grep -c 'Number of pd_neighbor_pairs in peridynamic evaluation' "$TMP/HALF.log")"
echo "OFF5PCT_SPACING_WARNINGS=$(grep -ciE 'spacing|mismatch|does not match' "$TMP/OFF5PCT.log")"
echo "HALF_SPACING_WARNINGS=$(grep -ciE 'spacing|mismatch|does not match' "$TMP/HALF.log")"

# ...but the answer is gone.
echo "OFF5PCT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/OFF5PCT.log")"
grep -m1 -F "is WRONG --> actresult= 7.84206011165002970e+00" "$TMP/OFF5PCT.log"
echo "HALF_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/HALF.log")"
grep -m1 -F "is WRONG --> actresult= 7.59349841769532219e+00" "$TMP/HALF.log"
exit 0
