#!/bin/bash
# Tier-2 for fourc::particles#2 — a PD lattice with different spacing in x and y
# is accepted without a word and changes the answer.
#
# Upstream particle_sph_2d_pdbody_gravity.4C.yaml packs its 162 pdphase
# particles on a square lattice at dx = dy = 1.0, declares
# PERIDYNAMIC_GRID_SPACING: 1.0 and INTERACTION_HORIZON: 3.0.  The mutant keeps
# the same 162 particles and the same declared spacing but halves every pdphase
# y-coordinate, so the lattice becomes dx = 1.0, dy = 0.5: the horizon now
# reaches 3 particle rows in x and 6 in y.
#
# 4C says nothing.  No spacing check, no packing check, no neighbour-count
# complaint — the only trace is that the bond list grows by more than half,
# because each particle acquires twice as many partners along y as along x while
# every one of those bonds is still weighted with the single scalar dx^4 volume
# correction.  That is the anisotropy the entry warns about, in countable form.
# The run then completes and fails 8 of its 10 result tests.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_gravity.4C.yaml) || exit 3
grep -q "  INTERACTION_HORIZON: 3.0" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "  PERIDYNAMIC_GRID_SPACING: 1.0" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$BASE" "$TMP/uniform.yaml" "$TMP/aniso.yaml" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
out, n = [], 0
for l in t.split("\n"):
    m = re.match(r'(\s*- "TYPE pdphase POS )(\S+) (\S+) (\S+)(.*)$', l)
    if m:
        y = float(m.group(3)) * 0.5          # halve only the y spacing
        n += 1
        out.append("%s%s %.10E %s%s" % (m.group(1), m.group(2), y, m.group(4), m.group(5)))
    else:
        out.append(l)
assert n > 100, "upstream deck no longer carries the pdphase lattice"
open(sys.argv[2], "w").write(t)
open(sys.argv[3], "w").write("\n".join(out))
print("PDPHASE_PARTICLES=%d" % n)
# horizon 3.0 over the two lattice pitches: the neighbourhood is no longer round
print("NEIGHBOUR_ROWS_X=%d" % int(3.0 / 1.0))
print("NEIGHBOUR_ROWS_Y=%d" % int(3.0 / 0.5))
PY

probe UNIFORM "$TMP/uniform.yaml"
probe ANISO   "$TMP/aniso.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/UNIFORM.log"
echo "UNIFORM_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/UNIFORM.log")"
grep -m1 -oE "Number of initialized peridynamic bonds on this proc: [0-9]+" "$TMP/UNIFORM.log" | sed 's/.*: /BONDS_UNIFORM=/'
grep -m1 -oE "Number of initialized peridynamic bonds on this proc: [0-9]+" "$TMP/ANISO.log"   | sed 's/.*: /BONDS_ANISO=/'

# Same particle count, same declared PERIDYNAMIC_GRID_SPACING, far more bonds:
# the neighbourhood has become elliptical and 4C never mentions it.
echo "ANISO_SPACING_WARNINGS=$(grep -ciE 'spacing|non.?uniform|packing|anisotrop' "$TMP/ANISO.log")"
echo "ANISO_STEPS_RUN=$(grep -c 'Number of pd_neighbor_pairs in peridynamic evaluation' "$TMP/ANISO.log")"
echo "ANISO_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/ANISO.log")"
grep -m1 -F "is WRONG --> actresult= 8.02483754744231881e+00" "$TMP/ANISO.log"
exit 0
