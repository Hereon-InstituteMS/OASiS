#!/bin/bash
# Tier-2 for fourc::particle_sph#5 — non-uniform initial spacing corrupts the
# density field immediately, and 4C never mentions it.
#
# The claim is about zeroth-order kernel error at t = 0, so the deck is cut to a
# single step (NUMSTEP 1), the initial velocity field is dropped and the result
# block is replaced by one probe: density at an interior fluid particle,
# expected to equal INITDENSITY.  Everything else is upstream.
#
#   UNIFORM  : the 1-D lattice at INITRADIUS = 2*dx reproduces INITDENSITY to
#              abs(diff) = 1.11022302462515654e-16, i.e. exactly.  That is the
#              control: a correctly packed lattice is not approximately right,
#              it is bit-right.
#   JITTERED : shift alternate fluid particles by +-25% of dx — same particle
#              count, same total length, same mass per particle, because 4C takes
#              the mass from INITIALPARTICLESPACING and not from the actual
#              positions — and the very first density evaluation returns
#              1.16662840136084323e+00, a 16.7% error before any dynamics.
#
# So the entry's "more than 5%" is if anything conservative, and there is no
# diagnostic: no spacing check, no neighbour-count complaint, nothing.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_1d_pressurewave_boundary_densitysummation_cubicspline_adami.4C.yaml) || exit 3
grep -q "NUMSTEP: 2500" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q 'INITIAL_VELOCITY_FIELD: "phase1 1"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$BASE" "$TMP/uniform.yaml" "$TMP/jittered.yaml" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
head, parts = t.split("PARTICLES:\n")
head = head.replace("  NUMSTEP: 2500", "  NUMSTEP: 1").replace("  RESULTSEVERY: 100", "  RESULTSEVERY: 1")
head = head.replace('  INITIAL_VELOCITY_FIELD: "phase1 1"\n', '')
head = head[:head.index("RESULT DESCRIPTION:")] + '''RESULT DESCRIPTION:
  - PARTICLE:
      ID: 25
      QUANTITY: "density"
      VALUE: 1.0
      TOLERANCE: 1e-12
'''
lines = [l for l in parts.split("\n") if l.strip()]
open(sys.argv[2], "w").write(head + "PARTICLES:\n" + "\n".join(lines) + "\n")

out, k = [], 0
for l in lines:
    m = re.search(r'TYPE (\w+) POS +(\S+) +(\S+) +(\S+)', l)
    if m and m.group(1) == "phase1":
        x = float(m.group(2)) + (0.001 if k % 2 == 0 else -0.001)   # +-25% of dx
        k += 1
        out.append('  - "TYPE phase1 POS %.6f 0.0 0.0"' % x)
    else:
        out.append(l)
assert k > 40, "upstream deck no longer carries the 1-D fluid column"
open(sys.argv[3], "w").write(head + "PARTICLES:\n" + "\n".join(out) + "\n")
print("JITTERED_FLUID_PARTICLES=%d" % k)
PY

probe UNIFORM  "$TMP/uniform.yaml"
probe JITTERED "$TMP/jittered.yaml"

# A correctly packed lattice reproduces INITDENSITY exactly.
grep -m1 -F "processor 0 finished normally" "$TMP/UNIFORM.log"
echo "UNIFORM_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/UNIFORM.log")"
grep -m1 -F "is CORRECT, abs(diff)= 1.11022302462515654e-16" "$TMP/UNIFORM.log"

# A jittered one does not, at the very first evaluation.
echo "JITTERED_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/JITTERED.log")"
grep -m1 -F "is WRONG --> actresult= 1.16662840136084323e+00" "$TMP/JITTERED.log"
echo "JITTERED_SPACING_WARNINGS=$(grep -ciE 'spacing|non.?uniform|packing' "$TMP/JITTERED.log")"

python3 - "$TMP/JITTERED.log" <<'PY'
import re, sys
for l in open(sys.argv[1]):
    if "density" in l and "actresult=" in l:
        v = float(re.search(r"actresult=\s*(-?[0-9.]+e[+-][0-9]+)", l).group(1))
        print("JITTERED_DENSITY_ERROR_PERCENT=%.1f" % (abs(v - 1.0) * 100))
        print("DENSITY_ERROR_EXCEEDS_FIVE_PERCENT_AT_T0=%s" % ("yes" if abs(v - 1.0) > 0.05 else "no"))
        break
PY
exit 0
