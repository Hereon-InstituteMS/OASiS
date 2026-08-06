#!/bin/bash
# Tier-2 for fourc::particle_sph#4 — MAT_ParticleSPHBoundary's INITDENSITY has to
# match the fluid's, and getting it wrong is completely silent.
#
# The upstream 1-D hydrostatic deck runs AdamiBoundaryFormulation with both
# INITDENSITY set to 1.  Raise only the boundary material's to 100 — the fluid
# is untouched — and 4C accepts it without a word: no warning about the density
# ratio, no mention of the boundary formulation, "processor 0 finished normally"
# is still printed.  The wall pressure the Adami formulation extrapolates is
# then referenced to the wrong density, and the column settles to a different
# place: posx moves from 4.81167849295583760 to 4.70238984534387594, velx and
# density move with it, and three of the deck's five result tests fail.
#
# Both arms use the SAME kernel, the same boundary formulation and the same
# fluid material, so the only thing separating them is the density ratio.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_1d_hydrostatic_freesurface_densityintegration_cubicspline_adami.4C.yaml) || exit 3
grep -q 'BOUNDARYPARTICLEFORMULATION: "AdamiBoundaryFormulation"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/matched.yaml"
python3 - "$BASE" "$TMP/mismatched.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
old = """  - MAT: 2
    MAT_ParticleSPHBoundary:
      INITRADIUS: 0.2
      INITDENSITY: 1"""
assert old in t, "upstream deck no longer gives the boundary material the fluid density"
open(sys.argv[2], "w").write(t.replace(old, old.replace("INITDENSITY: 1", "INITDENSITY: 100")))
PY
# Only the boundary material moved; the fluid density is still 1 in both.
echo "FLUID_DENSITY_UNCHANGED=$( a=$(grep -c 'MAT_ParticleSPHFluid' "$TMP/mismatched.yaml"); [ "$a" = "1" ] && echo yes || echo no )"
echo "MISMATCH_INJECTED=$(grep -c 'INITDENSITY: 100' "$TMP/mismatched.yaml")"

probe MATCHED    "$TMP/matched.yaml"
probe MISMATCHED "$TMP/mismatched.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/MATCHED.log"
echo "MATCHED_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/MATCHED.log")"

# Accepted in silence: the deck parses, runs the whole time loop and reaches the
# result-test manager without a single remark about the density ratio.
echo "MISMATCH_REACHED_RESULT_TEST=$(grep -c 'is WRONG --> actresult=\|is CORRECT, abs' "$TMP/MISMATCHED.log")"
echo "MISMATCH_WARNINGS=$(grep -ciE 'densit(y|ies).*(mismatch|differ|ratio)|adami.*(warn|inconsistent)' "$TMP/MISMATCHED.log")"
# ...and it moves the wall pressure, so the column settles somewhere else.
echo "MISMATCHED_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/MISMATCHED.log")"
grep -m1 -F "is WRONG --> actresult= 4.70238984534387594e+00" "$TMP/MISMATCHED.log"
grep -m1 -E "density .*is WRONG --> actresult=" "$TMP/MISMATCHED.log"
exit 0
