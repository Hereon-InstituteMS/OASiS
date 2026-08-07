#!/bin/bash
# Tier-2 for fourc::particle_pd#13 — PD_DIMENSION is a constitutive choice, not a
# label, and 4C will not tell you if you pick the wrong one.
#
# The entry gives two critical-stretch formulas, one for plane strain and one
# for plane stress, and warns that mixing them puts fracture initiation at the
# wrong load.  The half an agent can actually check in a log is upstream of the
# stretch formula: the SAME CRITICAL_STRETCH and the SAME horizon under
# Peridynamic_2DPlaneStrain and Peridynamic_2DPlaneStress are two different
# problems, because the bond micromodulus is normalised by 5/(pi*delta^3) in
# plane strain and 6/(pi*delta^3) in plane stress.
#
# Flipping only that one enum on the upstream 2-D PD deck:
#   * initialises the identical 1512 bonds, so nothing about the discretisation
#     or the neighbourhood changed and no count betrays it;
#   * emits no warning of any kind;
#   * and still breaks eight of the deck's ten result tests.
#
# So the enum is silent, geometric diagnostics cannot catch it, and the only
# evidence is the answer itself.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_gravity.4C.yaml) || exit 3
grep -q "PD_DIMENSION: Peridynamic_2DPlaneStrain" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "CRITICAL_STRETCH: 295262.0" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/strain.yaml"
sed 's|PD_DIMENSION: Peridynamic_2DPlaneStrain|PD_DIMENSION: Peridynamic_2DPlaneStress|' "$BASE" > "$TMP/stress.yaml"

probe PLANESTRAIN "$TMP/strain.yaml"
probe PLANESTRESS "$TMP/stress.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/PLANESTRAIN.log"
echo "PLANESTRAIN_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/PLANESTRAIN.log")"

# Identical geometry: same neighbourhood, same bond count.
grep -m1 -F "Number of initialized peridynamic bonds on this proc: 1512" "$TMP/PLANESTRAIN.log"
grep -m1 -F "Number of initialized peridynamic bonds on this proc: 1512" "$TMP/PLANESTRESS.log"
echo "BOND_COUNTS_DIFFER=$( a=$(grep -m1 -oE 'bonds on this proc: [0-9]+' "$TMP/PLANESTRAIN.log"); \
  b=$(grep -m1 -oE 'bonds on this proc: [0-9]+' "$TMP/PLANESTRESS.log"); \
  [ "$a" = "$b" ] && echo no || echo yes )"

# ...but a different constitutive problem, and 4C never says so.
echo "PLANESTRESS_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/PLANESTRESS.log")"
grep -m1 -F "Result check failed with 8 errors out of 10 tests" "$TMP/PLANESTRESS.log"
echo "PLANESTRESS_WARNINGS=$(grep -ciE 'plane (stress|strain).*(warn|mismatch|inconsistent|check)' "$TMP/PLANESTRESS.log")"
grep -m1 -E "posx .*is WRONG --> actresult=" "$TMP/PLANESTRESS.log"
exit 0
