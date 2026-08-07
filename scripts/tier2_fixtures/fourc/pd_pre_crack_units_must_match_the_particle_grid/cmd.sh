#!/bin/bash
# Tier-2 for fourc::particle_pd#5 — PRE_CRACKS coordinates in the wrong units
# are accepted without complaint and break (almost) nothing.
#
# NOTE ON PROVENANCE.  PRE_CRACKS and the deck used here,
# particle_sph_2d_pdbody_precrack_fixedflag.4C.yaml, are NOT in upstream 4C
# main.  They come from local branch work on bond-based peridynamics in the
# checkout this was run against.  The behaviour below is real and reproducible,
# but it is branch behaviour, not upstream behaviour, and the fixture aborts
# loudly rather than silently passing if the deck is absent.
#
# The deck cuts a horizontal pre-crack from (-5, 0) to (0, 0) through a 10 x 10
# plate of unit-spaced particles.  4C reports "Number of pre-crack segments: 1"
# and the initialised bond count drops from 1058 (no crack) to 974, i.e. 84
# bonds are broken at t = 0 and the probe particle reports pd_damage_phi = 0.357.
#
# Rescale only the crack endpoints by 1e-3, the classic mm-vs-m slip, and every
# one of those signals except the damage stays reassuring: the segment is still
# parsed, "Number of pre-crack segments: 1" is still printed, no warning is
# emitted — and the bond count comes back 1056, so 2 bonds are broken instead of
# 84 and the damage field is flat zero.  The only way to notice is to compare
# the bond count against the no-crack run, which is what this fixture does.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_precrack_fixedflag.4C.yaml) || exit 3
grep -q 'PRE_CRACKS: "-5.0 0.0 0.0 0.0"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/right.yaml"
sed 's|PRE_CRACKS: "-5.0 0.0 0.0 0.0"|PRE_CRACKS: "-0.005 0.0 0.0 0.0"|' "$BASE" > "$TMP/wrong.yaml"
grep -v 'PRE_CRACKS' "$BASE" > "$TMP/none.yaml"

probe RIGHTUNITS "$TMP/right.yaml"
probe WRONGUNITS "$TMP/wrong.yaml"
probe NOCRACK    "$TMP/none.yaml"

bonds() { grep -m1 -oE 'Number of initialized peridynamic bonds on this proc: [0-9]+' "$1" | grep -oE '[0-9]+$'; }
B_RIGHT=$(bonds "$TMP/RIGHTUNITS.log"); B_WRONG=$(bonds "$TMP/WRONGUNITS.log"); B_NONE=$(bonds "$TMP/NOCRACK.log")

grep -m1 -F "processor 0 finished normally" "$TMP/RIGHTUNITS.log"
echo "RIGHTUNITS_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/RIGHTUNITS.log")"
echo "BONDS_NOCRACK=$B_NONE"
echo "BONDS_RIGHTUNITS=$B_RIGHT"
echo "BONDS_WRONGUNITS=$B_WRONG"
echo "BROKEN_RIGHTUNITS=$((B_NONE - B_RIGHT))"
echo "BROKEN_WRONGUNITS=$((B_NONE - B_WRONG))"

# The wrong-units deck still looks accepted: same segment count, no warning.
grep -m1 -F "Number of pre-crack segments: 1" "$TMP/WRONGUNITS.log"
echo "WRONGUNITS_WARNINGS=$(grep -ciE 'pre.?crack.*(ignor|warn|outside|no bond|invalid)' "$TMP/WRONGUNITS.log")"
# ...but the damage field is flat zero and the deck's own tests fail.
grep -m1 -E "pd_damage_phi.*is WRONG --> actresult= 0\.00000000000000000e\+00" "$TMP/WRONGUNITS.log"
echo "WRONGUNITS_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/WRONGUNITS.log")"
exit 0
