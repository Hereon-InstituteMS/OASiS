#!/bin/bash
# Tier-2 for fourc::particle_sph#0 — KERNEL_SPACE_DIM must match the physical
# dimension, and the reason is bigger than kernel normalisation.
#
# Claimed:  a dimension mismatch gives "wrong kernel normalization"; the signal
#           is hydrostatic pressure divided by the 3-D normalisation and a
#           density field "deviating ~30% from INITDENSITY at the first step".
# Observed: the mismatch is accepted in silence — no warning, no note, the run
#           completes — and the deviation is nothing like 30%: on 4C's own 1-D
#           hydrostatic deck the tested density moves from 1.00015521027962739
#           to 1.00020166835008029, i.e. the excess over INITDENSITY grows by
#           about a third while the density itself stays within 0.02% of it.  An
#           agent told to look for a 30% density error will conclude the deck is
#           fine.
#
#           The mechanism worth knowing is not normalisation: KERNEL_SPACE_DIM
#           also sets the particle VOLUME.  4C computes mass as
#           INITDENSITY * INITIALPARTICLESPACING^KERNEL_SPACE_DIM, so bumping
#           1D -> 2D on a spacing of 0.1 divides every particle's mass by ten.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_1d_hydrostatic_freesurface_densityintegration_cubicspline_adami.4C.yaml) || exit 3
grep -q 'KERNEL_SPACE_DIM: "Kernel1D"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q 'INITIALPARTICLESPACING: 0.1' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/dim1.yaml"
sed 's|KERNEL_SPACE_DIM: "Kernel1D"|KERNEL_SPACE_DIM: "Kernel2D"|' "$BASE" > "$TMP/dim2.yaml"

probe KERNEL1D "$TMP/dim1.yaml"
probe KERNEL2D "$TMP/dim2.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/KERNEL1D.log"
echo "KERNEL1D_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/KERNEL1D.log")"

# The wrong dimension is accepted without a word and changes the physics.
echo "KERNEL2D_WARNINGS=$(grep -ciE 'kernel.*(dim|mismatch|inconsistent)|dimension.*mismatch' "$TMP/KERNEL2D.log")"
echo "KERNEL2D_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/KERNEL2D.log")"
grep -m1 -E "density .*is WRONG --> actresult= 1\.00020166835008029e\+00" "$TMP/KERNEL2D.log"
grep -m1 -E "velx .*is WRONG --> actresult=" "$TMP/KERNEL2D.log"

# The claimed magnitude is not what happens: the density stays inside 1%.
python3 - "$TMP/KERNEL2D.log" <<'PY'
import re, sys
for line in open(sys.argv[1]):
    if "density" in line and "actresult=" in line:
        v = float(re.search(r"actresult=\s*(-?[0-9.]+e[+-][0-9]+)", line).group(1))
        print("KERNEL2D_DENSITY_DEVIATION_FROM_INITDENSITY_PERCENT=%.4f" % (abs(v - 1.0) * 100))
        print("CLAIMED_30_PERCENT_DENSITY_DEVIATION_OBSERVED=%s" % ("yes" if abs(v - 1.0) > 0.15 else "no"))
        break
PY
exit 0
