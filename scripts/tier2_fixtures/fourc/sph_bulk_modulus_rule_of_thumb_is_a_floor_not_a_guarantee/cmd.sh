#!/bin/bash
# Tier-2 for fourc::particle_sph#3 — BULK_MODULUS >= 100 * rho * v_max^2 does NOT
# on its own keep the density inside 1%.
#
# The upstream 1-D pressure-wave deck is the ideal test: rho = 1, the initial
# velocity field peaks at v_max = 2 * 0.0005 = 1e-3, so the entry's own rule of
# thumb evaluates to BULK_MODULUS >= 100 * 1 * (1e-3)^2 = 1e-4.  The deck ships
# BULK_MODULUS = 1e-2, a hundredfold above that floor.
#
#   AMPLE  (1e-2, Mach 0.01) : density 1.00395124660444535 -> 0.40% excess, passes
#   ATRULE (1e-4, Mach 0.10) : density drops to 9.91447028694634080e-01 at one
#                              tested particle and 9.62504573511519079e-01 at the
#                              other, i.e. 0.9% and 3.7% off INITDENSITY
#
# So sitting exactly ON the quoted threshold already violates the 1% target the
# same sentence promises, by nearly a factor of four.  100 is a floor to clear,
# not a value to design to.  4C says nothing either way — no Mach number, no
# density-variation warning; the only reason this is visible at all is that the
# deck result-tests density directly.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_1d_pressurewave_boundary_densitysummation_cubicspline_adami.4C.yaml) || exit 3
grep -q "BULK_MODULUS: 0.01" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "INITDENSITY: 1" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "0.0005\*(1+cos" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/ample.yaml"
sed 's/      BULK_MODULUS: 0.01/      BULK_MODULUS: 0.0001/' "$BASE" > "$TMP/atrule.yaml"

probe AMPLE  "$TMP/ample.yaml"
probe ATRULE "$TMP/atrule.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/AMPLE.log"
echo "AMPLE_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/AMPLE.log")"

echo "ATRULE_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/ATRULE.log")"
grep -m1 -F "is WRONG --> actresult= 9.91447028694634080e-01" "$TMP/ATRULE.log"
grep -m1 -F "is WRONG --> actresult= 9.62504573511519079e-01" "$TMP/ATRULE.log"
# 4C never reports a Mach number or a density-variation check.
echo "ATRULE_COMPRESSIBILITY_WARNINGS=$(grep -ciE 'mach|density variation|compressib' "$TMP/ATRULE.log")"

python3 - "$TMP/AMPLE.log" "$TMP/ATRULE.log" <<'PY'
import re, sys
def dens(p):
    out = []
    for l in open(p):
        if "density" in l and "actresult=" in l:
            out.append(float(re.search(r"actresult=\s*(-?[0-9.]+e[+-][0-9]+)", l).group(1)))
        elif "density" in l and "is CORRECT" in l:
            out.append(None)
    return out
# the ample arm passes, so read its reference values from the deck instead
worst = max(abs(v - 1.0) for v in dens(sys.argv[2]) if v is not None)
print("ATRULE_WORST_DENSITY_EXCURSION_PERCENT=%.2f" % (worst * 100))
print("RULE_OF_THUMB_KEEPS_DENSITY_INSIDE_ONE_PERCENT=%s" % ("yes" if worst < 0.01 else "no"))
PY
exit 0
