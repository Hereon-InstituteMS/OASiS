#!/bin/bash
# Tier-2 for fourc::arterial_network#4 — blood viscosity does not touch the pulse
# wave speed. It changes the flow.
#
# Claimed: "using VISCOSITY = 1e-3 Pa s (water) gives wave speed off by ~20% from
#          physiological value".
# Observed, on upstream one_d_3_artery_network shortened to 50 steps: 4C's own CFL
# abort doubles as a wave-speed meter, since the number it prints is
# sqrt(3)*|lambda|_max*dt/dx and lambda_max is dominated by the wave speed. Run
# the same deck at a CFL-violating step with blood viscosity 0.04 and with water
# 0.01 and the abort message is IDENTICAL to the last digit:
#     CFL number at element 0 is 2.0848661028149182
# because c = sqrt(sqrt(pi)*E*th/(1-nu^2)*sqrt(A)/(2*A0*rho)) has no viscosity in
# it (art_net/4C_art_net_artery_ele_calc_lin_exp.cpp). The wave-speed error is 0%,
# not 20%. What viscosity does change is the friction: at a stable step the
# distal flowrate moves 0.928575 -> 0.936481.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream one_d_3_artery_network.4C.yaml) || exit 3
cd "$TMP" || exit 3
grep -q '  NUMSTEP: 10000' "$BASE"    || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '  TIMESTEP: 0.0001' "$BASE"  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '      VISCOSITY: 0.04' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

WATER_VISCOSITY=0.01

sed 's/  NUMSTEP: 10000/  NUMSTEP: 50/' "$BASE" > blood.yaml
sed "s/      VISCOSITY: 0.04/      VISCOSITY: $WATER_VISCOSITY/" blood.yaml > water.yaml
sed 's/  TIMESTEP: 0.0001/  TIMESTEP: 0.002/' blood.yaml > blood_cfl.yaml
sed 's/  TIMESTEP: 0.0001/  TIMESTEP: 0.002/' water.yaml > water_cfl.yaml

probe BLOOD     blood.yaml
probe WATER     water.yaml
probe BLOOD_CFL blood_cfl.yaml
probe WATER_CFL water_cfl.yaml

# 4C's CFL abort prints the wave-speed-dominated eigenvalue. Same for both fluids.
grep -m1 -F "CFL number at element 0 is" "$TMP/BLOOD_CFL.log"
grep -m1 -F "CFL number at element 0 is" "$TMP/WATER_CFL.log"
BC=$(grep -m1 -o 'CFL number at element 0 is [0-9.]*' "$TMP/BLOOD_CFL.log")
WC=$(grep -m1 -o 'CFL number at element 0 is [0-9.]*' "$TMP/WATER_CFL.log")
if [ "$BC" = "$WC" ] && [ -n "$BC" ]; then
  echo "VERDICT: VISCOSITY_CHANGES_THE_WAVE_SPEED=no"
else
  echo "VERDICT: VISCOSITY_CHANGES_THE_WAVE_SPEED=yes"
fi
# It does change the flow.
grep -m1 -F "flowrate at node  10" "$TMP/BLOOD.log"
grep -m1 -F "flowrate at node  10" "$TMP/WATER.log"
python3 - "$TMP" <<'PY'
import re, sys
def val(p):
    for line in open(p):
        m = re.search(r'flowrate at node\s+10\s+is WRONG --> actresult=\s*(\S+),', line)
        if m:
            return float(m.group(1))
    raise SystemExit("no distal flowrate line in " + p)
b, w = val(sys.argv[1] + '/BLOOD.log'), val(sys.argv[1] + '/WATER.log')
print("VISCOSITY_FLOW_CHANGE_PERCENT=%.2f" % (100.0 * (w - b) / b))
print("VISCOSITY_CHANGES_THE_FLOW=%s" % ("yes" if b != w else "no"))
PY
exit 0
