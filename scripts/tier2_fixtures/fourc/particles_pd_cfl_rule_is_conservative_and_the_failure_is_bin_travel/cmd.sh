#!/bin/bash
# Tier-2 for fourc::particles#8 — the CFL formula is not the stability limit, and
# the failure mode is not a NaN.
#
# Claimed:  "CFL condition for PD: dt < 0.5 * dx / c_wave where
#           c_wave = sqrt(E/rho).  Signal: dt > CFL gives NaN within ~10 time
#           steps (typical 'energy not conserved' message)."
# Observed: 4C's own regression deck breaks that bound and passes.  Upstream
#           particle_sph_2d_pdbody_gravity.4C.yaml has YOUNG 190.0e3,
#           INITDENSITY 8000.0e-9 and dx 1.0, so the formula gives a limit of
#           3.24e-06 s — and the deck runs at 8.0e-06 s, 2.47 times larger, for
#           3000 steps with all ten result tests exact.  Raising dt to 1.0e-05,
#           3.08 times the "limit", still completes every step.  The real
#           boundary is somewhere above three times the quoted one, so an agent
#           that applies this rule literally will shrink a working time step by
#           a factor of three and pay for it.
#
#           When dt really is too large the run does not produce a NaN and does
#           not mention energy.  At 1.6e-04 it is killed after 8 steps by
#             a particle of phase 'pdphase' traveled more than one bin on this
#             processor!
#           from particle/src/algorithm/4C_particle_algorithm.cpp.  No NaN
#           appears anywhere in any of the logs, and the string "energy not
#           conserved" does not exist in 4C.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_gravity.4C.yaml) || exit 3
grep -q "  TIMESTEP: 8.0e-6" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

# Evaluate the entry's own formula against the deck's own material data.
python3 - "$BASE" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
E   = float(re.search(r"YOUNG:\s*(\S+)", t).group(1))
rho = float(re.search(r"INITDENSITY:\s*(8000\S+)", t).group(1))
dx  = float(re.search(r"PERIDYNAMIC_GRID_SPACING:\s*(\S+)", t).group(1))
dt  = float(re.search(r"TIMESTEP:\s*(\S+)", t).group(1))
c   = (E / rho) ** 0.5
lim = 0.5 * dx / c
print("CLAIMED_CFL_LIMIT_SECONDS=%.2e" % lim)
print("UPSTREAM_TIMESTEP_OVER_CLAIMED_LIMIT=%.2f" % (dt / lim))
print("STABLE_TIMESTEP_1E5_OVER_CLAIMED_LIMIT=%.2f" % (1.0e-5 / lim))
PY

cp "$BASE" "$TMP/upstream.yaml"
sed 's/  TIMESTEP: 8.0e-6/  TIMESTEP: 1.0e-5/' "$BASE" > "$TMP/dt1e5.yaml"
sed 's/  TIMESTEP: 8.0e-6/  TIMESTEP: 1.6e-4/' "$BASE" > "$TMP/dt16e4.yaml"

probe UPSTREAM "$TMP/upstream.yaml"
probe DT1E5    "$TMP/dt1e5.yaml"
probe DT16E4   "$TMP/dt16e4.yaml"

# The shipped deck sits well above the quoted limit and is exact.
grep -m1 -F "processor 0 finished normally" "$TMP/UPSTREAM.log"
echo "UPSTREAM_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/UPSTREAM.log")"
echo "UPSTREAM_EXACT_ZERO_DIFFS=$(grep -c 'is CORRECT, abs(diff)= 0.00000000000000000e+00' "$TMP/UPSTREAM.log")"
echo "UPSTREAM_STEPS_RUN=$(grep -c 'Number of pd_neighbor_pairs in peridynamic evaluation' "$TMP/UPSTREAM.log")"

# Three times the quoted limit is still stable: every step is taken.
echo "DT1E5_STEPS_RUN=$(grep -c 'Number of pd_neighbor_pairs in peridynamic evaluation' "$TMP/DT1E5.log")"
echo "DT1E5_TRAVEL_ABORTS=$(grep -c 'traveled more than one bin' "$TMP/DT1E5.log")"

# When it does break, this is what you actually see.
grep -m1 -F "a particle of phase 'pdphase' traveled more than one bin on this processor!" "$TMP/DT16E4.log"
grep -m1 -oE "4C_particle_algorithm\.cpp, line [0-9]+" "$TMP/DT16E4.log"
echo "DT16E4_STEPS_BEFORE_ABORT=$(grep -c 'Number of pd_neighbor_pairs in peridynamic evaluation' "$TMP/DT16E4.log")"
echo "NAN_IN_ANY_LOG=$(cat "$TMP"/UPSTREAM.log "$TMP"/DT1E5.log "$TMP"/DT16E4.log | grep -c -i 'nan')"
echo "CLAIMED_ENERGY_NOT_CONSERVED_TEXT=$(cat "$TMP"/UPSTREAM.log "$TMP"/DT1E5.log "$TMP"/DT16E4.log | grep -ci 'energy not conserved')"
echo "DT16E4_DIAGNOSTIC_MENTIONS_CFL_OR_TIMESTEP=$(grep -ciE 'traveled more than one bin.*(cfl|time step|timestep)' "$TMP/DT16E4.log")"
exit 0
