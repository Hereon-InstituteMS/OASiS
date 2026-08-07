#!/bin/bash
# Tier-2 for fourc::cardiovascular0d#3 — the 0D initial pressures are not cosmetic
# and there is no warm-up detection.
#
# Claim: "Initial conditions: set initial pressures in the 0D model ... default
#        zero pressure with a physiological elastance gives a transient that takes
#        5-10 cardiac cycles to stabilise."
# Observed, on upstream cardiovascular0d_syspulcirculation_0d_heart, whose
# SYS-PUL CIRCULATION PARAMETERS carry a full set of p_*_0 seeds (p_at_l_0,
# p_v_l_0, p_ar_sys_0, ...): zeroing the left-heart and systemic-arterial seeds
# and running exactly the same 0.9 s cardiac cycle leaves 16 of the 24 result
# tests wrong. Ventricular pressure p_v_l ends at 0.319739 instead of 1.046055 --
# a third of the converged value after a full cycle -- and mitral inflow q_vin_l
# goes from -0.033332 to +2.6255e+04. The run converges cleanly every step and
# says nothing: there is no "not yet periodic" or "still in warm-up" check.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream cardiovascular0d_syspulcirculation_0d_heart.4C.yaml) || exit 3
cd "$TMP" || exit 3
cp "$BASE" base.yaml
for k in 'p_at_l_0: 0.599950804034' 'p_v_l_0: 0.599950804034' 'p_ar_sys_0: 9.68378038166'; do
  grep -q "  $k" base.yaml || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
done

ZEROED_P_AR_SYS_0=0.0

python3 - "$ZEROED_P_AR_SYS_0" <<'PY'
import sys
t = open('base.yaml').read()
open('zeroed.yaml', 'w').write(
    t.replace('  p_at_l_0: 0.599950804034', '  p_at_l_0: 0.0')
     .replace('  p_v_l_0: 0.599950804034', '  p_v_l_0: 0.0')
     .replace('  p_ar_sys_0: 9.68378038166', '  p_ar_sys_0: ' + sys.argv[1]))
PY

probe BASE   base.yaml
probe ZEROED zeroed.yaml

echo "BASE_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/BASE.log")"
echo "ZEROED_TESTS_WRONG=$(grep -c 'is WRONG' "$TMP/ZEROED.log")"
echo "ZEROED_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/ZEROED.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
grep -m1 -E "p_v_l.*is WRONG --> actresult=" "$TMP/ZEROED.log"
grep -m1 -E "p_ar_sys.*is WRONG --> actresult=" "$TMP/ZEROED.log"
python3 - "$TMP" <<'PY'
import re, sys
for line in open(sys.argv[1] + '/ZEROED.log'):
    m = re.search(r'p_v_l\s+is WRONG --> actresult=\s*(\S+),', line)
    if m:
        print("ZEROED_VENTRICULAR_PRESSURE_FRACTION_OF_SETTLED=%.2f"
              % (float(m.group(1)) / 1.0460547554921127))
        break
else:
    raise SystemExit("no p_v_l failure line")
PY
echo "WARMUP_DIAGNOSTICS=$(grep -ciE 'not periodic|warm-?up|transient not|initial condition' "$TMP/ZEROED.log")"
exit 0
