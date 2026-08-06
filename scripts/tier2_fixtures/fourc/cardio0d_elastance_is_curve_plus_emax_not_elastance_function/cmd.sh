#!/bin/bash
# Tier-2 for fourc::cardiovascular0d#1 — the time-varying elastance is
# E_*_min/E_*_max plus an activation CURVE. There is no ELASTANCE_FUNCTION.
#
# Claimed: "Time-varying elastance requires cardiac cycle TIMING parameters (T_S
#          systole, T_D diastole) ... For active cardiac models,
#          ELASTANCE_FUNCTION over the heart cycle is required."
# Observed, on upstream cardiovascular0d_syspulcirculation_0d_heart:
#   * there is no ELASTANCE_FUNCTION parameter, and no T_S or T_D either. Adding
#     ELASTANCE_FUNCTION to CARDIOVASCULAR 0D-STRUCTURE COUPLING/SYS-PUL
#     CIRCULATION PARAMETERS fails to match the section. The real inputs are
#     E_at_min_l/E_at_max_l, E_v_min_l/E_v_max_l (and their right-heart twins)
#     plus Atrium_act_curve_l / Ventricle_act_curve_l, which point at FUNCT
#     blocks that carry the systolic and diastolic timing.
#   * collapsing every max onto its min -- constant elastance, no contraction --
#     is accepted silently and does destroy the pumping: 16 of 24 result tests go
#     wrong, ventricular pressure p_v_l falls 1.04605 -> 0.83208 and the mitral
#     inflow q_vin_l flips from -0.03333 to +1.4856e+04.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream cardiovascular0d_syspulcirculation_0d_heart.4C.yaml) || exit 3
cd "$TMP" || exit 3
cp "$BASE" heart.yaml
for k in 'E_at_max_l: 2.9e-05' 'E_at_min_l: 9e-06' 'E_v_max_l: 7e-05' 'E_v_min_l: 1.2e-05' \
         'E_at_max_r: 1.8e-05' 'E_v_max_r: 3e-05' 'Ventricle_act_curve_l: 3'; do
  grep -q "  $k" heart.yaml || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
done

CONSTANT_E_V_MAX_L=1.2e-05

python3 - "$CONSTANT_E_V_MAX_L" <<'PY'
import sys
t = open('heart.yaml').read()
c = (t.replace('  E_v_max_l: 7e-05', '  E_v_max_l: ' + sys.argv[1])
      .replace('  E_v_max_r: 3e-05', '  E_v_max_r: 1e-05')
      .replace('  E_at_max_l: 2.9e-05', '  E_at_max_l: 9e-06')
      .replace('  E_at_max_r: 1.8e-05', '  E_at_max_r: 8e-06'))
open('constant_elastance.yaml', 'w').write(c)
open('elastance_function.yaml', 'w').write(
    t.replace('  VENTRICLE_MODEL: "0D"', '  VENTRICLE_MODEL: "0D"\n  ELASTANCE_FUNCTION: 3'))
PY

probe BASE      heart.yaml
probe CONSTANTE constant_elastance.yaml
probe ELASTFUN  elastance_function.yaml

echo "BASE_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/BASE.log")"
echo "CONSTANTE_TESTS_WRONG=$(grep -c 'is WRONG' "$TMP/CONSTANTE.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
grep -m1 -F "Result check failed with 16 errors out of 24 tests" "$TMP/CONSTANTE.log"
grep -m1 -E "p_v_l.*is WRONG --> actresult=" "$TMP/CONSTANTE.log"
grep -m1 -E "q_vin_l.*is WRONG --> actresult=" "$TMP/CONSTANTE.log"
# the parameter the entry names does not exist
grep -m1 -F "Could not match this input" "$TMP/ELASTFUN.log"
grep -m1 -F "CARDIOVASCULAR 0D-STRUCTURE COUPLING/SYS-PUL CIRCULATION PARAMETERS:" "$TMP/ELASTFUN.log"
N_EF=$(grep -c "Matched parameter 'ELASTANCE_FUNCTION'" "$TMP/ELASTFUN.log")
echo "ELASTANCE_FUNCTION_IS_A_PARAMETER=$N_EF"
echo "T_S_OR_T_D_IN_UPSTREAM_DECK=$(grep -cE '^  T_[SD]:' heart.yaml)"
echo "ACTIVATION_CURVES_IN_UPSTREAM_DECK=$(grep -cE '^  (Atrium|Ventricle)_act_curve_[lr]:' heart.yaml)"
exit 0
