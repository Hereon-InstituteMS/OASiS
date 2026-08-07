#!/bin/bash
# Tier-2 for fourc::thermal#6 — you cannot confirm a thermal tolerance was read
# by looking for it in the log, because the thermal integrator is not NOX and
# never prints one.  Three probes.
#
# ECHO   upstream tsi_lincompression_1waydisp with four DISTINCTIVE tolerances
#        injected (structural TOLRES 3.7e-07 / TOLDISP 4.2e-09, thermal TOLTEMP
#        5.3e-09 / TOLRES 6.1e-07).  Both fields run in the same log.  The NOX
#        status block echoes the structural pair back verbatim:
#            Converged....Structure-Update-Norm = 0.000e+00 < 4.2e-09
#            Converged....Structure-F-Norm = 0.000e+00 < 3.700e-07
#        The thermo block prints only
#            Predictor thermo absolute res-norm <r>
#            numiter      abs-res-norm     abs-temp-norm           wct
#        and neither thermal number appears anywhere in the file.
#
# TIGHT  upstream thermo3D_FBC_ost with TOLTEMP 1e-30 and MAXITER 3.  The only
#        place a thermal tolerance surfaces is the failure:
#            Newton unconverged in 3 iterations   (thermo_timint_impl.cpp)
#
# COMBI  the same deck with NORMCOMBI_RESFTEMP: "Or" converges in fewer Newton
#        steps than the baseline — 1 per step against 2 — which is the other
#        behavioural way to confirm the keys were read.
. "$(dirname "$0")/../_lib/preamble.sh"

TSI=$(upstream tsi_lincompression_1waydisp.4C.yaml) || exit 3
OST=$(upstream thermo3D_FBC_ost.4C.yaml) || exit 3

python3 - "$TSI" "$TMP/echo.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
old = "  M_DAMP: 0.5\n  TOLRES: 2e-08\n"
if old not in t or "THERMAL DYNAMIC:\n" not in t:
    print("FIXTURE_ABORT=upstream_deck_changed"); sys.exit(3)
t = t.replace(old, "  M_DAMP: 0.5\n  TOLRES: 3.7e-07\n  TOLDISP: 4.2e-09\n")
t = t.replace("THERMAL DYNAMIC:\n",
              "THERMAL DYNAMIC:\n  TOLTEMP: 5.3e-09\n  TOLRES: 6.1e-07\n", 1)
open(sys.argv[2], "w").write(t)
PY
[ -s "$TMP/echo.yaml" ] || exit 3

python3 - "$OST" "$TMP/tight.yaml" "$TMP/combi.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
if "  LINEAR_SOLVER: 1\n" not in t:
    print("FIXTURE_ABORT=upstream_deck_changed"); sys.exit(3)
open(sys.argv[2], "w").write(
    t.replace("  LINEAR_SOLVER: 1\n",
              "  LINEAR_SOLVER: 1\n  TOLTEMP: 1e-30\n  MAXITER: 3\n", 1))
open(sys.argv[3], "w").write(
    t.replace("  LINEAR_SOLVER: 1\n",
              "  LINEAR_SOLVER: 1\n  NORMCOMBI_RESFTEMP: \"Or\"\n", 1))
PY
[ -s "$TMP/tight.yaml" ] && [ -s "$TMP/combi.yaml" ] || exit 3

cp "$OST" "$TMP/base.yaml"
probe ECHO  "$TMP/echo.yaml"
probe BASE  "$TMP/base.yaml"
probe TIGHT "$TMP/tight.yaml"
probe COMBI "$TMP/combi.yaml"

# The structural field hands your tolerances straight back.
grep -m1 -F "Converged....Structure-Update-Norm = 0.000e+00 < 4.2e-09" "$TMP/ECHO.log"
grep -m1 -F "Converged....Structure-F-Norm = 0.000e+00 < 3.700e-07" "$TMP/ECHO.log"
# The thermal field never mentions its own.
echo "THERMAL_TOLTEMP_IN_LOG=$(grep -cE '5\.3e-09|5\.300e-09' "$TMP/ECHO.log")"
echo "THERMAL_TOLRES_IN_LOG=$(grep -cE '6\.1e-07|6\.100e-07' "$TMP/ECHO.log")"
echo "THERMAL_KEYWORDS_IN_LOG=$(grep -ciE 'toltemp|normcombi_resftemp' "$TMP/ECHO.log")"
grep -m1 -oF "numiter      abs-res-norm     abs-temp-norm           wct" "$TMP/ECHO.log"
grep -m1 -oE "Predictor thermo absolute res-norm" "$TMP/ECHO.log"

# The one place a thermal tolerance does surface.
grep -m1 -F "Newton unconverged in 3 iterations" "$TMP/TIGHT.log"
grep -m1 -oF "4C_thermo_timint_impl.cpp" "$TMP/TIGHT.log"
# And the behavioural confirmation that NORMCOMBI_RESFTEMP was read.
echo "BASE_STEPS_AT_NUMITER_1=$(grep -c 'numiter 1' "$TMP/BASE.log")"
echo "BASE_STEPS_AT_NUMITER_2=$(grep -c 'numiter 2' "$TMP/BASE.log")"
echo "COMBI_OR_STEPS_AT_NUMITER_1=$(grep -c 'numiter 1' "$TMP/COMBI.log")"
echo "COMBI_OR_STEPS_AT_NUMITER_2=$(grep -c 'numiter 2' "$TMP/COMBI.log")"
exit 0
