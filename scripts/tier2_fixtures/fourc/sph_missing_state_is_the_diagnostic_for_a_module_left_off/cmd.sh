#!/bin/bash
# Tier-2 for fourc::particle_sph#11 — the diagnostic for a module you did not
# switch on is a message about a missing STATE, printed at the END of the run.
#
# Also separates it from the neighbouring message a genuine typo produces, so
# the two are not confused.
#
# T2_MUTATE=1 removes both edits; the deck keeps its temperature evaluation and
# its valid quantity, nothing aborts and the fixture must go red.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

BASE=$(upstream particle_sph_1d_thermalconduction_boundary_temperatureintegration_quinticspline.4C.yaml) || exit 3
grep -q 'TEMPERATUREEVALUATION: "TemperatureIntegration"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$BASE" "$TMP" "$MUTATE" <<'PY'
import sys
src, tmp, mutate = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3] == "1"
line = '  TEMPERATUREEVALUATION: "TemperatureIntegration"\n'
# module off, result test still asking for its state
open(tmp + "/nomodule.yaml", "w").write(src if mutate else src.replace(line, ""))
# a genuine typo in the QUANTITY string, module left on
open(tmp + "/typo.yaml", "w").write(
    src if mutate else src.replace('QUANTITY: "temperature"', 'QUANTITY: "temperatur"', 1))
PY

probe NOMODULE "$TMP/nomodule.yaml"
probe TYPO     "$TMP/typo.yaml"
probe BASE     "$BASE"

grep -m1 -E "^OK \(" "$TMP/BASE.log" && echo "BASE_PASSES=yes"
# A module left off: the state is missing, and it is reported at the END.
grep -m1 -F "state 'temperature' not found in container!" "$TMP/NOMODULE.log"   && echo "MODULE_OFF_REPORTS_A_MISSING_STATE=yes" || echo "MODULE_OFF_REPORTS_A_MISSING_STATE=no"
# It really is at the end: the whole time loop ran first.
echo "NOMODULE_STEPS_RUN=$(grep -c '^TIME:' "$TMP/NOMODULE.log")"
echo "NOMODULE_RAN_THE_WHOLE_LOOP=$([ "$(grep -c '^TIME:' "$TMP/NOMODULE.log")" -gt 0 ] && echo yes || echo no)"
# Nothing at setup mentioned the temperature module.
echo "NOMODULE_SETUP_DIAGNOSTICS=$(sed -n '1,/^TIME:/p' "$TMP/NOMODULE.log" | grep -ciE 'temperature|thermal')"
# A genuine typo gives a DIFFERENT message, so the two cannot be confused.
grep -m1 -F "result check failed with unknown quantity" "$TMP/TYPO.log"   && echo "TYPO_GIVES_A_DIFFERENT_MESSAGE=yes" || echo "TYPO_GIVES_A_DIFFERENT_MESSAGE=no"
echo "TYPO_ALSO_SAYS_STATE_NOT_FOUND=$(grep -c 'not found in container' "$TMP/TYPO.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
exit 0
