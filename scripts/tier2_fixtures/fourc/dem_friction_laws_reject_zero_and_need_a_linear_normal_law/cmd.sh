#!/bin/bash
# Tier-2 for fourc::particle_dem#6 (and #7, which it also exercises) — the
# tangential law needs a LINEAR normal law, and a friction coefficient of ZERO
# is rejected rather than honoured.
#
# T2_MUTATE=1 removes every edit, so all probe decks are the untouched upstream
# one, nothing aborts and the fixture must go red.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

TAN=$(upstream particle_dem_2d_tangentialcontact_linspringdamp.4C.yaml) || exit 3
ROL=$(upstream particle_dem_2d_rollingcontact_coulomb.4C.yaml)          || exit 3
grep -q 'FRICT_COEFF_TANG: 0.2' "$TAN" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$TAN" "$ROL" "$TMP" "$MUTATE" <<'PY'
import sys
tan, rol = open(sys.argv[1]).read(), open(sys.argv[2]).read()
tmp, mutate = sys.argv[3], sys.argv[4] == "1"
def w(n, src, t): open("%s/%s.yaml" % (tmp, n), "w").write(src if mutate else t)
# a non-linear normal law under the tangential one
w("hertz", tan, tan.replace('  NORMALCONTACTLAW: "NormalLinearSpringDamp"',
                            '  NORMALCONTACTLAW: "NormalHertz"')
                   .replace("  COEFF_RESTITUTION: 0.8\n", "")
                   .replace("  DAMP_REG_FAC: 0.2\n", ""))
# friction switched off the way people expect: set it to zero
w("zerotang", tan, tan.replace("  FRICT_COEFF_TANG: 0.2", "  FRICT_COEFF_TANG: 0.0"))
# and the rolling twin. The DEM section value only, NOT the wall material's.
head, sep, rest = rol.partition("FUNCT1:")
w("zeroroll", rol, head.replace("  FRICT_COEFF_ROLL: 0.5", "  FRICT_COEFF_ROLL: 0.0") + sep + rest)
# POISSON_RATIO is validated by the tangential handler, not by any normal law
w("nopoisson", tan, tan.replace("  POISSON_RATIO: 0.3\n", ""))
PY

probe HERTZ     "$TMP/hertz.yaml"
probe ZEROTANG  "$TMP/zerotang.yaml"
probe ZEROROLL  "$TMP/zeroroll.yaml"
probe NOPOISSON "$TMP/nopoisson.yaml"
probe TANBASE   "$TAN"

grep -m1 -F "tangential contact law only valid with linear normal contact law!" "$TMP/HERTZ.log"   && echo "NONLINEAR_NORMAL_REFUSED=yes" || echo "NONLINEAR_NORMAL_REFUSED=no"
grep -m1 -F "invalid input parameter FRICT_COEFF_TANG for this kind of contact law!" "$TMP/ZEROTANG.log"   && echo "ZERO_TANG_FRICTION_REFUSED=yes" || echo "ZERO_TANG_FRICTION_REFUSED=no"
grep -m1 -F "invalid input parameter FRICT_COEFF_ROLL for this kind of contact law!" "$TMP/ZEROROLL.log"   && echo "ZERO_ROLL_FRICTION_REFUSED=yes" || echo "ZERO_ROLL_FRICTION_REFUSED=no"
grep -m1 -F "invalid input parameter POISSON_RATIO (expected in range ]-1.0; 0.5])!" "$TMP/NOPOISSON.log"   && echo "POISSON_IS_A_FRICTION_KEY=yes" || echo "POISSON_IS_A_FRICTION_KEY=no"
# The refusal is worded as if the value were malformed, not out of range, and
# is identical whether the key is zero or absent.
echo "ZERO_AND_ABSENT_GIVE_ONE_MESSAGE=$( { grep -m1 -F 'invalid input parameter FRICT_COEFF_TANG' "$TMP/ZEROTANG.log"; } | wc -l)"
# All four abort at setup, before the first step.
echo "STEPS_BEFORE_ABORT=$(cat "$TMP/HERTZ.log" "$TMP/ZEROTANG.log" "$TMP/ZEROROLL.log" "$TMP/NOPOISSON.log" | grep -c '^TIME:')"
grep -m1 -F "processor 0 finished normally" "$TMP/TANBASE.log" && echo "TANBASE_CLEAN=yes"
exit 0
