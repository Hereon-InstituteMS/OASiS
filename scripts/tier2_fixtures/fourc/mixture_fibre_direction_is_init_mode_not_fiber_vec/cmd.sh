#!/bin/bash
# Tier-2 for fourc::mixture#1 — the advice is right, the key it names is invented.
#
# Claimed: "Use a per-element FIBER_VEC vector field" instead of "a uniform
#          FIBER_VEC across all elements".
#
# Observed: there is no FIBER_VEC anywhere in 4C.  Anisotropic summands such as
# ELAST_CoupAnisoExpo select their fibre source with the integer INIT, and the
# per-element data goes on the ELEMENT LINE as FIBER1.  Four arms:
#
#   ANGLE     upstream as shipped: INIT: 0, direction from the GAMMA angle
#   FIBERVEC  add FIBER_VEC to the material -> "Could not match this input"
#             from global_data/4C_global_data_read.cpp: the key does not exist
#   INIT1BARE INIT: 1 (per-element fibres) with no FIBER1 on any element ->
#             "Could not find element coordinate system or element fibers!"
#             from mat/4C_mat_anisotropy_extension_default.cpp
#   INIT1FIB  INIT: 1 with FIBER1 written on every element line -> runs, and
#             every result test moves, which is the real point: the fibre
#             direction is per-element data and it changes the answer.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream mixture_elast_hyper_dynamic.4C.yaml) || exit 3
grep -q "      INIT: 0" "$BASE"                      || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "ELAST_CoupAnisoExpo" "$BASE"                || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "MAT 1 KINEM nonlinear" "$BASE"              || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/angle.yaml"
python3 - "$BASE" "$TMP/fibervec.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
anchor = "      STR_TENS_ID: 1000\n      INIT: 0\n"
if anchor not in t:
    print("FIXTURE_ABORT=upstream_deck_changed"); sys.exit(3)
open(sys.argv[2], "w").write(
    t.replace(anchor, anchor + "      FIBER_VEC: [1, 0, 0]\n", 1))
PY
[ -f "$TMP/fibervec.yaml" ] || exit 3
sed 's/      INIT: 0/      INIT: 1/' "$BASE" > "$TMP/init1bare.yaml"
python3 - "$TMP/init1bare.yaml" "$TMP/init1fib.yaml" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
t2, n = re.subn(r'(- "\d+ SOLID HEX8 [\d ]+MAT 1 KINEM nonlinear)"',
                r'\1 FIBER1 0.0 1.0 0.0"', t)
if n == 0:
    print("FIXTURE_ABORT=upstream_deck_changed"); sys.exit(3)
open(sys.argv[2], "w").write(t2)
PY
[ -f "$TMP/init1fib.yaml" ] || exit 3

probe ANGLE     "$TMP/angle.yaml"
probe FIBERVEC  "$TMP/fibervec.yaml"
probe INIT1BARE "$TMP/init1bare.yaml"
probe INIT1FIB  "$TMP/init1fib.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/ANGLE.log"
# FIBER_VEC is not a 4C material key.
grep -m1 -F "Could not match this input" "$TMP/FIBERVEC.log"
grep -m1 -oF "4C_global_data_read.cpp" "$TMP/FIBERVEC.log"
# INIT: 1 means "take the fibres from the element", and says so when there are none.
grep -m1 -F "Could not find element coordinate system or element fibers!" "$TMP/INIT1BARE.log"
grep -m1 -oF "4C_mat_anisotropy_extension_default.cpp" "$TMP/INIT1BARE.log"
# With FIBER1 on the element lines it runs — and the answer is a different one.
echo "PER_ELEMENT_FIBRE_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/INIT1FIB.log")"
grep -m1 -F "is WRONG --> actresult=" "$TMP/INIT1FIB.log"
exit 0
