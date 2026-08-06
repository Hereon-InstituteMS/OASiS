#!/bin/bash
# Tier-2 for fourc::beams#3 — a BEAM3R LINE3 element line lists its nodes as
# endpoint1 endpoint2 midpoint, not in sequential order along the beam.
#
# Upstream beam3r_line3_static_test1 writes "1 3 2", "3 5 4", ... The bad arm
# rewrites every element to sequential "1 2 3", "3 4 5", ... and nothing
# complains: no parse error, no warning, ten load steps finalised, Newton
# converged. The ONLY thing that reveals it is the deck's own result test, which
# pins the tip node to 1e-08 and reports three failures.
#
# The entry claimed "element length is halved, stiffness is wrong by factor
# 2-4". Measured on this deck the tip displacement moves by 5.0e-01 / 5.3e-01 /
# 4.2e-01 on components of magnitude 2.4e+01 / 1.4e+01 / 5.4e+01 — a couple of
# per cent, not a factor of two to four. A convergence study would not notice.
# That is the actual danger and why WRONG_ORDER_STILL_CONVERGES=yes is asserted.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream beam3r_line3_static_test1.4C.yaml) || exit 3
grep -q "1 BEAM3R LINE3 1 3 2 MAT 1" "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/endendmid.yaml"

python3 - "$BASE" "$TMP/sequential.yaml" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
# "<e> BEAM3R LINE3 n1 n3 n2" -> "<e> BEAM3R LINE3 n1 n2 n3"
out = re.sub(r'(BEAM3R LINE3 )(\d+) (\d+) (\d+)',
             lambda m: m.group(1) + f"{m.group(2)} {m.group(4)} {m.group(3)}", t)
assert out != t, "upstream deck no longer carries BEAM3R LINE3 element lines"
open(sys.argv[2], "w").write(out)
PY

probe ENDENDMID  "$TMP/endendmid.yaml"
probe SEQUENTIAL "$TMP/sequential.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/ENDENDMID.log"
# The mis-ordered deck runs the whole simulation without a single complaint...
echo "SEQUENTIAL_STEPS=$(grep -c 'Finalised step' "$TMP/SEQUENTIAL.log")"
echo "SEQUENTIAL_PARSE_ERRORS=$(grep -ciE 'node ordering|invalid connectivity|negative jacobian' "$TMP/SEQUENTIAL.log")"
if [ "$(grep -c 'Finalised step' "$TMP/SEQUENTIAL.log")" = "10" ]; then
  echo "WRONG_ORDER_STILL_CONVERGES=yes"
else
  echo "WRONG_ORDER_STILL_CONVERGES=no"
fi
# ...and is caught only by the deck's own result test.
echo "SEQUENTIAL_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/SEQUENTIAL.log")"
echo "ENDENDMID_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/ENDENDMID.log")"
grep -m1 -F "is WRONG --> actresult=" "$TMP/SEQUENTIAL.log"
grep -m1 -F "Result check failed with 3 errors out of 3 tests" "$TMP/SEQUENTIAL.log"
exit 0
