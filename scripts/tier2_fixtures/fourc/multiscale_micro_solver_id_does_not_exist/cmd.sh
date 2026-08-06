#!/bin/bash
# Tier-2 for fourc::multiscale#2 — a FALSIFICATION plus the real failure shape.
#
# Claimed: parser warns `MICRO_SOLVER_ID X not found among macro SOLVER
#          definitions`, or a runtime abort `null pointer to micro Belos solver`.
# Observed: there is no MICRO_SOLVER_ID key anywhere in 4C — zero occurrences in
#          the binary's own --parameters schema. The micro input file is a
#          COMPLETE 4C input in its own right and names its own solver through
#          its own LINEAR_SOLVER / SOLVER n blocks; nothing links it to the macro
#          deck's solver numbering.
#
#          Point the micro deck's LINEAR_SOLVER at an id it does not define and
#          4C produces NO error block at all. It dies inside Teuchos with
#          'Error!  The parameter "SOLVER" does not exist' for sublist
#          "ROOT->SOLVER 7", an unhandled C++ exception and shell status 134.
#          That matters: there is no `PROC 0 ERROR` banner to grep for, so the
#          usual way of reading a 4C failure finds nothing.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream sohex8_multiscale_macro.4C.yaml) || exit 3
MICRO=$(upstream sohex8_multiscale_micro.mat.4C.yaml) || exit 3
cd "$TMP" || exit 3
cp "$BASE" macro.yaml
cp "$MICRO" micro.yaml
sed -i 's|MICROFILE: "sohex8_multiscale_micro.mat.4C.yaml"|MICROFILE: "micro.yaml"|g' macro.yaml
grep -q 'MICROFILE: "micro.yaml"' macro.yaml || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "  LINEAR_SOLVER: 1" micro.yaml || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

sed 's/  LINEAR_SOLVER: 1/  LINEAR_SOLVER: 7/' micro.yaml > micro_badsolver.yaml
sed 's|MICROFILE: "micro.yaml"|MICROFILE: "micro_badsolver.yaml"|g' macro.yaml > macro_bad.yaml

probe GOOD macro.yaml
probe BAD  macro_bad.yaml

echo "GOOD_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/GOOD.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/GOOD.log"
grep -m1 -F "terminate called after throwing an instance of 'Teuchos::Exceptions::InvalidParameterName'" "$TMP/BAD.log"
grep -m1 -F 'Error!  The parameter "SOLVER" does not exist' "$TMP/BAD.log"
grep -m1 -F 'in the parameter (sub)list "ROOT->SOLVER 7".' "$TMP/BAD.log"
# No 4C diagnostic banner at all on the failing arm.
echo "BAD_FOURC_ERROR_BLOCKS=$(grep -c 'PROC 0 ERROR' "$TMP/BAD.log")"
# The claimed key does not exist.
"$BIN" --parameters 2>/dev/null > params.json
echo "MICRO_SOLVER_ID_IN_SCHEMA=$(grep -c 'MICRO_SOLVER_ID' params.json)"
echo "CLAIMED_NULL_POINTER_TEXT=$(grep -ci 'null pointer to micro' "$TMP/BAD.log")"
exit 0
