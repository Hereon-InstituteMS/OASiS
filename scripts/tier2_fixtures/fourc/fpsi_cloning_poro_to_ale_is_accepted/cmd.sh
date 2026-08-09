#!/bin/bash
# Tier-2 for fourc::fpsi#5 — cloning the porous structure to ALE as well as the
# free fluid does NOT raise anything.  4C accepts the extra entry and produces a
# bit-identical answer.
#
# Claimed:  "trying to clone porous -> ALE raises 'porous field already has
#            Lagrangian motion' from 4C_fpsi_factory.cpp".
# Observed: upstream fpsi_ofsiinterface.4C.yaml maps structure -> porofluid and
#           fluid -> ale.  Add a third entry, structure -> ale, and the run
#           finishes normally, exits 0 and passes both of the deck's result
#           tests.  No abort, no warning, no mention of Lagrangian motion, and no
#           file named 4C_fpsi_factory.cpp exists in 4C.
#
# So the advice ("do not clone the porous domain to ALE") is sound but
# unenforced: getting it wrong costs you nothing visible here, which is exactly
# why it needs to be written down rather than discovered from a diagnostic.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fpsi_ofsiinterface.4C.yaml) || exit 3
grep -q '^CLONING MATERIAL MAP:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_cloning_map_changed"; exit 3; }

# The pathology: an extra porous-structure -> ale clone entry.
EXTRA_CLONE='  - SRC_FIELD: "structure"\n    SRC_MAT: 1\n    TAR_FIELD: "ale"\n    TAR_MAT: 5\n'

cp "$BASE" "$TMP/clean.yaml"
python3 - "$BASE" "$TMP/poroale.yaml" "$EXTRA_CLONE" <<'PY'
import sys
t = open(sys.argv[1]).read()
anchor = ('  - SRC_FIELD: "fluid"\n    SRC_MAT: 4\n'
          '    TAR_FIELD: "ale"\n    TAR_MAT: 5\n')
assert anchor in t, "upstream deck no longer carries the fluid -> ale clone entry"
extra = sys.argv[3].encode().decode("unicode_escape")
open(sys.argv[2], "w").write(t.replace(anchor, anchor + extra, 1))
PY
echo "ALE_CLONE_ENTRIES=$(grep -c 'TAR_FIELD: "ale"' "$TMP/poroale.yaml")"

probe CLEAN   "$TMP/clean.yaml"
probe POROALE "$TMP/poroale.yaml"

grep -m1 -F "OK (2)" "$TMP/CLEAN.log"
grep -m1 -F "OK (2)" "$TMP/POROALE.log"
grep -m1 -F "processor 0 finished normally" "$TMP/POROALE.log"
grep -m1 -F "is CORRECT, abs(diff)=" "$TMP/POROALE.log"

echo "POROALE_FAILED_TESTS=$(grep -c 'is WRONG' "$TMP/POROALE.log")"
echo "POROALE_PASSED_TESTS=$(grep -c 'is CORRECT' "$TMP/POROALE.log")"
# Nothing is said about it.
echo "CLAIMED_LAGRANGIAN_TEXT=$(grep -ci 'already has Lagrangian motion' "$TMP/POROALE.log")"
echo "CLAIMED_FPSI_FACTORY_FILE=$(grep -c '4C_fpsi_factory' "$TMP/POROALE.log")"
echo "POROALE_CLONE_WARNINGS=$(grep -ciE 'clon(e|ing).*(ignor|duplicate|already|invalid)' "$TMP/POROALE.log")"
# Still exactly one ALE discretisation was built, so the extra entry did nothing.
echo "POROALE_FIELDS_BUILT=$(grep -c 'fill_complete() on discretization' "$TMP/POROALE.log")"
echo "CLEAN_FIELDS_BUILT=$(grep -c 'fill_complete() on discretization' "$TMP/CLEAN.log")"
exit 0
