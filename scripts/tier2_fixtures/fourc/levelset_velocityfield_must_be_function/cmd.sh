#!/bin/bash
# Tier-2 for fourc::level_set#2 -- omitting VELOCITYFIELD does not silently give
# you a stationary interface.  A Level_Set problem refuses to start without it.
#
# Claimed: "omitting VELOCITYFIELD defaults to ZERO -- no advection, the interface
#          STAYS PUT even with non-zero time stepping".
# Observed: levelset_dyn checks the value before anything is built and throws
#          "Other velocity fields than a field given by a function not yet
#          supported for level-set problems" from 4C_levelset_dyn.cpp.  The
#          default (zero) is rejected, and so is Navier_Stokes: for PROBLEMTYPE
#          Level_Set, VELOCITYFIELD: function with VELFUNCNO is the ONLY option.
. "$(dirname "$0")/../_lib/preamble.sh"

# 4C resolves IFPACK_XML_FILE relative to the INPUT FILE's directory, so a copied
# deck needs the upstream xml tree beside it.
link_xml() { ln -sfn "$(dirname "$1")/xml" "$TMP/xml" 2>/dev/null || cp -r "$(dirname "$1")/xml" "$TMP/xml"; }

BASE=$(upstream levelset_gaussian_hill_pbc.4C.yaml) || exit 3
grep -q '  VELOCITYFIELD: "function"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "  VELFUNCNO: 1" "$BASE"              || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
link_xml "$BASE"

cp "$BASE" "$TMP/func.yaml"
python3 - "$BASE" "$TMP/omitted.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
drive = '  VELOCITYFIELD: "function"\n  VELFUNCNO: 1\n'
assert drive in t, "upstream deck no longer carries the prescribed velocity field"
open(sys.argv[2], "w").write(t.replace(drive, ""))
PY
sed 's/  VELOCITYFIELD: "function"/  VELOCITYFIELD: "Navier_Stokes"/' "$BASE" > "$TMP/navsto.yaml"

probe FUNC    "$TMP/func.yaml"
probe OMITTED "$TMP/omitted.yaml"
probe NAVSTO  "$TMP/navsto.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/FUNC.log"
echo "FUNC_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/FUNC.log")"
grep -m1 -F "Other velocity fields than a field given by a function not yet supported for level-set problems" "$TMP/OMITTED.log"
grep -m1 -F "4C_levelset_dyn.cpp" "$TMP/OMITTED.log"
# Navier_Stokes is rejected by the same check -- 'function' really is the only value.
echo "NAVSTO_SAME_ABORT=$(grep -c 'not yet supported for level-set problems' "$TMP/NAVSTO.log")"
# No stationary-interface run ever happens.
echo "OMITTED_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/OMITTED.log")"
echo "OMITTED_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/OMITTED.log")"
exit 0
