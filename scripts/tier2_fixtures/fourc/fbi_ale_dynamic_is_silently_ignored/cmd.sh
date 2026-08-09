#!/bin/bash
# Tier-2 for fourc::fbi#0 — FBI really is Eulerian-fluid + immersed beams, but
# putting an ALE DYNAMIC section into an FBI input does NOT abort.  It is
# swallowed without a word.
#
# Claimed:  "including ALE DYNAMIC in an FBI input aborts with
#            'FBI is incompatible with ALE' from 4C_fbi_factory.cpp".
# Observed: upstream fbi_mortar_solidcoupling.4C.yaml plus a complete ALE DYNAMIC
#           block (TIMESTEP/NUMSTEP/MAXTIME/LINEAR_SOLVER) runs to completion,
#           exits 0, and passes all six of the deck's own result tests to 1e-17.
#           No abort, no warning, no ALE discretisation: the string "ALE"/"ale"
#           never appears as a field banner in the log.  Neither
#           'FBI is incompatible with ALE' nor any file named 4C_fbi_factory.cpp
#           exists in 4C.
#
# This is the silent-wrong shape that matters: an author who adds ALE DYNAMIC
# believing it enables mesh motion gets a clean, converged, plausible run in
# which the fluid mesh never moved and nothing said so.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fbi_mortar_solidcoupling.4C.yaml) || exit 3
grep -q '^FSI DYNAMIC:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
# The deck's NOX status test XML is resolved relative to the CWD, not the deck.
cp "$(dirname "$BASE")/beam_flow_solver.xml" "$TMP/" \
  || { echo "FIXTURE_ABORT=missing_nox_xml"; exit 3; }
cd "$TMP" || exit 3

ADD_ALE_SECTION=yes

cp "$BASE" "$TMP/noale.yaml"
python3 - "$BASE" "$TMP/withale.yaml" "$ADD_ALE_SECTION" <<'PY'
import sys
t = open(sys.argv[1]).read()
if sys.argv[3] == "yes":
    blk = ("ALE DYNAMIC:\n  TIMESTEP: 0.01\n  NUMSTEP: 2\n"
           "  MAXTIME: 0.05\n  LINEAR_SOLVER: 1\n")
    assert "FSI DYNAMIC:\n" in t, "upstream deck no longer has FSI DYNAMIC"
    t = t.replace("FSI DYNAMIC:\n", blk + "FSI DYNAMIC:\n", 1)
open(sys.argv[2], "w").write(t)
PY
grep -q "^ALE DYNAMIC:" "$TMP/withale.yaml" && echo "ALE_SECTION_IN_DECK=yes" \
                                           || echo "ALE_SECTION_IN_DECK=no"

probe NOALE   "$TMP/noale.yaml"
probe WITHALE "$TMP/withale.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/WITHALE.log"
grep -m1 -F "OK (6)" "$TMP/WITHALE.log"
grep -m1 -F "is CORRECT, abs(diff)=" "$TMP/WITHALE.log"

# Not one of the claimed strings, and no complaint of any kind.
echo "CLAIMED_INCOMPATIBLE_TEXT=$(grep -ci 'incompatible with ALE' "$TMP/WITHALE.log")"
echo "CLAIMED_FBI_FACTORY_FILE=$(grep -c '4C_fbi_factory' "$TMP/WITHALE.log")"
echo "ALE_WARNINGS=$(grep -ciE 'ale.*(ignor|unus|not used|no effect|incompatib)' "$TMP/WITHALE.log")"
# No ALE field was ever built: 4C prints one 'fill_complete() on discretization
# <name>' banner per field, and 'ale' is not among them.
echo "ALE_DISCRETISATIONS_BUILT=$(grep -c 'fill_complete() on discretization ale' "$TMP/WITHALE.log")"
echo "FIELDS_BUILT=$(grep -c 'fill_complete() on discretization' "$TMP/WITHALE.log")"
# And the answer is bit-for-bit the same as without the section.
echo "RESULT_TESTS_FAILED_WITH_ALE=$(grep -c 'is WRONG' "$TMP/WITHALE.log")"
echo "RESULT_TESTS_FAILED_WITHOUT_ALE=$(grep -c 'is WRONG' "$TMP/NOALE.log")"
exit 0
