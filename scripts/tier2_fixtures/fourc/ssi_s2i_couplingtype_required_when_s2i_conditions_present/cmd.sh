#!/bin/bash
# Tier-2 for fourc::ssi#2 — and a FALSIFICATION of how it was worded.
#
# Claimed: omitting the S2I COUPLING settings "gives zero current across the
#          interface", i.e. a quietly decoupled scatra field.
#
# Observed: if DESIGN S2I KINETICS conditions exist, dropping
# `SCALAR TRANSPORT DYNAMIC/S2I COUPLING` (which is where COUPLINGTYPE lives)
# is a hard abort:
#
#     Type of mortar meshtying for scatra-scatra interface coupling not recognized!
#     src/scatra/4C_scatra_timint_meshtying_strategy_s2i.cpp
#
# from MeshtyingStrategyS2I::setup_meshtying(), exit 1.  COUPLINGTYPE has no
# usable default: the strategy object is built, finds no coupling type, and
# stops.  Nothing decouples silently — asserted as a zero result-test count.
#
# The trap worth knowing is the reverse one, also shown here: on a deck with NO
# S2I conditions the same section is entirely inert, so deleting it is harmless.
# The section only bites when interface conditions are present.
. "$(dirname "$0")/../_lib/preamble.sh"

WITHS2I=$(upstream ssi_mono_3D_2hex8_scatra_s2i_constperm.4C.yaml) || exit 3
NOS2I=$(upstream ssi_mono_3D_1hex8_elch_funct_growthlaw.4C.yaml)   || exit 3
grep -q "^DESIGN S2I KINETICS SURF CONDITIONS:" "$WITHS2I" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

strip_s2i() {  # $1 = source deck, $2 = destination
python3 - "$1" "$2" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = 'SCALAR TRANSPORT DYNAMIC/S2I COUPLING:\n  COUPLINGTYPE: "MatchingNodes"\n'
if blk not in t:
    print("FIXTURE_ABORT=upstream_deck_changed"); sys.exit(3)
open(sys.argv[2], "w").write(t.replace(blk, "", 1))
PY
}

cp "$WITHS2I" "$TMP/kept.yaml"
strip_s2i "$WITHS2I" "$TMP/stripped.yaml"
strip_s2i "$NOS2I"   "$TMP/nointerface.yaml"
[ -f "$TMP/stripped.yaml" ] && [ -f "$TMP/nointerface.yaml" ] || exit 3

probe KEPT        "$TMP/kept.yaml"
probe STRIPPED    "$TMP/stripped.yaml"
probe NOINTERFACE "$TMP/nointerface.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/KEPT.log"
grep -m1 -F "Type of mortar meshtying for scatra-scatra interface coupling not recognized!" "$TMP/STRIPPED.log"
grep -m1 -oF "4C_scatra_timint_meshtying_strategy_s2i.cpp" "$TMP/STRIPPED.log"
echo "FAILS_IN_SETUP_MESHTYING=$(grep -c 'setup_meshtying' "$TMP/STRIPPED.log")"
# No silent decoupling: the run stops before any result is produced.
echo "STRIPPED_RESULT_TESTS_PERFORMED=$(grep -c 'is WRONG --> actresult=' "$TMP/STRIPPED.log")"
# ...and on a deck with no S2I conditions the very same deletion is inert.
echo "NOINTERFACE_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/NOINTERFACE.log")"
exit 0
