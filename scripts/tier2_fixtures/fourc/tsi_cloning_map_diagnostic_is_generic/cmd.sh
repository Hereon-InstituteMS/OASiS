#!/bin/bash
# Tier-2 for fourc::tsi#1 — TSI builds its thermo field by cloning the
# structural mesh, so CLONING MATERIAL MAP is mandatory: delete it from
# upstream tsi_lincompression_1waydisp and 4C stops during setup_tsi, exit 1.
#
# The rule holds.  The Signal was fabricated.  Claimed:
#     'cannot clone material for thermo field'  from 4C_adapter_str_factory.cpp
# Neither the string nor that origin occurs in the run.  What 4C prints is the
# generic message from the shared cloning utility, which names neither TSI nor
# the thermo field and still spells the section in the retired --SECTION dat
# form:
#     At least one material pairing required in --CLONING MATERIAL MAP.
#     core/fem/src/general/utils/4C_fem_general_utils_createdis.hpp
# The only place "TSI" or "thermo" appears is the demangled C++ backtrace
# (TSI::Utils::ThermoStructureCloneStrategy, TSI::Utils::setup_tsi), not the
# message.  Both the real text and the absence of the claimed text are pinned.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream tsi_lincompression_1waydisp.4C.yaml) || exit 3
grep -q "^CLONING MATERIAL MAP:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
cp "$BASE" "$TMP/mapped.yaml"

python3 - "$BASE" "$TMP/unmapped.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
i = t.find("CLONING MATERIAL MAP:")
j = t.find("FUNCT1:")
if i < 0 or j < 0 or j <= i:
    print("FIXTURE_ABORT=upstream_deck_changed"); sys.exit(3)
open(sys.argv[2], "w").write(t[:i] + t[j:])
PY
[ -s "$TMP/unmapped.yaml" ] || exit 3

probe MAPPED   "$TMP/mapped.yaml"
probe UNMAPPED "$TMP/unmapped.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/MAPPED.log"
grep -m1 -F "At least one material pairing required in --CLONING MATERIAL MAP." "$TMP/UNMAPPED.log"
grep -m1 -oF "4C_fem_general_utils_createdis.hpp" "$TMP/UNMAPPED.log"
# The stack, not the message, is what tells you this was the TSI clone.
grep -m1 -oF "ThermoStructureCloneStrategy" "$TMP/UNMAPPED.log"
grep -m1 -oF "setup_tsi" "$TMP/UNMAPPED.log"
# The catalogued wording and origin do not exist.
echo "CLAIMED_CANNOT_CLONE_TEXT=$(grep -ci 'cannot clone material for thermo' "$TMP/UNMAPPED.log")"
echo "CLAIMED_STR_FACTORY_ORIGIN=$(grep -ci '4C_adapter_str_factory' "$TMP/UNMAPPED.log")"
echo "DIAGNOSTIC_LINE_NAMES_THERMO=$(grep -c 'At least one material pairing.*thermo' "$TMP/UNMAPPED.log")"
# The standard form the upstream deck uses: structure -> thermo.
echo "UPSTREAM_SRC_FIELD_STRUCTURE=$(grep -c 'SRC_FIELD: "structure"' "$BASE")"
echo "UPSTREAM_TAR_FIELD_THERMO=$(grep -c 'TAR_FIELD: "thermo"' "$BASE")"
exit 0
