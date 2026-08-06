#!/bin/bash
# Tier-2 for fourc::sti#0 — STI builds its thermo field by cloning the
# scatra mesh, so CLONING MATERIAL MAP is mandatory.  Delete it and 4C stops.
#
# Claimed:  'cannot clone material for thermo'.
# Observed: no such string.  The message is the generic one from the shared
# cloning utility, and it names neither STI nor thermo:
#
#     At least one material pairing required in --CLONING MATERIAL MAP.
#     core/fem/src/general/utils/4C_fem_general_utils_createdis.hpp
#
# It also still spells the section in the retired --SECTION dat form.  Both the
# real text and the absence of the claimed text are asserted.
#
# Note the map here is scatra -> thermo (one entry per material group), NOT
# structure -> anything: in STI there is no structural field.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream sti_mono_3D_hex8_elch_s2i_butlervolmerpeltier_adiabatic_mortar_standard.4C.yaml) || exit 3
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
[ -f "$TMP/unmapped.yaml" ] || exit 3

probe MAPPED   "$TMP/mapped.yaml"
probe UNMAPPED "$TMP/unmapped.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/MAPPED.log"
grep -m1 -F "At least one material pairing required in --CLONING MATERIAL MAP." "$TMP/UNMAPPED.log"
grep -m1 -oF "4C_fem_general_utils_createdis.hpp" "$TMP/UNMAPPED.log"
# The diagnostic is field-agnostic: it says neither STI nor thermo.
echo "DIAGNOSTIC_NAMES_THERMO=$(grep -c 'At least one material pairing.*thermo' "$TMP/UNMAPPED.log")"
echo "CLAIMED_CANNOT_CLONE_THERMO_TEXT=$(grep -ci 'cannot clone material for thermo' "$TMP/UNMAPPED.log")"
# In STI the pairing is scatra -> thermo; there is no structural field to map.
echo "UPSTREAM_MAPS_FROM_SCATRA=$(grep -c 'SRC_FIELD: "scatra"' "$BASE")"
echo "UPSTREAM_MAPS_FROM_STRUCTURE=$(grep -c 'SRC_FIELD: "structure"' "$BASE")"
exit 0
