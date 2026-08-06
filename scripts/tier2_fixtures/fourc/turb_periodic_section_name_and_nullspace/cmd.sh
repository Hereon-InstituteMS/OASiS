#!/bin/bash
# Tier-2 for fourc::fluid_turbulence#3 -- the section is not called DESIGN
# PERIODIC CONDITIONS, and dropping periodicity does not give you "blockage and
# incorrect mean flow".  It stops the run.
#
# Real names: DESIGN LINE PERIODIC BOUNDARY CONDITIONS and DESIGN SURF PERIODIC
#             BOUNDARY CONDITIONS, each entry pairing a Master and a Slave
#             surface through a shared ID, with PLANE / LAYER / ANGLE.
# Observed:   the claimed name is rejected outright as an invalid section, and
#             deleting the real block from the upstream channel deck aborts with
#             "Nullspace check for sysmat_ failed" -- the streamwise direction had
#             no Dirichlet data, so the pressure nullspace no longer matches the
#             Krylov projection.  You never get as far as a wrong mean profile.
. "$(dirname "$0")/../_lib/preamble.sh"

# 4C resolves MUELU_XML_FILE / IFPACK_XML_FILE relative to the INPUT FILE's
# directory, so a copied deck needs the upstream xml tree beside it.
link_xml() { ln -sfn "$(dirname "$1")/xml" "$TMP/xml" 2>/dev/null || cp -r "$(dirname "$1")/xml" "$TMP/xml"; }

BASE=$(upstream f3_cha_8x8x8_recongradl2.4C.yaml) || exit 3
grep -q "^DESIGN SURF PERIODIC BOUNDARY CONDITIONS:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "^DESIGN PATCH RECOVERY BOUNDARY SURF CONDITIONS:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
link_xml "$BASE"

cp "$BASE" "$TMP/base.yaml"
sed 's|^DESIGN SURF PERIODIC BOUNDARY CONDITIONS:|DESIGN PERIODIC CONDITIONS: []\nDESIGN SURF PERIODIC BOUNDARY CONDITIONS:|' "$BASE" > "$TMP/badname.yaml"
python3 - "$BASE" "$TMP/noperiodic.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
i = t.index("DESIGN SURF PERIODIC BOUNDARY CONDITIONS:")
j = t.index("DESIGN PATCH RECOVERY BOUNDARY SURF CONDITIONS:")
open(sys.argv[2], "w").write(t[:i] + t[j:])
PY

probe BASE       "$TMP/base.yaml"
probe BADNAME    "$TMP/badname.yaml"
probe NOPERIODIC "$TMP/noperiodic.yaml"

grep -m1 -F "Section 'DESIGN PERIODIC CONDITIONS' is not a valid section name." "$TMP/BADNAME.log"
grep -m1 -F "4C_io_input_file.cpp" "$TMP/BADNAME.log"
grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
echo "BASE_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/BASE.log")"
# Removing the real block is fatal, not merely inaccurate.
grep -m1 -F "Nullspace check for sysmat_ failed" "$TMP/NOPERIODIC.log"
grep -m1 -F "4C_fluid_implicit_integration.cpp" "$TMP/NOPERIODIC.log"
echo "NOPERIODIC_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/NOPERIODIC.log")"
echo "CLAIMED_BLOCKAGE_TEXT=$(grep -ciE 'blockage|incorrect mean flow' "$TMP/NOPERIODIC.log")"
exit 0
