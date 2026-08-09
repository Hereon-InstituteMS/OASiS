#!/bin/bash
# Tier-2 for fourc::low_mach#5 -- the entry is right that there is no GRAVITY
# section, and wrong about how to replace it.
#
# Confirmed: a GRAVITY: block is rejected -- "Section 'GRAVITY' is not a valid
#            section name."  (The only GRAVITY_* keys in 4C live in PARTICLE
#            DYNAMIC and do nothing for a fluid.)
# Falsified: the suggested remedy, a Neumann condition with FORCE: [0, 0, -9.81],
#            is also rejected -- Neumann conditions take NUMDOF / ONOFF / VAL /
#            FUNCT / TYPE, and there is no FORCE key.  The upstream heated-channel
#            deck applies buoyancy as DESIGN SURF NEUMANN with VAL and
#            TYPE: "Dead"; deleting it stops the solve outright.
. "$(dirname "$0")/../_lib/preamble.sh"

# 4C resolves MUELU_XML_FILE relative to the INPUT FILE's directory, so a copied
# deck needs the upstream xml tree beside it.
link_xml() { ln -sfn "$(dirname "$1")/xml" "$TMP/xml" 2>/dev/null || cp -r "$(dirname "$1")/xml" "$TMP/xml"; }
BASE=$(upstream loma_2d_heated_chan_30x100.4C.yaml) || exit 3
link_xml "$BASE"

BUOY='DESIGN SURF NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 6
    ONOFF: [0, 1, 0, 0, 0, 0]
    VAL: [0, -9.81, 0, 0, 0, 0]
    FUNCT: [0, 0, 0, 0, 0, 0]
    TYPE: "Dead"
'
grep -q '    TYPE: "Dead"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/base.yaml"
sed 's/^LOMA CONTROL:/GRAVITY:\n  ACCELERATION: [0, -9.81, 0]\nLOMA CONTROL:/' "$BASE" > "$TMP/gravsec.yaml"
python3 - "$BASE" "$TMP" "$BUOY" <<'PY'
import sys, os
t, TMP, buoy = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3]
assert buoy in t, "upstream deck no longer carries the buoyancy Neumann condition"
open(os.path.join(TMP, "forcekey.yaml"), "w").write(
    t.replace("    VAL: [0, -9.81, 0, 0, 0, 0]", "    FORCE: [0, -9.81, 0, 0, 0, 0]"))
open(os.path.join(TMP, "nobuoy.yaml"), "w").write(t.replace(buoy, ""))
PY

probe BASE     "$TMP/base.yaml"
probe GRAVSEC  "$TMP/gravsec.yaml"
probe FORCEKEY "$TMP/forcekey.yaml"
probe NOBUOY   "$TMP/nobuoy.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
echo "BASE_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/BASE.log")"
# confirmed half
grep -m1 -F "Section 'GRAVITY' is not a valid section name." "$TMP/GRAVSEC.log"
grep -m1 -F "4C_io_input_file.cpp" "$TMP/GRAVSEC.log"
# falsified half: FORCE is not a Neumann key either
grep -m1 -F "Failed to match condition specification in section 'DESIGN SURF NEUMANN CONDITIONS'" "$TMP/FORCEKEY.log"
grep -m1 -F "4C_fem_condition_definition.cpp" "$TMP/FORCEKEY.log"
echo "FORCEKEY_UNUSED=$(grep -c 'FORCE:' "$TMP/FORCEKEY.log")"
# and the VAL-based condition is really what carries the buoyancy
echo "NOBUOY_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/NOBUOY.log")"
exit 0
