#!/bin/bash
# Tier-2 for fourc::structural_mechanics#4 — a TSI structural mesh must be
# written with SOLIDSCATRA elements; plain SOLID cannot be cloned into the
# thermal field.
#
# Upstream deck tsi_heatflux_iterstaggaitken.4C.yaml, one token changed:
#
#   SOLIDSCATRA HEX8 ... KINEM linear TYPE Undefined   -> runs, exit 0
#   SOLID       HEX8 ... KINEM linear                  -> Unsupported solid
#                                                         element type!
#   SOLID       HEX8 ... KINEM linear TYPE Undefined   -> the deck does not even
#                                                         parse, because TYPE is
#                                                         a SOLIDSCATRA key
#
# The middle arm is the one an agent will hit, and its message names neither
# SOLIDSCATRA nor the thermal field.  The entry used to quote 'no SCATRA
# discretisation found' from a file called 4C_tsi_factory.cpp; neither exists,
# and CLAIMED_TSI_TEXTS=0 keeps that correction pinned.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream tsi_heatflux_iterstaggaitken.4C.yaml) || exit 3
for needle in 'SOLIDSCATRA HEX8' ' TYPE Undefined' 'PROBLEMTYPE: "Thermo_Structure_Interaction"'; do
  grep -qF "$needle" "$BASE" || {
    echo "FIXTURE_ABORT=upstream_deck_changed (missing: $needle)"; exit 3; }
done

cp "$BASE" "$TMP/base.yaml"
sed 's/SOLIDSCATRA HEX8/SOLID HEX8/; s/ TYPE Undefined//' "$BASE" > "$TMP/solid.yaml"
sed 's/SOLIDSCATRA HEX8/SOLID HEX8/'                      "$BASE" > "$TMP/solid_keeps_type.yaml"

probe BASE           "$TMP/base.yaml"
probe SOLID          "$TMP/solid.yaml"
probe SOLID_WITHTYPE "$TMP/solid_keeps_type.yaml"

# The control couples both fields and finishes.
grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
echo "BASE_SOLIDSCATRA_ELEMENTS=$(grep -c 'SOLIDSCATRA HEX8' "$TMP/base.yaml")"

# Plain SOLID: the abort comes from the TSI clone utility and names only the
# element type, not the missing coupling.
grep -m1 -F "Unsupported solid element type!" "$TMP/SOLID.log"
grep -m1 -F "4C_tsi_utils.cpp" "$TMP/SOLID.log"

# Keeping TYPE with plain SOLID does not even reach that point: TYPE belongs to
# the SOLIDSCATRA element line, so the element line itself stops parsing.
grep -m1 -F "After parsing, the line still contains 'TYPE Undefined'." "$TMP/SOLID_WITHTYPE.log"

# The strings the entry used to quote are in no log and no source file.
python3 - "$TMP/SOLID.log" "$TMP/SOLID_WITHTYPE.log" <<'PY'
import sys
n = 0
for p in sys.argv[1:]:
    t = open(p, "rb").read().decode("utf-8", "replace").lower()
    n += (t.count("no scatra discretisation found")
          + t.count("no scatra discretization found")
          + t.count("4c_tsi_factory.cpp"))
print("CLAIMED_TSI_TEXTS=%d" % n)
PY
exit 0
