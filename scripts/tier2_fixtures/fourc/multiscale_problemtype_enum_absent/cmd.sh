#!/bin/bash
# Tier-2 for fourc::multiscale#4 — there is no PROBLEMTYPE 'Multiscale'.
#
# The rule holds; the Signal the entry carried did not. 4C never says 'unknown
# problem type'. It says "Could not match this input", then prints the ENTIRE
# allowed enum from 4C_io_input_spec_builders.cpp, and 'Multiscale' is not in it
# while 'Structure' is.
#
# The fixture also pins the route that DOES work, because a rejection on its own
# leaves an agent stuck: PROBLEMTYPE Structure plus a MAT_Struct_Multiscale
# material, which 4C's own `--parameters` schema carries together with MICROFILE
# and MICRODIS_NUM.
. "$(dirname "$0")/../_lib/preamble.sh"

cat > "$TMP/bad.yaml" <<'YAML'
PROBLEM TYPE:
  PROBLEMTYPE: "Multiscale"
YAML

probe BAD "$TMP/bad.yaml"

grep -m1 -F "Could not match this input" "$TMP/BAD.log"
grep -m1 -F "4C_io_input_spec_builders.cpp" "$TMP/BAD.log"
grep -m1 -oF "has wrong value, possible values:" "$TMP/BAD.log"
# The claimed diagnostic is not what 4C emits.
echo "CLAIMED_UNKNOWN_PROBLEM_TYPE_TEXT=$(grep -ci 'unknown problem type' "$TMP/BAD.log")"

# Note the 0-9 in the class: without it the enum truncates at
# Reduced_Lung_1D_Pipe_Flow and the membership tests below silently lie.
ENUM=$(grep -m1 -o "possible values: [A-Za-z_|0-9]*" "$TMP/BAD.log")
echo "ENUM_LENGTH_OVER_900=$([ "${#ENUM}" -gt 900 ] && echo yes || echo no)"
echo "ENUM_HAS_MULTISCALE=$(printf '%s|' "$ENUM" | grep -c '|Multiscale|')"
echo "ENUM_HAS_STRUCTURE=$(printf '%s|' "$ENUM" | grep -c '|Structure|')"

# ...and the material that carries multiscale really is MAT_Struct_Multiscale.
"$BIN" --parameters 2>/dev/null > "$TMP/params.json"
echo "MAT_STRUCT_MULTISCALE_IN_SCHEMA=$(grep -c 'MAT_Struct_Multiscale' "$TMP/params.json")"
echo "MICROFILE_IN_SCHEMA=$(grep -c 'MICROFILE' "$TMP/params.json")"
exit 0
