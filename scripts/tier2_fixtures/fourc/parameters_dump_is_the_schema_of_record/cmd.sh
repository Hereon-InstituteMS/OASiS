#!/bin/bash
# Tier-2 for fourc::input_format#28 — `4C --parameters` is the version-exact
# input schema of the INSTALLED binary, and it is the right thing to consult
# before writing a section, a key or an element line.  It exits 0, writes YAML
# to stdout beginning with `metadata:` / `commit_hash:` / `version:`, and has
# exactly seven top-level keys.
#
# Two practical points the fixture pins:
#   * Redirect stdout ALONE.  4C's stderr carries unrelated host noise (an X11
#     cookie warning on this box), so `2>&1` corrupts any size or hash you take.
#     Both are measured here and the difference is reported.
#   * "Is section X in the dump?" is the correct way to ask whether a section
#     exists in this build.  A name absent from the dump does not exist,
#     whatever any documentation says -- and the fixture checks that with a
#     section that does exist and one that does not.
. "$(dirname "$0")/../_lib/preamble.sh"

"$BIN" --parameters 2>/dev/null > "$TMP/clean.yaml"
echo "EXIT_PARAMETERS=$?"
"$BIN" --parameters > "$TMP/noisy.yaml" 2>&1

echo "CLEAN_BYTES=$(wc -c < "$TMP/clean.yaml")"
echo "NOISY_MINUS_CLEAN_BYTES=$(( $(wc -c < "$TMP/noisy.yaml") - $(wc -c < "$TMP/clean.yaml") ))"
# The 2>&1 capture can only ever be LARGER; on this host it carries 30 bytes of
# X11 warning.  How much noise there is depends on the box, so only the
# direction is asserted -- but the direction is the whole warning.
echo "NOISY_IS_AT_LEAST_CLEAN=$([ "$(wc -c < "$TMP/noisy.yaml")" -ge "$(wc -c < "$TMP/clean.yaml")" ] && echo yes || echo no)"
echo "CLEAN_DUMP_IS_PURE_YAML=$(head -c 9 "$TMP/clean.yaml")"

head -3 "$TMP/clean.yaml"
echo "TOP_LEVEL_KEYS=$(grep -c '^[A-Za-z_$][A-Za-z_$]*:' "$TMP/clean.yaml")"
echo "TOP_LEVEL_KEY_LIST=$(grep -o '^[A-Za-z_$][A-Za-z_$]*:' "$TMP/clean.yaml" | tr -d ':' | tr '\n' ' ')"

body() { awk -v k="^$1:$" '$0 ~ k {f=1;next} /^[A-Za-z_$]/{f=0} f' "$TMP/clean.yaml"; }
echo "SECTION_COUNT=$(body sections | grep -c '^    - name: ')"
echo "LEGACY_ELEMENT_COUNT=$(body legacy_element_specs | grep -c '^  [A-Za-z0-9_]*:$')"

# Every parameter carries a name, a type and a required flag.
echo "HAS_REQUIRED_FLAGS=$(grep -c '^ *required: ' "$TMP/clean.yaml")"

# Asking the dump whether a section exists is the whole point.
present() { body sections | grep -c "^    - name: $1\$"; }
echo "DUMP_HAS_STRUCTURAL_DYNAMIC=$(present 'STRUCTURAL DYNAMIC')"
echo "DUMP_HAS_INVENTED_SECTION=$(present 'STRUCTURAL DYNAMICS')"

# A section the dump does not know is rejected by the binary that produced it.
cat > "$TMP/invented.yaml" <<'YAML'
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMICS:
  DYNAMICTYPE: "Statics"
YAML
probe INVENTED "$TMP/invented.yaml"
grep -m1 -F "Section 'STRUCTURAL DYNAMICS' is not a valid section name." "$TMP/INVENTED.log"
exit 0
