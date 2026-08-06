#!/bin/bash
# Tier-2 for fourc::fs3i#3 — a nine-decade diffusivity contrast between the two
# scalar fields does NOT make the coupling "iterate between two non-converging
# states", and the under-relaxation the entry prescribes does not exist.
#
# Upstream fs3i_part_1wc_infperm.4C.yaml has MAT 5 (fluid scalar) DIFFUSIVITY 1.6
# and MAT 6 (structure scalar) DIFFUSIVITY 1.  Drop the structure scalar to 1e-09
# — the drug-transport contrast the entry describes — and:
#   * the run converges normally and completes all its steps,
#   * no "did not converge" appears anywhere,
#   * the answer just changes: the structure-side concentration moves by ~1.1e-01
#     and the fluid-side by ~2.3e-02, caught only by the deck's own result tests.
#
# And the remedy is unavailable: FS3I DYNAMIC/PARTITIONED accepts exactly
# COUPALGO, CONVTOL and ITEMAX.  Adding STARTOMEGA — the name 4C uses for fixed
# relaxation in its other partitioned couplers — is rejected at parse time by
# core/io/src/4C_io_input_spec_builders.cpp with "Could not match this input".
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fs3i_part_1wc_infperm.4C.yaml) || exit 3
grep -q 'DIFFUSIVITY: 1.6' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_fluid_scalar_diffusivity_changed"; exit 3; }
grep -q 'FS3I DYNAMIC/STRUCTURE SCALAR STABILIZATION:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_fs3i_stab_section_changed"; exit 3; }
grep -q 'STARTOMEGA' "$BASE" \
  && { echo "FIXTURE_ABORT=upstream_now_sets_startomega"; exit 3; }
cp -r "$(dirname "$BASE")/xml" "$TMP/" \
  || { echo "FIXTURE_ABORT=missing_teko_xml"; exit 3; }
cd "$TMP" || exit 3

# The pathology, and the remedy the entry prescribes.
LOW_STRUCT_DIFFUSIVITY=1e-09
RELAX_BLOCK='FS3I DYNAMIC/PARTITIONED:\n  STARTOMEGA: 0.5\n'

cp "$BASE" "$TMP/matched.yaml"
python3 - "$BASE" "$TMP/contrast.yaml" "$LOW_STRUCT_DIFFUSIVITY" <<'PY'
import sys
t = open(sys.argv[1]).read()
old = "  - MAT: 6\n    MAT_scatra:\n      DIFFUSIVITY: 1\n"
assert old in t, "upstream deck no longer sets the structure scalar diffusivity to 1"
open(sys.argv[2], "w").write(
    t.replace(old, "  - MAT: 6\n    MAT_scatra:\n      DIFFUSIVITY: %s\n" % sys.argv[3], 1))
PY
python3 - "$BASE" "$TMP/relax.yaml" "$RELAX_BLOCK" <<'PY'
import sys
t = open(sys.argv[1]).read()
anchor = "FS3I DYNAMIC/STRUCTURE SCALAR STABILIZATION:"
assert anchor in t
blk = sys.argv[3].encode().decode("unicode_escape")
open(sys.argv[2], "w").write(t.replace(anchor, blk + anchor, 1))
PY
grep -A2 '  - MAT: 6' "$TMP/contrast.yaml" | grep DIFFUSIVITY | tr -d ' ' \
  | sed 's/^/CONTRAST_ARM_[/;s/$/]/'
echo "RELAX_ARM_HAS_OMEGA_KEY=$(grep -c 'STARTOMEGA' "$TMP/relax.yaml")"

probe MATCHED  "$TMP/matched.yaml"
probe CONTRAST "$TMP/contrast.yaml"
probe RELAX    "$TMP/relax.yaml"

grep -m1 -F "OK (3)" "$TMP/MATCHED.log"
grep -m1 -F "processor 0 finished normally" "$TMP/MATCHED.log"
grep -m1 -F "Result check failed with 2 errors out of 3 tests" "$TMP/CONTRAST.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/CONTRAST.log"
grep -m1 -F "Could not match this input" "$TMP/RELAX.log"
grep -m1 -F "4C_io_input_spec_builders.cpp" "$TMP/RELAX.log"

# The contrast run is stable, not oscillating.
echo "CONTRAST_NONCONVERGENCE=$(grep -ci 'did not converge' "$TMP/CONTRAST.log")"
echo "CONTRAST_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/CONTRAST.log")"
echo "CONTRAST_FAILED_TESTS=$(grep -c 'is WRONG' "$TMP/CONTRAST.log")"
echo "CONTRAST_SCATRA_TESTS_MOVED=$(grep -c 'SCATRA .*is WRONG' "$TMP/CONTRAST.log")"
# The prescribed relaxation key does not exist.
echo "RELAX_KEY_ACCEPTED=$(grep -c 'Checking results of' "$TMP/RELAX.log")"
exit 0
