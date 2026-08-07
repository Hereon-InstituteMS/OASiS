#!/bin/bash
# Tier-2 for fourc::fs3i#2 — the fluid-side scalar stabilisation is live and
# silent.  Upstream fs3i_part_1wc_infperm.4C.yaml ships with
# SCALAR TRANSPORT DYNAMIC/STABILIZATION STABTYPE: "no_stabilization" and passes
# its three result tests; switch it to SUPG with DEFINITION_TAU: "Codina" — the
# entry's own recommendation — and both scalar results move while the fluid
# velocity does not.
#
# What is worth pinning is that 4C says NOTHING either way: no Peclet number is
# reported, no warning that an advection-dominated field is running unstabilised,
# and both runs converge.  The only thing that distinguishes them is the deck's
# own 1e-08 result test.  So the "visible oscillations" of the entry are not a
# diagnostic you can wait for — the difference has to be looked for on purpose.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fs3i_part_1wc_infperm.4C.yaml) || exit 3
grep -q 'SCALAR TRANSPORT DYNAMIC/STABILIZATION:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_scatra_stab_section_changed"; exit 3; }
grep -q 'DIFFUSIVITY: 1.6' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_fluid_scalar_diffusivity_changed"; exit 3; }
cp -r "$(dirname "$BASE")/xml" "$TMP/" \
  || { echo "FIXTURE_ABORT=missing_teko_xml"; exit 3; }
cd "$TMP" || exit 3

# The contrast: what the fluid-side scalar stabilisation is set to.
STABILISED_TYPE='  STABTYPE: "SUPG"\n  DEFINITION_TAU: "Codina"'

cp "$BASE" "$TMP/nostab.yaml"
python3 - "$BASE" "$TMP/supg.yaml" "$STABILISED_TYPE" <<'PY'
import sys
t = open(sys.argv[1]).read()
old = 'SCALAR TRANSPORT DYNAMIC/STABILIZATION:\n  STABTYPE: "no_stabilization"'
assert old in t, "upstream deck no longer runs the fluid scalar unstabilised"
new = 'SCALAR TRANSPORT DYNAMIC/STABILIZATION:\n' + \
      sys.argv[3].encode().decode("unicode_escape")
open(sys.argv[2], "w").write(t.replace(old, new, 1))
PY
grep -A1 'SCALAR TRANSPORT DYNAMIC/STABILIZATION:' "$TMP/supg.yaml" | grep STABTYPE \
  | tr -d ' ' | sed 's/^/SUPG_ARM_[/;s/$/]/'

probe NOSTAB "$TMP/nostab.yaml"
probe SUPG   "$TMP/supg.yaml"

grep -m1 -F "OK (3)" "$TMP/NOSTAB.log"
grep -m1 -F "processor 0 finished normally" "$TMP/NOSTAB.log"
grep -m1 -F "Result check failed with 2 errors out of 3 tests" "$TMP/SUPG.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/SUPG.log"

echo "NOSTAB_FAILED_TESTS=$(grep -c 'is WRONG' "$TMP/NOSTAB.log")"
echo "SUPG_FAILED_TESTS=$(grep -c 'is WRONG' "$TMP/SUPG.log")"
# Both scalar fields moved; the fluid velocity did not.
echo "SUPG_SCATRA_TESTS_MOVED=$(grep -c 'SCATRA .*is WRONG' "$TMP/SUPG.log")"
echo "SUPG_FLUID_TESTS_MOVED=$(grep -c 'FLUID .*is WRONG' "$TMP/SUPG.log")"
# Neither run says a word about Peclet numbers or missing stabilisation.
echo "NOSTAB_PECLET_OR_STAB_WARNINGS=$(grep -ciE 'peclet|advection.?dominated|unstabili|no_stabilization is' "$TMP/NOSTAB.log")"
echo "NOSTAB_NONCONVERGENCE=$(grep -ci 'did not converge' "$TMP/NOSTAB.log")"
echo "SUPG_NONCONVERGENCE=$(grep -ci 'did not converge' "$TMP/SUPG.log")"
exit 0
