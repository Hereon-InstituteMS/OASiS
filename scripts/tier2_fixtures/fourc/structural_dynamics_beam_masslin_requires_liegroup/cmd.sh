#!/bin/bash
# Tier-2 for fourc::structural_dynamics#2 — for beam elements the pairing
# DYNAMICTYPE: GenAlphaLieGroup + MASSLIN: rotations is not a recommendation,
# it is a hard requirement, and BOTH halves of it are enforced — one loudly,
# one not at all.
#
# Upstream deck beam3r_herm2line3_genalpha_liegroup_lineload_dynamic, shortened
# to a handful of steps and with its RESULT DESCRIPTION removed so the control
# arm can exit 0.  Three arms:
#
#   LIEGROUP            GenAlphaLieGroup + MASSLIN rotations -> runs, exit 0
#   CLASSICAL_GENALPHA  GenAlpha         + MASSLIN rotations -> explicit throw
#                       naming the scheme you should have used
#   LIEGROUP_NO_MASSLIN GenAlphaLieGroup + MASSLIN none      -> SEGFAULT inside
#                       Beam3r's inertia/mass assembly, with no message at all
#
# The second failure mode is the one worth knowing: forgetting MASSLIN on a
# Lie-group beam run does not produce a diagnostic, it produces signal 11.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream beam3r_herm2line3_genalpha_liegroup_lineload_dynamic.4C.yaml) || exit 3
for token in 'DYNAMICTYPE: "GenAlphaLieGroup"' 'MASSLIN: "rotations"' '  MAXTIME: 2' 'RESULT DESCRIPTION:'; do
  grep -qF "$token" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed ($token)"; exit 3; }
done

python3 - "$BASE" "$TMP" <<'PY'
import sys, re
base, tmp = sys.argv[1], sys.argv[2]
t = open(base).read()
t = t.replace("  MAXTIME: 2", "  MAXTIME: 0.05")
# Drop the RESULT DESCRIPTION block: at 0.05 s none of its pinned values apply,
# and the control arm has to be able to exit 0.
t = re.sub(r"RESULT DESCRIPTION:\n(?:  [-\s].*\n)+", "", t, count=1)
assert "RESULT DESCRIPTION" not in t
open(tmp + "/liegroup.4C.yaml", "w").write(t)
open(tmp + "/classical.4C.yaml", "w").write(
    t.replace('DYNAMICTYPE: "GenAlphaLieGroup"', 'DYNAMICTYPE: "GenAlpha"'))
open(tmp + "/nomasslin.4C.yaml", "w").write(
    t.replace('MASSLIN: "rotations"', 'MASSLIN: "none"'))
PY

probe LIEGROUP "$TMP/liegroup.4C.yaml"
probe CLASSICAL_GENALPHA "$TMP/classical.4C.yaml"
# This arm dies on SIGSEGV; the shell's job message is locale dependent.
( probe LIEGROUP_NO_MASSLIN "$TMP/nomasslin.4C.yaml" ) 2>/dev/null

grep -m1 -F "processor 0 finished normally" "$TMP/LIEGROUP.log"
echo "LIEGROUP_STEPS=$(grep -c '^Finalised step' "$TMP/LIEGROUP.log")"
# Loud half: the scheme you should have used is named in the message.
grep -m1 -F "MASSLIN=ml_rotations is not supported by classical GenAlpha! Choose GenAlphaLieGroup instead!" "$TMP/CLASSICAL_GENALPHA.log"
grep -m1 -F "4C_structure_new_impl_genalpha.cpp" "$TMP/CLASSICAL_GENALPHA.log"
# Silent half: no 4C diagnostic at all, just a signal.
echo "NO_MASSLIN_STEPS=$(grep -c '^Finalised step' "$TMP/LIEGROUP_NO_MASSLIN.log")"
echo "NO_MASSLIN_4C_ERROR_LINES=$(grep -c 'PROC 0 ERROR in' "$TMP/LIEGROUP_NO_MASSLIN.log")"
grep -m1 -F "Signal: Segmentation fault (11)" "$TMP/LIEGROUP_NO_MASSLIN.log"
grep -m1 -oF "GenAlphaLieGroup10post_setup" "$TMP/LIEGROUP_NO_MASSLIN.log"
exit 0
