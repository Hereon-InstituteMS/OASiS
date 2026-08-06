#!/bin/bash
# Tier-2 for fourc::beams#6 — MASSLIN: rotations really is required alongside
# DYNAMICTYPE GenAlphaLieGroup, and the way you find out is the worst possible
# one.
#
# Claimed:  it "aborts with 'inconsistent mass linearisation for Lie-group
#           integrator' at setup".
# Observed: no such string exists in 4C and none is printed. MASSLIN defaults to
#           'none', GenAlphaLieGroup accepts that without comment, and the
#           process dies of SIGSEGV during post_setup ->
#           compute_mass_matrix_and_init_acc -> Beam3r::calc_inertia_force_and_
#           mass_matrix. Exit status 139, zero "PROC 0 ERROR" lines, no hint of
#           which key was wrong.
#
# Two arms differing in exactly one line of the upstream 3D-twist-moment deck.
. "$(dirname "$0")/../_lib/preamble.sh"
ulimit -c 0   # the bad arm dies of SIGSEGV; do not leave core files behind

BASE=$(upstream beam3r_line3_genalpha_liegroup_3Dtwistmoment.4C.yaml) || exit 3
grep -q 'MASSLIN: "rotations"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/rotations.yaml"
sed 's/MASSLIN: "rotations"/MASSLIN: "none"/' "$BASE" > "$TMP/ml_none.yaml"

probe ROTATIONS "$TMP/rotations.yaml"
probe NONE      "$TMP/ml_none.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/ROTATIONS.log"
echo "ROTATIONS_STEPS=$(grep -c 'Finalised step' "$TMP/ROTATIONS.log")"

# The bad arm: crash, not abort.
grep -m1 -F "Signal: Segmentation fault (11)" "$TMP/NONE.log"
grep -m1 -F "GenAlphaLieGroup" "$TMP/NONE.log"
grep -m1 -F "compute_mass_matrix_and_init_acc" "$TMP/NONE.log"
grep -m1 -F "calc_inertia_force_and_mass_matrix" "$TMP/NONE.log"
# 4C says nothing at all: no diagnostic and no step.
# The three absences are read straight off the bytes of the log, so that the
# verdict cannot depend on which grep implementation is on PATH.
python3 - "$TMP/NONE.log" <<'PY'
import sys
log = open(sys.argv[1], "rb").read().decode("utf-8", "replace").lower()
print("NONE_4C_DIAGNOSTICS=%d" % log.count("proc 0 error"))
print("NONE_STEPS=%d" % log.count("finalised step"))
print("CLAIMED_INCONSISTENT_MASS_LIN_TEXT=%d" % log.count("inconsistent mass linearisation"))
PY
exit 0
