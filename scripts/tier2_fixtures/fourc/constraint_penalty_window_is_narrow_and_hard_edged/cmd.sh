#!/bin/bash
# Tier-2 for fourc::constraint#1 — the penalty parameter of a penalty-enforced
# coupling constraint really does trade accuracy against solvability, and the
# usable window is much narrower and much harder-edged than the entry suggested.
#
# Upstream deck: beam3r_herm2line3_beam_to_beam_point_coupling_elbow_offset_
# indirect_global_assembly — two beams joined by a BEAM TO BEAM POINT COUPLING
# condition with CONSTRAINT_ENFORCEMENT penalty_indirect, POSITIONAL_PENALTY_
# PARAMETER 1.2, and a result test on the tip pinned to 1e-12.
#
# Four arms, one number changed:
#
#   1.2      (as shipped)  -> converges, result test passes
#   1.2e-03  too soft      -> converges, and the joint has drifted: the tip is
#                             out by ~8e-2 on a coordinate of order 1e-1
#   1.2e+03  too stiff     -> "The nonlinear solver did not converge!"
#   1.2e+14  far too stiff -> same
#
# Two corrections fall out. First, the failure at the stiff end is a hard abort
# from NOX, not the "Trilinos iterative solver stagnates" the entry described —
# this deck solves with a direct solver and still fails, so the mechanism is the
# Newton problem, not the linear one. Second, "alpha ~ 1e3 - 1e5 times the
# stiffness diagonal" is not a safe recipe: three decades above the shipped value
# is already past the edge here. The lagrange_multiplier variant of the same
# problem is included as the control the entry recommends, and it converges.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream beam3r_herm2line3_beam_to_beam_point_coupling_elbow_offset_indirect_global_assembly.4C.yaml) || exit 3
LAG=$(upstream beam3r_herm2line3_beam_to_beam_point_coupling_elbow_offset_indirect_global_assembly_lagrange.4C.yaml) || exit 3
grep -q "POSITIONAL_PENALTY_PARAMETER: 1.2" "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

# Both decks name a NOX status-test XML by bare filename, resolved against the
# working directory, so it has to travel with them.
DECKDIR=$(dirname "$BASE")
NOXXML="beam3r_herm2line3_beam_to_beam_point_coupling_elbow.xml"
[ -f "$DECKDIR/$NOXXML" ] || { echo "FIXTURE_ABORT=missing_upstream_aux"; exit 3; }
cp "$DECKDIR/$NOXXML" "$TMP/"

cp "$BASE" "$TMP/tuned.yaml"
cp "$LAG"  "$TMP/lagrange.yaml"
sed 's/POSITIONAL_PENALTY_PARAMETER: 1.2/POSITIONAL_PENALTY_PARAMETER: 1.2e-3/'  "$BASE" > "$TMP/soft.yaml"
sed 's/POSITIONAL_PENALTY_PARAMETER: 1.2/POSITIONAL_PENALTY_PARAMETER: 1.2e3/'   "$BASE" > "$TMP/stiff.yaml"
sed 's/POSITIONAL_PENALTY_PARAMETER: 1.2/POSITIONAL_PENALTY_PARAMETER: 1.2e14/'  "$BASE" > "$TMP/verystiff.yaml"

cd "$TMP" || exit 3
probe TUNED     "$TMP/tuned.yaml"
probe SOFT      "$TMP/soft.yaml"
probe STIFF     "$TMP/stiff.yaml"
probe VERYSTIFF "$TMP/verystiff.yaml"
probe LAGRANGE  "$TMP/lagrange.yaml"

# Tuned and Lagrange both meet the 1e-12 result test.
echo "TUNED_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/TUNED.log")"
echo "LAGRANGE_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/LAGRANGE.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/TUNED.log"

# Too soft: it converges, and the constraint has drifted.
echo "SOFT_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/SOFT.log")"
grep -m1 -F "is WRONG --> actresult=" "$TMP/SOFT.log"

# Too stiff: NOX gives up. Same at 1e14.
grep -m1 -F "The nonlinear solver did not converge!" "$TMP/STIFF.log"
grep -m1 -F "4C_solver_nonlin_nox_problem.cpp" "$TMP/STIFF.log"
grep -m1 -F "The nonlinear solver did not converge!" "$TMP/VERYSTIFF.log"

python3 - "$TMP/STIFF.log" "$TMP/VERYSTIFF.log" <<'PY'
import sys
n = 0
for p in sys.argv[1:]:
    t = open(p, "rb").read().decode("utf-8", "replace").lower()
    n += t.count("condition number") + t.count("stagnat")
print("CLAIMED_CONDITIONING_WARNING=%d" % n)
PY
exit 0
