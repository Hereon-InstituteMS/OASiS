#!/bin/bash
# Tier-2 for fourc::fsi#4 — SHAPEDERIVATIVES is NOT required for monolithic FSI,
# it is off by default, and switching it on costs Newton iterations here.
#
# Claimed: "SHAPEDERIVATIVES: true is REQUIRED in FSI DYNAMIC/MONOLITHIC SOLVER
#           for monolithic schemes ... with SHAPEDERIVATIVES: false, the
#           monolithic Newton iteration is missing a term and shows linear (not
#           quadratic) convergence."
# Observed: the 4C default is false (4C_inpar_fsi.cpp declares the parameter with
#           .default_value = false), and the upstream 2D driven-cavity monolithic
#           benchmark fsi_dc_mono_fs_ga_ga.4C.yaml does not set the key at all.
#           It converges every one of its 10 time steps and matches all nine
#           pinned results with the flag OFF, taking 81 Newton steps.  Adding
#           SHAPEDERIVATIVES: true to the same deck also converges and also
#           matches all nine results — in 87 Newton steps, i.e. six MORE.  So the
#           flag is optional, it is off unless you ask for it, and on this
#           benchmark asking for it makes the solve longer rather than shorter.
#
# The deck names its MueLu preconditioner file by a path relative to the deck
# directory, so both arms are rewritten to an absolute path; that edit is
# identical in both arms and is not the variable under test.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fsi_dc_mono_fs_ga_ga.4C.yaml) || exit 3
grep -q '^FSI DYNAMIC/MONOLITHIC SOLVER:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_has_no_monolithic_solver_section"; exit 3; }
grep -q 'COUPALGO: "iter_monolithicfluidsplit"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_is_not_monolithic"; exit 3; }
echo "UPSTREAM_SETS_SHAPEDERIVATIVES=$(grep -c 'SHAPEDERIVATIVES' "$BASE")"

# The pathology: leave the "REQUIRED" flag at its 4C default of false.
SHAPEDERIV_IN_BAD_ARM=false

python3 - "$BASE" "$TMP" "$DECKS" "$SHAPEDERIV_IN_BAD_ARM" <<'PY'
import sys
src, tmp, decks, flag = sys.argv[1:5]
t = open(src).read()
rel = 'MUELU_XML_FILE: "xml/multigrid/fluid_solid_ale.xml"'
assert rel in t, "upstream deck no longer uses the relative MueLu xml path"
t = t.replace(rel, 'MUELU_XML_FILE: "%s/xml/multigrid/fluid_solid_ale.xml"' % decks)
open(tmp + "/default.yaml", "w").write(
    t.replace("FSI DYNAMIC/MONOLITHIC SOLVER:",
              "FSI DYNAMIC/MONOLITHIC SOLVER:\n  SHAPEDERIVATIVES: " + flag, 1))
open(tmp + "/shapederiv.yaml", "w").write(
    t.replace("FSI DYNAMIC/MONOLITHIC SOLVER:",
              "FSI DYNAMIC/MONOLITHIC SOLVER:\n  SHAPEDERIVATIVES: true", 1))
PY

probe DEFAULT    "$TMP/default.yaml"
probe SHAPEDERIV "$TMP/shapederiv.yaml"

# Both converge and both match every pinned result.
grep -m1 -F "OK (9)" "$TMP/DEFAULT.log"
grep -m1 -F "OK (9)" "$TMP/SHAPEDERIV.log"
grep -m1 -F "processor 0 finished normally" "$TMP/DEFAULT.log"

D=$(grep -c 'Nonlinear Solver Step' "$TMP/DEFAULT.log")
S=$(grep -c 'Nonlinear Solver Step' "$TMP/SHAPEDERIV.log")
echo "NEWTON_STEPS_SHAPEDERIVATIVES_OFF=$D"
echo "NEWTON_STEPS_SHAPEDERIVATIVES_ON=$S"
echo "DEFAULT_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/DEFAULT.log")"
echo "DEFAULT_NONCONVERGENCE=$(grep -ci 'did not converge' "$TMP/DEFAULT.log")"
if [ "$S" -gt "$D" ]; then
  echo "VERDICT: SHAPEDERIVATIVES_ON_COSTS_MORE_NEWTON_STEPS=yes"
else
  echo "VERDICT: SHAPEDERIVATIVES_ON_COSTS_MORE_NEWTON_STEPS=no"
fi
exit 0
