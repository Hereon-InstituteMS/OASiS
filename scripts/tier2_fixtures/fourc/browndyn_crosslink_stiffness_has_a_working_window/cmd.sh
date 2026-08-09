#!/bin/bash
# Tier-2 for fourc::brownian_dynamics#2 — cross-link stiffness has a working
# window, and both walls of it are hard aborts in different subsystems.
#
# The stiffness knob is not on MAT_Crosslinker itself: MAT_Crosslinker points at
# a beam material through MATNUM, and that beam's YOUNG is k_xl.  Upstream's
# periodic-RVE crosslinking deck uses 1.3e+09 there (MAT 3, separate from the
# filament material MAT 1 even though the two happen to carry the same numbers).
#
#   1.3e+09  upstream: runs, passes its own result test
#   1.3e+03  too soft: the linker stretches past half the periodic box and 4C
#            aborts in the crosslinking submodel evaluator with
#              You are trying to set the binding spot positions of this crosslinker in at least one direction
#            — the network has lost its elastic constraint, exactly the
#            "deforms like a viscous fluid" end of the claim
#   1.3e+15  too stiff: the Newton solve for the constrained network stops
#            converging and 4C aborts with
#              The nonlinear solver did not converge!
#            from solver_nonlin_nox/4C_solver_nonlin_nox_problem.cpp — the
#            "rigid, fluctuations locked out" end
#
# Only MAT 3 is touched; the filament material is left alone in every arm.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream beam3r_line2_backweuler_browndyn_periodic_rve_dirich_crosslinking.4C.yaml) || exit 3
grep -q "MAT_Crosslinker" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "MATNUM: 3"       "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

arm() {  # $1 = linker YOUNG, $2 = label
  python3 - "$BASE" "$TMP/$2.yaml" "$1" <<'PY'
import sys
t = open(sys.argv[1]).read()
old = "  - MAT: 3\n    MAT_BeamReissnerElastHyper:\n      YOUNG: 1.3e+09"
assert old in t, "upstream deck no longer defines the crosslinker beam material as MAT 3"
open(sys.argv[2], "w").write(t.replace(old, old.replace("YOUNG: 1.3e+09", "YOUNG: %s" % sys.argv[3])))
PY
  probe "$2" "$TMP/$2.yaml"
}

arm 1.3e+09 ASSHIPPED
arm 1.3e+03 TOOSOFT
arm 1.3e+15 TOOSTIFF

echo "FILAMENT_MATERIAL_UNTOUCHED=$( a=$(grep -A2 '  - MAT: 1' "$TMP/ASSHIPPED.yaml" | grep -c 'YOUNG: 1.3e+09'); \
  b=$(grep -A2 '  - MAT: 1' "$TMP/TOOSOFT.yaml" | grep -c 'YOUNG: 1.3e+09'); \
  c=$(grep -A2 '  - MAT: 1' "$TMP/TOOSTIFF.yaml" | grep -c 'YOUNG: 1.3e+09'); \
  [ "$a" = "$b" ] && [ "$b" = "$c" ] && echo yes || echo no )"

grep -m1 -F "processor 0 finished normally" "$TMP/ASSHIPPED.log"
echo "ASSHIPPED_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/ASSHIPPED.log")"
grep -m1 -F "Time step 0.01 form Structural Dynamic section used for crosslinking." "$TMP/ASSHIPPED.log"

# Too soft: the linker outruns the periodic box.
grep -m1 -F "You are trying to set the binding spot positions of this crosslinker in at least one direction" "$TMP/TOOSOFT.log"
grep -m1 -oE "4C_beaminteraction_crosslinking_submodel_evaluator\.cpp, line [0-9]+" "$TMP/TOOSOFT.log"

# Too stiff: the network Newton stops converging.
grep -m1 -F "The nonlinear solver did not converge!" "$TMP/TOOSTIFF.log"
grep -m1 -oE "4C_solver_nonlin_nox_problem\.cpp, line [0-9]+" "$TMP/TOOSTIFF.log"

if [ "$(grep -c 'is WRONG --> actresult=' "$TMP/ASSHIPPED.log")" = "0" ]; then
  echo "VERDICT: CROSSLINK_STIFFNESS_IS_BRACKETED_BY_TWO_ABORTS=yes"
else
  echo "VERDICT: CROSSLINK_STIFFNESS_IS_BRACKETED_BY_TWO_ABORTS=no"
fi
exit 0
