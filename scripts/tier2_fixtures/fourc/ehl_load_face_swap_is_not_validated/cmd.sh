#!/bin/bash
# Tier-2 for fourc::ehl#4 — moving the structural load to the wrong face is
# accepted without comment and produces exactly the "looks reasonable" answer the
# entry warns about.
#
# Upstream ehl3d_mixed.4C.yaml presses the upper body down with a 3-DOF
# DESIGN SURF DIRICH on DSURFACE 1 — its back face — while DSURFACE 2 is the
# Slave side of the EHL mortar contact.  Move that same condition onto DSURFACE
# 2, i.e. drive the contact face directly instead of loading it through the body,
# and 4C:
#   * parses it, builds every field, runs all 20 steps, prints no diagnostic
#     about the condition being applied to a coupling surface,
#   * gives a deformation that LOOKS right — the contact nodes move by exactly
#     the prescribed -5.00e-2 in y, so a plot shows a deflected body,
#   * and gets 5 of the deck's 7 result tests wrong, including both x
#     displacements collapsing to exactly 0.
#
# That is the whole pitfall: the load face is never validated, and the wrong
# answer is visually plausible.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ehl3d_mixed.4C.yaml) || exit 3
grep -q '^DESIGN SURF DIRICH CONDITIONS:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_surf_dirich_section_changed"; exit 3; }
grep -q '    Side: "Slave"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_slave_side_changed"; exit 3; }

# The pathology: which design surface carries the pressing Dirichlet.
LOAD_SURFACE=2

cp "$BASE" "$TMP/rightface.yaml"
python3 - "$BASE" "$TMP/wrongface.yaml" "$LOAD_SURFACE" <<'PY'
import sys
t = open(sys.argv[1]).read()
old = ('DESIGN SURF DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 3\n'
       '    ONOFF: [1, 1, 1]\n    VAL: [0, -1, 0]\n    FUNCT: [0, 2, 0]\n')
assert old in t, "upstream deck no longer drives DSURFACE 1 with a 3-DOF Dirichlet"
new = old.replace('  - E: 1\n', '  - E: %s\n' % sys.argv[3])
open(sys.argv[2], "w").write(t.replace(old, new, 1))
PY
python3 - "$TMP/wrongface.yaml" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
m = re.search(r'DESIGN SURF DIRICH CONDITIONS:\n  - E: (\d+)\n', t)
print("LOADED_SURFACE_IN_DECK=%s" % m.group(1))
PY

probe RIGHTFACE "$TMP/rightface.yaml"
probe WRONGFACE "$TMP/wrongface.yaml"

grep -m1 -F "OK (7)" "$TMP/RIGHTFACE.log"
grep -m1 -F "processor 0 finished normally" "$TMP/RIGHTFACE.log"
grep -m1 -F "Result check failed with 5 errors out of 7 tests" "$TMP/WRONGFACE.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/WRONGFACE.log"

echo "WRONGFACE_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/WRONGFACE.log")"
echo "WRONGFACE_FAILED_TESTS=$(grep -c 'is WRONG' "$TMP/WRONGFACE.log")"
# Not a word about the condition sitting on a coupling surface.
echo "LOAD_FACE_WARNINGS=$(grep -ciE 'dirichlet.*(coupling|contact|slave|master) (surface|side)|condition.*wrong (face|surface)' "$TMP/WRONGFACE.log")"
# The deformation is nonzero and looks like a deflection: the driven nodes take
# exactly the prescribed value while the x displacements collapse to zero.
if grep -qE "dispy +at node +8[24].*actresult=-5\.00000000000000028e-02" "$TMP/WRONGFACE.log"; then
  echo "WRONGFACE_DEFORMATION=nonzero_and_plausible"
else
  echo "WRONGFACE_DEFORMATION=other"
fi
if grep -qE "dispx +at node +82.*actresult= 0\.00000000000000000e\+00" "$TMP/WRONGFACE.log" \
   && grep -qE "dispx +at node +84.*actresult= 0\.00000000000000000e\+00" "$TMP/WRONGFACE.log"; then
  echo "WRONGFACE_TANGENTIAL_DISPLACEMENT=exactly_zero"
else
  echo "WRONGFACE_TANGENTIAL_DISPLACEMENT=nonzero"
fi
exit 0
