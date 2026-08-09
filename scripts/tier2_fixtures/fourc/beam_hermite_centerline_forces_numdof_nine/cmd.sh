#!/bin/bash
# Tier-2 for fourc::beams#4 — HERMITE_CENTERLINE true really does take a BEAM3R
# node from 6 to 9 DOFs, but the consequence of not following it is not the one
# the entry described.
#
# Claimed:  "the tangent DOFs are silently allocated but DESIGN ... DIRICH BCs
#           with NUMDOF=6 leave them free, and you get spurious tangent growth
#           at constrained nodes" — i.e. a run that completes with bad physics.
# Observed: it does not run. The DBC reader compares the condition's NUMDOF
#           against the DOFs the node actually has and aborts:
#
#     6 DOFs given but 9 expected in Point Dirichlet boundary condition
#     .../core/fem/src/discretization/4C_fem_discretization_utils_dbc.cpp
#
# Upstream beam3r_herm2line2_static_test1 is the same cantilever as
# beam3r_line2_static_test1 with HERMITE_CENTERLINE true, and its Dirichlet and
# Neumann blocks carry NUMDOF 9 with nine-entry ONOFF/VAL/FUNCT vectors — that
# contrast is the whole content of the pitfall. The bad arm cuts the Dirichlet
# block back to the non-Hermite NUMDOF 6.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream beam3r_herm2line2_static_test1.4C.yaml) || exit 3
grep -q "HERMITE_CENTERLINE" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/nine.yaml"

python3 - "$BASE" "$TMP/six.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
nine = """  - E: 1
    NUMDOF: 9
    ONOFF: [1, 1, 1, 1, 1, 1, 0, 0, 0]
    VAL: [0, 0, 0, 0, 0, 0, 0, 0, 0]
    FUNCT: [0, 0, 0, 0, 0, 0, 0, 0, 0]
"""
six = """  - E: 1
    NUMDOF: 6
    ONOFF: [1, 1, 1, 1, 1, 1]
    VAL: [0, 0, 0, 0, 0, 0]
    FUNCT: [0, 0, 0, 0, 0, 0]
"""
assert nine in t, "upstream Hermite deck no longer carries the 9-DOF point Dirichlet"
open(sys.argv[2], "w").write(t.replace(nine, six, 1))
PY

probe NINE "$TMP/nine.yaml"
probe SIX  "$TMP/six.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/NINE.log"
grep -m1 -F "6 DOFs given but 9 expected in Point Dirichlet boundary condition" "$TMP/SIX.log"
grep -m1 -F "4C_fem_discretization_utils_dbc.cpp" "$TMP/SIX.log"
# No step is taken, so there is nothing "silent" and no tangent growth to see.
echo "SIX_STEPS_TAKEN=$(grep -c 'Finalised step' "$TMP/SIX.log")"
echo "CLAIMED_SPURIOUS_TANGENT_TEXT=$(grep -ciE 'tangent growth|silently allocated' "$TMP/SIX.log")"
exit 0
