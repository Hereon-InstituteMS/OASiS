#!/bin/bash
# Tier-2 for fourc::fsi_xfem#2 -- a non-watertight cutter surface is caught, but
# by the integration rule, not by a classification check.
#
# Claimed: the cut algorithm aborts with 'inside/outside classification
#          inconsistent', or silently lets the fluid permeate the structure.
# Observed: no such string exists.  Drop one of the four coupling sides of the
#          cutter body in the upstream mesh-cutter XFEM deck -- leaving an open
#          ring instead of a closed one -- and 4C aborts with "negative volume
#          predicted by the DirectDivergence integration rule;" from
#          4C_cut_direct_divergence.cpp.  It is a geometry/quadrature message
#          that never says the surface is open, and there is no silent
#          permeation: the run stops.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream xfluid_mesh_neumann_inflow_stab.4C.yaml) || exit 3
grep -q '  - "SIDE structure y- DSURFACE 4"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/closed.yaml"
python3 - "$BASE" "$TMP/open.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
disp = """  - E: 4
    COUPLINGID: 1
    EVALTYPE: "zero"
    NUMDOF: 3
    ONOFF: [0, 0, 0]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
"""
neum = """  - E: 4
    COUPLINGID: 1
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [-5, -5, 0]
    FUNCT: [1, 1, 1]
    INFLOW_STAB: true
"""
topo = '  - "SIDE structure y- DSURFACE 4"\n'
for s in (disp, neum, topo):
    assert s in t, "upstream deck no longer carries the fourth cutter side"
open(sys.argv[2], "w").write(t.replace(disp, "").replace(neum, "").replace(topo, ""))
PY

probe CLOSED "$TMP/closed.yaml"
probe OPEN   "$TMP/open.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/CLOSED.log"
echo "CLOSED_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/CLOSED.log")"
grep -m1 -F "negative volume predicted by the DirectDivergence integration rule" "$TMP/OPEN.log"
grep -m1 -F "4C_cut_direct_divergence.cpp" "$TMP/OPEN.log"
# no classification wording, and no quiet continuation
echo "CLAIMED_CLASSIFICATION_TEXT=$(grep -ciE 'inside/outside classification|classification inconsistent' "$TMP/OPEN.log")"
echo "OPEN_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/OPEN.log")"
exit 0
