#!/bin/bash
# Tier-2 for fourc::ale#1 — a Neumann condition on a stand-alone ALE field is
# accepted by the parser and then does nothing.  The claim is that this is
# SILENT, which is the dangerous part, so the fixture has to demonstrate both
# halves: no diagnostic, and no motion.
#
# Baseline (upstream ale2d_solid) drives DNODE 2 with a Dirichlet ramp and
# result-tests node 3 at dispx=-6.302e-2, dispy=0.25.  The bad arm replaces that
# Dirichlet with a Neumann of 1000 on the same node set.  4C parses it, runs two
# steps, exits... with node 3 at EXACTLY zero in both components.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ale2d_solid.4C.yaml) || exit 3

python3 - "$BASE" "$TMP/neumann.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
drive = """  - E: 2
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0, 1]
    FUNCT: [0, 1]
"""
assert drive in t, "upstream deck no longer carries the DNODE 2 Dirichlet drive"
t = t.replace(drive, "")
t = t.replace("DNODE-NODE TOPOLOGY:", """DESIGN POINT NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0, 1000]
    FUNCT: [0, 1]
    TYPE: "Live"
DNODE-NODE TOPOLOGY:""")
open(sys.argv[2], "w").write(t)
PY

cp "$BASE" "$TMP/dirichlet.yaml"
probe DIRICHLET "$TMP/dirichlet.yaml"
probe NEUMANN   "$TMP/neumann.yaml"

# Did 4C say anything at all about the Neumann condition being inert?
echo "NEUMANN_WARNINGS=$(grep -ciE 'neumann.*(ignor|unus|not applied|no effect)' "$TMP/NEUMANN.log")"
# The parser accepted the section: the run reached the result test.
echo "NEUMANN_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/NEUMANN.log")"
# And the mesh did not move.
grep -m2 -F "is WRONG --> actresult=" "$TMP/NEUMANN.log"
if grep -qE "dispx +at node +3.*actresult= 0\.00000000000000000e\+00" "$TMP/NEUMANN.log" \
   && grep -qE "dispy +at node +3.*actresult= 0\.00000000000000000e\+00" "$TMP/NEUMANN.log"; then
  echo "NEUMANN_NODE3_DISPLACEMENT=exactly_zero"
else
  echo "NEUMANN_NODE3_DISPLACEMENT=nonzero"
fi
exit 0
