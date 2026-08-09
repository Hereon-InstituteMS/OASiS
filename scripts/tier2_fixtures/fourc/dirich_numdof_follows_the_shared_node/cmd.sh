#!/bin/bash
# Tier-2 for fourc::input_format#2 — in a multi-field problem a DESIGN ...
# DIRICH CONDITIONS block is attached to NODES, and every discretisation that
# contains those nodes reads it.  A node shared between two fields therefore
# receives the OTHER field's Dirichlet block, whose NUMDOF is that field's DOF
# count, and the DOF check refuses it.
#
# The probe changes ONE THING in an upstream monolithic FSI deck: the structure
# element is rewired onto the fluid's interface nodes 13-16 instead of its own
# coincident copies 17-20 (which are then removed and renamed in the topology).
# Not a single condition block is edited.  The deck goes from exit 0 to
#
#     3 DOFs given but 4 expected in Volume Dirichlet boundary condition
#
# from core/fem/.../4C_fem_discretization_utils_dbc.cpp -- the structure's
# NUMDOF-3 volume Dirichlet being read on the 4-DOF fluid discretisation, purely
# because the node is now in both.
#
# And there is no NUMDOF that rescues it: setting that same block to NUMDOF 4
# just moves the complaint to the next condition on the shared nodes.
#
# Two corrections to the entry are pinned: the string 'inconsistent NUMDOF on
# shared node' does not exist, and the failure does not come from
# 4C_io_input_spec.cpp -- it is a runtime DOF check, not a parse check.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fsi_fp_mono_fs_ga_ga.4C.yaml) || exit 3
grep -q '"1 SOLID HEX8 17 18 19 20 21 22 23 24 MAT 1 KINEM nonlinear"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/separate.yaml"
python3 - "$BASE" "$TMP" <<'PY'
import re, sys
base, tmp = sys.argv[1], sys.argv[2]
b = open(base).read()

# The ONLY edit: the structure element reuses the fluid's interface nodes.
s = b.replace('"1 SOLID HEX8 17 18 19 20 21 22 23 24 MAT 1 KINEM nonlinear"',
              '"1 SOLID HEX8 13 14 15 16 21 22 23 24 MAT 1 KINEM nonlinear"')
assert s != b
for i in (17, 18, 19, 20):                       # drop the now-orphaned copies
    s = re.sub(r'  - "NODE %d COORD [^\n]*\n' % i, '', s)
for a, c in ((17, 13), (18, 14), (19, 15), (20, 16)):
    s = s.replace('"NODE %d D' % a, '"NODE %d D' % c)
open(tmp + "/shared.yaml", "w").write(s)

# Same mesh, but the structure volume Dirichlet declared with the fluid's NUMDOF.
s4 = s.replace('''  - E: 2
    NUMDOF: 3
    ONOFF: [0, 1, 1]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
DESIGN SURF DIRICH CONDITIONS:''', '''  - E: 2
    NUMDOF: 4
    ONOFF: [0, 1, 1, 0]
    VAL: [0, 0, 0, 0]
    FUNCT: [0, 0, 0, 0]
DESIGN SURF DIRICH CONDITIONS:''')
assert s4 != s
open(tmp + "/shared_numdof4.yaml", "w").write(s4)

# Prove the conditions themselves were not touched.
def conds(t):
    return t[t.index('DESIGN POINT DIRICH CONDITIONS:'):t.index('DNODE-NODE TOPOLOGY:')]
print("ONLY_THE_MESH_WAS_EDITED=%s" % ("yes" if conds(b) == conds(s) else "no"))
PY

probe SEPARATE "$TMP/separate.yaml"
probe SHARED   "$TMP/shared.yaml"
probe SHARED4  "$TMP/shared_numdof4.yaml"

# Duplicated interface nodes: the deck runs.
grep -m1 -F "processor 0 finished normally" "$TMP/SEPARATE.log"
echo "SEPARATE_CORRECT=$(grep -c 'is CORRECT' "$TMP/SEPARATE.log")"
# Shared interface nodes: the structure's NUMDOF-3 block lands on 4-DOF nodes.
grep -m1 -F "3 DOFs given but 4 expected in Volume Dirichlet boundary condition" "$TMP/SHARED.log"
grep -m1 -F "4C_fem_discretization_utils_dbc.cpp" "$TMP/SHARED.log"
# Declaring the fluid's NUMDOF instead just moves the complaint along.
grep -m1 -F "3 DOFs given but 4 expected in Point Dirichlet boundary condition" "$TMP/SHARED4.log"
echo "SHARED4_STILL_FAILS=$(grep -c 'DOFs given but' "$TMP/SHARED4.log")"
# Corrections to the entry.
echo "CLAIMED_INCONSISTENT_NUMDOF_TEXT=$(cat "$TMP/SHARED.log" "$TMP/SHARED4.log" | grep -ci 'inconsistent NUMDOF on shared node')"
echo "BLAMED_INPUT_SPEC_CPP=$(grep -c '4C_io_input_spec.cpp' "$TMP/SHARED.log")"
exit 0
