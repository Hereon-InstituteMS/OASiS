#!/bin/bash
# Tier-2 for fourc::fsi#12 — separate interface nodes ARE needed, but 4C never
# says "no FSI interface nodes found" and does not run uncoupled.  It couples,
# runs, and the monolithic Newton stalls part-way through.
#
# Claimed: "a single Gmsh mesh shares nodes, and 4C reports 'no FSI interface
#           nodes found' or runs without coupling (fluid and solid never exchange
#           forces)."
# Observed: upstream fsi_fp_mono_fs_ga_ga.4C.yaml puts the fluid interface on
#           nodes 13..16 and the structure interface on nodes 17..20 at the SAME
#           coordinates.  Re-pointing the structure element and the DNODE/DSURF/
#           DVOL 2 topology rows at 13..16 makes the two fields share those four
#           nodes.  4C accepts the mesh, sets up the coupling and integrates —
#           until it aborts with
#             "Nonlinear solver did not converge in 20 iterations in time step 7."
#           from fsi/src/monolithic/4C_fsi_monolithic.cpp line 740, raised by
#           FSI::Monolithic::non_lin_error_check().  Neither the claimed sentence
#           nor any interface-node warning is printed anywhere.
#
# Both arms drive the structure from its FREE end (DNODE 3) instead of the
# interface, so that the shared node does not also trip the slave-Dirichlet
# check, and both have RESULT DESCRIPTION removed since the drive moved; the
# separate-node arm is therefore the honest control at exit 0 / OK (0).
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fsi_fp_mono_fs_ga_ga.4C.yaml) || exit 3
grep -q '"1 SOLID HEX8 17 18 19 20 21 22 23 24 MAT 1 KINEM nonlinear"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_structure_element_changed"; exit 3; }

# The pathology: let the structure element reuse the fluid's interface nodes.
STRUCT_INTERFACE_NODES="13 14 15 16"

python3 - "$BASE" "$TMP" "$STRUCT_INTERFACE_NODES" <<'PY'
import sys
src, tmp, nodes = sys.argv[1], sys.argv[2], sys.argv[3].split()
t = open(src).read()
i = t.index("RESULT DESCRIPTION:")
t = t[:i]
drive = ('  - E: 2\n    NUMDOF: 3\n    ONOFF: [1, 1, 1]\n'
         '    VAL: [1, 0, 0]\n    FUNCT: [1, 0, 0]')
assert drive in t
t = t.replace(drive, drive.replace("- E: 2", "- E: 3"), 1)
open(tmp + "/separate.yaml", "w").write(t)

u = t.replace('"1 SOLID HEX8 17 18 19 20 21 22 23 24 MAT 1 KINEM nonlinear"',
              '"1 SOLID HEX8 %s 21 22 23 24 MAT 1 KINEM nonlinear"' % " ".join(nodes), 1)
for old, new in zip((17, 18, 19, 20), (int(n) for n in nodes)):
    for tag in ("DNODE 2", "DSURFACE 2", "DVOL 2"):
        u = u.replace('  - "NODE %d %s"' % (old, tag), '  - "NODE %d %s"' % (new, tag))
# the DVOL 2 Dirichlet declares 3 entries; the shared nodes carry 4 fluid dofs
u = u.replace('  - E: 2\n    NUMDOF: 3\n    ONOFF: [0, 1, 1]\n'
              '    VAL: [0, 0, 0]\n    FUNCT: [0, 0, 0]',
              '  - E: 2\n    NUMDOF: 4\n    ONOFF: [0, 1, 1, 0]\n'
              '    VAL: [0, 0, 0, 0]\n    FUNCT: [0, 0, 0, 0]', 1)
open(tmp + "/shared.yaml", "w").write(u)
PY
python3 - "$TMP/shared.yaml" "$TMP/separate.yaml" <<'PY'
import re, sys
for path, label in ((sys.argv[1], "SHARED"), (sys.argv[2], "SEPARATE")):
    t = open(path).read()
    fluid = set(re.findall(r'FLUID HEX8 ((?:\d+ )+)MAT', t))
    fl = {int(x) for grp in fluid for x in grp.split()}
    st = {int(x) for grp in re.findall(r'SOLID HEX8 ((?:\d+ )+)MAT', t) for x in grp.split()}
    print("%s_NODES_IN_BOTH_FIELDS=%d" % (label, len(fl & st)))
PY

probe SEPARATE "$TMP/separate.yaml"
probe SHARED   "$TMP/shared.yaml"

# Control: disjoint interface nodes integrate the whole window.
grep -m1 -F "processor 0 finished normally" "$TMP/SEPARATE.log"
echo "SEPARATE_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/SEPARATE.log")"
echo "SEPARATE_NONCONVERGENCE=$(grep -ci 'did not converge' "$TMP/SEPARATE.log")"

# Shared nodes: the coupling is built, the run starts, the Newton stalls.
grep -m1 -F "Created discretization ale as a clone of discretization fluid" "$TMP/SHARED.log"
grep -m1 -F "Nonlinear solver did not converge in 20 iterations in time step 7." "$TMP/SHARED.log"
grep -m1 -F "4C_fsi_monolithic.cpp" "$TMP/SHARED.log"
grep -m1 -F "non_lin_error_check" "$TMP/SHARED.log"
echo "SHARED_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/SHARED.log")"
echo "SHARED_CLAIMED_TEXT=$(grep -ciE 'no FSI interface nodes' "$TMP/SHARED.log")"
echo "SHARED_ANY_INTERFACE_WARNING=$(grep -ciE 'interface.*(node|empty|degenerate).*(missing|not found|zero)' "$TMP/SHARED.log")"
exit 0
