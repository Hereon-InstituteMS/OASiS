#!/bin/bash
# Tier-2 for fourc::fsi#6 — the FSI coupling condition is NOT tied to the spatial
# dimension, and getting the geometry entity wrong is never silent.
#
# Claimed: "2D uses DESIGN FSI COUPLING LINE CONDITIONS, 3D uses DESIGN FSI
#           COUPLING SURF CONDITIONS.  Signal: a 2D problem with SURF CONDITIONS
#           (or vice versa) silently has ZERO coupling nodes — the FSI interface
#           is degenerate and structure / fluid evolve independently; neither one
#           diverges."
# Observed: FSI::Monolithic::setup_system looks the interface up by CONDITION
#           NAME ("FSICoupling"), not by geometry type.  Two arms on the 3D
#           upstream deck fsi_fp_mono_fs_ga_ga.4C.yaml prove both halves wrong:
#
#   LINE_WITH_TOPOLOGY  rename the section to DESIGN FSI COUPLING LINE
#                       CONDITIONS and add a DLINE-NODE TOPOLOGY carrying the
#                       same eight interface nodes -> the 3D problem couples
#                       perfectly through LINE conditions: exit 0, OK (6),
#                       identical to the SURF baseline.
#   LINE_NO_TOPOLOGY    rename the section and leave the DLine set undefined ->
#                       a hard abort before time step 1,
#                         "DLine 0 not in range [0:0["
#                         "DLine condition on non existent DLine?Could not read
#                          set from entity type."
#                       from core/fem/src/condition/4C_fem_condition.cpp line 120.
#
# So the entity type must EXIST, but which entity type you use is not fixed by
# the dimension, and a wrong one aborts loudly instead of quietly decoupling.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fsi_fp_mono_fs_ga_ga.4C.yaml) || exit 3
grep -q '^DESIGN FSI COUPLING SURF CONDITIONS:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_has_no_surf_fsi_condition"; exit 3; }
grep -q '^  DIM: 2' "$BASE" && { echo "FIXTURE_ABORT=upstream_deck_is_not_3d"; exit 3; }

# The pathology: declare the FSI interface of a 3D problem on LINE entities.
FSI_COND_ENTITY=LINE

cp "$BASE" "$TMP/surf.yaml"
python3 - "$BASE" "$TMP" "$FSI_COND_ENTITY" <<'PY'
import sys
src, tmp, ent = sys.argv[1:4]
t = open(src).read()
swapped = t.replace("DESIGN FSI COUPLING SURF CONDITIONS:",
                    "DESIGN FSI COUPLING %s CONDITIONS:" % ent, 1)
open(tmp + "/line_no_topology.yaml", "w").write(swapped)
topo = "DLINE-NODE TOPOLOGY:\n" + "".join(
    '  - "NODE %d DLINE %d"\n' % (n, 1 if n < 17 else 2) for n in range(13, 21))
assert "DNODE-NODE TOPOLOGY:" in swapped
open(tmp + "/line_with_topology.yaml", "w").write(
    swapped.replace("DNODE-NODE TOPOLOGY:", topo + "DNODE-NODE TOPOLOGY:", 1))
PY
echo "BAD_ARMS_STILL_HAVE_SURF_CONDITION=$(grep -c 'FSI COUPLING SURF CONDITIONS' "$TMP/line_with_topology.yaml")"
echo "BAD_ARMS_USE_ENTITY=$(grep -o 'DESIGN FSI COUPLING [A-Z]* CONDITIONS' "$TMP/line_with_topology.yaml" | awk '{print $4}')"

probe SURF        "$TMP/surf.yaml"
probe LINETOPO    "$TMP/line_with_topology.yaml"
probe LINENOTOPO  "$TMP/line_no_topology.yaml"

# Baseline.
grep -m1 -F "processor 0 finished normally" "$TMP/SURF.log"
grep -m1 -F "OK (6)" "$TMP/SURF.log"

# A 3D problem coupled through LINE conditions is accepted and gives the same
# answer, so "3D uses SURF" is not a rule 4C enforces.
grep -m1 -F "processor 0 finished normally" "$TMP/LINETOPO.log"
grep -m1 -F "OK (6)" "$TMP/LINETOPO.log"
echo "LINETOPO_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/LINETOPO.log")"

# Naming an entity type the deck never defines is a hard abort, not silence.
grep -m1 -F "DLine 0 not in range [0:0[" "$TMP/LINENOTOPO.log"
grep -m1 -F "DLine condition on non existent DLine?" "$TMP/LINENOTOPO.log"
grep -m1 -F "4C_fem_condition.cpp" "$TMP/LINENOTOPO.log"
echo "LINENOTOPO_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/LINENOTOPO.log")"
echo "LINENOTOPO_ZERO_COUPLING_NODE_MESSAGE=$(grep -ciE 'zero coupling nodes|no coupling nodes|degenerate interface' "$TMP/LINENOTOPO.log")"
exit 0
