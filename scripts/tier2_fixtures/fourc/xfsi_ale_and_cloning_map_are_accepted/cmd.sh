#!/bin/bash
# Tier-2 for fourc::fsi_xfem#0 -- FSI-XFEM does not reject ALE DYNAMIC or
# CLONING MATERIAL MAP.  It accepts both and ignores them.
#
# Claimed: including ALE DYNAMIC or CLONING MATERIAL MAP aborts with
#          'XFEM and ALE are mutually exclusive' from 4C_xfem_fluid_setup.cpp.
# Observed: no such string, and no such source file.  The upstream monolithic
#          XFSI deck with BOTH sections added runs to "processor 0 finished
#          normally" and matches all seven of its pinned results.  An agent that
#          leaves an ALE block in an XFSI input gets no feedback whatsoever.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream xfsi_3D_boxes.4C.yaml) || exit 3
grep -q "^XFEM GENERAL:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/clean.yaml"
python3 - "$BASE" "$TMP/aled.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
t = t.replace("XFEM GENERAL:", """ALE DYNAMIC:
  TIMESTEP: 0.05
  NUMSTEP: 1
  LINEAR_SOLVER: 2
CLONING MATERIAL MAP:
  - SRC_FIELD: "structure"
    SRC_MAT: 2
    TAR_FIELD: "ale"
    TAR_MAT: 2
XFEM GENERAL:""", 1)
open(sys.argv[2], "w").write(t)
PY

probe CLEAN "$TMP/clean.yaml"
probe ALED  "$TMP/aled.yaml"

# both forbidden sections really are in the deck 4C accepted
echo "ALED_HAS_ALE_DYNAMIC=$(grep -c '^ALE DYNAMIC:' "$TMP/aled.yaml")"
echo "ALED_HAS_CLONING_MAP=$(grep -c '^CLONING MATERIAL MAP:' "$TMP/aled.yaml")"
grep -m1 -F "processor 0 finished normally" "$TMP/ALED.log"
echo "ALED_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/ALED.log")"
echo "CLAIMED_MUTUALLY_EXCLUSIVE_TEXT=$(grep -ciE 'mutually exclusive|xfem_fluid_setup' "$TMP/ALED.log")"
exit 0
