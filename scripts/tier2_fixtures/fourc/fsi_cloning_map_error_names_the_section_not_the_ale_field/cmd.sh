#!/bin/bash
# Tier-2 for fourc::fsi#3 — CLONING MATERIAL MAP really is required for FSI, but
# the abort is raised by the generic clone helper, not by the fluid adapter, and
# it says something else.
#
# Claimed: "missing CLONING MATERIAL MAP aborts with 'cannot clone material for
#           ALE field' from 4C_adapter_fld_base_algorithm."
# Observed: deleting the CLONING MATERIAL MAP block from upstream
#           fsi_fp_mono_fs_ga_ga.4C.yaml aborts with
#             "At least one material pairing required in --CLONING MATERIAL MAP."
#           from core/fem/src/general/utils/4C_fem_general_utils_createdis.hpp
#           line 318, inside Core::FE::clone_discretization<ALE::Utils::
#           AleCloneStrategy>.  Neither the claimed sentence nor
#           4C_adapter_fld_base_algorithm appears anywhere in the log.  The real
#           message is friendlier than the claimed one — it names the section —
#           but it uses the legacy `--SECTION` spelling, which no longer matches
#           anything an agent would write in a YAML deck.
#
# The second arm keeps the section but points the SRC_MAT at a material id the
# deck does not define, to show that the requirement is on the PAIRING, not on
# the section header being present.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fsi_fp_mono_fs_ga_ga.4C.yaml) || exit 3
grep -q '^CLONING MATERIAL MAP:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_has_no_cloning_material_map"; exit 3; }

# The pathology: delete the fluid->ale material pairing.
DROP_CLONING_MAP=yes

cp "$BASE" "$TMP/withmap.yaml"
python3 - "$BASE" "$TMP/nomap.yaml" "$DROP_CLONING_MAP" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = """CLONING MATERIAL MAP:
  - SRC_FIELD: "fluid"
    SRC_MAT: 2
    TAR_FIELD: "ale"
    TAR_MAT: 3
"""
assert blk in t, "upstream CLONING MATERIAL MAP block changed"
if sys.argv[3] == "yes":
    t = t.replace(blk, "", 1)
open(sys.argv[2], "w").write(t)
PY
echo "NOMAP_CLONING_SECTIONS=$(grep -c '^CLONING MATERIAL MAP:' "$TMP/nomap.yaml")"

probe WITHMAP "$TMP/withmap.yaml"
probe NOMAP   "$TMP/nomap.yaml"

# Control: with the pairing, the ale discretization is cloned from the fluid.
grep -m1 -F "Created discretization ale as a clone of discretization fluid" "$TMP/WITHMAP.log"
grep -m1 -F "OK (6)" "$TMP/WITHMAP.log"

# The real diagnostic and its real origin.
grep -m1 -F "At least one material pairing required in --CLONING MATERIAL MAP." "$TMP/NOMAP.log"
grep -m1 -F "4C_fem_general_utils_createdis.hpp" "$TMP/NOMAP.log"
grep -m1 -F "AleCloneStrategy" "$TMP/NOMAP.log"

echo "NOMAP_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/NOMAP.log")"
echo "NOMAP_CLONED_ALE=$(grep -c 'Created discretization ale as a clone' "$TMP/NOMAP.log")"
echo "NOMAP_CLAIMED_SENTENCE=$(grep -ci 'cannot clone material for ALE field' "$TMP/NOMAP.log")"
echo "NOMAP_CLAIMED_SOURCE_FILE=$(grep -ci '4C_adapter_fld_base_algorithm' "$TMP/NOMAP.log")"
exit 0
