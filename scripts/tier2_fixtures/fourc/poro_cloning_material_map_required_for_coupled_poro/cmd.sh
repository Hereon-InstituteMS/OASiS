#!/bin/bash
# Tier-2 for fourc::porous_media#5 — a coupled poroelasticity problem needs a
# CLONING MATERIAL MAP, and leaving it out is loud, not silent.
#
# Claimed:  `CloningMaterialMap missing for porofluid -> solid`, OR a silent
#           decoupled run whose structure matches a dry-block reference.
# Observed: neither.  4C aborts with
#
#     At least one material pairing required in --CLONING MATERIAL MAP.
#     core/fem/src/general/utils/4C_fem_general_utils_createdis.hpp
#
# raised from Core::FE::clone_discretization<PoroElast::Utils::PoroelastCloneStrategy>,
# i.e. the second field is never created at all — there is no decoupled run to
# mistake for a coupled one.  Note the message still names the section in the
# retired --SECTION dat spelling, which is why grepping for the YAML key
# "CLONING MATERIAL MAP:" finds it but grepping for a camel-case symbol does not.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream poro_2D_quad4_linporo.4C.yaml) || exit 3
grep -q "^CLONING MATERIAL MAP:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
cp "$BASE" "$TMP/mapped.yaml"

python3 - "$BASE" "$TMP/unmapped.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = ('CLONING MATERIAL MAP:\n'
       '  - SRC_FIELD: "structure"\n'
       '    SRC_MAT: 1\n'
       '    TAR_FIELD: "porofluid"\n'
       '    TAR_MAT: 3\n')
if blk not in t:
    print("FIXTURE_ABORT=upstream_deck_changed"); sys.exit(3)
open(sys.argv[2], "w").write(t.replace(blk, "", 1))
PY
[ -f "$TMP/unmapped.yaml" ] || exit 3

probe MAPPED   "$TMP/mapped.yaml"
probe UNMAPPED "$TMP/unmapped.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/MAPPED.log"
grep -m1 -F "At least one material pairing required in --CLONING MATERIAL MAP." "$TMP/UNMAPPED.log"
grep -m1 -oF "4C_fem_general_utils_createdis.hpp" "$TMP/UNMAPPED.log"
echo "FAILS_IN_POROELAST_CLONE_STRATEGY=$(grep -c 'PoroelastCloneStrategy' "$TMP/UNMAPPED.log")"
# No silent decoupled run: the porofluid field never exists, so no time step
# and no result test happens.
echo "UNMAPPED_RESULT_TESTS_PERFORMED=$(grep -c 'is WRONG --> actresult=' "$TMP/UNMAPPED.log")"
# And the quoted diagnostic is not real.
echo "CLAIMED_CLONINGMATERIALMAP_TEXT=$(grep -ci 'CloningMaterialMap missing' "$TMP/UNMAPPED.log")"
exit 0
