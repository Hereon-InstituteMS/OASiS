#!/bin/bash
# Tier-2 for fourc::fpsi#2 — MAT_StructPoro's MATID really must point at a
# structural material, and the two ways of getting it wrong give two DIFFERENT
# diagnostics, neither of them the one the entry quoted.
#
# Claimed:  "wrong MATID (pointing to a fluid material or undefined ID) aborts
#            with 'StructPoro inner material must be elastic' from
#            4C_mat_structporo.cpp".
# Observed, on upstream fpsi_ofsiinterface.4C.yaml (MAT 1 = MAT_StructPoro with
# MATID 6 -> MAT_ElastHyper):
#   MATID 4  (a MAT_fluid): "Mat::StructPoro: underlying material should be of
#             type Mat::So3Material" — right file, mat/4C_mat_structporo.cpp
#             line 109, but the wording is about So3Material, not "elastic".
#   MATID 99 (undefined):   "Material 'MAT 99' could not be found" from
#             mat/4C_mat_par_bundle.cpp line 36 — a different file entirely, so
#             grepping for the structporo file after this typo finds nothing.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fpsi_ofsiinterface.4C.yaml) || exit 3
grep -q '    MAT_StructPoro:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_structporo_changed"; exit 3; }
grep -q '      MATID: 6' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_structporo_matid_changed"; exit 3; }

# The two pathologies: what MATID points at.
FLUID_MATID=4
UNDEFINED_MATID=99

cp "$BASE" "$TMP/good.yaml"
python3 - "$BASE" "$TMP/fluidmat.yaml" "$FLUID_MATID" <<'PY'
import sys
t = open(sys.argv[1]).read()
old = "    MAT_StructPoro:\n      MATID: 6"
assert old in t
open(sys.argv[2], "w").write(
    t.replace(old, "    MAT_StructPoro:\n      MATID: %s" % sys.argv[3], 1))
PY
python3 - "$BASE" "$TMP/undefmat.yaml" "$UNDEFINED_MATID" <<'PY'
import sys
t = open(sys.argv[1]).read()
old = "    MAT_StructPoro:\n      MATID: 6"
assert old in t
open(sys.argv[2], "w").write(
    t.replace(old, "    MAT_StructPoro:\n      MATID: %s" % sys.argv[3], 1))
PY
grep -A1 '    MAT_StructPoro:' "$TMP/fluidmat.yaml" | grep 'MATID' | tr -d ' ' | sed 's/^/FLUID_ARM_/'
grep -A1 '    MAT_StructPoro:' "$TMP/undefmat.yaml" | grep 'MATID' | tr -d ' ' | sed 's/^/UNDEF_ARM_/'

probe GOOD     "$TMP/good.yaml"
probe FLUIDMAT "$TMP/fluidmat.yaml"
probe UNDEFMAT "$TMP/undefmat.yaml"

grep -m1 -F "OK (2)" "$TMP/GOOD.log"
grep -m1 -F "Mat::StructPoro: underlying material should be of type Mat::So3Material" "$TMP/FLUIDMAT.log"
grep -m1 -F "4C_mat_structporo.cpp" "$TMP/FLUIDMAT.log"
grep -m1 -F "Material 'MAT 99' could not be found" "$TMP/UNDEFMAT.log"
grep -m1 -F "4C_mat_par_bundle.cpp" "$TMP/UNDEFMAT.log"

# The claimed wording appears in neither arm.
echo "CLAIMED_MUST_BE_ELASTIC_TEXT=$(grep -ci 'inner material must be elastic' "$TMP/FLUIDMAT.log")$(grep -ci 'inner material must be elastic' "$TMP/UNDEFMAT.log")"
# The two arms abort in different files, so a reader who greps for the
# structporo file after an undefined-ID typo finds nothing.
echo "UNDEFMAT_IN_STRUCTPORO_FILE=$(grep -c '4C_mat_structporo.cpp' "$TMP/UNDEFMAT.log")"
echo "FLUIDMAT_IN_STRUCTPORO_FILE=$(grep -c '4C_mat_structporo.cpp' "$TMP/FLUIDMAT.log")"
exit 0
