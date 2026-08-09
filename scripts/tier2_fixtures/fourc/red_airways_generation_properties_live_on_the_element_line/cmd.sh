#!/bin/bash
# Tier-2 for fourc::reduced_airways#5 — per-generation airway properties are set
# on the ELEMENT line, not by giving each generation its own material.
#
# Claimed: "using a single uniform MAT_redAirway across all 16+ generations gives
#          wrong impedance spectrum ... Specify a different MAT_redAirway per
#          generation (or per element)."
# Observed, on upstream red_airway_3airway_2acinus_awacinter (generations 0, 1, 1):
#   * all three airways already share ONE material, MAT 1 = MAT_fluid, and that is
#     how the upstream regression deck is written. MAT_redAirway does not exist:
#     renaming MAT_fluid to it fails to match section 'MATERIALS'.
#   * the properties that vary by generation are element-line tokens. Tapering
#     Area 1.0 -> 0.5 on the two generation-1 airways moves node 2 pressure
#     29.8681 -> 29.9613 and the acinar volumes, with the material untouched.
#   * WallElasticity is likewise per element and does bite on this
#     ConvectiveViscoElasticRLC airway: 500 -> 5000 moves node 2 to 29.9913.
# So the fix is per-element geometry, and 4C says nothing when you leave it uniform.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream red_airway_3airway_2acinus_awacinter.4C.yaml) || exit 3
cd "$TMP" || exit 3
cp "$BASE" base.yaml
grep -q 'MAT_fluid:' base.yaml || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
echo "AIRWAY_MATERIAL_IDS_IN_USE=$(grep -c 'RED_AIRWAY LINE2' base.yaml)"
echo "DISTINCT_AIRWAY_MATERIALS=$(grep -o 'RED_AIRWAY LINE2 [0-9]* [0-9]* MAT [0-9]*' base.yaml | awk '{print $NF}' | sort -u | wc -l)"

python3 - <<'PY'
t = open('base.yaml').read()
gen1 = ['''  - "%d RED_AIRWAY LINE2 2 %d MAT 1 ElemSolvingType NonLinear TYPE ConvectiveViscoElasticRLC Resistance
    Poiseuille PowerOfVelocityProfile 2 WallElasticity 500.0 PoissonsRatio 0.4 ViscousTs 2.0 ViscousPhaseShift
    0.13 WallThickness 0.1 Area 1.0 Generation 1"''' % p for p in ((2, 3), (3, 4))]
taper = t
for blk in gen1:
    assert blk in taper, "upstream generation-1 airway line changed"
    taper = taper.replace(blk, blk.replace('Area 1.0 Generation 1', 'Area 0.5 Generation 1'))
open('taper.yaml', 'w').write(taper)
PY
sed 's/WallElasticity 500.0/WallElasticity 5000.0/'  base.yaml > wall.yaml
sed 's/    MAT_fluid:/    MAT_redAirway:/'           base.yaml > matred.yaml

probe BASE   base.yaml
probe TAPER  taper.yaml
probe WALL   wall.yaml
probe MATRED matred.yaml

echo "BASE_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/BASE.log")"
echo "TAPER_TESTS_WRONG=$(grep -c 'is WRONG' "$TMP/TAPER.log")"
echo "WALL_TESTS_WRONG=$(grep -c 'is WRONG' "$TMP/WALL.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
grep -m1 -F "pressure at node   2" "$TMP/TAPER.log"
grep -m1 -F "pressure at node   2" "$TMP/WALL.log"
grep -m1 -F "Failed to match specification in section 'MATERIALS'." "$TMP/MATRED.log"
echo "MAT_REDAIRWAY_EXISTS=$(grep -c 'Expected group .MAT_redAirway' "$TMP/MATRED.log")"
echo "UNIFORM_MATERIAL_WARNINGS=$(grep -ciE 'generation|impedance|uniform material' "$TMP/BASE.log")"
exit 0
