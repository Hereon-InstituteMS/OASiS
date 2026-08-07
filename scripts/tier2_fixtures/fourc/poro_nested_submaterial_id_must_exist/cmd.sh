#!/bin/bash
# Tier-2 for fourc::porous_media#1 — the porofluid material hierarchy really is
# four levels deep (MAT_FluidPoroMultiPhase -> MAT_FluidPoroSinglePhase ->
# density / viscosity / relative-permeability law + DoF type -> phase law), and
# deleting ONE leaf entry is fatal.  What the abort actually says is
#
#     Material 'MAT 103' could not be found        (mat/4C_mat_par_bundle.cpp)
#
# and NOT `referenced material ID X not found in MATERIALS` — that wording is
# nowhere in 4C.  Both are asserted, the fabricated one as a zero count.
#
# The stack trace is the part worth reading: it names
# Mat::PAR::PoroDensityLaw::create_density_law, i.e. the failure surfaces two
# levels below the entry the user wrote, which is why the nesting matters.
#
# Arms: the deck as shipped, then the same deck with MAT 103
# (MAT_PoroDensityLawExp, referenced by MAT 10's DENSITYLAWID) removed.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream porofluid_pressure_based_2D_quad4.4C.yaml) || exit 3
ln -s "$(dirname "$BASE")/xml" "$TMP/xml"
cp "$BASE" "$TMP/full.yaml"

python3 - "$BASE" "$TMP/gap.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
leaf = """  - MAT: 103
    MAT_PoroDensityLawExp:
      BULKMODULUS: 100
"""
if leaf not in t:
    print("FIXTURE_ABORT=upstream_deck_changed"); sys.exit(3)
open(sys.argv[2], "w").write(t.replace(leaf, "", 1))
PY
[ -f "$TMP/gap.yaml" ] || exit 3
# The dangling reference must still be there, otherwise nothing is proven.
grep -q "DENSITYLAWID: 103" "$TMP/gap.yaml" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

probe FULL "$TMP/full.yaml"
probe GAP  "$TMP/gap.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/FULL.log"
grep -m1 -F "Material 'MAT 103' could not be found" "$TMP/GAP.log"
grep -m1 -oF "4C_mat_par_bundle.cpp" "$TMP/GAP.log"
# The failure is raised from inside the density-law factory, two levels below
# the MATERIALS entry the user actually edited.
echo "FAILS_INSIDE_DENSITY_LAW_FACTORY=$(grep -c 'PoroDensityLaw::create_density_law' "$TMP/GAP.log")"
# The wording the entry used to quote does not exist anywhere in the output.
echo "CLAIMED_NOT_FOUND_IN_MATERIALS_TEXT=$(grep -ci 'not found in MATERIALS' "$TMP/GAP.log")"
exit 0
