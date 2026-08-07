#!/bin/bash
# Tier-2 for fourc::particle_dem#0 — MAT_ParticleDEM carries geometry and mass
# and nothing else; every contact property is a PARTICLE DYNAMIC/DEM key.
#
# Claimed: putting YOUNG (or any contact property) in MAT_ParticleDEM is a
#          MATERIALS parse error that names neither the key nor the material,
#          while the SAME quantity set in the DEM section is accepted.
#
# T2_MUTATE=1 removes the pathology: the material is left untouched, so no
# parse error occurs, EXIT_MATYOUNG becomes 0 and the fixture must go red.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

BASE=$(upstream particle_dem_1d_normalcontact_linspring_stiffset.4C.yaml) || exit 3
grep -q "MAT_ParticleDEM" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$BASE" "$TMP" "$MUTATE" <<'PY'
import sys
src, tmp, mutate = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3] == "1"
mat = "      INITDENSITY: 0.005"
assert mat in src, "upstream deck no longer sets INITDENSITY on MAT_ParticleDEM"
# the pathology: a contact property written into the material
bad = src.replace(mat, mat + "\n      YOUNG: 210000")
open(tmp + "/matyoung.yaml", "w").write(src if mutate else bad)
# the same quantity in the SECTION, which is where it belongs
sec = "  NORMAL_STIFF: 3.5e-05"
assert sec in src
open(tmp + "/secyoung.yaml", "w").write(
    src.replace(sec, sec + "\n  YOUNG_MODULUS: 210000\n  POISSON_RATIO: 0.3"))
PY

probe MATYOUNG "$TMP/matyoung.yaml"
probe SECYOUNG "$TMP/secyoung.yaml"

grep -m1 -F "Failed to match specification in section 'MATERIALS'" "$TMP/MATYOUNG.log"   && echo "MATERIAL_REJECTS_CONTACT_KEY=yes" || echo "MATERIAL_REJECTS_CONTACT_KEY=no"
grep -m1 -F "Could not match this input" "$TMP/MATYOUNG.log"
# What the abort actually is: a candidate list over every material 4C knows,
# each block re-echoing the offending entry as unused data. Counted directly,
# not inferred from a proxy.
echo "DIAGNOSTIC_LINES=$(wc -l < "$TMP/MATYOUNG.log")"
echo "CANDIDATE_MATERIALS_LISTED=$(grep -c "Expected group" "$TMP/MATYOUNG.log")"
echo "DIAGNOSTIC_IS_A_WALL=$([ "$(grep -c "Expected group" "$TMP/MATYOUNG.log")" -gt 100 ] && echo yes || echo no)"
# and none of it says the key is unknown, nor lists MAT_ParticleDEM as a candidate
echo "SAYS_KEY_IS_UNKNOWN=$(grep -ciE 'unknown parameter|not a parameter|unexpected key' "$TMP/MATYOUNG.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
echo "OFFERS_MAT_PARTICLEDEM_AS_CANDIDATE=$(grep -c "Expected group 'MAT_ParticleDEM'" "$TMP/MATYOUNG.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
# The section accepts the same quantity and the run completes.
grep -m1 -F "processor 0 finished normally" "$TMP/SECYOUNG.log" && echo "SECTION_ACCEPTS=yes"
# MAT_ParticleDEM has exactly two parameters upstream.
echo "MATERIAL_PARAM_COUNT=$(sed -n '/MAT_ParticleDEM:/,/^  - MAT\|^RESULT/p' "$BASE" | grep -cE '^      [A-Z_]+:')"
exit 0
