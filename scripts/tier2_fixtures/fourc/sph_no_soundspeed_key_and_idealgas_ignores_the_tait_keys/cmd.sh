#!/bin/bash
# Tier-2 for fourc::particle_sph#16 — there is no SOUNDSPEED and no
# SMOOTHING_LENGTH key, and under EQUATIONOFSTATE: IdealGas the still-mandatory
# REFDENSFAC and EXPONENT are read and ignored.
#
# T2_MUTATE=1 removes the pathology: the two IdealGas decks are built WITHOUT
# the equation-of-state switch, so they are GenTait decks, the Tait keys are
# live, the identity breaks and the fixture must go red.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

BASE=$(upstream particle_sph_1d_pressurewave_boundary_densitysummation_cubicspline_adami.4C.yaml) || exit 3
grep -q "REFDENSFAC: 1" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

# Neither name exists anywhere in 4C's source or in ANY of its decks. Counted,
# not asserted from memory.
SRC="${FOURC_SRC:-$HOME/4C/src}"
if [ -d "$SRC" ]; then
  echo "SOUNDSPEED_IN_SOURCE=$(grep -rl 'SOUNDSPEED' "$SRC" 2>/dev/null | wc -l)"
  echo "SMOOTHING_LENGTH_IN_SOURCE=$(grep -rl 'SMOOTHING_LENGTH' "$SRC" 2>/dev/null | wc -l)"
else
  echo "FIXTURE_ABORT=no_source_tree"; exit 3
fi
echo "SOUNDSPEED_IN_DECKS=$(grep -rl 'SOUNDSPEED' "$DECKS" 2>/dev/null | wc -l)"
echo "EQUATIONOFSTATE_IN_UPSTREAM_SPH_DECKS=$(grep -l 'EQUATIONOFSTATE' "$DECKS"/particle_sph_*.4C.yaml 2>/dev/null | wc -l)"

python3 - "$BASE" "$TMP" "$MUTATE" <<'PY'
import sys
src, tmp, mutate = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3] == "1"
a = "  INITIALPARTICLESPACING: 0.004"
assert a in src
ideal = src.replace(a, '  EQUATIONOFSTATE: "IdealGas"\n' + a)
perturbed = ideal.replace("      REFDENSFAC: 1", "      REFDENSFAC: 0.5").replace(
                          "      EXPONENT: 1", "      EXPONENT: 7")
# under mutation both decks stay GenTait, so the Tait keys are live
open(tmp + "/ig.yaml", "w").write(src if mutate else ideal)
open(tmp + "/igp.yaml", "w").write(
    src.replace("      REFDENSFAC: 1", "      REFDENSFAC: 0.5").replace(
        "      EXPONENT: 1", "      EXPONENT: 7") if mutate else perturbed)
PY

probe IG  "$TMP/ig.yaml"
probe IGP "$TMP/igp.yaml"

# Guard: the deck under test really carries the switch (or really does not).
echo "IDEALGAS_SELECTED=$(grep -c 'IdealGas' "$TMP/ig.yaml" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
echo "TAIT_KEYS_PERTURBED=$( { grep -q 'REFDENSFAC: 0.5' "$TMP/igp.yaml" && grep -q 'EXPONENT: 7' "$TMP/igp.yaml"; } && echo yes || echo no)"

grep -E "actresult=" "$TMP/IG.log"  > "$TMP/v_ig"
grep -E "actresult=" "$TMP/IGP.log" > "$TMP/v_igp"
echo "VERDICT_LINES=$(wc -l < "$TMP/v_ig")"
cmp -s "$TMP/v_ig" "$TMP/v_igp" && echo "TAIT_KEYS_ARE_INERT=yes" || echo "TAIT_KEYS_ARE_INERT=no"
# REFDENSFAC and EXPONENT remain REQUIRED by the material parser even though
# IdealGas ignores them.
python3 - "$TMP/ig.yaml" "$TMP/norefdens.yaml" <<'PY'
import sys
open(sys.argv[2], "w").write(open(sys.argv[1]).read().replace("      REFDENSFAC: 0.5\n", "").replace("      REFDENSFAC: 1\n", ""))
PY
probe NOREFDENS "$TMP/norefdens.yaml"
grep -m1 -F "Failed to match specification in section 'MATERIALS'" "$TMP/NOREFDENS.log"   && echo "IGNORED_KEY_IS_STILL_REQUIRED=yes" || echo "IGNORED_KEY_IS_STILL_REQUIRED=no"
exit 0
