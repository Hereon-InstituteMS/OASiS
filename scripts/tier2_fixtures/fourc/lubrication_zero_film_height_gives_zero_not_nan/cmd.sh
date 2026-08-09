#!/bin/bash
# Tier-2 for fourc::lubrication#2 — the film height really must be supplied, but
# a ZERO film height does not produce the NaN the entry claimed.
#
# Three arms on upstream lubrication_sb_2d.4C.yaml:
#   BASELINE : h(x) = 0.045 - 5e-4 x  -> result test CORRECT, exit 0
#   ZEROH    : the height function replaced by "0.0"
#   NOHFUNC  : HEIGHTFEILD / HFUNCNO deleted from LUBRICATION DYNAMIC
#
# Claimed:  "h(x) -> 0 gives pressure NaN at the first time step (1/h^3 ...)".
# Observed: h = 0 gives pressure EXACTLY 0.0.  Not NaN, not Inf, no solver
#           complaint, no zero-pivot, five clean Newton tables; the only failure
#           is the deck's own result test noticing the answer changed.  1/h^3
#           never divides anything, because h enters the assembled operator
#           multiplicatively (h^3 * grad p), so h = 0 kills the whole stiffness
#           term rather than blowing it up.  h = 1e-14 behaves identically.
# Observed: omitting the height field is what actually aborts, and the diagnostic
#           is about the FUNCT id, not about the film: "Function with index -1
#           (i.e. input FUNCT-1) not available." from
#           core/utils/src/functions/4C_utils_function_manager.hpp, raised from
#           Lubrication::TimIntImpl::set_height_field_pure_lub.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream lubrication_sb_2d.4C.yaml) || exit 3
grep -q '"(0.045)-(x\*0.5e-3)"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '  HEIGHTFEILD: "function"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_heightfield_key_changed"; exit 3; }

ZERO_HEIGHT='0.0'

cp "$BASE" "$TMP/baseline.yaml"
sed "s|(0.045)-(x\*0.5e-3)|$ZERO_HEIGHT|" "$BASE" > "$TMP/zeroh.yaml"
sed "s|(0.045)-(x\*0.5e-3)|1.0e-14|"      "$BASE" > "$TMP/tinyh.yaml"
python3 - "$BASE" "$TMP/nohfunc.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = '  HEIGHTFEILD: "function"\n  HFUNCNO: 3\n'
assert blk in t, "upstream deck no longer declares HEIGHTFEILD/HFUNCNO"
open(sys.argv[2], "w").write(t.replace(blk, ""))
PY

probe BASELINE "$TMP/baseline.yaml"
probe ZEROH    "$TMP/zeroh.yaml"
probe TINYH    "$TMP/tinyh.yaml"
probe NOHFUNC  "$TMP/nohfunc.yaml"

grep -m1 -F "is CORRECT, abs(diff)=" "$TMP/BASELINE.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/ZEROH.log"

# What h -> 0 actually yields.
PZ=$(grep -m1 -oE 'actresult=[ ]*[-0-9.eE+]+' "$TMP/ZEROH.log" | tr -d ' ' | cut -d= -f2)
PT=$(grep -m1 -oE 'actresult=[ ]*[-0-9.eE+]+' "$TMP/TINYH.log" | tr -d ' ' | cut -d= -f2)
echo "ZERO_HEIGHT_PRESSURE=$PZ"
echo "TINY_HEIGHT_PRESSURE=$PT"
echo "ZERO_HEIGHT_PRESSURE_IS_EXACTLY_ZERO=$(python3 -c "print('yes' if $PZ == 0.0 else 'no')")"
# The claimed NaN is nowhere: 'inf' only ever appears in the 'pre-res-inf'
# column header, so match the bare words with word boundaries.
echo "CLAIMED_NAN_OR_INF=$(grep -ciE '(^|[^a-z-])(nan|inf)([^a-z-]|$)' "$TMP/ZEROH.log")"
echo "ZERO_HEIGHT_SOLVER_COMPLAINTS=$(grep -ciE 'zero pivot|singular|breakdown|did not converge' "$TMP/ZEROH.log")"
echo "ZERO_HEIGHT_NEWTON_LINES=$(grep -c '\[L_2 \]' "$TMP/ZEROH.log")"
# Omitting the height field is the abort, and it names the FUNCT id.
grep -m1 -F "Function with index -1 (i.e. input FUNCT-1) not available." "$TMP/NOHFUNC.log"
grep -m1 -F "4C_utils_function_manager.hpp" "$TMP/NOHFUNC.log"
grep -m1 -F "set_height_field_pure_lub" "$TMP/NOHFUNC.log"
exit 0
