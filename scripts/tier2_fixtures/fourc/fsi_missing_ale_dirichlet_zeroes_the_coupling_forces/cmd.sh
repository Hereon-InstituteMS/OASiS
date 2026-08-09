#!/bin/bash
# Tier-2 for fourc::fsi#2 — dropping the ALE Dirichlet conditions from an FSI
# input does NOT invert elements and does NOT abort.  It runs to the end and
# silently reports a different answer, with the FSI interface forces gone.
#
# Claimed: "missing ALE Dirichlet on an outflow / outer wall lets the ALE mesh
#           drift freely there, producing inverted elements within ~5-20 steps
#           and 'det(J) < 0' from the ALE solver — simulation aborts."
# Observed: upstream fsi_fp_mono_fs_ga_ga.4C.yaml pins its ALE field with
#           DESIGN SURF ALE DIRICH CONDITIONS (E 3, the inflow face) and
#           DESIGN VOL ALE DIRICH CONDITIONS (E 1, the fluid volume).  Delete
#           both and all 10 time steps still run, no Jacobian complaint of any
#           kind appears, and the run ends on its own result test: 4 of 6 pinned
#           values move.  The interesting ones are the FSI Lagrange multipliers
#           at interface node 16, which collapse from -5.5e-1 / -8.0e-1 / -8.0e-1
#           to ~1e-13 — the interface transmits no force at all — while the
#           fluid pressure at the same node goes from 4 to 6.  Nothing in the log
#           says why.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fsi_fp_mono_fs_ga_ga.4C.yaml) || exit 3
grep -q '^DESIGN SURF ALE DIRICH CONDITIONS:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_has_no_surf_ale_dirich"; exit 3; }
grep -q '^DESIGN VOL ALE DIRICH CONDITIONS:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_has_no_vol_ale_dirich"; exit 3; }

# The pathology: remove every ALE Dirichlet condition from the deck.
DROP_ALE_DIRICH=yes

cp "$BASE" "$TMP/pinned.yaml"
python3 - "$BASE" "$TMP/drifting.yaml" "$DROP_ALE_DIRICH" <<'PY'
import sys
t = open(sys.argv[1]).read()
surf = """DESIGN SURF ALE DIRICH CONDITIONS:
  - E: 3
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
"""
vol = """DESIGN VOL ALE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [0, 1, 1]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
"""
assert surf in t and vol in t, "upstream ALE Dirichlet blocks changed"
if sys.argv[3] == "yes":
    t = t.replace(surf, "").replace(vol, "")
open(sys.argv[2], "w").write(t)
PY
echo "DRIFTING_ALE_DIRICH_BLOCKS=$(grep -c 'ALE DIRICH CONDITIONS:' "$TMP/drifting.yaml")"

probe PINNED   "$TMP/pinned.yaml"
probe DRIFTING "$TMP/drifting.yaml"

# Control.
grep -m1 -F "processor 0 finished normally" "$TMP/PINNED.log"
grep -m1 -F "OK (6)" "$TMP/PINNED.log"

# The unpinned ALE mesh runs every step and never trips a Jacobian check.
echo "DRIFTING_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/DRIFTING.log")"
echo "DRIFTING_JACOBIAN_COMPLAINTS=$(grep -ciE 'det\(J\)|NEGATIVE JACOBIAN|inverted' "$TMP/DRIFTING.log")"
echo "DRIFTING_ALE_WARNINGS=$(grep -ciE 'ale.*(drift|distort|quality|not pinned)' "$TMP/DRIFTING.log")"

# It ends on the deck's own result test instead, and the FSI interface forces
# are the thing that changed.
grep -m1 -F "Result check failed with 4 errors out of 6 tests" "$TMP/DRIFTING.log"
grep -m1 -E "lambdax .*is WRONG --> actresult=" "$TMP/DRIFTING.log"
if grep -qE "lambdax +at node +16.*actresult=-?[0-9]\.[0-9]+e-1[0-9]" "$TMP/DRIFTING.log" \
   && grep -qE "lambday +at node +16.*actresult=-?[0-9]\.[0-9]+e-1[0-9]" "$TMP/DRIFTING.log" \
   && grep -qE "lambdaz +at node +16.*actresult=-?[0-9]\.[0-9]+e-1[0-9]" "$TMP/DRIFTING.log"; then
  echo "DRIFTING_INTERFACE_LAMBDA=collapsed_to_roundoff"
else
  echo "DRIFTING_INTERFACE_LAMBDA=still_finite"
fi
exit 0
