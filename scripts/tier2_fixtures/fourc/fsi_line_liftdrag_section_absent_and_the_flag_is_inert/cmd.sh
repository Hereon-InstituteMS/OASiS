#!/bin/bash
# Tier-2 for fourc::fsi#9 — the section really does not exist (confirmed, with
# the exact wording), but the suggested substitute does nothing on its own.
#
# Claimed: "DESIGN FLUID LINE LIFT&DRAG does NOT exist in 4C for 2D.  Signal:
#           writing it ... raises \"Section 'DESIGN FLUID LINE LIFT&DRAG' is not
#           a valid section name.\" from core/io/src/4C_io_input_file.cpp, exit 1
#           at parse. ... For 2D lift/drag, set LIFTDRAG: true in FLUID DYNAMIC —
#           4C computes it automatically from the no-slip boundaries."
# Observed: the abort and its wording are exactly right (line 546).  The advice
#           is not.  FLD::Utils::lift_drag() starts with
#           dis->get_condition("LIFTDRAG", ldconds) and the only definition that
#           registers that condition name is DESIGN FLUID SURF LIFT&DRAG
#           (src/inpar/4C_inpar_fluid.cpp).  Setting LIFTDRAG: true with no
#           condition parses, runs, matches every pinned result and prints not a
#           single "lift'n'drag" line — no forces are computed from anything.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fsi_fp_mono_fs_ga_ga.4C.yaml) || exit 3
grep -q '^FLUID DYNAMIC:' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_has_no_fluid_dynamic"; exit 3; }

# The pathology: ask for lift and drag the two ways the entry describes.
LIFTDRAG_SECTION="DESIGN FLUID LINE LIFT&DRAG"

cp "$BASE" "$TMP/plain.yaml"
python3 - "$BASE" "$TMP" "$LIFTDRAG_SECTION" <<'PY'
import sys
src, tmp, sec = sys.argv[1:4]
t = open(src).read()
assert "DNODE-NODE TOPOLOGY:" in t
open(tmp + "/linesection.yaml", "w").write(
    t.replace("DNODE-NODE TOPOLOGY:",
              "%s:\n  - E: 3\n    LABEL: 1\nDNODE-NODE TOPOLOGY:" % sec, 1))
blk = "FLUID DYNAMIC:\n  LINEAR_SOLVER: 1"
assert blk in t
open(tmp + "/flagonly.yaml", "w").write(
    t.replace(blk, "FLUID DYNAMIC:\n  LIFTDRAG: true\n  LINEAR_SOLVER: 1", 1))
PY
echo "SURF_VARIANT_EXISTS_IN_UPSTREAM_TREE=$(grep -lc 'DESIGN FLUID SURF LIFT&DRAG' "$DECKS"/*.yaml 2>/dev/null | wc -l)"
echo "LINE_VARIANT_EXISTS_IN_UPSTREAM_TREE=$(grep -l 'DESIGN FLUID LINE LIFT&DRAG' "$DECKS"/*.yaml 2>/dev/null | wc -l)"

probe PLAIN       "$TMP/plain.yaml"
probe LINESECTION "$TMP/linesection.yaml"
probe FLAGONLY    "$TMP/flagonly.yaml"

grep -m1 -F "OK (6)" "$TMP/PLAIN.log"

# Confirmed: the section name is rejected, with this wording and this file.
grep -m1 -F "Section 'DESIGN FLUID LINE LIFT&DRAG' is not a valid section name." "$TMP/LINESECTION.log"
grep -m1 -F "4C_io_input_file.cpp" "$TMP/LINESECTION.log"
echo "LINESECTION_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/LINESECTION.log")"

# Falsified: the flag alone computes nothing.
grep -m1 -F "processor 0 finished normally" "$TMP/FLAGONLY.log"
grep -m1 -F "OK (6)" "$TMP/FLAGONLY.log"
echo "FLAGONLY_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/FLAGONLY.log")"
echo "FLAGONLY_LIFTDRAG_OUTPUT_LINES=$(grep -ci \"lift'n'drag\" "$TMP/FLAGONLY.log")"
echo "FLAGONLY_LIFTDRAG_WARNINGS=$(grep -ciE 'liftdrag.*(no condition|ignored|nothing)' "$TMP/FLAGONLY.log")"
exit 0
