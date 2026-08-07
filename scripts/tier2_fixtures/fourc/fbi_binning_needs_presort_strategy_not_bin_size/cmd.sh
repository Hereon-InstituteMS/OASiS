#!/bin/bash
# Tier-2 for fourc::fbi#3 — the entry's advice ("BINNING STRATEGY may be needed
# ... set BIN_SIZE_LOWER_BOUND appropriately") does nothing on its own.
#
# 4C picks the beam-fluid pre-sort in FBI::GeometryCouplerFactory from
# FLUID BEAM INTERACTION/PRESORT_STRATEGY, an enum whose choices are
# 'bruteforce' (the DEFAULT) and 'binning'.  A BINNING STRATEGY section only
# parameterises the binning coupler once that switch has been thrown.
#
# The switch is directly observable, because the two couplers register different
# Teuchos TimeMonitor timers and 4C prints the timer table at the end of every
# run:
#   bruteforce -> "FBI::FBICoupler::Search"        only
#   binning    -> "FBI::FBIBinningCoupler::Search" as well
#
# Three arms on upstream fbi_mortar_solidcoupling.4C.yaml, which sets no
# PRESORT_STRATEGY at all:
#   BASE     : untouched                                  -> no binning timer
#   BINSIZE  : + BINNING STRATEGY/BIN_SIZE_LOWER_BOUND     -> no binning timer,
#              i.e. the entry's advice followed literally changes NOTHING, and
#              4C does not say the section was ignored
#   PRESORT  : + PRESORT_STRATEGY: binning                 -> binning timer appears
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fbi_mortar_solidcoupling.4C.yaml) || exit 3
grep -q '^FLUID BEAM INTERACTION:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_fbi_section_changed"; exit 3; }
grep -q '  COUPLING: "solid"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_coupling_key_changed"; exit 3; }
grep -q 'PRESORT_STRATEGY' "$BASE" \
  && { echo "FIXTURE_ABORT=upstream_already_sets_presort"; exit 3; }
cp "$(dirname "$BASE")/beam_flow_solver.xml" "$TMP/" \
  || { echo "FIXTURE_ABORT=missing_nox_xml"; exit 3; }
cd "$TMP" || exit 3

# What the entry told the reader to add, and what actually throws the switch.
CLAIMED_KNOB='BINNING STRATEGY:\n  BIN_SIZE_LOWER_BOUND: 0.25\n'
REAL_KNOB='  PRESORT_STRATEGY: binning'

cp "$BASE" "$TMP/base.yaml"
python3 - "$BASE" "$TMP/binsize.yaml" "$CLAIMED_KNOB" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = sys.argv[3].encode().decode("unicode_escape")
assert "FSI DYNAMIC:\n" in t
open(sys.argv[2], "w").write(t.replace("FSI DYNAMIC:\n", blk + "FSI DYNAMIC:\n", 1))
PY
sed "s|^  COUPLING: \"solid\"|  COUPLING: \"solid\"\n$REAL_KNOB|" "$BASE" > "$TMP/presort.yaml"

probe BASE    "$TMP/base.yaml"
probe BINSIZE "$TMP/binsize.yaml"
probe PRESORT "$TMP/presort.yaml"

grep -m1 -F "OK (6)" "$TMP/BASE.log"
grep -m1 -F "processor 0 finished normally" "$TMP/BINSIZE.log"
grep -m1 -F "FBI::FBICoupler::Search" "$TMP/BASE.log"
grep -m1 -F "FBI::FBIBinningCoupler::Search" "$TMP/PRESORT.log"

echo "BASE_BINNING_TIMER=$(grep -c 'FBI::FBIBinningCoupler::Search' "$TMP/BASE.log")"
echo "BINSIZE_BINNING_TIMER=$(grep -c 'FBI::FBIBinningCoupler::Search' "$TMP/BINSIZE.log")"
echo "PRESORT_BINNING_TIMER=$(grep -c 'FBI::FBIBinningCoupler::Search' "$TMP/PRESORT.log")"
# Adding BINNING STRATEGY alone is accepted in silence — no "ignored" notice.
echo "BINSIZE_IGNORED_WARNINGS=$(grep -ciE 'binning.*(ignor|unus|no effect|not used)|bin_size.*ignor' "$TMP/BINSIZE.log")"
# ...and the answer is identical, which is why it looks like it worked.
echo "BINSIZE_FAILED_TESTS=$(grep -c 'is WRONG' "$TMP/BINSIZE.log")"
echo "PRESORT_FAILED_TESTS=$(grep -c 'is WRONG' "$TMP/PRESORT.log")"
exit 0
