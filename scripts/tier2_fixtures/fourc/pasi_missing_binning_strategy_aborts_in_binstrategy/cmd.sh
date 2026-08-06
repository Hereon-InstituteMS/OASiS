#!/bin/bash
# Tier-2 for fourc::pasi#0 — BINNING STRATEGY is indeed mandatory for a PASI run,
# and the message you get says nothing about PASI, binning or contact search.
#
# Claimed:  "omitting BINNING STRATEGY aborts with 'no binning strategy defined
#           for PASI' at setup".
# Observed: the section is declared optional in 4C's input spec, so the deck
#           parses, both fields are set up, the time-stepping banner is printed —
#           and then it dies inside the binning strategy on an empty domain with
#             We need a discretization at this point.
#           from core/binstrategy/4C_binstrategy.cpp.  That sentence contains
#           neither "binning" nor "PASI" nor "strategy"; an agent grepping the
#           error for any word from the claim finds nothing, and the natural
#           reading ("some discretisation is missing") points at the mesh rather
#           than at the missing section.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream pasi_twoway_norelax_particle_dem_1d_normalcontact_linspring_walldiscretcond.4C.yaml) || exit 3

cp "$BASE" "$TMP/withbinning.yaml"
python3 - "$BASE" "$TMP/nobinning.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = '''BINNING STRATEGY:
  BIN_SIZE_LOWER_BOUND: 1
  DOMAINBOUNDINGBOX: "-1.0 -1.0 -1.0 1.0 1.0 1.0"
'''
assert blk in t, "upstream PASI deck no longer carries the BINNING STRATEGY block"
open(sys.argv[2], "w").write(t.replace(blk, ""))
PY
echo "BINNING_SECTIONS_REMOVED=$(( $(grep -c '^BINNING STRATEGY:' "$TMP/withbinning.yaml") - $(grep -c '^BINNING STRATEGY:' "$TMP/nobinning.yaml") ))"

probe WITHBINNING "$TMP/withbinning.yaml"
probe NOBINNING   "$TMP/nobinning.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/WITHBINNING.log"
echo "WITHBINNING_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/WITHBINNING.log")"

# The deck parses far enough to print the coupled time-stepping banner.
grep -m1 -F "Overview of chosen time stepping" "$TMP/NOBINNING.log"
# ...and then dies with a sentence that names none of the right things.
grep -m1 -F "We need a discretization at this point." "$TMP/NOBINNING.log"
grep -m1 -oE "4C_binstrategy\.cpp, line [0-9]+" "$TMP/NOBINNING.log"
echo "CLAIMED_NO_BINNING_STRATEGY_TEXT=$(grep -ci 'no binning strategy defined' "$TMP/NOBINNING.log")"
if grep -qiE "We need a discretization at this point.*(bin|pasi|contact)" "$TMP/NOBINNING.log"; then
  echo "DIAGNOSTIC_NAMES_BINNING_OR_PASI=yes"
else
  echo "DIAGNOSTIC_NAMES_BINNING_OR_PASI=no"
fi
exit 0
