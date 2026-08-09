#!/bin/bash
# Tier-2 for fourc::fsi#19 — the entry says "copy verbatim from this list", and
# two of the six names on that list are not COUPALGO values at all.
#
# Claimed: valid partitioned values are "iter_stagg_AITKEN_rel_force (default),
#          iter_stagg_fixed_rel_force"; a mis-spelling "aborts with 'unknown
#          coupling algorithm' from 4C_fsi_adapter.cpp".
# Observed: the real names end in _rel_param, not _rel_force.  Feeding
#           iter_stagg_AITKEN_rel_force to upstream fsi_fp_mono_fs_ga_ga.4C.yaml
#           aborts at parse with "Could not match this input" from
#           core/io/src/4C_io_input_spec_builders.cpp line 633 — not with the
#           claimed sentence, and not from 4C_fsi_adapter.cpp, which does not
#           exist in the tree.
#
#           The useful part is what 4C prints instead: a candidate line reading
#             "[!] Candidate deprecated_selection 'COUPALGO' has wrong value,
#                  possible values: ...|iter_stagg_AITKEN_rel_param|..."
#           i.e. the binary hands over the authoritative enumeration on the spot,
#           so no agent ever needs a memorised list.  The fixture pins that the
#           _rel_param spellings appear in it and the _rel_force spellings do not,
#           and then runs the corrected value to show it is accepted.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fsi_fp_mono_fs_ga_ga.4C.yaml) || exit 3
grep -q 'COUPALGO: "iter_monolithicfluidsplit"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_coupalgo_changed"; exit 3; }

# The pathology: use the partitioned COUPALGO name the entry recommends.
CLAIMED_PARTITIONED_ALGO=iter_stagg_AITKEN_rel_force

python3 - "$BASE" "$TMP" "$CLAIMED_PARTITIONED_ALGO" <<'PY'
import sys
src, tmp, algo = sys.argv[1:4]
t = open(src).read()
old = '  COUPALGO: "iter_monolithicfluidsplit"'
assert old in t
open(tmp + "/claimed.yaml", "w").write(t.replace(old, '  COUPALGO: "%s"' % algo, 1))
# the corrected value selects a PARTITIONED scheme, whose answer is not what the
# deck's monolithic RESULT DESCRIPTION pins, so that block is dropped from this
# arm; it is only here to show the spelling is accepted and runs.
real = (t.replace(old, '  COUPALGO: "iter_stagg_AITKEN_rel_param"', 1)
         .replace("  NUMSTEP: 10", "  NUMSTEP: 2", 1))
open(tmp + "/real.yaml", "w").write(real[:real.index("RESULT DESCRIPTION:")])
PY

probe MONO    "$BASE"
probe CLAIMED "$TMP/claimed.yaml"
probe REAL    "$TMP/real.yaml"

grep -m1 -F "OK (6)" "$TMP/MONO.log"

# The name the entry tells the reader to copy is rejected.
grep -m1 -F "Could not match this input" "$TMP/CLAIMED.log"
grep -m1 -F "4C_io_input_spec_builders.cpp" "$TMP/CLAIMED.log"
grep -m1 -F "Candidate deprecated_selection 'COUPALGO' has wrong value, possible values:" "$TMP/CLAIMED.log"
echo "CLAIMED_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/CLAIMED.log")"
echo "CLAIMED_SENTENCE_UNKNOWN_COUPLING_ALGORITHM=$(grep -ci 'unknown coupling algorithm' "$TMP/CLAIMED.log")"
echo "CLAIMED_SOURCE_FILE_FSI_ADAPTER=$(grep -ci '4C_fsi_adapter.cpp' "$TMP/CLAIMED.log")"

# The enumeration 4C prints contains _rel_param and not _rel_force.
L=$(grep -m1 -F "possible values:" "$TMP/CLAIMED.log")
for n in iter_stagg_AITKEN_rel_param iter_stagg_fixed_rel_param \
         iter_monolithicfluidsplit iter_monolithicstructuresplit \
         iter_mortar_monolithicfluidsplit iter_sliding_monolithicfluidsplit; do
  case "$L" in *"$n"*) echo "OFFERED_$n=1";; *) echo "OFFERED_$n=0";; esac
done
for n in iter_stagg_AITKEN_rel_force iter_stagg_fixed_rel_force; do
  case "$L" in *"$n"*) echo "OFFERED_$n=1";; *) echo "OFFERED_$n=0";; esac
done

# The corrected spelling is accepted and runs.
grep -m1 -F "processor 0 finished normally" "$TMP/REAL.log"
grep -m1 -F "FSI::Partitioned" "$TMP/REAL.log"
echo "REAL_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/REAL.log")"
exit 0
