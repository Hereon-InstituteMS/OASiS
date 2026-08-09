#!/bin/bash
# Tier-2 for fourc::fs3i#5 — the outer scalar-coupling iteration limit is real,
# but it is NOT a key of FS3I DYNAMIC.
#
# Claimed:  "setting ITEMAX in FSI DYNAMIC but not in FS3I DYNAMIC limits the
#            inner FSI iterations but lets the outer scalar coupling iterate
#            indefinitely".
# Observed: FS3I DYNAMIC has no ITEMAX at all.  Writing one aborts at parse time
#           in core/io/src/4C_io_input_spec_builders.cpp line 633 with "Could not
#           match this input", the section echoed back, and the offending key
#           listed under "[!] The following data remains unused:".
#           The real home is the FS3I DYNAMIC/PARTITIONED subsection — which also
#           carries COUPALGO and CONVTOL, and whose ITEMAX already defaults to 10,
#           so the outer loop is bounded whether you set it or not.  Writing it
#           there on upstream fs3i_part_1wc_infperm.4C.yaml is accepted and the
#           deck still passes all three of its result tests.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fs3i_part_1wc_infperm.4C.yaml) || exit 3
grep -q '^FS3I DYNAMIC:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_fs3i_dynamic_changed"; exit 3; }
grep -q '  TIMESTEP: 12.5' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_fs3i_timestep_changed"; exit 3; }
grep -q 'ITEMAX' "$BASE" \
  && { echo "FIXTURE_ABORT=upstream_now_sets_itemax"; exit 3; }
cp -r "$(dirname "$BASE")/xml" "$TMP/" \
  || { echo "FIXTURE_ABORT=missing_teko_xml"; exit 3; }
cd "$TMP" || exit 3

# Where the entry says to put it, and where it actually belongs.
CLAIMED_PLACEMENT='FS3I DYNAMIC:\n  ITEMAX: 5\n'
REAL_PLACEMENT='FS3I DYNAMIC/PARTITIONED:\n  ITEMAX: 5\n'

cp "$BASE" "$TMP/plain.yaml"
python3 - "$BASE" "$TMP/claimed.yaml" "$CLAIMED_PLACEMENT" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = sys.argv[3].encode().decode("unicode_escape")
assert "FS3I DYNAMIC:\n" in t
open(sys.argv[2], "w").write(t.replace("FS3I DYNAMIC:\n", blk, 1) if blk else t)
PY
python3 - "$BASE" "$TMP/real.yaml" "$REAL_PLACEMENT" <<'PY'
import sys
t = open(sys.argv[1]).read()
anchor = "FS3I DYNAMIC/STRUCTURE SCALAR STABILIZATION:"
assert anchor in t
blk = sys.argv[3].encode().decode("unicode_escape")
open(sys.argv[2], "w").write(t.replace(anchor, blk + anchor, 1))
PY
echo "CLAIMED_ARM_HAS_ITEMAX=$(grep -c 'ITEMAX' "$TMP/claimed.yaml")"
echo "REAL_ARM_HAS_PARTITIONED_SECTION=$(grep -c '^FS3I DYNAMIC/PARTITIONED:' "$TMP/real.yaml")"

probe PLAIN   "$TMP/plain.yaml"
probe CLAIMED "$TMP/claimed.yaml"
probe REAL    "$TMP/real.yaml"

grep -m1 -F "OK (3)" "$TMP/PLAIN.log"
grep -m1 -F "processor 0 finished normally" "$TMP/PLAIN.log"
grep -m1 -F "Could not match this input" "$TMP/CLAIMED.log"
grep -m1 -F "4C_io_input_spec_builders.cpp" "$TMP/CLAIMED.log"
grep -m1 -F "The following data remains unused:" "$TMP/CLAIMED.log"
grep -m1 -F "OK (3)" "$TMP/REAL.log"

# The claimed placement never runs; the real one runs and changes nothing,
# because the outer loop was already bounded by the default.
echo "CLAIMED_PLACEMENT_ACCEPTED=$(grep -c 'Checking results of' "$TMP/CLAIMED.log")"
echo "REAL_PLACEMENT_FAILED_TESTS=$(grep -c 'is WRONG' "$TMP/REAL.log")"
echo "PLAIN_FAILED_TESTS=$(grep -c 'is WRONG' "$TMP/PLAIN.log")"
exit 0
