#!/bin/bash
# Tier-2 for fourc::fsi#10 — EVERY_ITERATION in the IO section is a parse error,
# and the same key one section deeper is perfectly legal.  Both halves executed.
#
# Claimed: "IO section has NO EVERY_ITERATION parameter ... Signal: 'Could not
#           match this input' from core/io/src/4C_io_input_spec_builders.cpp,
#           echoing the IO block and the candidate specification, exit 1. ...
#           Note EVERY_ITERATION IS a real key — it lives in IO/RUNTIME VTK
#           OUTPUT, not in IO."
# Observed: confirmed on both counts, at line 633 of that file.  The abort is
#           worth pinning because of what it echoes: 4C prints the offending
#           block back, then a candidate list in which EVERY_ITERATION appears
#           nowhere but 'Defaulted parameter' rows for the keys IO does accept
#           appear by the dozen — the reader has to notice an ABSENCE.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fsi_fp_mono_fs_ga_ga.4C.yaml) || exit 3
grep -q '^PROBLEM TYPE:' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_has_no_problem_type"; exit 3; }

# The pathology: put EVERY_ITERATION in the IO section.
EVERY_ITERATION_SECTION="IO"

cp "$BASE" "$TMP/plain.yaml"
python3 - "$BASE" "$TMP" "$EVERY_ITERATION_SECTION" <<'PY'
import sys
src, tmp, sec = sys.argv[1:4]
t = open(src).read()
assert "PROBLEM TYPE:" in t
head = "%s:\n  INTERVAL_STEPS: 1\n" % sec if sec != "IO" else "IO:\n"
open(tmp + "/badsection.yaml", "w").write(
    t.replace("PROBLEM TYPE:", head + "  EVERY_ITERATION: true\nPROBLEM TYPE:", 1))
open(tmp + "/goodsection.yaml", "w").write(
    t.replace("PROBLEM TYPE:",
              "IO/RUNTIME VTK OUTPUT:\n  INTERVAL_STEPS: 1\n"
              "  EVERY_ITERATION: false\nPROBLEM TYPE:", 1))
PY

probe PLAIN  "$TMP/plain.yaml"
probe BAD    "$TMP/badsection.yaml"
probe GOOD   "$TMP/goodsection.yaml"

grep -m1 -F "OK (6)" "$TMP/PLAIN.log"

# EVERY_ITERATION in IO: parse error, with the echoed block and candidate list.
grep -m1 -F "Could not match this input" "$TMP/BAD.log"
grep -m1 -F "4C_io_input_spec_builders.cpp" "$TMP/BAD.log"
grep -m1 -F "against the given input specification" "$TMP/BAD.log"
grep -m1 -F "Candidate group 'IO'" "$TMP/BAD.log"
echo "BAD_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/BAD.log")"
OFFERED=$(grep -c -F "parameter 'EVERY_ITERATION'" "$TMP/BAD.log")
LISTED=$(grep -c -F "Defaulted parameter" "$TMP/BAD.log")
echo "BAD_CANDIDATE_LIST_OFFERS_EVERY_ITERATION=$OFFERED"
echo "BAD_CANDIDATE_LIST_HAS_DEFAULTED_ROWS=$LISTED"

# The same key inside IO/RUNTIME VTK OUTPUT is accepted and changes nothing else.
grep -m1 -F "processor 0 finished normally" "$TMP/GOOD.log"
grep -m1 -F "OK (6)" "$TMP/GOOD.log"
echo "GOOD_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/GOOD.log")"
echo "CLAIMED_UNKNOWN_PARAMETER_TEXT=$(grep -ci 'unknown parameter EVERY_ITERATION' "$TMP/BAD.log")"
exit 0
