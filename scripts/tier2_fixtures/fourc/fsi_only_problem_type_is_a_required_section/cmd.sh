#!/bin/bash
# Tier-2 for fourc::fsi#0 — "an FSI input missing any of the required sections
# (PROBLEM TYPE / STRUCTURAL DYNAMIC / FLUID DYNAMIC / ALE DYNAMIC / FSI DYNAMIC
# / CLONING MATERIAL MAP) aborts at setup with 'missing required section' from
# 4C_io_input_file.cpp".
#
# Only ONE of those six behaves that way, and the wording is different.
# Deleting each section in turn from the upstream monolithic deck
# fsi_fp_mono_fs_ga_ga.4C.yaml gives five DIFFERENT outcomes:
#
#   PROBLEM TYPE       -> "Required section 'PROBLEM TYPE' not found in input
#                          file." from 4C_io_input_file.cpp        (the claim)
#   STRUCTURAL DYNAMIC -> "no linear solver defined for structural field..."
#                          from 4C_adapter_str_structure.cpp
#   FLUID DYNAMIC      -> "no linear solver defined for fluid problem..."
#                          from 4C_adapter_fld_base_algorithm.cpp
#   ALE DYNAMIC        -> "No linear solver defined for ALE problems..."
#                          from 4C_adapter_ale.cpp
#   FSI DYNAMIC        -> NO abort at all.  4C silently defaults COUPALGO to a
#                          PARTITIONED scheme and MAXTIME to 1000, runs 13 time
#                          steps and dies on a negative fluid Jacobian.
#
# The four DYNAMIC sections are not "required sections" — they are required
# only because LINEAR_SOLVER has no default, and each field complains in its own
# words from its own adapter.  An agent grepping for the claimed phrase finds
# nothing in any of the five logs.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fsi_fp_mono_fs_ga_ga.4C.yaml) || exit 3
for s in "PROBLEM TYPE" "STRUCTURAL DYNAMIC" "FLUID DYNAMIC" "ALE DYNAMIC" "FSI DYNAMIC"; do
  grep -q "^$s:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed_missing_$s"; exit 3; }
done

# The pathology: delete one top-level section per arm.
DELETE_SECTIONS=yes

drop() {  # $1 = section name, $2 = out file
python3 - "$BASE" "$2" "$1" "$DELETE_SECTIONS" <<'PY'
import sys
src, dst, sec, do = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
out, drop = [], False
for line in open(src).read().split("\n"):
    if do == "yes" and line.startswith(sec + ":"):
        drop = True
        continue
    if drop and line and not line[0].isspace():
        drop = False
    if not drop:
        out.append(line)
open(dst, "w").write("\n".join(out))
PY
}

drop "PROBLEM TYPE"       "$TMP/noprob.yaml"
drop "STRUCTURAL DYNAMIC" "$TMP/nostruct.yaml"
drop "FLUID DYNAMIC"      "$TMP/nofluid.yaml"
drop "ALE DYNAMIC"        "$TMP/noale.yaml"
drop "FSI DYNAMIC"        "$TMP/nofsi.yaml"
cp "$BASE" "$TMP/full.yaml"

probe FULL     "$TMP/full.yaml"
probe NOPROB   "$TMP/noprob.yaml"
probe NOSTRUCT "$TMP/nostruct.yaml"
probe NOFLUID  "$TMP/nofluid.yaml"
probe NOALE    "$TMP/noale.yaml"
probe NOFSI    "$TMP/nofsi.yaml"

# The complete deck is the control.
grep -m1 -F "processor 0 finished normally" "$TMP/FULL.log"
grep -m1 -F "OK (6)" "$TMP/FULL.log"

# Exactly one section really is a "required section", and this is its wording.
grep -m1 -F "Required section 'PROBLEM TYPE' not found in input file." "$TMP/NOPROB.log"
grep -m1 -F "4C_io_input_file.cpp" "$TMP/NOPROB.log"

# The three field DYNAMIC sections each complain about LINEAR_SOLVER instead,
# from three different adapter files.
grep -m1 -F "no linear solver defined for structural field" "$TMP/NOSTRUCT.log"
grep -m1 -F "4C_adapter_str_structure.cpp"                  "$TMP/NOSTRUCT.log"
grep -m1 -F "no linear solver defined for fluid problem"    "$TMP/NOFLUID.log"
grep -m1 -F "4C_adapter_fld_base_algorithm.cpp"             "$TMP/NOFLUID.log"
grep -m1 -F "No linear solver defined for ALE problems"     "$TMP/NOALE.log"
grep -m1 -F "4C_adapter_ale.cpp"                            "$TMP/NOALE.log"

# FSI DYNAMIC is not required at all: the run proceeds with defaults.
echo "NOFSI_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/NOFSI.log")"
echo "NOFSI_USES_PARTITIONED_SCHEME=$(grep -c 'FSI::Partitioned' "$TMP/NOFSI.log" | awk '{print ($1>0)?1:0}')"
echo "NOFSI_STEP_BUDGET=$(grep -m1 -o 'STEP =[ ]*1/[ ]*[0-9]*' "$TMP/NOFSI.log" | grep -o '[0-9]*$')"

# The claimed phrase appears in none of the five logs.
echo "CLAIMED_MISSING_REQUIRED_SECTION=$(cat "$TMP"/NOPROB.log "$TMP"/NOSTRUCT.log \
      "$TMP"/NOFLUID.log "$TMP"/NOALE.log "$TMP"/NOFSI.log | grep -ci 'missing required section')"
exit 0
