#!/bin/bash
# Tier-2 for fourc::fsi#11 — COMPONENT: 0 is OPTIONAL, and a VARIABLE is honoured
# with or without it.  The claimed silent miss does not happen.
#
# Claimed: "FUNCT with SYMBOLIC_FUNCTION_OF_SPACE_TIME + VARIABLE requires
#           COMPONENT: 0 in the same list item.  Signal: without COMPONENT, the
#           VARIABLE definition is silently ignored and the function returns
#           wrong values — an inflow ramp stays stuck at 0 instead of ramping up."
# Observed: two things.  (1) VARIABLE is not written "in the same list item" at
#           all — every upstream deck puts it in its OWN list entry, sibling to
#           the SYMBOLIC_FUNCTION_OF_SPACE_TIME entry; writing it inside the same
#           item is what actually fails to parse.  (2) Dropping COMPONENT: 0 from
#           the function entry changes nothing.
#
# Upstream fsi_fp_mono_fs_ga_ga.4C.yaml drives its structure with
# FUNCT1 = -t*t and pins six results.  Rewriting that as -t*t*a with a
# linearinterpolation VARIABLE a == 1 reproduces all six results BOTH with and
# without COMPONENT: 0.  Setting the same variable to ramp 1 -> 0.5, still
# without COMPONENT, moves every one of the six pinned values — so the variable
# is being read, not ignored.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fsi_fp_mono_fs_ga_ga.4C.yaml) || exit 3
grep -q 'SYMBOLIC_FUNCTION_OF_SPACE_TIME: "-t\*t"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_funct1_changed"; exit 3; }

# The pathology: omit COMPONENT: 0 from the function entry that uses a VARIABLE.
COMPONENT_LINE=""

python3 - "$BASE" "$TMP" "$COMPONENT_LINE" <<'PY'
import sys
src, tmp, comp = sys.argv[1], sys.argv[2], sys.argv[3]
t = open(src).read()
old = 'FUNCT1:\n  - COMPONENT: 0\n    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "-t*t"\n'
assert old in t
def funct(component, values):
    head = ("  - COMPONENT: 0\n    " if component else "  - ")
    return ('FUNCT1:\n' + head + 'SYMBOLIC_FUNCTION_OF_SPACE_TIME: "-t*t*a"\n'
            '  - VARIABLE: 0\n'
            '    NAME: "a"\n'
            '    TYPE: "linearinterpolation"\n'
            '    NUMPOINTS: 2\n'
            '    TIMES: [0, 1]\n'
            '    VALUES: [%s]\n' % values)
open(tmp + "/withcomp.yaml", "w").write(t.replace(old, funct("COMPONENT: 0", "1, 1"), 1))
open(tmp + "/nocomp.yaml", "w").write(t.replace(old, funct(comp, "1, 1"), 1))
open(tmp + "/nocomp_half.yaml", "w").write(t.replace(old, funct(comp, "1, 0.5"), 1))
PY
echo "NOCOMP_ARM_HAS_COMPONENT_KEY=$(grep -c 'COMPONENT: 0' "$TMP/nocomp.yaml")"
echo "NOCOMP_ARM_HAS_VARIABLE_KEY=$(grep -c 'VARIABLE: 0' "$TMP/nocomp.yaml")"

probe WITHCOMP   "$TMP/withcomp.yaml"
probe NOCOMP     "$TMP/nocomp.yaml"
probe NOCOMPHALF "$TMP/nocomp_half.yaml"

# With COMPONENT: the variable reproduces the deck's own drive exactly.
grep -m1 -F "OK (6)" "$TMP/WITHCOMP.log"
# Without COMPONENT: identical.  Not "silently ignored", not "stuck at 0".
grep -m1 -F "processor 0 finished normally" "$TMP/NOCOMP.log"
grep -m1 -F "OK (6)" "$TMP/NOCOMP.log"
echo "NOCOMP_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/NOCOMP.log")"
echo "NOCOMP_FUNCT_WARNINGS=$(grep -ciE 'funct.*(ignor|COMPONENT|variable not)' "$TMP/NOCOMP.log")"

# And the variable really is what drives the answer: halve it and every pinned
# result moves, still without COMPONENT.
grep -m1 -F "Result check failed with 6 errors out of 6 tests" "$TMP/NOCOMPHALF.log"
grep -m1 -E "velx +at node +5.*is WRONG --> actresult=" "$TMP/NOCOMPHALF.log"
exit 0
