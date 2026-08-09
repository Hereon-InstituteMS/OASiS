#!/bin/bash
# Tier-2 for fourc::multiscale#5 — the RVE queue length per MPI rank is decided
# by the macro partition, it is badly uneven, and it never changes during the
# run.  Measured in RVE COUNTS, never in seconds.
#
# Upstream sohex8_multiscale_macro.4C.yaml is 8 hex8 macro elements, each with 8
# Gauss points, each Gauss point carrying its own MAT_Struct_Multiscale RVE: 64
# RVE solves per macro Newton step in total.  Every one of them is solved by
# whichever rank owns the macro element, so "RVEs on this rank" is exactly
# 8 x (macro elements owned).  4C writes that ownership itself, as the
# element_owner cell array of its structure runtime VTK output.
#
#   1 rank   8 elements          -> one queue of 64
#   2 ranks  3 and 5 elements    -> queues of 24 and 40
#   3 ranks  1, 3 and 4 elements -> queues of  8, 24 and 32, a four-to-one spread
#
# on a mesh where a perfect split would be 2-3-3.  The spread is not a transient:
# the element_owner map at the last output step is identical to the one at the
# first, so nothing was redistributed while the run was going on, which is the
# entry's point.  And all three runs pass all three result tests with the same
# values, so the imbalance is invisible in the answer — the only way to see it is
# to count who owns what, exactly as done here.
#
# Nothing in this fixture reads a clock.  Wall time is not an admissible
# observable and is not used.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

command -v mpirun >/dev/null || { echo "FIXTURE_ABORT=no_mpirun"; exit 3; }

BASE=$(upstream sohex8_multiscale_macro.4C.yaml) || exit 3
MICRO=$(upstream sohex8_multiscale_micro.mat.4C.yaml) || exit 3
grep -q "  MICROFILE: \"sohex8_multiscale_micro.mat.4C.yaml\"" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "^  ELEMENTS: 8" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$MICRO" "$TMP/"
python3 - "$BASE" "$TMP/macro.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
old = "IO/RUNTIME VTK OUTPUT/STRUCTURE:\n  OUTPUT_STRUCTURE: true\n  DISPLACEMENT: true\n"
assert old in t, "upstream macro deck no longer carries the structure VTK block"
# instrumentation only, identical for every arm: ask 4C which rank owns which
# macro element.
open(sys.argv[2], "w").write(t.replace(old, old + "  ELEMENT_OWNER: true\n"))
print("MACRO_ELEMENTS=%d" % t.count('SOLID HEX8'))
print("GAUSS_POINTS_PER_ELEMENT=8")
print("TOTAL_RVES=%d" % (8 * t.count('SOLID HEX8')))
PY

mpiprobe() {  # $1 = label, $2 = number of ranks
  # --bind-to none matters: on a loaded box OpenMPI's default core binding can
  # fail outright under --oversubscribe, which would turn a scheduling
  # measurement into a spurious red.
  ( cd "$TMP" && mpirun -np "$2" --oversubscribe --bind-to none \
        "$BIN" "$TMP/macro.yaml" "$TMP/o_$1" ) > "$TMP/$1.log" 2>&1
  echo "EXIT_$1=$?"
}
mpiprobe NP1 1
mpiprobe NP2 2
mpiprobe NP3 3

grep -m1 -F "processor 0 finished normally" "$TMP/NP1.log"
for L in NP1 NP2 NP3; do
  echo "${L}_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/$L.log")"
done

python3 - "$TMP" <<'PY'
import collections, glob, re, sys
d = sys.argv[1]
def owners(tag, step):
    v = []
    for f in sorted(glob.glob("%s/o_%s-vtk-files/structure-%s-*.vtu" % (d, tag, step))):
        t = open(f, errors="replace").read()
        m = re.search(r'<DataArray[^>]*Name="element_owner"[^>]*>(.*?)</DataArray>', t, re.S)
        if m:
            v.extend(int(float(x)) for x in m.group(1).split())
    assert v, "no element_owner written for %s at step %s" % (tag, step)
    return collections.Counter(v)
for tag in ("NP1", "NP2", "NP3"):
    first, last = owners(tag, "00000"), owners(tag, "00003")
    q = sorted(8 * n for n in first.values())
    print("ELEMENTS_PER_RANK_%s=%s" % (tag, ",".join(str(first[k]) for k in sorted(first))))
    print("RVE_QUEUE_%s=%s" % (tag, ",".join(str(x) for x in q)))
    print("MAX_RVE_QUEUE_%s=%d" % (tag, max(q)))
    print("MIN_RVE_QUEUE_%s=%d" % (tag, min(q)))
    print("RVE_QUEUE_IMBALANCE_%s=%.2f" % (tag, max(q) / min(q)))
    print("OWNER_MAP_CHANGED_DURING_RUN_%s=%s" % (tag, "no" if first == last else "yes"))
PY
exit 0
