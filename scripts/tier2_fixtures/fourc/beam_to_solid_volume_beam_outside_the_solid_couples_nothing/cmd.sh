#!/bin/bash
# Tier-2 for fourc::beam_interaction#3 — the rule is confirmed and the promised
# diagnostic does not exist.
#
# Claimed:  "Beams outside the solid domain are not coupled.  Signal:
#           BeamToSolidMeshtying diagnostic prints `0 of N beam segments
#           coupled`; the beam displaces freely as if no solid mesh existed."
# Observed: the physics is exactly right, and 4C prints nothing at all.  Nothing
#           in the log matches "segments coupled", and no line reports how many
#           pairs were found or lost.  The only place the coupling shows up is
#           the beam-to-solid runtime VTK output, which the deck has to have
#           switched on: the segmentation file's point count is the coupled
#           segment endpoints and the integration-points file's count is the
#           coupling quadrature points.
#
# Upstream beam3r_herm2line3_static_beam_to_solid_volume_meshtying_beam_in_solid_column_segmentation.4C.yaml
# threads a beam up the axis of a hex8 column.  Translating only the nine beam
# node coordinates by +1.0 in x — still well inside the binning box — takes the
# beam out of the solid and:
#   * the segmentation output falls from 26 points (13 coupled segments) to 0,
#     at every one of the writes, and the integration points from 78 to 0;
#   * the solid stops moving entirely — its probed node reports displacement
#     exactly 0.00000000000000000e+00, i.e. the beam is loading nothing;
#   * the beam itself bends further than the coupled reference, which is the
#     "displaces freely" half of the claim;
#   * and the run still exits through the result test, not through any coupling
#     check, because there is no coupling check.
#
# NOTE ON THE BUILD: this checkout is FOUR_C_WITH_ARBORX=OFF.  That only rules
# out SEARCH_STRATEGY: bounding_volume_hierarchy; the default
# bruteforce_with_binning path used here needs no ArborX and runs in seconds.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream beam3r_herm2line3_static_beam_to_solid_volume_meshtying_beam_in_solid_column_segmentation.4C.yaml) || exit 3
grep -q "  SEGMENTATION: true" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '  DOMAINBOUNDINGBOX: "-2 -2 -2 2 2 2"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/inside.yaml"
python3 - "$BASE" "$TMP/outside.yaml" <<'PY'
import re, sys
out, n = [], 0
for l in open(sys.argv[1]):
    l = l.rstrip("\n")
    m = re.match(r'(\s*- "NODE )(4[5-9]|5[0-3])( COORD )(\S+) (\S+) (\S+)"$', l)
    if m:
        n += 1
        out.append('%s%s%s%.10e %s %s"' % (m.group(1), m.group(2), m.group(3),
                                           float(m.group(4)) + 1.0, m.group(5), m.group(6)))
    else:
        out.append(l)
assert n == 9, "upstream deck no longer carries the nine beam nodes 45..53"
open(sys.argv[2], "w").write("\n".join(out) + "\n")
print("BEAM_NODES_MOVED=%d" % n)
PY

probe INSIDE  "$TMP/inside.yaml"
probe OUTSIDE "$TMP/outside.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/INSIDE.log"
echo "INSIDE_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/INSIDE.log")"
echo "OUTSIDE_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/OUTSIDE.log")"

# Count the coupling 4C actually wrote out.
python3 - "$TMP" <<'PY'
import glob, re, sys
d = sys.argv[1]
def peak(tag, kind):
    v = []
    for f in sorted(glob.glob("%s/o_%s-vtk-files/*%s*-0.vtu" % (d, tag, kind))):
        m = re.search(r'NumberOfPoints="(\d+)"', open(f, errors="replace").read(4000))
        v.append(int(m.group(1)))
    assert v, "no %s output written for %s" % (kind, tag)
    return v
for tag in ("INSIDE", "OUTSIDE"):
    seg = peak(tag, "segmentation")
    ip = peak(tag, "integration-points")
    print("MAX_SEGMENT_POINTS_%s=%d" % (tag, max(seg)))
    print("MAX_COUPLED_SEGMENTS_%s=%d" % (tag, max(seg) // 2))
    print("MAX_INTEGRATION_POINTS_%s=%d" % (tag, max(ip)))
    print("SEGMENTATION_WRITES_WITH_COUPLING_%s=%d" % (tag, sum(1 for x in seg if x)))
PY

# The solid is not loaded at all any more.
grep -m1 -F "is WRONG --> actresult= 0.00000000000000000e+00, givenresult= 2.12196301048432762e-01" "$TMP/OUTSIDE.log"
# And 4C never says a word about it.
echo "CLAIMED_SEGMENTS_COUPLED_TEXT=$(grep -ci 'segments coupled' "$TMP/OUTSIDE.log")"
echo "OUTSIDE_COUPLING_WARNINGS=$(grep -ciE 'no .*(pair|coupling) found|not coupled|outside the solid' "$TMP/OUTSIDE.log")"
grep -m1 -oE "4C_utils_result_test\.cpp, line [0-9]+" "$TMP/OUTSIDE.log"
exit 0
