#!/bin/bash
# Tier-2 for fourc::beam_interaction#4 — falsified.  4C's beam-to-solid surface
# contact gap ALREADY subtracts the beam cross-section radius.
#
# Claimed:  "The contact detection uses the beam centerline distance to the
#           surface, not the beam radius.  The gap offset must account for the
#           beam cross-section.  Signal: visualize shows the beam centerline at
#           the solid surface (gap = 0) instead of one beam-radius offset."
# Observed: gap = 0 means the beam's OUTER SURFACE touches, not its centerline,
#           and the user has no offset to add.
#
# Upstream beam3r_herm2line3_static_beam_to_solid_surface_contact_ironing_penalty_gap_variation_segmentation.4C.yaml
# is run three ways, differing only in the beam interaction radius:
#   RDEFAULT  no INTERACTIONRADIUS, so 4C derives it from MOMIN2 as
#             (4*Iyy/pi)^(1/4) = 0.4        -> exit 0, all six result tests pass
#   REXPLICIT INTERACTIONRADIUS: 0.4        -> byte-for-byte the same verdict,
#             which pins the derived default at 0.4
#   RHALF     INTERACTIONRADIUS: 0.2        -> exit 1, all six result tests fail
#
# If the gap were the bare centerline distance the radius could not matter at
# all.  It matters, and by exactly the right amount: at the first output, before
# the two arms have deformed apart, every one of the 18 reported gap values in
# RHALF exceeds its RDEFAULT counterpart by exactly 0.2 = 0.4 - 0.2.  The gap
# 4C reports is therefore (r_beam - r_surface).n - R_beam, and with the correct
# radius the initial minimum gap is zero to machine precision: the beam surface
# is set down exactly on the solid surface.
#
# Both arms are instrumented identically (ascii VTK so the gap array is
# readable, per-step rather than per-iteration output, absolute path to the
# deck's NOX xml); the only difference between them is the radius.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream beam3r_herm2line3_static_beam_to_solid_surface_contact_ironing_penalty_gap_variation_segmentation.4C.yaml) || exit 3
XML=$(dirname "$BASE")/beam3r_herm2line3_static_beam_to_solid_surface_contact_ironing.xml
[ -f "$XML" ] || { echo "FIXTURE_ABORT=no_upstream_decks (missing $XML)"; exit 3; }
grep -q "      MOMIN2: 0.02010619298297468" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "  CONTACT_TYPE: gap_variation" "$BASE"     || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "INTERACTIONRADIUS" "$BASE" && { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$BASE" "$XML" "$TMP" <<'PY'
import os, sys
t, xml, d = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3]
old_io = "IO/RUNTIME VTK OUTPUT:\n  INTERVAL_STEPS: 1\n  EVERY_ITERATION: true\n"
assert old_io in t
t = t.replace(old_io, "IO/RUNTIME VTK OUTPUT:\n  INTERVAL_STEPS: 1\n"
                      "  EVERY_ITERATION: false\n  OUTPUT_DATA_FORMAT: ascii\n")
old_xml = '  XML File: "beam3r_herm2line3_static_beam_to_solid_surface_contact_ironing.xml"'
assert old_xml in t
t = t.replace(old_xml, '  XML File: "%s"' % xml)
anchor = "      MOMIN3: 0.02010619298297468"
open(os.path.join(d, "rdefault.yaml"),  "w").write(t)
open(os.path.join(d, "rexplicit.yaml"), "w").write(t.replace(anchor, anchor + "\n      INTERACTIONRADIUS: 0.4"))
open(os.path.join(d, "rhalf.yaml"),     "w").write(t.replace(anchor, anchor + "\n      INTERACTIONRADIUS: 0.2"))
print("DERIVED_DEFAULT_RADIUS=%.1f" % (4.0 * 0.02010619298297468 / 3.141592653589793) ** 0.25)
PY

probe RDEFAULT  "$TMP/rdefault.yaml"
probe REXPLICIT "$TMP/rexplicit.yaml"
probe RHALF     "$TMP/rhalf.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/RDEFAULT.log"
grep -m1 -F "PID  0 currently monitors    16 beam contact pairs" "$TMP/RDEFAULT.log"
echo "RDEFAULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/RDEFAULT.log")"
echo "REXPLICIT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/REXPLICIT.log")"
echo "RHALF_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/RHALF.log")"

# Read the gap array 4C wrote at the first output, where both arms are still in
# the same configuration, and compare them point by point.
python3 - "$TMP" <<'PY'
import re, sys
d = sys.argv[1]
def gaps(tag):
    f = "%s/o_%s-vtk-files/beam-to-solid-surface-contact-integration-points-00001-0.vtu" % (d, tag)
    t = open(f, errors="replace").read()
    m = re.search(r'<DataArray[^>]*Name="gap"[^>]*>(.*?)</DataArray>', t, re.S)
    assert m, "no ascii gap array in %s" % f
    return [float(x) for x in m.group(1).split()]
a, b = gaps("RDEFAULT"), gaps("RHALF")
assert len(a) == len(b)
print("GAP_POINTS_STEP1=%d" % len(a))
shift = sorted({round(y - x, 9) for x, y in zip(a, b)})
print("GAP_SHIFT_STEP1_IS_UNIFORM=%s" % ("yes" if len(shift) == 1 else "no"))
print("GAP_SHIFT_STEP1=%.6f" % shift[0])
print("MIN_GAP_STEP1_RDEFAULT_IS_ZERO_TO_1E-12=%s" % ("yes" if abs(min(a)) < 1e-12 else "no"))
print("MIN_GAP_STEP1_RHALF=%.6f" % min(b))
print("CLAIMED_GAP_IGNORES_BEAM_RADIUS=%s" % ("yes" if shift == [0.0] else "no"))
PY
exit 0
