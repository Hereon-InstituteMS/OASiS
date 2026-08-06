#!/bin/bash
# Tier-2 for fourc::brownian_dynamics#3 — periodicity for an RVE is not a
# refinement, it is what makes the load reach the network at all.
#
# The entry says a free-boundary RVE "produces non-physical edge effects" and
# gets the average stress wrong by 5-20%, and points at DESIGN PERIODIC
# CONDITIONS.  Two things are worth correcting.
#
# First, the knob in 4C is PERIODICONOFF inside BINNING STRATEGY, alongside
# DOMAINBOUNDINGBOX — not a DESIGN condition block.  Upstream's RVE deck drives
# the box through PERIODIC BOUNDINGBOX ELEMENTS plus DESIGN SURF DIRICH
# CONDITIONS on its corner nodes.
#
# Second, the consequence is not a 5-20% stress error.  Delete the single
# PERIODICONOFF line and the filament stops being loaded at all: node 2's
# displacement comes back 0.00000000000000000e+00 in every component, against a
# prescribed -2.47500871887900376e-01 axially.  The RVE deformation never
# reaches the network, the run exits normally, and 4C prints no warning.  A
# "20% off" screen would pass this; a zero-response check catches it at once.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream beam3r_line2_backweuler_browndyn_periodic_rve_dirich_element.4C.yaml) || exit 3
grep -q 'PERIODICONOFF: "1 1 1"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "^PERIODIC BOUNDINGBOX ELEMENTS:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
# The knob really does live in BINNING STRATEGY, not in a DESIGN block.
echo "PERIODICONOFF_IS_IN_BINNING_STRATEGY=$(awk '/^BINNING STRATEGY:/{f=1;next}/^[A-Z]/{f=0}f&&/PERIODICONOFF/{c++}END{print c+0}' "$BASE")"
echo "DESIGN_PERIODIC_CONDITIONS_SECTIONS=$(grep -c 'DESIGN PERIODIC CONDITIONS' "$BASE")"

cp "$BASE" "$TMP/periodic.yaml"
grep -v 'PERIODICONOFF' "$BASE" > "$TMP/free.yaml"
echo "PERIODICONOFF_LINES_REMOVED=$(( $(grep -c PERIODICONOFF "$TMP/periodic.yaml") - $(grep -c PERIODICONOFF "$TMP/free.yaml") ))"

probe PERIODIC "$TMP/periodic.yaml"
probe FREE     "$TMP/free.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/PERIODIC.log"
echo "PERIODIC_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/PERIODIC.log")"

# Without periodicity the drive never reaches the filament.
echo "FREE_WARNINGS=$(grep -ciE 'periodic|bounding ?box.*(ignor|inactive)' "$TMP/FREE.log")"
echo "FREE_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/FREE.log")"
grep -m1 -F "is WRONG --> actresult= 0.00000000000000000e+00" "$TMP/FREE.log"
python3 - "$TMP/FREE.log" <<'PY'
import re, sys
vals = []
for l in open(sys.argv[1]):
    m = re.search(r"disp[xyz]\s+at node\s+2\s+is (?:WRONG --> actresult=\s*(-?[0-9.e+-]+)|CORRECT, abs\(diff\)=\s*([0-9.e+-]+))", l)
    if m:
        vals.append(float(m.group(1) if m.group(1) else m.group(2)))
print("FREE_NODE2_COMPONENTS=%d" % len(vals))
print("FREE_RVE_RESPONSE_IS_EXACTLY_ZERO=%s"
      % ("yes" if len(vals) == 3 and all(v == 0.0 for v in vals) else "no"))
PY
exit 0
