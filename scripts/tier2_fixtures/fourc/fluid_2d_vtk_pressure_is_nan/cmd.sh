#!/bin/bash
# Tier-2 for fourc::input_format#8 — in 2D, the runtime VTK output of the fluid
# field writes nan for PRESSURE at every point while the simulation itself is
# fine.  The mechanism is sharper than "garbage in the z component": in 2D a
# node carries three DOFs (vx, vy, p), and the writer copies all three into the
# 3-vector called "velocity" and then reads a fourth DOF for "pressure" that
# does not exist.
#
# So the pressure IS in the file -- as the third component of "velocity" -- and
# the array named "pressure" is entirely nan.  The fixture proves that by
# matching vtu points back to input node ids by coordinate and comparing the
# third velocity component against the pressures the deck's own RESULT
# DESCRIPTION pins (-0.5, 0.25, 0.5 at nodes 1, 19, 25).
#
# 3D is unaffected: the same sections on a 3D Stokes deck give a pressure array
# with no nan at all.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE2D=$(upstream f2_stokes_residualbased.4C.yaml) || exit 3
BASE3D=$(upstream f3_stokes_residualbased_rotboxgeom.4C.yaml) || exit 3
grep -q 'QUANTITY: "pressure"' "$BASE2D" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '"NODE 1 COORD -0.5 -0.5 0.0"' "$BASE2D" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

VTK='IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: ascii
IO/RUNTIME VTK OUTPUT/FLUID:
  OUTPUT_FLUID: true
  VELOCITY: true
  PRESSURE: true
'
mkdir -p "$TMP/d2" "$TMP/d3"
{ printf '%s' "$VTK"; cat "$BASE2D"; } > "$TMP/d2/in.4C.yaml"
{ printf '%s' "$VTK"; cat "$BASE3D"; } > "$TMP/d3/in.4C.yaml"

run4c "$TMP/d2/in.4C.yaml" "$TMP/d2/out" > "$TMP/d2/log" 2>&1; echo "EXIT_2D=$?"
run4c "$TMP/d3/in.4C.yaml" "$TMP/d3/out" > "$TMP/d3/log" 2>&1; echo "EXIT_3D=$?"

# The 2D run is healthy: it converges and every pinned result test passes.
grep -m1 -F "processor 0 finished normally" "$TMP/d2/log"
echo "TESTS_CORRECT_2D=$(grep -c 'is CORRECT' "$TMP/d2/log")"
echo "TESTS_WRONG_2D=$(grep -c 'is WRONG' "$TMP/d2/log")"
echo "SOLVER_COMPLAINED_ABOUT_NAN=$(grep -ci 'nan' "$TMP/d2/log")"

V2=$(find "$TMP/d2" -name 'fluid-*.vtu' | head -1)
V3=$(find "$TMP/d3" -name 'fluid-*.vtu' | head -1)
[ -n "$V2" ] && [ -n "$V3" ] || { echo "FIXTURE_ABORT=no_vtu_written"; exit 3; }

python3 - "$V2" "$V3" "$BASE2D" <<'PY'
import re, sys

def arr(text, name):
    m = re.search(r'<DataArray[^>]*Name="%s"[^>]*>(.*?)</DataArray>' % name, text, re.S)
    return m.group(1).split() if m else []

t2 = open(sys.argv[1]).read()
t3 = open(sys.argv[2]).read()
deck = open(sys.argv[3]).read()

p2, v2 = arr(t2, "pressure"), arr(t2, "velocity")
p3 = arr(t3, "pressure")
nan2 = sum(1 for x in p2 if x.lower().startswith(("nan", "-nan")))
nan3 = sum(1 for x in p3 if x.lower().startswith(("nan", "-nan")))
print("VTU_2D_PRESSURE_POINTS=%d" % len(p2))
print("VTU_2D_PRESSURE_ALL_NAN=%s" % ("yes" if len(p2) and nan2 == len(p2) else "no"))
print("VTU_2D_VELOCITY_HAS_NAN=%s" % ("yes" if any(x.lower().startswith(("nan", "-nan")) for x in v2) else "no"))
print("VTU_3D_PRESSURE_POINTS=%d" % len(p3))
print("VTU_3D_PRESSURE_NAN_COUNT=%d" % nan3)

pts = re.search(r'<Points>.*?<DataArray[^>]*>(.*?)</DataArray>', t2, re.S).group(1).split()
pts = [tuple(round(float(x), 9) for x in pts[3 * i:3 * i + 3]) for i in range(len(pts) // 3)]
coord = {}
for m in re.finditer(r'"NODE (\d+) COORD ([-\d.eE+ ]+)"', deck):
    coord[int(m.group(1))] = tuple(round(float(x), 9) for x in m.group(2).split())

# Pressures the deck itself pins, node -> value.
want = {1: -0.5, 19: 0.25, 25: 0.5}
hits = 0
for node, p in want.items():
    i = pts.index(coord[node])
    third = float(v2[3 * i + 2])
    if abs(third - p) < 1e-9:
        hits += 1
print("VTU_2D_VELOCITY_Z_HOLDS_THE_PRESSURE=%d/%d" % (hits, len(want)))
PY
exit 0
