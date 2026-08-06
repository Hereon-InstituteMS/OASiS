#!/bin/bash
# Tier-2 for fourc::fsi#15 — CONFIRMED, in full, including the "garbage vz"
# detail: 4C's runtime VTK writer emits an all-NaN pressure array and a nonzero
# third velocity component for a 2D fluid, while the solve itself is fine.
#
# Claimed: "2D fluid VTK output may show NaN pressure and garbage vz component —
#           this is a VTK output artifact, NOT divergence. ... pressure = NaN
#           over the entire 2D domain while the simulation logs report
#           convergence ... Check vx/vy (correct in 2D) and convergence logs."
# Observed: upstream 2D deck fsi_dc_part_ait_ga_ga.4C.yaml already carries an
#           IO/RUNTIME VTK OUTPUT/FLUID block; adding VELOCITY: true and
#           PRESSURE: true to it and decoding the base64+zlib payload of the last
#           step's .vtu gives, over all 4096 fluid points:
#             pressure   4096 / 4096 NaN
#             velocity   comp0, comp1 finite; comp2 spans about -1.4e-1 .. 1.4e-1
#                        in a problem with no third dimension
#             displacement, grid-velocity: comp2 identically 0, so the writer is
#                        not simply padding every vector with noise
#           and the run itself exits 0 with OK (4) — one of those four pinned
#           tests being fluid PRESSURE at node 2155, matched to 1e-10.  The
#           solver's pressure is right; only the file is wrong.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fsi_dc_part_ait_ga_ga.4C.yaml) || exit 3
grep -q '^  DIM: 2' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_is_not_2d"; exit 3; }
grep -q '^IO/RUNTIME VTK OUTPUT/FLUID:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_has_no_runtime_vtk_fluid_block"; exit 3; }
grep -q 'QUANTITY: "pressure"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_no_longer_result_tests_pressure"; exit 3; }

# The pathology: ask the runtime VTK writer for the 2D fluid pressure field.
REQUEST_VTK_PRESSURE=true

python3 - "$BASE" "$TMP/vtk.yaml" "$REQUEST_VTK_PRESSURE" <<'PY'
import sys
t = open(sys.argv[1]).read()
old = ("IO/RUNTIME VTK OUTPUT/FLUID:\n  OUTPUT_FLUID: true\n"
       "  DISPLACEMENT: true\n  GRIDVELOCITY: true")
assert old in t, "upstream runtime VTK fluid block changed"
t = t.replace(old, old + "\n  VELOCITY: true\n  PRESSURE: " + sys.argv[3], 1)
open(sys.argv[2], "w").write(t)
PY

probe VTK "$TMP/vtk.yaml"

# The solve converged and matched its pinned results — including the pressure.
grep -m1 -F "OK (4)" "$TMP/VTK.log"
grep -m1 -F "processor 0 finished normally" "$TMP/VTK.log"
echo "VTK_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/VTK.log")"
echo "VTK_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/VTK.log")"

# Now read the file 4C wrote.
python3 - "$TMP" <<'PY'
import base64, glob, math, os, re, struct, sys
tmp = sys.argv[1]
files = sorted(glob.glob(os.path.join(tmp, "o_VTK-vtk-files", "fluid-*-0.vtu")))
if not files:
    print("FIXTURE_ABORT=no_fluid_vtu_written")
    raise SystemExit(3)
print("FLUID_VTU_FILES_WRITTEN=%d" % len(files))
t = open(files[-1]).read()

def decode(payload):
    i = payload.index("==") + 2
    hdr, body = base64.b64decode(payload[:i]), base64.b64decode(payload[i:])
    n = struct.unpack("<III", hdr[:12])[0]
    sizes = struct.unpack("<%dI" % n, hdr[12:12 + 4 * n])
    import zlib
    out, off = b"", 0
    for s in sizes:
        out += zlib.decompress(body[off:off + s]); off += s
    return struct.unpack("<%dd" % (len(out) // 8), out)

arrays = {}
for m in re.finditer(r'<DataArray type="Float64" Name="([^"]+)"([^>]*)'
                     r'format="binary">\s*([A-Za-z0-9+/=]+)', t):
    nc = 1
    mm = re.search(r'NumberOfComponents="(\d+)"', m.group(2))
    if mm:
        nc = int(mm.group(1))
    arrays[m.group(1)] = (nc, decode(m.group(3)))

if "pressure" not in arrays:
    print("VTU_PRESSURE_POINTS=0")
    print("VTU_PRESSURE_NAN_POINTS=0")
    print("VTU_PRESSURE_ALL_NAN=no")
    raise SystemExit(0)
p_nc, p = arrays["pressure"]
nan = sum(1 for v in p if math.isnan(v))
print("VTU_PRESSURE_POINTS=%d" % len(p))
print("VTU_PRESSURE_NAN_POINTS=%d" % nan)
print("VTU_PRESSURE_ALL_NAN=%s" % ("yes" if nan == len(p) and len(p) else "no"))

v_nc, v = arrays["velocity"]
def rng(comp):
    c = [x for x in v[comp::v_nc] if not math.isnan(x)]
    return min(c), max(c)
for c in (0, 1):
    lo, hi = rng(c)
    print("VTU_VELOCITY_COMP%d_FINITE=%s" % (c, "yes" if (lo == lo and hi == hi) else "no"))
lo, hi = rng(2)
print("VTU_VELOCITY_VZ_IS_NONZERO_IN_2D=%s" % ("yes" if max(abs(lo), abs(hi)) > 1e-6 else "no"))
for name in ("displacement", "grid-velocity"):
    nc, a = arrays[name]
    z = max(abs(x) for x in a[2::nc])
    print("VTU_%s_COMP2_IS_ZERO=%s" % (name.replace("-", "_").upper(),
                                       "yes" if z == 0.0 else "no"))
PY
exit 0
