#!/bin/bash
# Tier-2 for fourc::pasi#2 — the symptom is real and measurable, the advice is
# one-sided: the interface mesh size has a WINDOW, not a floor.
#
# Upstream pasi_twoway_disprelaxaitken_particle_dem_1d_normalcontact_linspring_walldiscretcond.4C.yaml
# drops one DEM particle of radius 0.5 (diameter 1.0) onto a clamped membrane
# spanning 1.75 x 1.75.  Everything is kept except the membrane, which is
# regenerated as an N x N QUAD4 patch over the same square, so the only thing
# that varies is h_structure.  All arms are instrumented identically (ascii
# runtime VTK for the structure, particle-wall interaction output on) and the
# particle result tests are kept; the upstream STRUCTURE result tests are
# dropped because they name node ids of the upstream numbering.
#
#   N =  2   h/d = 0.875   the partitioned PASI loop DIVERGES
#   N =  3   h/d = 0.583   converges; 4 free interface nodes; every moving node
#                          deflects by the same amount, peak/mean = 1.000 — this
#                          is the entry's "stress field looks uniform", literally
#   N =  7   h/d = 0.250   converges and reproduces the upstream answer exactly,
#                          all six particle result tests correct; 36 free
#                          interface nodes and a genuine contact patch,
#                          peak/mean deflection about 5
#   N = 12   h/d = 0.146   DIVERGES AGAIN
#
# So the entry's "target h_structure ~ 0.5 * particle_diameter" lands inside the
# admissible band, but reading it as a floor and refining further is fatal: at
# h/d = 0.146 the run dies with
#   The partitioned PASI solver did not converge in ITEMAX steps!
# from pasi/4C_pasi_partitioned_twowaycoup.cpp, the same message the too-coarse
# arm produces.  One diagnostic, two opposite causes, and nothing in it mentions
# the mesh, the particle diameter or the interface.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream pasi_twoway_disprelaxaitken_particle_dem_1d_normalcontact_linspring_walldiscretcond.4C.yaml) || exit 3
grep -q '  - "TYPE phase1 POS 0.0 0.0 0.5"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "      INITRADIUS: 0.5" "$BASE"               || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cat > "$TMP/gen.py" <<'PY'
import sys
t = open(sys.argv[1]).read()
head = t[:t.index("RESULT DESCRIPTION:")]
res = t[t.index("RESULT DESCRIPTION:"):t.index("PARTICLES:")]
res = res[:res.index("  - STRUCTURE:")]          # keep the PARTICLE probes only
head = head.replace("PARTICLE DYNAMIC/DEM:\n",
                    "PARTICLE DYNAMIC/DEM:\n  WRITE_PARTICLE_WALL_INTERACTION: true\n")
head = head.replace("STRUCTURAL DYNAMIC:\n",
                    "IO/RUNTIME VTK OUTPUT:\n  INTERVAL_STEPS: 10\n  OUTPUT_DATA_FORMAT: ascii\n"
                    "IO/RUNTIME VTK OUTPUT/STRUCTURE:\n  OUTPUT_STRUCTURE: true\n  DISPLACEMENT: true\n"
                    "STRUCTURAL DYNAMIC:\n")
tail = '''PARTICLES:
  - "TYPE phase1 POS 0.0 0.0 0.5"
DESIGN POINT DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
DESIGN SURFACE PARTICLE WALL:
  - E: 1
    MAT: -1
'''
n, out = int(sys.argv[2]), sys.argv[3]
L = 1.75
nid = lambda i, j: j * (n + 1) + i + 1
co, dn, ds, el = [], [], [], []
for j in range(n + 1):
    for i in range(n + 1):
        co.append('  - "NODE %d COORD %.12e %.12e 0.0"'
                  % (nid(i, j), -0.875 + L * i / n, -0.875 + L * j / n))
        ds.append('  - "NODE %d DSURFACE 1"' % nid(i, j))
        if i in (0, n) or j in (0, n):
            dn.append('  - "NODE %d DNODE 1"' % nid(i, j))
e = 0
for j in range(n):
    for i in range(n):
        e += 1
        el.append('  - "%d MEMBRANE4 QUAD4 %d %d %d %d MAT 3 KINEM nonlinear THICK 0.01 '
                  'STRESS_STRAIN plane_stress"'
                  % (e, nid(i, j), nid(i+1, j), nid(i+1, j+1), nid(i, j+1)))
open(out, "w").write(head + res + tail +
                     "DNODE-NODE TOPOLOGY:\n" + "\n".join(dn) + "\n" +
                     "DSURF-NODE TOPOLOGY:\n" + "\n".join(ds) + "\n" +
                     "NODE COORDS:\n" + "\n".join(co) + "\n" +
                     "STRUCTURE ELEMENTS:\n" + "\n".join(el) + "\n")
print("H_OVER_PARTICLE_DIAMETER_N%d=%.3f" % (n, (L / n) / 1.0))
print("FREE_INTERFACE_NODES_N%d=%d" % (n, (n - 1) ** 2))
PY

for N in 2 3 7 12; do python3 "$TMP/gen.py" "$BASE" "$N" "$TMP/n$N.yaml"; done

probe N2  "$TMP/n2.yaml"
probe N3  "$TMP/n3.yaml"
probe N7  "$TMP/n7.yaml"
probe N12 "$TMP/n12.yaml"

# The regenerated mesh at the upstream resolution reproduces the shipped answer.
grep -m1 -F "processor 0 finished normally" "$TMP/N7.log"
echo "N7_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/N7.log")"
echo "N7_CORRECT_TESTS=$(grep -c 'is CORRECT, abs(diff)=' "$TMP/N7.log")"

# Too coarse and too fine both die, with the same sentence.
grep -m1 -F "The partitioned PASI solver did not converge in ITEMAX steps!" "$TMP/N2.log"
grep -m1 -oE "4C_pasi_partitioned_twowaycoup\.cpp, line [0-9]+" "$TMP/N2.log"
for L in N2 N3 N7 N12; do
  echo "DIVERGED_$L=$(grep -c 'did not converge in ITEMAX steps!' "$TMP/$L.log")"
done
echo "DIAGNOSTIC_NAMES_THE_MESH=$(grep -ciE 'did not converge in ITEMAX.*(mesh|element|interface|particle diameter)' "$TMP/N12.log")"

# How the contact is spread over the interface, from 4C's own displacement field.
python3 - "$TMP" <<'PY'
import re, sys
d = sys.argv[1]
def peak_over_mean(tag):
    f = "%s/o_%s-vtk-files/structure-00050-0.vtu" % (d, tag)
    t = open(f, errors="replace").read()
    pts = [float(x) for x in re.search(
        r'<Points>.*?<DataArray[^>]*>(.*?)</DataArray>', t, re.S).group(1).split()]
    dsp = [float(x) for x in re.search(
        r'<DataArray[^>]*Name="displacement"[^>]*>(.*?)</DataArray>', t, re.S).group(1).split()]
    uniq = {}
    for i in range(0, len(dsp), 3):
        uniq[(round(pts[i], 9), round(pts[i + 1], 9))] = abs(dsp[i + 2])
    mv = [v for v in uniq.values() if v > 1e-14]
    return len(uniq), len(mv), max(uniq.values()) / (sum(mv) / len(mv))
for tag in ("N3", "N7"):
    nn, nmv, r = peak_over_mean(tag)
    print("INTERFACE_NODES_%s=%d" % (tag, nn))
    print("MOVING_INTERFACE_NODES_%s=%d" % (tag, nmv))
    print("DEFLECTION_PEAK_OVER_MEAN_%s=%.3f" % (tag, r))
print("COARSE_INTERFACE_RESPONSE_IS_UNIFORM=%s"
      % ("yes" if abs(peak_over_mean("N3")[2] - 1.0) < 1e-6 else "no"))
PY
exit 0
