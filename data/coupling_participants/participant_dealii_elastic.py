"""deal.II VECTOR participant for the OASiS `couple` driver.

Plane-strain linear elasticity  -div(sigma(u)) = 0  on ONE rectangular
subdomain of a domain split by a straight interface at x = IFACE_X. The
exchanged interface state is a VECTOR on BOTH channels:

    values        = displacement       u = (u_x, u_y)   at the interface nodes
    normal_fluxes = interface traction export, q_out = -(sigma . n_own)

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

Pure glue: the PDE solve is done by the compiled deal.II executable
`elast_iface_dealii` (elast_iface_dealii.cc), which handles BOTH the Dirichlet
and the Neumann side of the interface. This wrapper converts the partner's
InterfaceData into the solver's plain-text input file and its plain-text output
back into exports.json. deal.II has no Python API, so unlike every other
backend there is a BUILD STEP before this can run at all:

    cmake -S <this directory> -B <build> -DDEAL_II_DIR=<deal.II install>
    make -C <build> elast_iface_dealii

and DEALII_EXE below must point at the result.
"""
import json
import subprocess
import sys
from pathlib import Path

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side.
SIDE      = "dirichlet"   # "dirichlet" (import u, export traction) | "neumann"
PARTNER   = "right"       # name of the partner participant in couple(...)
X0, X1    = 0.0, 0.55
Y0, Y1    = 0.0, 0.4
IFACE_X   = 0.55
E_MOD     = 1000.0        # Young's modulus
NU        = 0.3           # Poisson ratio (PLANE STRAIN)
# Prescribed displacement on this subdomain's WHOLE non-interface boundary
# (its outer x-face and both y-faces), as a polynomial in (x, y):
#     u_x = UDX[0] + UDX[1]*x + UDX[2]*y + UDX[3]*y*y
#     u_y = UDY[0] + UDY[1]*x + UDY[2]*y + UDY[3]*y*y
UDX = (0.0, 0.0, 0.0, 0.0)
UDY = (0.0, 0.0, 0.0, 0.0)
NX, NY    = 24, 16
UI_X, UI_Y = 0.0, 0.0     # iteration-1 fallback interface displacement
TI_X, TI_Y = 0.0, 0.0     # iteration-1 fallback interface traction export
DEALII_EXE = "./elast_iface_dealii"   # the compiled solver; see the build note
# ─────────────────────────────────────────────────────────────────────────

DEGREE = 1                # FE_Q degree used inside the FESystem


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return d.get(PARTNER) or None


def sample(imp, key, fallback):
    """Partner samples as sorted (y, vx, vy) triples; constant fallback if
    absent. The deal.II solver interpolates them piecewise-linearly PER
    COMPONENT (clamped outside the range) onto its own interface nodes — the
    same semantics as numpy.interp applied to each component separately."""
    if imp and imp.get("coordinates"):
        ys = [float(c[1]) for c in imp["coordinates"]]
        raw = imp.get(key) or []
        vs = []
        for v in raw:
            if isinstance(v, (list, tuple)):
                vs.append((float(v[0]), float(v[1])))
            else:                       # a scalar partner on a vector interface
                vs = []
                break
        if len(vs) == len(ys) and len(ys) > 0:
            return sorted((y, v[0], v[1]) for y, v in zip(ys, vs))
    fx, fy = float(fallback[0]), float(fallback[1])
    return [(float(Y0), fx, fy), (float(Y1), fx, fy)]


imp = read_imports()
if SIDE == "dirichlet":
    side_flag, triples = 0, sample(imp, "values", (UI_X, UI_Y))
else:
    side_flag, triples = 1, sample(imp, "normal_fluxes", (TI_X, TI_Y))

lines = [f"{side_flag} {E_MOD!r} {NU!r} {X0!r} {X1!r} {Y0!r} {Y1!r} "
         f"{IFACE_X!r} {NX} {NY} {DEGREE}",
         " ".join(repr(float(c)) for c in UDX),
         " ".join(repr(float(c)) for c in UDY),
         str(len(triples))]
lines += [f"{y:.16g} {vx:.16g} {vy:.16g}" for y, vx, vy in triples]
Path("dealii_input.txt").write_text("\n".join(lines) + "\n")

out_txt = Path("dealii_output.txt")
if out_txt.exists():
    out_txt.unlink()
r = subprocess.run([DEALII_EXE, "dealii_input.txt", "dealii_output.txt"],
                   capture_output=True, text=True)
if r.returncode != 0 or not out_txt.is_file():
    sys.stderr.write("deal.II elasticity solver failed (rc=%s)\n%s\n%s\n"
                     % (r.returncode, r.stdout[-2000:], r.stderr[-2000:]))
    sys.exit(1)

coords, disp, trac = [], [], []
for line in out_txt.read_text().splitlines():
    tok = line.split()
    if len(tok) != 5:
        continue
    coords.append([float(IFACE_X), float(tok[0])])
    disp.append([float(tok[1]), float(tok[2])])
    trac.append([float(tok[3]), float(tok[4])])
if not coords:
    sys.stderr.write("deal.II solver produced no interface points\n")
    sys.exit(1)

Path("exports.json").write_text(json.dumps({
    "field_name": "displacement",
    "n_points": len(coords),
    "coordinates": coords,
    "values": disp,
    "normal_fluxes": trac,
}, indent=2))
