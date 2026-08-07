"""deal.II participant for the OASiS `couple` driver.

Steady heat conduction  -div(k grad T) = f  on one rectangular subdomain.
CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

Pure glue: the PDE solve is done by the compiled deal.II executable
`heat_iface_dealii` (heat_iface_dealii.cc), which handles BOTH the Dirichlet
and the Neumann side of the interface.  This wrapper converts the partner's
InterfaceData into the solver's plain-text input file and its plain-text
output back into exports.json.
"""
import json
import subprocess
import sys
from pathlib import Path

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side; the payload that served
#    this script gives the exact block for the RIGHT / Neumann side.
SIDE      = "dirichlet"   # "dirichlet" | "neumann"
PARTNER   = "right"       # name of the partner participant in couple(...)
X0, X1    = 0.0, 0.6
Y0, Y1    = 0.0, 0.4
IFACE_X   = 0.6
K         = 0.8
F_SRC     = 0.0           # volumetric source
T_OUTER   = 320.0
NX, NY    = 24, 16
T_INIT    = 310.0
Q_INIT    = 0.0           # iteration-1 fallback interface flux
DEALII_EXE = "./heat_iface_dealii"   # the compiled solver; see the build note# ─────────────────────────────────────────────────────────────────────────

DEGREE = 1                # FE_Q degree used by the deal.II solver


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
    """Partner samples as sorted (y, value) pairs; constant fallback if absent.

    The deal.II solver interpolates them piecewise-linearly (clamped outside
    the range) onto its own interface nodes — same semantics as numpy.interp.
    """
    if imp and imp.get("coordinates"):
        ys = [float(c[1]) for c in imp["coordinates"]]
        vs = [float(v) for v in (imp.get(key) or [])]
        if len(vs) == len(ys) and len(ys) > 0:
            return sorted(zip(ys, vs))
    return [(float(Y0), float(fallback)), (float(Y1), float(fallback))]


imp = read_imports()
if SIDE == "dirichlet":
    side_flag, pairs = 0, sample(imp, "values", T_INIT)
else:
    side_flag, pairs = 1, sample(imp, "normal_fluxes", Q_INIT)

header = (f"{side_flag} {K!r} {X0!r} {X1!r} {Y0!r} {Y1!r} {IFACE_X!r} "
          f"{T_OUTER!r} {F_SRC!r} {NX} {NY} {DEGREE}")
lines = [header, str(len(pairs))]
lines += [f"{y:.16g} {v:.16g}" for y, v in pairs]
Path("dealii_input.txt").write_text("\n".join(lines) + "\n")

out_txt = Path("dealii_output.txt")
if out_txt.exists():
    out_txt.unlink()
r = subprocess.run([DEALII_EXE, "dealii_input.txt", "dealii_output.txt"],
                   capture_output=True, text=True)
if r.returncode != 0 or not out_txt.is_file():
    sys.stderr.write("deal.II solver failed (rc=%s)\n%s\n%s\n"
                     % (r.returncode, r.stdout[-2000:], r.stderr[-2000:]))
    sys.exit(1)

coords, temps, fluxes = [], [], []
for line in out_txt.read_text().splitlines():
    tok = line.split()
    if len(tok) != 3:
        continue
    coords.append([float(IFACE_X), float(tok[0])])
    temps.append(float(tok[1]))
    fluxes.append(float(tok[2]))
if not coords:
    sys.stderr.write("deal.II solver produced no interface points\n")
    sys.exit(1)

Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": len(coords),
    "coordinates": coords,
    "values": temps,
    "normal_fluxes": fluxes,
}, indent=2))
