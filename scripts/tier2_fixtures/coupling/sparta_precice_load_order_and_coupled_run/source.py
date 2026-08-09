"""SPARTA and preCICE: which load order actually works, and a real coupled run.

TWO SERVED TEXTS CONTRADICTED EACH OTHER
----------------------------------------
`src/backends/sparta/backend.py`'s `precice_participant()` served, under the
docstring "Verified pattern", a snippet that does

    import precice, numpy as np
    from sparta import sparta
    spa = sparta(name='serial')

`src/tools/coupling_knowledge.py` served, for the same backend, "There is
exactly one way to load it, and every other way segfaults", naming that exact
ordering as the one that crashes inside `PMPI_Type_size`.

Both were served to agents; at most one could be true. This fixture settles it
BY EXECUTION — it runs every ordering in a subprocess and reports what each
one does — and then runs a real preCICE coupling with SPARTA in it.

WHAT THE ORDERINGS DO, MEASURED
-------------------------------
`libsparta_serial.so` DEFINES ITS OWN `MPI_*` STUB SYMBOLS and links no real
MPI. `import precice` pulls a real libmpi into the GLOBAL symbol namespace, so
SPARTA's stub calls get interposed by it and SPARTA dies with MPI never
initialised. Loading SPARTA first with `RTLD_GLOBAL` (what the stock
`sparta.py` wrapper does) fails the other way: SPARTA's stubs then interpose
preCICE's, and `import precice` cannot find the real libmpi. Deep binding
resolves it — SPARTA's own symbols win inside SPARTA, preCICE's inside preCICE.

THE COUPLED RUN, AND WHY IT USES THE BINARY
-------------------------------------------
The coupled run drives SPARTA as a SUBPROCESS, one invocation per time window,
which is the shipped participant's route. That is not a way around the load
order — it is the honest consequence of it plus a second limitation the
`backend.py` snippet's own notes state: the SPARTA Python library exposes
command/extract_global/extract_compute/extract_variable and NO per-surf
scatter, so an in-process participant could exchange only a SCALAR, while the
deck can carry a per-element wall temperature through `custom surf ... file`.
The load-order facts above are what an agent needs if it drives SPARTA
in-process anyway, so they are proved here rather than asserted.

The physics is conjugate heat transfer: a rarefied argon stream past a
cylinder, coupled to a lumped thermal shell around the same surface.
SPARTA is necessarily the DIRICHLET side — it has no native flux BC — so it
imports a per-element wall temperature and exports the per-element net energy
flux `etot`. The shell imports that flux and returns
T_wall = T_OUTER + qbar / C_SHELL.

HOW A MONTE-CARLO COUPLING IS GRADED
------------------------------------
DSMC output carries sampling noise that does NOT shrink with coupling
iterations, so an implicit scheme's convergence measure cannot be driven below
it and the scheme here is EXPLICIT — `couple_precice` reports that as
unmeasured rather than as converged, which is correct and is why this fixture
does not check `converged` at all.

What it checks instead is the fixed point, against SPARTA run STANDALONE at
uniform wall temperatures BRACKETING the coupled answer. That reference never
touches preCICE, the orchestrator or the shell participant, so agreement
between it and the coupled answer is evidence about the coupling rather than a
restatement of it. The bracket is deliberate: a wide fit would put the DSMC
flux curve's curvature into the reference.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import statistics
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                     # noqa: E402


# ── the conjugate-heat-transfer problem ────────────────────────────────────

SURF, SPECIES, VSS = "circle.surf", "ar.species", "ar.vss"
T_OUTER = 300.0            # the shell's outer face, K
C_SHELL = 11.0             # shell conductance, W/(m^2 K) — chosen so the wall
                           # temperature is set by the GAS response and not by
                           # T_OUTER: a large C pins the wall at T_OUTER and
                           # the coupling stops being a test of anything.
T_INIT = 950.0             # deliberately far from the answer, so the run has
                           # to travel there
N_WINDOWS = 10
SETTLE = 2                 # windows discarded as transient before averaging
NRUN, NAVE = 8000, 4000    # DSMC steps / sampling window
PROBE_T = (600.0, 750.0)   # the standalone reference, BRACKETING the answer
PROBE_REPEATS = 3

# Tolerance, and where it comes from. The measured per-window scatter of the
# mean interface flux at a FIXED wall temperature is ~0.5% at these sampling
# settings, i.e. ~2 K once divided by C_SHELL, and the fixture averages 8
# windows. 15 K is several times that and still ~40x below every pathology it
# must catch: a coupling that never exchanges leaves the wall at T_INIT (950 K)
# or at T_OUTER (300 K), and a sign error sends it out of the bracket entirely.
T_ATOL = 15.0
# The two sides must report the same interface flux with opposite signs.
BALANCE_RTOL = 1e-9

# The sign with which the shell applies the flux it imported. Named here
# because it is this fixture's mutation point: this is a NEUMANN participant
# and the served rule is that the partner's number is applied UNCHANGED.
APPLY_SIGN = 1.0


# ── PART A: the load orderings, run rather than described ──────────────────

_ORDERINGS = {
    # name: (source, what the served knowledge says must happen)
    "precice_first_stock_wrapper": ('''
import precice
print("step: precice imported", flush=True)
from sparta import sparta
print("step: sparta module imported", flush=True)
spa = sparta(name="serial")
print("step: sparta instance created", flush=True)
spa.command("dimension 2")
print("ORDERING_OK", flush=True)
''', "segfault"),
    "mpi4py_then_precice_then_sparta": ('''
from mpi4py import MPI
print("step: mpi4py imported", flush=True)
import precice
print("step: precice imported", flush=True)
from sparta import sparta
spa = sparta(name="serial")
print("ORDERING_OK", flush=True)
''', "segfault"),
    "sparta_first_rtld_global": ('''
from sparta import sparta
spa = sparta(name="serial")
print("step: sparta instance created", flush=True)
import precice
print("ORDERING_OK", flush=True)
''', "import_error"),
    "deepbind_local_then_precice": ('''
import ctypes, os
mode = os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_DEEPBIND
lib = ctypes.CDLL(LIBSPARTA, mode=mode)
print("step: libsparta loaded DEEPBIND|LOCAL", flush=True)
import precice
print("step: precice imported", flush=True)
spa = ctypes.c_void_p()
lib.sparta_open_no_mpi(0, None, ctypes.byref(spa))
lib.sparta_command(spa, b"dimension 2")
print("ORDERING_OK", flush=True)
''', "works"),
}


def libsparta(binary: str) -> Path:
    """The shared library next to the SPARTA binary the registry resolved."""
    src = Path(binary).parent
    for name in ("libsparta_serial.so", "libsparta.so"):
        p = src / name
        if p.is_file():
            return p
    raise L.Absent(f"no libsparta*.so beside {binary}; the in-process load "
                   f"orderings cannot be tested without one")


def run_ordering(name: str, code: str, python: str, binary: str,
                 lib: Path, lib_dir: str, work: Path) -> str:
    """Run one ordering in its OWN process and report what it did:
    'works' | 'segfault' | 'import_error' | 'other_failure'."""
    src = work / f"order_{name}.py"
    src.write_text(f"LIBSPARTA = {str(lib)!r}\n" + textwrap.dedent(code))
    env = {**os.environ,
           "PYTHONPATH": str(Path(binary).parent.parent / "python"),
           "LD_LIBRARY_PATH": f"{lib_dir}:{Path(binary).parent}:"
                              + os.environ.get("LD_LIBRARY_PATH", "")}
    try:
        r = subprocess.run([python, str(src)], capture_output=True, text=True,
                           timeout=600, cwd=str(work), env=env)
    except subprocess.SubprocessError as e:
        return f"other_failure({type(e).__name__})"
    out = (r.stdout or "") + (r.stderr or "")
    if "ORDERING_OK" in out and r.returncode == 0:
        return "works"
    if r.returncode < 0 or "Segmentation fault" in out or r.returncode == 139:
        return "segfault"
    if "ImportError" in out or "cannot open shared object file" in out:
        return "import_error"
    return f"other_failure(rc={r.returncode})"


def load_order(python: str, binary: str, lib_dir: str) -> None:
    work = L.workroot("sparta_order")
    lib = libsparta(binary)
    print(f"libsparta={lib}")
    got = {}
    for name, (code, want) in _ORDERINGS.items():
        got[name] = run_ordering(name, code, python, binary, lib, lib_dir,
                                 work)
        print(f"ordering[{name}]={got[name]} (served text says: {want})")
        verdict(got[name] == want, f"ordering_{name}_matches_served_text",
                f"ran it: {got[name]!r}, served text says {want!r}")
    n_work = sum(1 for v in got.values() if v == "works")
    print(f"orderings_that_work={n_work}_of_{len(got)}")
    verdict(n_work == 1, "exactly_one_ordering_works",
            f"{n_work} of {len(got)} orderings ran to completion; the served "
            f"text says there is exactly one")
    verdict(got.get("deepbind_local_then_precice") == "works",
            "rtld_deepbind_is_the_one_that_works",
            "the ordering the coupling_knowledge text prescribes is not the "
            "one that runs")
    verdict(got.get("precice_first_stock_wrapper") == "segfault",
            "precice_first_stock_wrapper_segfaults",
            "the ordering backend.py used to call a 'Verified pattern' did "
            "not segfault after all")


# ── PART B: the participants ───────────────────────────────────────────────

_GAS = '''\
"""GAS side, SPARTA (DSMC), DIRICHLET role: reads Wall-Temperature, writes
Heat-Flux, driven from inside a preCICE coupling loop.

SPARTA HAS NO NATIVE FLUX BC — none of the nine surf_collide styles takes a
prescribed heat flux, and the four thermal ones all take a temperature — so
this is necessarily the Dirichlet-side participant.

SPARTA runs as a SUBPROCESS. Its Python library defines its own MPI stubs and
cannot share an address space with preCICE unless it is deep-bound, and even
deep-bound it exposes no per-surf scatter, so an in-process participant could
exchange only a scalar. A subprocess exchanges the full per-element field
through `custom surf ... file`.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import precice

NAME, MESH = "Gas", "Gas-Mesh"
WRITE_DATA, READ_DATA = "Heat-Flux", "Wall-Temperature"
SPARTA = "spa_serial"
SURF_FILE = "circle.surf"
SPECIES = "ar.species"
VSS = "ar.vss"
T_INIT = 950.0
NRUN = 8000
NAVE = 4000
SEED = 12345
OUT = "gas_interface.json"

DECK, TSURF_IN, FLUX_OUT = "cpl.sparta", "tsurf.in", "flux.out"


def read_surf_elements():
    """Parse SPARTA's surf file -> (n_elem, centroids[n,2])."""
    txt = Path(SURF_FILE).read_text().splitlines()
    pts, lines, sec = {}, [], None
    for ln in txt:
        s = ln.split("#")[0].strip()
        if not s:
            continue
        low = s.lower()
        if low == "points":
            sec = "p"; continue
        if low == "lines":
            sec = "l"; continue
        if low.endswith("points") or low.endswith("lines"):
            continue
        f = s.split()
        if sec == "p" and len(f) >= 3:
            pts[int(f[0])] = (float(f[1]), float(f[2]))
        elif sec == "l" and len(f) >= 3:
            lines.append((int(f[0]), int(f[-2]), int(f[-1])))
    lines.sort()
    cen = np.array([[0.5 * (pts[a][0] + pts[b][0]),
                     0.5 * (pts[a][1] + pts[b][1])] for (_, a, b) in lines])
    return len(lines), cen


def write_tsurf(t):
    """SPARTA `custom surf ... file` format: comment, blank, 'N M', 'id v'."""
    rows = ["# per-surf wall temperature from the preCICE participant", "",
            f"{len(t)} 1"]
    rows += [f"{i} {float(v):.10g}" for i, v in enumerate(t, 1)]
    Path(TSURF_IN).write_text("\\n".join(rows) + "\\n")


def write_deck(seed):
    Path(DECK).write_text(f"""\\
seed                {seed}
dimension           2
global              nrho 4.247e19 fnum 7e14 gridcut 0.01 comm/style all comm/sort yes
timestep            3.5e-7
boundary            o ro p
create_box          -0.2 0.65 0.0 0.4 -0.5 0.5
create_grid         30 15 1 block * * *
species             {SPECIES} Ar
mixture             all vstream 2634.1 0 0 temp 200.0
collide             vss all {VSS}
collide_modify      vremax 1000 yes
read_surf           {SURF_FILE} group 1

custom              surf create tsurf float 0 file {TSURF_IN} 1 tsurf
surf_collide        1 diffuse s_tsurf 1.0
surf_modify         1 collide 1

fix                 in emit/face all xlo twopass
create_particles    all n 0 twopass

compute             1 surf all all etot
fix                 1 ave/surf all 1 {NAVE} {NAVE} c_1[1] ave one
dump                1 surf all {NRUN} {FLUX_OUT} id v1x v1y v2x v2y f_1 s_tsurf
dump_modify         1 pad 0

stats               {NRUN}
stats_style         step cpu np nscoll
run                 {NRUN}
""")


def parse_dump(path, n):
    """Read the LAST snapshot of a SPARTA ASCII surf dump."""
    txt = Path(path).read_text().splitlines()
    starts = [i for i, l in enumerate(txt) if l.startswith("ITEM: SURFS")]
    if not starts:
        raise SystemExit(f"no 'ITEM: SURFS' block in {path}")
    rows = []
    for l in txt[starts[-1] + 1:]:
        if l.startswith("ITEM:"):
            break
        f = l.split()
        if len(f) >= 7:
            rows.append([float(x) for x in f[:7]])
    a = np.asarray(rows, float)
    a = a[np.argsort(a[:, 0])]
    if len(a) != n:
        raise SystemExit(f"dump has {len(a)} rows, expected {n}")
    return a


n_elem, cen = read_surf_elements()
_it = {"n": 0}


def advance(t_wall):
    """One DSMC run at the imported wall temperature -> per-element flux."""
    _it["n"] += 1
    write_tsurf(t_wall)
    # A NEW seed each window: with a fixed seed the run is bit-reproducible and
    # a fixed-point iteration can look converged when only the RNG is repeating.
    write_deck(SEED + 1000 * _it["n"])
    Path(FLUX_OUT).unlink(missing_ok=True)
    r = subprocess.run([SPARTA, "-in", DECK], capture_output=True, text=True,
                       timeout=3600)
    if r.returncode != 0 or not Path(FLUX_OUT).is_file():
        sys.stderr.write(f"SPARTA failed rc={r.returncode}\\n"
                         f"{(r.stdout or '')[-2000:]}\\n"
                         f"{(r.stderr or '')[-800:]}\\n")
        raise SystemExit(1)
    a = parse_dump(FLUX_OUT, n_elem)
    return a[:, 5], a[:, 6]        # etot (out of the gas), T actually used


p = precice.Participant(NAME, "precice-config.xml", 0, 1)
vid = p.set_mesh_vertices(MESH, cen)
if p.requires_initial_data():
    p.write_data(MESH, WRITE_DATA, vid, np.zeros(len(vid)))
p.initialize()

q_last = np.zeros(len(vid))
t_last = np.full(len(vid), T_INIT)
n_it = 0
while p.is_coupling_ongoing():
    if p.requires_writing_checkpoint():
        pass                       # each window is a fresh DSMC sample
    dt = p.get_max_time_step_size()
    t_in = np.asarray(p.read_data(MESH, READ_DATA, vid, dt), float).ravel()
    if n_it == 0 and not np.any(t_in):
        t_in = np.full(len(vid), T_INIT)
    q_last, t_last = advance(t_in)
    p.write_data(MESH, WRITE_DATA, vid, q_last)
    p.advance(dt)
    n_it += 1
    print(f"gas window {n_it}: T={t_last.min():.2f} q_mean={q_last.mean():.6e}",
          flush=True)
    if p.requires_reading_checkpoint():
        pass
p.finalize()

with open(OUT, "w") as f:
    json.dump({"participant": NAME, "sub_iterations": n_it,
               "coordinates": [[float(x), float(y)] for x, y in cen],
               "values": [float(v) for v in t_last],
               "normal_fluxes": [float(v) for v in q_last]}, f, indent=2)
'''


_SHELL = '''\
"""SOLID side, lumped thermal shell, NEUMANN role: reads Heat-Flux, writes
Wall-Temperature.

Steady lumped shell, isothermal in-plane: a thin shell whose tangential
conductivity is high compared with its through-thickness one holds ONE wall
temperature, set by the TOTAL heat it must pass to its outer face,
    T_wall = T_OUTER + qbar / C_SHELL.
That is what makes the standalone reference exact rather than approximate: the
reference probes SPARTA at a UNIFORM wall temperature, and here the wall really
is uniform. A shell that reacted element by element would sit at a different
fixed point than any uniform-wall probe, by an amount nothing could bound.
"""
import json
from pathlib import Path

import numpy as np
import precice

NAME, MESH = "Solid", "Solid-Mesh"
WRITE_DATA, READ_DATA = "Wall-Temperature", "Heat-Flux"
SURF_FILE = "circle.surf"
T_OUTER = 300.0
C_SHELL = 11.0
T_INIT = 950.0
APPLY_SIGN = 1.0
OUT = "solid_interface.json"


def centroids():
    txt = Path(SURF_FILE).read_text().splitlines()
    pts, lines, sec = {}, [], None
    for ln in txt:
        s = ln.split("#")[0].strip()
        if not s:
            continue
        low = s.lower()
        if low == "points":
            sec = "p"; continue
        if low == "lines":
            sec = "l"; continue
        if low.endswith("points") or low.endswith("lines"):
            continue
        f = s.split()
        if sec == "p" and len(f) >= 3:
            pts[int(f[0])] = (float(f[1]), float(f[2]))
        elif sec == "l" and len(f) >= 3:
            lines.append((int(f[0]), int(f[-2]), int(f[-1])))
    lines.sort()
    return np.array([[0.5 * (pts[a][0] + pts[b][0]),
                      0.5 * (pts[a][1] + pts[b][1])] for (_, a, b) in lines])


cen = centroids()
p = precice.Participant(NAME, "precice-config.xml", 0, 1)
vid = p.set_mesh_vertices(MESH, cen)
if p.requires_initial_data():
    p.write_data(MESH, WRITE_DATA, vid, np.full(len(vid), T_INIT))
p.initialize()

t_last = np.full(len(vid), T_INIT)
q_last = np.zeros(len(vid))
n_it = 0
while p.is_coupling_ongoing():
    if p.requires_writing_checkpoint():
        pass                                 # steady lumped shell: no state
    dt = p.get_max_time_step_size()
    q_in = APPLY_SIGN * np.asarray(
        p.read_data(MESH, READ_DATA, vid, dt), float).ravel()
    qbar = float(np.mean(q_in))
    t_last = np.full(len(vid), T_OUTER + qbar / C_SHELL)
    # Exported with respect to THIS side's outward normal, which points the
    # other way, so the two sides carry opposite signs.
    q_last = -q_in
    p.write_data(MESH, WRITE_DATA, vid, t_last)
    p.advance(dt)
    n_it += 1
    print(f"solid window {n_it}: qbar={qbar:.6e} T_wall={t_last[0]:.4f}",
          flush=True)
    if p.requires_reading_checkpoint():
        pass
p.finalize()

with open(OUT, "w") as f:
    json.dump({"participant": NAME, "sub_iterations": n_it,
               "coordinates": [[float(x), float(y)] for x, y in cen],
               "values": [float(v) for v in t_last],
               "normal_fluxes": [float(v) for v in q_last]}, f, indent=2)
'''


# ── the real `couple_precice` tool, reached the way the server reaches it ───

_TOOL_MANAGER = None


def _tool_manager():
    global _TOOL_MANAGER
    if _TOOL_MANAGER is None:
        from mcp.server.fastmcp import FastMCP
        from core.registry import load_all_backends
        from tools.consolidated import register_consolidated_tools
        m = FastMCP("tier2-precice-sparta")
        register_consolidated_tools(m)
        load_all_backends()
        _TOOL_MANAGER = m._tool_manager
    return _TOOL_MANAGER


def call_tool(name: str, args: dict) -> str:
    res = asyncio.run(_tool_manager().call_tool(name, args))
    if isinstance(res, tuple) and len(res) >= 1:
        res = res[0]
    if isinstance(res, list):
        return "\n".join(getattr(b, "text", str(b)) for b in res)
    return getattr(res, "text", str(res))


def verdict(ok: bool, label: str, detail: str = "") -> bool:
    if L.check(bool(ok), f"{label}_violated", detail):
        print(f"{label}=yes")
        return True
    return False


# ── locating SPARTA and its data files ─────────────────────────────────────

def sparta_bits() -> tuple[str, dict]:
    from backends.sparta.backend import _find_sparta_binary, _sparta_data_dirs
    binary = _find_sparta_binary()
    if not binary:
        raise L.Absent("no SPARTA binary resolved from the registry")
    found: dict = {}
    for root in _sparta_data_dirs(binary):
        for name in (SURF, SPECIES, VSS):
            if name not in found:
                hits = sorted(Path(root).rglob(name))
                if hits:
                    found[name] = hits[0]
    missing = [n for n in (SURF, SPECIES, VSS) if n not in found]
    if missing:
        raise L.Absent(f"SPARTA data files not in the distribution: {missing}")
    return binary, found


def gas_source(binary: str) -> str:
    text = _GAS.replace('SPARTA = "spa_serial"', f'SPARTA = "{binary}"')
    if f'SPARTA = "{binary}"' not in text:
        raise AssertionError("the SPARTA binary substitution matched nothing")
    return text


# ── PART C: the coupled run ────────────────────────────────────────────────

def coupled_run(binary: str, data: dict, python: str) -> tuple[list, dict]:
    work = L.workroot("sparta_precice")
    print(f"sparta_precice_work_dir={work}")
    for src in data.values():
        shutil.copy(src, work / src.name)
    (work / "participant_gas.py").write_text(gas_source(binary))
    shell = _SHELL.replace("APPLY_SIGN = 1.0", f"APPLY_SIGN = {APPLY_SIGN}")
    if f"APPLY_SIGN = {APPLY_SIGN}" not in shell:
        raise AssertionError("the apply-sign substitution matched nothing")
    (work / "participant_solid.py").write_text(shell)

    participants = [
        {"name": "Gas", "mesh": "Gas-Mesh", "writes": ["Heat-Flux"],
         "reads": ["Wall-Temperature"],
         "command": [python, "participant_gas.py"]},
        {"name": "Solid", "mesh": "Solid-Mesh", "writes": ["Wall-Temperature"],
         "reads": ["Heat-Flux"],
         "command": [python, "participant_solid.py"]},
    ]
    fields = [{"name": "Wall-Temperature", "type": "scalar"},
              {"name": "Heat-Flux", "type": "scalar"}]
    exchanges = [{"data": "Wall-Temperature", "from": "Solid", "to": "Gas"},
                 {"data": "Heat-Flux", "from": "Gas", "to": "Solid"}]

    raw = call_tool("couple_precice", {
        "participants": json.dumps(participants), "data": json.dumps(fields),
        "exchanges": json.dumps(exchanges), "work_dir": str(work),
        # EXPLICIT on purpose: DSMC noise does not shrink with sub-iteration,
        # so an implicit measure could never be met and "converged" would be
        # meaningless. `couple_precice` reports an explicit scheme as
        # unmeasured, which is the honest verdict.
        "scheme": "serial-explicit", "dimensions": 2,
        "max_time": float(N_WINDOWS), "time_window": 1.0, "timeout": 3600,
        "critic_approved": True})
    try:
        res = json.loads(raw)
    except json.JSONDecodeError:
        raise AssertionError(f"couple_precice returned non-JSON: {raw[:400]}")

    print(f"sparta_returncodes={json.dumps(res.get('returncodes') or {}, sort_keys=True)}")
    rcs = res.get("returncodes") or {}
    if not verdict(bool(rcs) and all(int(v) == 0 for v in rcs.values()),
                   "sparta_all_participant_returncodes_zero",
                   str(res.get("error"))[:300]):
        for name, tail in (res.get("logs") or {}).items():
            print(f"--- participant log tail [{name}] ---\n{str(tail)[-1500:]}")
    verdict(bool(res.get("exchanged")), "sparta_participants_exchanged_data",
            "preCICE's own per-participant record shows no exchange")
    # An explicit scheme measures no convergence at all, and the tool says so
    # rather than defaulting to True. Asserting that is asserting the tool's
    # honesty about what it did not measure.
    verdict(res.get("converged") is None,
            "explicit_scheme_reports_convergence_unmeasured",
            f"couple_precice returned converged={res.get('converged')!r} for a "
            f"scheme that measures no convergence")

    ex = {}
    for side, fn in (("gas", "gas_interface.json"),
                     ("solid", "solid_interface.json")):
        f = work / fn
        if not f.is_file():
            L.check(False, f"sparta_{side}_wrote_no_interface_file", str(f))
            for name, tail in (res.get("logs") or {}).items():
                print(f"--- log [{name}] ---\n{str(tail)[-1500:]}")
            raise L.Absent("the coupled run produced no interface files")
        ex[side] = json.loads(f.read_text())

    trace = []
    for line in (work / "Solid.out").read_text().splitlines():
        if "T_wall=" in line:
            trace.append(float(line.rsplit("T_wall=", 1)[1].split()[0]))
    print(f"sparta_windows={len(trace)}")
    print(f"sparta_T_wall_trace={[round(v, 2) for v in trace]}")
    verdict(len(trace) == N_WINDOWS, "sparta_all_time_windows_ran",
            f"{len(trace)} of {N_WINDOWS} windows produced a wall temperature")
    verdict(abs(trace[0] - T_INIT) > T_ATOL, "sparta_coupling_moved_the_wall",
            f"the wall stayed at its initial {T_INIT} K, so nothing the gas "
            f"computed ever reached the solid")
    return trace, ex


def flux_balance(ex: dict) -> None:
    qg = [float(v) for v in ex["gas"]["normal_fluxes"]]
    qs = [float(v) for v in ex["solid"]["normal_fluxes"]]
    mg, ms = sum(qg) / len(qg), sum(qs) / len(qs)
    rel = abs(mg + ms) / max(abs(mg), abs(ms), 1e-30)
    print(f"sparta_q_gas={mg:.6e} sparta_q_solid={ms:.6e}")
    print(f"sparta_flux_balance_rel={rel:.3e}")
    verdict(rel <= BALANCE_RTOL, "sparta_interface_flux_balanced",
            f"the two sides report {mg:.6e} and {ms:.6e}; what leaves the gas "
            f"must enter the solid, with the opposite sign")


# ── PART D: the independent reference ──────────────────────────────────────

def standalone_flux(binary: str, data: dict, root: Path, t_wall: float,
                    seed: int) -> float:
    """SPARTA ALONE at a UNIFORM wall temperature -> mean interface flux.

    Runs the participant's OWN deck writer and dump parser with preCICE removed,
    so the reference and the coupled run differ in exactly one thing: whether
    preCICE is in the loop."""
    wd = root / f"probe_{t_wall:.0f}_{seed}"
    wd.mkdir(parents=True, exist_ok=True)
    for src in data.values():
        shutil.copy(src, wd / src.name)
    code = gas_source(binary).split("p = precice.Participant")[0] \
        .replace("import precice", "precice = None") \
        .replace("SEED = 12345", f"SEED = {seed}")
    ns: dict = {}
    cwd = os.getcwd()
    os.chdir(wd)
    try:
        exec(compile(code, "gas_standalone", "exec"), ns)
        import numpy as np
        q, _ = ns["advance"](np.full(ns["n_elem"], t_wall))
        return float(q.mean())
    finally:
        os.chdir(cwd)


def reference_fixed_point(binary: str, data: dict) -> float:
    """Fit q(T) from standalone probes BRACKETING the coupled answer, then
    solve the shell's own balance T = T_OUTER + q(T)/C for T."""
    root = L.workroot("sparta_reference")
    qs = []
    for i, t in enumerate(PROBE_T):
        draws = [standalone_flux(binary, data, root, t, 777 + 131 * i + 17 * r)
                 for r in range(PROBE_REPEATS)]
        qs.append(statistics.fmean(draws))
        sd = statistics.stdev(draws) if len(draws) > 1 else 0.0
        print(f"reference_probe_T={t:.0f} q_mean={qs[-1]:.6e} "
              f"draw_sd={sd:.3e}")
    (t0, t1), (q0, q1) = PROBE_T[:2], qs[:2]
    a = (q1 - q0) / (t1 - t0)
    b = q0 - a * t0
    print(f"reference_dq_dT={a:.6e}")
    verdict(a < 0.0, "hotter_wall_takes_less_heat",
            f"dq/dT = {a:.6e} >= 0: a wall that gets hotter must absorb LESS "
            f"from the gas, or the gas-side physics is not what it claims")
    t_fp = (T_OUTER + b / C_SHELL) / (1.0 - a / C_SHELL)
    print(f"reference_fixed_point_T={t_fp:.4f}")
    verdict(min(PROBE_T) <= t_fp <= max(PROBE_T),
            "reference_fixed_point_inside_the_probe_bracket",
            f"{t_fp:.4f} K is outside [{min(PROBE_T)}, {max(PROBE_T)}], so the "
            f"linear fit is an extrapolation and its curvature error is not "
            f"bounded by anything here")
    return t_fp


# ── the fixture body ───────────────────────────────────────────────────────

def body() -> None:
    L.require_available("sparta")
    from core.precice_config import PRECICE_LIB_DIR, check_precice_available
    if not Path(PRECICE_LIB_DIR).is_dir():
        raise L.Absent(f"no preCICE lib directory at {PRECICE_LIB_DIR}")
    ok, msg = check_precice_available()
    if not ok:
        raise L.Absent(f"preCICE is not usable from this install ({msg[:200]})")
    print(f"precice_lib_dir={PRECICE_LIB_DIR}")

    binary, data = sparta_bits()
    print(f"sparta_binary={binary}")
    # SPARTA is a WRAPPER backend: the participant is a plain Python with numpy
    # and preCICE, and the solver binary goes into a constant inside the script.
    # That interpreter is the one running this fixture.
    python = sys.executable
    print(f"sparta_participant_interpreter={python}")

    load_order(python, binary, PRECICE_LIB_DIR)

    trace, ex = coupled_run(binary, data, python)
    flux_balance(ex)

    settled = trace[SETTLE:]
    t_coupled = statistics.fmean(settled)
    spread = max(settled) - min(settled)
    print(f"sparta_settled_windows={len(settled)}")
    print(f"sparta_coupled_T_wall={t_coupled:.4f}")
    print(f"sparta_settled_spread={spread:.4f}")

    t_ref = reference_fixed_point(binary, data)
    err = abs(t_coupled - t_ref)
    print(f"sparta_T_wall_vs_reference_err={err:.4f}")
    verdict(err <= T_ATOL, "sparta_coupled_fixed_point_matches_reference",
            f"|{t_coupled:.4f} - {t_ref:.4f}| = {err:.4f} K > {T_ATOL} K — the "
            f"coupled answer and SPARTA run standalone at the same wall "
            f"temperature disagree by more than the sampling noise allows")


L.main(body)
