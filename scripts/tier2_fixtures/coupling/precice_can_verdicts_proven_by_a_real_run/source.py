"""The preCICE "CAN" side, proven by a real two-participant coupled run.

WHICH OF THE TWO FIXTURES THIS IS, AND WHY
------------------------------------------
This is the STRONG one: a REAL preCICE coupling, driven through the REGISTERED
`couple_precice` tool, between two backends that live in two DIFFERENT
interpreters — scikit-fem in the OASiS venv and FEniCSx in its own conda env —
with the exchanged interface fields checked against a closed form. The weaker
fallback (prove only the import gate) was not needed, so the import gate is
folded in as this fixture's PRECONDITION rather than as its whole content: the
gate is what decides whether a backend can use `couple_precice` at all, and
`_PRECICE_CORE` states it in a form nothing else tests.

THE CLAIMS UNDER TEST
---------------------
From `knowledge(topic='precice', solver=...)`:

  scikit-fem: "CAN — proven by a real coupled run ... `import precice` works in
              the interpreter that carries scikit-fem, and a scikit-fem
              participant was coupled to a second code end to end."
  FEniCSx:    "CAN — proven by a real coupled run ... `import precice` works in
              the same interpreter as `dolfinx` ... You do NOT need
              `fenicsprecice`."
  NGSolve:    "CAN — ... `import precice` works in the interpreter that carries
              NGSolve." (the GATE only; NGSolve shares the venv with
              scikit-fem, so it is named in the same probe rather than left to
              scikit-fem's result standing in for it. The NGSolve payload's own
              two silent-wrong traps are not what this fixture is about.)
  DUNE-fem:   "put the site-packages of the interpreter that HAS `pyprecice` on
              `PYTHONPATH` and the preCICE lib on `LD_LIBRARY_PATH`, and the
              DUNE interpreter imports both."

From `_PRECICE_CORE`:

  "`libprecice` must be loadable, and `LD_LIBRARY_PATH` is NOT optional:
   without it `import precice` fails with `libprecice.so.3: cannot open shared
   object file`."
  "every participant must be able to `import precice` (or link libprecice) IN
   ITS OWN INTERPRETER. That is the gate."
  "`converged` in the returned JSON is EXIT CODES plus preCICE's own per-window
   verdict ... It is still not a check on the VALUES."

WHY CHECKING `converged` WOULD NOT BE A TEST
--------------------------------------------
The knowledge says outright that `converged` never looks at the exchanged
fields, and the fixture's own mutation demonstrates the consequence: flip the
sign with which the Neumann participant applies the incoming flux and the run
still exits 0 on both sides, preCICE still logs "All converged", and the
coupling still reaches a fixed point — a DIFFERENT one, ~22 K away from the
right answer, with both sides exporting the SAME sign of flux so nothing is
conserved across the interface. Every signal `couple_precice` returns stays
green. So this fixture asserts the VALUES: the interface temperature from both
sides, the interface flux density from both sides with the two outward normals'
opposite signs, the flux balance, and the absence of y-variation the exact
solution forbids — each against a closed form computed here, not against a
number recorded from a previous run.

THE PLACEHOLDER PROBLEM AND ITS CLOSED FORM
-------------------------------------------
Steady conduction with no source on a rectangle split at x = XI. No
y-variation, so the exact solution is piecewise linear in x, the interface
temperature is the conductance-weighted mean of the two outer Dirichlet values
and the interface flux follows from either side. "Conductance" is k divided by
the distance from the interface to that subdomain's own outer boundary. The two
subdomains are meshed INDEPENDENTLY and do not match at the interface (16
points against 12 here), which is the case preCICE's mapping exists for.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

# The shared library lives in a sibling `_lib/` DIRECTORY, not as a bare file,
# because scripts/mutate_tier2_fixtures.py stages a fixture into a scratch tree
# and copies only sibling directories whose name starts with `_`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                     # noqa: E402


# ── the split conduction problem, and the reference this fixture checks against
#
# Recomputed here from the geometry and the material data rather than imported,
# so the reference is independent of anything the coupling machinery does. A
# fixture that compared a run against a number a previous run printed would be
# checking reproducibility, not correctness.

XL, XI, XR = 0.0, 0.6, 1.1        # outer left boundary, interface, outer right
Y0, Y1 = 0.0, 0.4
KL, KR = 0.8, 1.5                 # conductivities,      W/(m K)
TL, TR = 320.0, 300.0             # outer Dirichlet values, K
MESH_L = (20, 15)                 # the two subdomains are meshed independently
MESH_R = (14, 11)                 # and DO NOT match at the interface

CL = KL / (XI - XL)               # left interface conductance,  W/(m^2 K)
CR = KR / (XR - XI)               # right interface conductance
T_EXACT = (CL * TL + CR * TR) / (CL + CR)
Q_EXACT = CL * (TL - T_EXACT)     # +x-ward flux density at the interface, W/m^2

# TOLERANCES, and why these and not the numbers the run happens to print.
#
# `couple_precice` hard-codes preCICE's relative convergence measure at 1e-6 and
# does not pass a tolerance through, so the fixed point is reached to about
# |T| * 1e-6 / (1 - |slope|) ~ 5e-4 K, where the slope of the Dirichlet-Neumann
# map is -CL/CR. These thresholds sit an order of magnitude above that bound and
# three orders BELOW every pathology the fixture exists to catch: the sign error
# in the mutation is 22 K and 30 W/m^2, a unit mismatch is a factor of 1e3.
T_ATOL = 1e-2                     # K
Q_ATOL = 1e-2                     # W/m^2
UNIFORM_ATOL = 1e-6               # the exact interface field has NO y-variation
BALANCE_RTOL = 1e-4

SCHEME = "serial-implicit"
TIME_WINDOW = 1.0
MAX_TIME = 2.0                    # -> 2 time windows
N_WINDOWS = int(round(MAX_TIME / TIME_WINDOW))


# ── the participant scripts ────────────────────────────────────────────────
#
# Both follow the loop `_PRECICE_CORE` prints, checkpoint calls included: the
# scheme here is implicit, and a participant that skips them does not hang, it
# aborts with "The required actions write-iteration-checkpoint are not
# fulfilled". The steady solve has no state, so the checkpoint bodies are empty
# — the CALLS are what preCICE requires.
#
# Each participant writes its final interface data to its own file in work_dir.
# That is not decoration: the orchestrator returns exit codes and log tails and
# never sees the exchanged fields, so a file is the only way the values reach a
# check at all.

_LEFT = '''\
"""LEFT subdomain, scikit-fem, DIRICHLET role: reads Temperature, writes flux."""
import json
import numpy as np
import precice
from skfem import (Basis, BilinearForm, ElementTriP1, LinearForm, MeshTri,
                   condense, solve)
from skfem.helpers import dot, grad

NAME, MESH = "Left", "Left-Mesh"
WRITE_DATA, READ_DATA = "Heat-Flux", "Temperature"
X0, X1 = 0.0, 0.6
Y0, Y1 = 0.0, 0.4
IFACE_X = 0.6
K = 0.8
T_OUTER = 320.0
NX, NY = 20, 15
OUT = "left_interface.json"

ON_RIGHT = abs(IFACE_X - X1) < abs(IFACE_X - X0)
OUTER_X = X0 if ON_RIGHT else X1
S = 1.0 if ON_RIGHT else -1.0            # outward normal at interface = S * e_x
TOL = 1e-9

mesh = MeshTri.init_tensor(np.linspace(X0, X1, NX + 1),
                           np.linspace(Y0, Y1, NY + 1))
elem = ElementTriP1()
basis = Basis(mesh, elem)
n2d = basis.nodal_dofs[0]
px, py = mesh.p[0], mesh.p[1]
iface_n = np.where(np.abs(px - IFACE_X) < TOL)[0]
iface_n = iface_n[np.argsort(py[iface_n])]
y_if = py[iface_n]
iface_dofs = n2d[iface_n]
outer_dofs = n2d[np.where(np.abs(px - OUTER_X) < TOL)[0]]
if len(iface_dofs) == 0:
    raise SystemExit("no interface nodes at x=%r" % IFACE_X)


@BilinearForm
def stiffness(u, v, w):
    return K * dot(grad(u), grad(v))


@BilinearForm
def mass(u, v, w):
    return u * v


@LinearForm
def proj_rhs(v, w):
    return (-K * S) * w["uh"].grad[0] * v


A = stiffness.assemble(basis)
b = basis.zeros()
M = mass.assemble(basis)
D = np.concatenate([outer_dofs, iface_dofs])


def advance(t_iface):
    """Impose the incoming interface temperature, return (T, q_out) there."""
    sol = basis.zeros()
    sol[outer_dofs] = T_OUTER
    sol[iface_dofs] = t_iface
    sol = solve(*condense(A, b, x=sol, D=D))
    qh = solve(M, proj_rhs.assemble(basis, uh=basis.interpolate(sol)))
    return sol[iface_dofs], qh[iface_dofs]


coords = np.column_stack([np.full(len(y_if), IFACE_X), y_if])
p = precice.Participant(NAME, "precice-config.xml", 0, 1)
vid = p.set_mesh_vertices(MESH, coords)
if p.requires_initial_data():
    p.write_data(MESH, WRITE_DATA, vid, np.zeros(len(vid)))
p.initialize()

t_last = np.zeros(len(vid))
q_last = np.zeros(len(vid))
n_it = 0
while p.is_coupling_ongoing():
    if p.requires_writing_checkpoint():
        pass                                 # steady solve: no state to save
    dt = p.get_max_time_step_size()
    t_in = np.asarray(p.read_data(MESH, READ_DATA, vid, dt), float).ravel()
    t_last, q_last = advance(t_in)
    p.write_data(MESH, WRITE_DATA, vid, q_last)
    p.advance(dt)
    n_it += 1
    if p.requires_reading_checkpoint():
        pass                                 # ... and none to restore
p.finalize()

with open(OUT, "w") as f:
    json.dump({"participant": NAME, "sub_iterations": n_it,
               "coordinates": [[float(IFACE_X), float(y)] for y in y_if],
               "values": [float(t) for t in t_last],
               "normal_fluxes": [float(q) for q in q_last]}, f, indent=2)
'''

_RIGHT = '''\
"""RIGHT subdomain, FEniCSx, NEUMANN role: reads the flux, writes Temperature."""
import json
import numpy as np
import precice
import ufl
from dolfinx import default_scalar_type, fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

NAME, MESH = "Right", "Right-Mesh"
WRITE_DATA, READ_DATA = "Temperature", "Heat-Flux"
X0, X1 = 0.6, 1.1
Y0, Y1 = 0.0, 0.4
IFACE_X = 0.6
K = 1.5
T_OUTER = 300.0
NX, NY = 14, 11
OUT = "right_interface.json"

OUTER_X = X0 if abs(IFACE_X - X1) < abs(IFACE_X - X0) else X1
S = 1.0 if IFACE_X > OUTER_X else -1.0   # outward normal at interface = S * e_x

domain = dmesh.create_rectangle(MPI.COMM_WORLD, [[X0, Y0], [X1, Y1]],
                                [NX, NY], dmesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)

xy = V.tabulate_dof_coordinates()
iface_dofs = np.where(np.abs(xy[:, 0] - IFACE_X) < 1e-10)[0]
iface_dofs = iface_dofs[np.argsort(xy[iface_dofs, 1])]
y_if = xy[iface_dofs, 1]
if len(iface_dofs) == 0:
    raise SystemExit("no interface DOFs at x=%r" % IFACE_X)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = fem.Constant(domain, default_scalar_type(K)) * \\
    ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
outer = dmesh.locate_entities_boundary(domain, fdim,
                                       lambda x: np.isclose(x[0], OUTER_X))
bcs = [fem.dirichletbc(default_scalar_type(T_OUTER),
                       fem.locate_dofs_topological(V, fdim, outer), V)]
g = fem.Function(V)
facets = dmesh.locate_entities_boundary(domain, fdim,
                                        lambda x: np.isclose(x[0], IFACE_X))
tags = dmesh.meshtags(domain, fdim, np.sort(facets),
                      np.full(len(facets), 7, dtype=np.int32))
ds_if = ufl.Measure("ds", domain=domain, subdomain_data=tags)(7)
L = g * v * ds_if
prob = LinearProblem(a, L, bcs=bcs, petsc_options_prefix="cpl",
                     petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = prob.solve()
p_, w_ = ufl.TrialFunction(V), ufl.TestFunction(V)
fprob = LinearProblem(p_ * w_ * ufl.dx, -K * S * uh.dx(0) * w_ * ufl.dx,
                      petsc_options_prefix="flx",
                      petsc_options={"ksp_type": "preonly", "pc_type": "lu"})


def advance(q_in):
    """Impose the incoming flux density, return (T, q_out) at the interface."""
    g.x.array[iface_dofs] = q_in
    g.x.scatter_forward()
    prob.solve()
    qh = fprob.solve()
    return uh.x.array[iface_dofs].copy(), qh.x.array[iface_dofs].copy()


coords = np.column_stack([np.full(len(y_if), IFACE_X), y_if])
p = precice.Participant(NAME, "precice-config.xml", 0, 1)
vid = p.set_mesh_vertices(MESH, coords)
if p.requires_initial_data():
    p.write_data(MESH, WRITE_DATA, vid, np.zeros(len(vid)))
p.initialize()

t_last = np.zeros(len(vid))
q_last = np.zeros(len(vid))
n_it = 0
while p.is_coupling_ongoing():
    if p.requires_writing_checkpoint():
        pass
    dt = p.get_max_time_step_size()
    q_in = np.asarray(p.read_data(MESH, READ_DATA, vid, dt), float).ravel()
    t_last, q_last = advance(q_in)
    p.write_data(MESH, WRITE_DATA, vid, t_last)
    p.advance(dt)
    n_it += 1
    if p.requires_reading_checkpoint():
        pass
p.finalize()

with open(OUT, "w") as f:
    json.dump({"participant": NAME, "sub_iterations": n_it,
               "coordinates": [[float(IFACE_X), float(y)] for y in y_if],
               "values": [float(t) for t in t_last],
               "normal_fluxes": [float(q) for q in q_last]}, f, indent=2)
'''


# ── the real `couple_precice` tool, reached the way the server reaches it ───
#
# The registered tool, not `run_precice_coupling`: the tool's own argument
# parsing, its availability gate and its post-run verdict are part of what the
# knowledge describes, and calling the function underneath would skip all three.

_TOOL_MANAGER = None


def _tool_manager():
    global _TOOL_MANAGER
    if _TOOL_MANAGER is None:
        from mcp.server.fastmcp import FastMCP
        from core.registry import load_all_backends
        from tools.consolidated import register_consolidated_tools
        m = FastMCP("tier2-precice")
        register_consolidated_tools(m)
        load_all_backends()
        _TOOL_MANAGER = m._tool_manager
    return _TOOL_MANAGER


def _text(res) -> str:
    if isinstance(res, tuple) and len(res) >= 1:
        res = res[0]
    if isinstance(res, list):
        return "\n".join(getattr(b, "text", str(b)) for b in res)
    return getattr(res, "text", str(res))


def call_tool(name: str, args: dict) -> str:
    return _text(asyncio.run(_tool_manager().call_tool(name, args)))


# ── assertion plumbing on top of couplinglib's ─────────────────────────────
#
# `L.check` prints `FAIL: ...` and records the failure; this prints a NAMED
# verdict on the passing path so every expectation in fixture.json can point at
# an assertion rather than at a bare `name=` prefix that any value satisfies.

def verdict(ok: bool, label: str, detail: str = "") -> bool:
    if L.check(bool(ok), f"{label}_violated", detail):
        print(f"{label}=yes")
        return True
    return False


def spans(vals) -> tuple[float, float]:
    v = [float(x) for x in vals]
    return min(v), max(v)


def net_flux(export: dict) -> float:
    """Net normal flux leaving through the interface, integrated over
    arclength. Recomputed here rather than taken from any OASiS helper, so the
    conservation check does not depend on that helper being right."""
    co = export["coordinates"]
    fl = export["normal_fluxes"]
    ys = [c[1] for c in co]
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    tot = 0.0
    for i, j in zip(order, order[1:]):
        tot += 0.5 * (float(fl[i]) + float(fl[j])) * (ys[j] - ys[i])
    return tot


# ── PART A: the import gate ────────────────────────────────────────────────

def _probe(python: str, code: str, ld: str | None,
           pythonpath: str | None = None) -> tuple[int, str, str]:
    env = os.environ.copy()
    # Set or REMOVE LD_LIBRARY_PATH outright rather than editing it: the tool
    # under test prepends the preCICE lib dir to this process's own
    # environment when it checks availability, so anything inherited is not a
    # reliable starting point for a probe about that very variable.
    if ld is None:
        env.pop("LD_LIBRARY_PATH", None)
    else:
        env["LD_LIBRARY_PATH"] = ld
    if pythonpath is None:
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = pythonpath
    try:
        r = subprocess.run([python, "-c", code], capture_output=True,
                           text=True, errors="replace", timeout=600, env=env)
    except (OSError, subprocess.SubprocessError) as e:
        return -1, "", f"{type(e).__name__}: {e}"
    return r.returncode, r.stdout.strip(), r.stderr.strip()


_MSG = "libprecice.so.3: cannot open shared object file"


def import_gate(lib_dir: str) -> None:
    """`LD_LIBRARY_PATH` is NOT optional — checked in every interpreter this
    install would actually launch a participant in."""
    from core.registry import load_all_backends
    load_all_backends()

    own_site = sysconfig.get_paths()["purelib"]
    # One entry per INTERPRETER, listing every backend module that interpreter
    # has to carry alongside preCICE. scikit-fem and NGSolve share one, and
    # both payloads make the same per-backend claim about it, so both modules
    # are named rather than one standing in for the other.
    #
    # The DUNE interpreter is the one whose payload gives a RECIPE: it has no
    # pyprecice of its own, and the claim is that pointing PYTHONPATH at the
    # interpreter that does have one is enough. So it is probed with that
    # recipe, and the other two on their own.
    targets = [
        ("oasis_venv", L.interpreter("skfem"), "skfem, ngsolve", None),
        ("fenicsx", L.interpreter("fenics"), "dolfinx", None),
        ("dune", L.interpreter("dune"), "dune.fem", own_site),
    ]
    n_failed_without = 0
    n_ok_with = 0
    n_msg = 0
    for tag, py, modules, pp in targets:
        if not py or not Path(py).is_file():
            raise L.Absent(f"no interpreter resolved for {tag!r}")
        print(f"interpreter[{tag}]={py}")

        rc, _, err = _probe(py, "import precice", None, pp)
        got_msg = _MSG in err
        print(f"{tag}_precice_without_ld_path="
              f"{'failed' if rc != 0 else 'imported'}")
        if rc != 0:
            n_failed_without += 1
        if got_msg:
            n_msg += 1
        else:
            print(f"{tag}_without_ld_path_stderr_tail="
                  f"{err.splitlines()[-1][:160] if err else '<empty>'}")

        rc, out, err = _probe(py, "import precice;print(precice."
                              "get_version_information())", lib_dir, pp)
        print(f"{tag}_precice_with_ld_path={'ok' if rc == 0 else 'failed'}")
        if rc == 0:
            n_ok_with += 1
        else:
            print(f"{tag}_with_ld_path_stderr_tail="
                  f"{err.splitlines()[-1][:200] if err else '<empty>'}")

        # THE REAL GATE: precice AND every backend module that interpreter
        # carries, in ONE interpreter.
        stmt = f"import precice, {modules}"
        rc, _, err = _probe(py, stmt, lib_dir, pp)
        print(f"{tag}_backend_modules={modules}")
        verdict(rc == 0, f"{tag}_precice_and_backend_modules_coexist",
                f"`{stmt}` failed in {py}: "
                f"{(err.splitlines()[-1] if err else '')[:200]} — a backend "
                f"whose interpreter cannot import both cannot use "
                f"`couple_precice` at all, whatever its payload claims")

    n = len(targets)
    print(f"interpreters_probed={n}")
    print(f"precice_import_without_ld_path="
          f"{'failed' if n_failed_without == n else 'imported'}")
    print(f"precice_import_with_ld_path={'ok' if n_ok_with == n else 'failed'}")
    verdict(n_failed_without == n, "every_interpreter_needs_the_lib_dir",
            f"only {n_failed_without}/{n} interpreters failed to `import "
            f"precice` with LD_LIBRARY_PATH removed — the served text says "
            f"LD_LIBRARY_PATH is NOT optional")
    verdict(n_msg == n, "libprecice_so_3_message_from_every_interpreter",
            f"only {n_msg}/{n} produced {_MSG!r}, which is the exact string "
            f"the served text tells an agent to expect")
    verdict(n_ok_with == n, "every_interpreter_imports_precice_with_the_lib_dir",
            f"only {n_ok_with}/{n} imported preCICE with the lib dir present; "
            f"the CAN verdicts require it in each backend's OWN interpreter")


# ── PART B: the coupled run, and the values it exchanged ───────────────────

def coupled_run(work: Path) -> dict:
    left = L.edit(_LEFT, {
        "X0, X1": f"{XL}, {XI}", "Y0, Y1": f"{Y0}, {Y1}",
        "IFACE_X": f"{XI}", "K": f"{KL}", "T_OUTER": f"{TL}",
        "NX, NY": f"{MESH_L[0]}, {MESH_L[1]}"})
    right = L.edit(_RIGHT, {
        "X0, X1": f"{XI}, {XR}", "Y0, Y1": f"{Y0}, {Y1}",
        "IFACE_X": f"{XI}", "K": f"{KR}", "T_OUTER": f"{TR}",
        "NX, NY": f"{MESH_R[0]}, {MESH_R[1]}"})
    (work / "participant_left.py").write_text(left)
    (work / "participant_right.py").write_text(right)

    py_left, py_right = L.interpreter("skfem"), L.interpreter("fenics")
    print(f"participant[Left]=skfem interpreter={py_left}")
    print(f"participant[Right]=fenics interpreter={py_right}")
    verdict(Path(py_left).resolve() != Path(py_right).resolve(),
            "two_distinct_interpreters",
            f"both participants would run in {py_left} — the claim being "
            f"tested is that preCICE couples codes living in SEPARATE "
            f"interpreters, and one interpreter does not test it")

    participants = [
        {"name": "Left", "mesh": "Left-Mesh", "writes": ["Heat-Flux"],
         "reads": ["Temperature"], "command": [py_left, "participant_left.py"]},
        {"name": "Right", "mesh": "Right-Mesh", "writes": ["Temperature"],
         "reads": ["Heat-Flux"], "command": [py_right, "participant_right.py"]},
    ]
    data = [{"name": "Temperature", "type": "scalar"},
            {"name": "Heat-Flux", "type": "scalar"}]
    # exchanges[0] MUST be the field written by the SECOND participant: OASiS
    # makes it both the convergence measure and the acceleration datum, and
    # preCICE allows only second-to-first data there.
    exchanges = [{"data": "Temperature", "from": "Right", "to": "Left"},
                 {"data": "Heat-Flux", "from": "Left", "to": "Right"}]

    raw = call_tool("couple_precice", {
        "participants": json.dumps(participants), "data": json.dumps(data),
        "exchanges": json.dumps(exchanges), "work_dir": str(work),
        "scheme": SCHEME, "dimensions": 2, "max_time": MAX_TIME,
        "time_window": TIME_WINDOW, "timeout": 900, "critic_approved": True})
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise AssertionError(f"couple_precice returned non-JSON: {raw[:400]}")


def check_values(res: dict, work: Path) -> None:
    rcs = res.get("returncodes") or {}
    print(f"participant_returncodes={json.dumps(rcs, sort_keys=True)}")
    ok_rc = bool(rcs) and all(int(v) == 0 for v in rcs.values())
    if not verdict(ok_rc, "all_participant_returncodes_zero",
                   str(res.get("error"))[:300]):
        _dump_logs(res)
    if not verdict(bool(res.get("converged")), "coupled_run_converged",
                   str(res.get("error"))[:300]):
        _dump_logs(res)

    # From here on the fixture stops trusting the tool and reads what the two
    # codes actually exchanged.
    ex = {}
    for side, name in (("left", "Left"), ("right", "Right")):
        f = work / f"{side}_interface.json"
        if not f.is_file():
            L.check(False, f"{name}_wrote_no_interface_file",
                    f"{f} is missing, so the participant did not finish its "
                    f"coupling loop and there is nothing to check")
            _dump_logs(res)
            return
        ex[side] = json.loads(f.read_text())

    nl, nr = len(ex["left"]["values"]), len(ex["right"]["values"])
    print(f"interface_points_left_right={nl}/{nr}")
    verdict(nl != nr, "interface_meshes_non_matching",
            f"both sides used {nl} interface points, so preCICE's mapping "
            f"between NON-matching surfaces was never exercised")

    subs = (int(ex["left"]["sub_iterations"]),
            int(ex["right"]["sub_iterations"]))
    print(f"sub_iterations_left_right={subs[0]}/{subs[1]}")
    verdict(min(subs) > N_WINDOWS, "implicit_subiteration_happened",
            f"{subs} solves over {N_WINDOWS} time windows — an implicit "
            f"scheme that never sub-iterates is an explicit one, and the "
            f"acceleration the config asks for did nothing")

    print(f"closed_form_T_iface={T_EXACT:.10g}")
    print(f"closed_form_q_iface={Q_EXACT:.10g}")
    for side in ("left", "right"):
        lo, hi = spans(ex[side]["values"])
        print(f"{side}_T_span=[{lo:.10g},{hi:.10g}]")
        err = abs(0.5 * (lo + hi) - T_EXACT)
        print(f"{side}_T_abs_error={err:.3e}")
        verdict(err <= T_ATOL, f"{side}_interface_T_matches_closed_form",
                f"|{0.5 * (lo + hi):.10g} - {T_EXACT:.10g}| = {err:.6e} K "
                f"> {T_ATOL:.1e}")
        verdict(hi - lo <= UNIFORM_ATOL, f"{side}_interface_T_uniform_in_y",
                f"the exact interface temperature has no y-variation; this "
                f"one spans {hi - lo:.3e} K")

    # The two sides export with respect to their OWN outward normals, which are
    # anti-parallel — so the right side's number must be the NEGATIVE of the
    # left's. A fixture that checked magnitudes would pass a sign error.
    for side, sign in (("left", +1.0), ("right", -1.0)):
        lo, hi = spans(ex[side]["normal_fluxes"])
        print(f"{side}_q_span=[{lo:.10g},{hi:.10g}]")
        err = abs(0.5 * (lo + hi) - sign * Q_EXACT)
        print(f"{side}_q_abs_error={err:.3e}")
        verdict(err <= Q_ATOL, f"{side}_interface_q_matches_closed_form",
                f"|{0.5 * (lo + hi):.10g} - {sign * Q_EXACT:.10g}| = "
                f"{err:.6e} W/m^2 > {Q_ATOL:.1e}")

    net_l, net_r = net_flux(ex["left"]), net_flux(ex["right"])
    scale = max(abs(net_l), abs(net_r), 1e-30)
    rel = abs(net_l + net_r) / scale
    print(f"flux_balance_rel={rel:.3e}")
    verdict(rel <= BALANCE_RTOL, "interface_flux_balanced",
            f"net(Left)={net_l:.6e} net(Right)={net_r:.6e} — what leaves one "
            f"subdomain must enter the other")


def _dump_logs(res: dict) -> None:
    for name, tail in (res.get("logs") or {}).items():
        print(f"--- participant log tail [{name}] ---")
        print(str(tail)[-1200:])


# ── the fixture body ───────────────────────────────────────────────────────

def body() -> None:
    L.require_available("skfem", "fenics", "dune")

    from core.precice_config import PRECICE_LIB_DIR, check_precice_available
    if not Path(PRECICE_LIB_DIR).is_dir():
        raise L.Absent(f"no preCICE lib directory at {PRECICE_LIB_DIR}; "
                       f"nothing here can be tested without libprecice")
    print(f"precice_lib_dir={PRECICE_LIB_DIR}")
    ok, msg = check_precice_available()
    if not ok:
        raise L.Absent(f"preCICE is not usable from this install ({msg[:200]})")

    import_gate(PRECICE_LIB_DIR)

    # A FRESH work_dir per run, and one per coupling: participants share it as
    # their cwd, and a stale `precice-run/` from a killed run makes the next
    # connect hang forever — preCICE has no connect timeout.
    work = L.workroot("precice_can")
    print(f"precice_work_dir={work}")
    res = coupled_run(work)
    check_values(res, work)


L.main(body)
