"""The four preCICE "CAN" verdicts that had NO coupled run behind them.

WHAT THIS FIXTURE EXISTS TO FIX
-------------------------------
`_PRECICE_BY_BACKEND` in src/tools/coupling_knowledge.py served
"CAN — proven by a real coupled run" for SEVEN backends, under a header comment
saying "Every CAN was established by running a real two-participant coupling
through OASiS's own preCICE orchestrator on this install."

Two of the seven had that: scikit-fem and FEniCSx, in
`precice_can_verdicts_proven_by_a_real_run`, whose own docstring downgrades
NGSolve and DUNE-fem to the IMPORT GATE ("the GATE only", "put the
site-packages ... on PYTHONPATH") and says nothing about the rest. deal.II,
Kratos and SPARTA had no preCICE fixture at all — the shipped deal.II
participant is a `couple` file-handshake wrapper and neither it nor its
CMakeLists mentioned preCICE, so a reviewer greping the tree for a
preCICE-linked C++ participant found nothing.

This fixture runs the four missing FEM couplings for real. (SPARTA is a DSMC
code with a different interface problem and its own load-order trap, so it has
its own fixture, `sparta_precice_load_order_and_coupled_run`.)

WHAT IS RUN
-----------
Four separate two-participant preCICE couplings, each driven through the
REGISTERED `couple_precice` tool, each pairing the SAME scikit-fem DIRICHLET
participant against one backend's NEUMANN participant:

    NGSolve    Python, netgen mesh          (shares the OASiS venv with skfem)
    DUNE-fem   Python, ALUGrid + JIT UFL    (its own conda env + the PYTHONPATH
                                             recipe its payload prescribes)
    deal.II    COMPILED C++ linking         (data/coupling_participants/
               libprecice directly          precice_heat_dealii.cc, built here)
    Kratos     Python, ConvectionDiffusion  (its own interpreter + a preCICE
               ThermalFace2D2N conditions    shim, see `precice_shim`)

Interface meshes are non-matching in every pair, the scheme is serial-implicit
so the sub-iteration is real, and the exchanged fields are checked against a
CLOSED FORM computed here — not against a number a previous run printed.

WHY THE VALUES AND NOT `converged`
----------------------------------
`couple_precice`'s own knowledge says `converged` is exit codes plus preCICE's
per-window verdict and never looks at the values. This fixture's mutation is
the proof: flip the sign with which the shared scikit-fem Dirichlet participant
EXPORTS its flux and all four couplings still exit 0, still log "All
converged", and still reach a fixed point — a different one, ~284 K against a
closed-form 306.15 K, with both sides carrying the same sign of flux so nothing
is conserved. Every signal the orchestrator returns stays green.

THE PLACEHOLDER PROBLEM AND ITS CLOSED FORM
-------------------------------------------
Steady conduction with no source on a rectangle split at x = XI. No
y-variation, so the exact solution is piecewise linear in x, the interface
temperature is the conductance-weighted mean of the two outer Dirichlet values
and the interface flux follows from either side. "Conductance" is k divided by
the distance from the interface to that subdomain's own outer boundary.
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                     # noqa: E402


# ── the split conduction problem, and the reference checked against ────────

XL, XI, XR = 0.0, 0.6, 1.1        # outer left boundary, interface, outer right
Y0, Y1 = 0.0, 0.4
KL, KR = 0.8, 1.5                 # conductivities,      W/(m K)
TL, TR = 320.0, 300.0             # outer Dirichlet values, K
MESH_L = (20, 15)                 # scikit-fem's own subdomain mesh

CL = KL / (XI - XL)               # left interface conductance,  W/(m^2 K)
CR = KR / (XR - XI)               # right interface conductance
T_EXACT = (CL * TL + CR * TR) / (CL + CR)
Q_EXACT = CL * (TL - T_EXACT)     # +x-ward flux density at the interface, W/m^2

# Tolerances, and why these. `couple_precice` hard-codes preCICE's relative
# convergence measure at 1e-6, so the fixed point is reached to about
# |T| * 1e-6 / (1 - |slope|) ~ 5e-4 K. These sit an order of magnitude above
# that and three orders BELOW every pathology the fixture exists to catch: the
# mutation's sign error is 22 K and 37 W/m^2, a unit mismatch is a factor 1e3.
T_ATOL = 1e-2                     # K
Q_ATOL = 1e-2                     # W/m^2
UNIFORM_ATOL = 1e-6               # the exact interface field has NO y-variation
BALANCE_RTOL = 1e-4

SCHEME = "serial-implicit"
TIME_WINDOW = 1.0
MAX_TIME = 2.0
N_WINDOWS = int(round(MAX_TIME / TIME_WINDOW))

# The sign with which the DIRICHLET participant exports the flux it computed.
# Named once, here, because it is this fixture's mutation point: the served
# FEniCSx pattern says the partner's number goes across UNCHANGED, and every
# one of the four couplings below goes through this one participant.
EXPORT_SIGN = 1.0


# ── the shared DIRICHLET participant: scikit-fem, LEFT subdomain ───────────

_SKFEM_LEFT = '''\
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
EXPORT_SIGN = 1.0
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
    # The consistent nodal flux goes across with the sign THIS side's outward
    # normal gives it. The partner applies it UNCHANGED.
    return sol[iface_dofs], EXPORT_SIGN * qh[iface_dofs]


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
        pass
p.finalize()

with open(OUT, "w") as f:
    json.dump({"participant": NAME, "sub_iterations": n_it,
               "coordinates": [[float(IFACE_X), float(y)] for y in y_if],
               "values": [float(t) for t in t_last],
               "normal_fluxes": [float(q) for q in q_last]}, f, indent=2)
'''


# ── the four NEUMANN participants ─────────────────────────────────────────

_RIGHT_NGSOLVE = '''\
"""RIGHT subdomain, NGSolve, NEUMANN role: reads Heat-Flux, writes Temperature.

Both NGSolve traps the payload names are avoided HERE, deliberately:
  * every Dirichlet value goes into ONE `Set` call, because two consecutive
    `Set(..., definedon=...)` calls cancel each other;
  * the outgoing flux is a VOLUME L2 projection, because
    `Integrate(grad(gfu)[0], ..., definedon=Boundaries)` returns exactly 0.0 —
    an H1 gradient has no boundary trace.
"""
import json
import numpy as np
import precice
from netgen.geom2d import SplineGeometry
from ngsolve import (H1, BilinearForm, GridFunction, LinearForm, Mesh,
                     TaskManager, ds, dx, grad)

NAME, MESH = "Right", "Right-Mesh"
WRITE_DATA, READ_DATA = "Temperature", "Heat-Flux"
X0, X1 = 0.6, 1.1
Y0, Y1 = 0.0, 0.4
IFACE_X = 0.6
K = 1.5
T_OUTER = 300.0
MAXH = 0.045
OUT = "right_interface.json"

ON_RIGHT = abs(IFACE_X - X1) < abs(IFACE_X - X0)
OUTER_X = X0 if ON_RIGHT else X1
S = 1.0 if ON_RIGHT else -1.0
TOL = 1e-9

geo = SplineGeometry()
geo.AddRectangle((X0, Y0), (X1, Y1),
                 bcs=(["bot", "iface", "top", "outer"] if ON_RIGHT
                      else ["bot", "outer", "top", "iface"]))
mesh = Mesh(geo.GenerateMesh(maxh=MAXH))

V = H1(mesh, order=1, dirichlet="outer")     # order 1 -> dof i == vertex i
pts = np.array([list(v.point) for v in mesh.vertices])
iface = np.where(np.abs(pts[:, 0] - IFACE_X) < TOL)[0]
iface = iface[np.argsort(pts[iface, 1])]
y_if = pts[iface, 1]
if len(iface) == 0:
    raise SystemExit("no interface vertices at x=%r" % IFACE_X)

u, v = V.TnT()
a = BilinearForm(V)
a += K * grad(u) * grad(v) * dx
a.Assemble()

gq = GridFunction(V)                          # incoming flux density, at MY dofs
f = LinearForm(V)
f += gq * v * ds(definedon=mesh.Boundaries("iface"))

mform = BilinearForm(V)
mform += u * v * dx
mform.Assemble()
minv = mform.mat.Inverse(freedofs=None, inverse="sparsecholesky")

gfu = GridFunction(V)
qh = GridFunction(V)


def advance(q_in):
    gq.vec[:] = 0.0
    for i, dof in enumerate(iface):
        gq.vec[dof] = float(q_in[i])
    f.Assemble()
    gfu.Set(T_OUTER, definedon=mesh.Boundaries("outer"))
    r = f.vec.CreateVector()
    r.data = f.vec - a.mat * gfu.vec
    gfu.vec.data += a.mat.Inverse(V.FreeDofs(),
                                  inverse="sparsecholesky") * r
    fl = LinearForm(V)
    fl += (-K * S) * grad(gfu)[0] * v * dx
    fl.Assemble()
    qh.vec.data = minv * fl.vec
    return (np.array([gfu.vec[d] for d in iface]),
            np.array([qh.vec[d] for d in iface]))


coords = np.column_stack([np.full(len(y_if), IFACE_X), y_if])
p = precice.Participant(NAME, "precice-config.xml", 0, 1)
vid = p.set_mesh_vertices(MESH, coords)
if p.requires_initial_data():
    p.write_data(MESH, WRITE_DATA, vid, np.zeros(len(vid)))
p.initialize()

t_last = np.zeros(len(vid))
q_last = np.zeros(len(vid))
n_it = 0
with TaskManager():
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

with open(OUT, "w") as fh:
    json.dump({"participant": NAME, "sub_iterations": n_it,
               "coordinates": [[float(IFACE_X), float(y)] for y in y_if],
               "values": [float(t) for t in t_last],
               "normal_fluxes": [float(q) for q in q_last]}, fh, indent=2)
'''


_RIGHT_DUNE = '''\
"""RIGHT subdomain, DUNE-fem, NEUMANN role: reads Heat-Flux, writes Temperature.

DUNE-fem JIT-compiles each distinct scheme, so BOTH schemes and the grid are
built AND COMPILED (on a throw-away solve) BEFORE precice.Participant(...) is
constructed: the partner blocks on the connection handshake, so anything
compiled after that point is time the partner spends waiting. The coupled datum
is a DISCRETE FUNCTION whose dofs are mutated each iteration, so no scheme is
ever rebuilt.
"""
import json
import numpy as np
from dune.grid import cartesianDomain
from dune.alugrid import aluConformGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import (SpatialCoordinate, TestFunction, TrialFunction, conditional,
                 ds, dx, grad, inner)

NAME, MESH = "Right", "Right-Mesh"
WRITE_DATA, READ_DATA = "Temperature", "Heat-Flux"
X0, X1 = 0.6, 1.1
Y0, Y1 = 0.0, 0.4
IFACE_X = 0.6
K = 1.5
T_OUTER = 300.0
NX, NY = 14, 11
OUT = "right_interface.json"

ON_RIGHT = abs(IFACE_X - X1) < abs(IFACE_X - X0)
OUTER_X = X0 if ON_RIGHT else X1
S = 1.0 if ON_RIGHT else -1.0
TOL = 1e-6

grid = aluConformGrid(cartesianDomain([X0, Y0], [X1, Y1], [NX, NY]))
space = lagrange(grid, order=1)
u, v = TrialFunction(space), TestFunction(space)
x = SpatialCoordinate(space)

gq = space.interpolate(0, name="gq")
uh = space.interpolate(0, name="uh")
qh = space.interpolate(0, name="qh")

on_iface = conditional(abs(x[0] - IFACE_X) < TOL, 1.0, 0.0)
bc = DirichletBC(space, T_OUTER, abs(x[0] - OUTER_X) < TOL)
scheme = galerkin([K * inner(grad(u), grad(v)) * dx == gq * on_iface * v * ds,
                   bc], space, solver=("suitesparse", "umfpack"))
proj = galerkin(u * v * dx == (-K * S) * grad(uh)[0] * v * dx, space,
                solver=("suitesparse", "umfpack"))

xc = space.interpolate(x[0], name="xc").as_numpy
yc = space.interpolate(x[1], name="yc").as_numpy
iface = np.where(np.abs(xc - IFACE_X) < 1e-9)[0]
iface = iface[np.argsort(yc[iface])]
y_if = yc[iface].copy()
if len(iface) == 0:
    raise SystemExit("no interface dofs at x=%r" % IFACE_X)

scheme.solve(target=uh)                # compile both schemes NOW
proj.solve(target=qh)


def advance(q_in):
    gq.as_numpy[:] = 0.0
    gq.as_numpy[iface] = np.asarray(q_in, float)
    scheme.solve(target=uh)
    proj.solve(target=qh)
    return uh.as_numpy[iface].copy(), qh.as_numpy[iface].copy()


import precice                                              # noqa: E402

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


_RIGHT_KRATOS = '''\
"""RIGHT subdomain, Kratos, NEUMANN role: reads Heat-Flux, writes Temperature.

Kratos drives its own time loop, so the preCICE loop wraps
InitializeSolutionStep / SolveSolutionStep / FinalizeSolutionStep on a strategy
BUILT ONCE, before the Participant is constructed. The incoming flux density
goes on as FACE_HEAT_FLUX on ThermalFace2D2N conditions along the interface —
ConvectionDiffusion's surface-source route.
"""
import json
import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication  # noqa: F401

NAME, MESH = "Right", "Right-Mesh"
WRITE_DATA, READ_DATA = "Temperature", "Heat-Flux"
X0, X1 = 0.6, 1.1
Y0, Y1 = 0.0, 0.4
IFACE_X = 0.6
K = 1.5
T_OUTER = 300.0
NX, NY = 14, 11
OUT = "right_interface.json"

ON_RIGHT = abs(IFACE_X - X1) < abs(IFACE_X - X0)
I_IF = NX if ON_RIGHT else 0
I_OUT = 0 if ON_RIGHT else NX
I_NEAR = I_IF - 1 if ON_RIGHT else 1

model = KM.Model()
mp = model.CreateModelPart("thermal")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
settings = KM.ConvectionDiffusionSettings()
settings.SetUnknownVariable(KM.TEMPERATURE)
settings.SetDiffusionVariable(KM.CONDUCTIVITY)
settings.SetVolumeSourceVariable(KM.HEAT_FLUX)
settings.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, settings)
for var in (KM.TEMPERATURE, KM.CONDUCTIVITY, KM.HEAT_FLUX, KM.FACE_HEAT_FLUX):
    mp.AddNodalSolutionStepVariable(var)
mp.SetBufferSize(1)

props = mp.CreateNewProperties(1)
nid, cnt = {}, 1
for j in range(NY + 1):
    for i in range(NX + 1):
        mp.CreateNewNode(cnt, X0 + (X1 - X0) * i / NX,
                         Y0 + (Y1 - Y0) * j / NY, 0.0)
        nid[(i, j)] = cnt
        cnt += 1
eid = 1
for j in range(NY):
    for i in range(NX):
        a, b, c, d = (nid[(i, j)], nid[(i + 1, j)], nid[(i + 1, j + 1)],
                      nid[(i, j + 1)])
        mp.CreateNewElement("LaplacianElement2D3N", eid, [a, b, d], props)
        eid += 1
        mp.CreateNewElement("LaplacianElement2D3N", eid, [b, c, d], props)
        eid += 1
for j in range(NY):
    mp.CreateNewCondition("ThermalFace2D2N", j + 1,
                          [nid[(I_IF, j)], nid[(I_IF, j + 1)]], props)

for node in mp.Nodes:
    node.SetSolutionStepValue(KM.CONDUCTIVITY, K)
    node.SetSolutionStepValue(KM.HEAT_FLUX, 0.0)
    node.SetSolutionStepValue(KM.FACE_HEAT_FLUX, 0.0)
for j in range(NY + 1):
    n = mp.Nodes[nid[(I_OUT, j)]]
    n.SetSolutionStepValue(KM.TEMPERATURE, T_OUTER)
    n.Fix(KM.TEMPERATURE)

KM.VariableUtils().AddDof(KM.TEMPERATURE, mp)
scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
builder = KM.ResidualBasedBlockBuilderAndSolver(
    KM.SkylineLUFactorizationSolver())
strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder,
                                          False, False, False, False)
strategy.Initialize()

y_if = np.array([Y0 + (Y1 - Y0) * j / NY for j in range(NY + 1)])
DX = abs(X1 - X0) / NX


def advance(q_in):
    for j in range(NY + 1):
        mp.Nodes[nid[(I_IF, j)]].SetSolutionStepValue(KM.FACE_HEAT_FLUX,
                                                      float(q_in[j]))
    mp.ProcessInfo[KM.TIME] = mp.ProcessInfo[KM.TIME] + 1.0
    mp.ProcessInfo[KM.STEP] = mp.ProcessInfo[KM.STEP] + 1
    strategy.InitializeSolutionStep()
    strategy.SolveSolutionStep()
    strategy.FinalizeSolutionStep()
    t_out = np.array([mp.Nodes[nid[(I_IF, j)]].GetSolutionStepValue(
        KM.TEMPERATURE) for j in range(NY + 1)])
    t_near = np.array([mp.Nodes[nid[(I_NEAR, j)]].GetSolutionStepValue(
        KM.TEMPERATURE) for j in range(NY + 1)])
    # The exact solution is linear in x, so the one-sided difference toward the
    # interior is the exact slope; q = -K dT/dn with n the outward normal.
    return t_out, -K * (t_out - t_near) / DX


import precice                                              # noqa: E402

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


# deal.II has no Python API: the participant is the COMPILED executable, and
# this is only the launcher that hands it its problem and lets it talk to
# preCICE itself. Nothing about the coupling passes through Python here.
_RIGHT_DEALII = '''\
"""RIGHT subdomain, deal.II, NEUMANN role — a launcher for the COMPILED C++
participant, which links libprecice and drives the coupling loop itself."""
import subprocess
import sys

DEALII_EXE = "./precice_heat_dealii"
SIDE = 1                            # 1 = Neumann
K = 1.5
X0, X1 = 0.6, 1.1
Y0, Y1 = 0.0, 0.4
IFACE_X = 0.6
T_OUTER = 300.0
F_SRC = 0.0
NX, NY = 14, 11
DEGREE = 1

with open("dealii_in.txt", "w") as f:
    f.write(f"{SIDE} {K} {X0} {X1} {Y0} {Y1} {IFACE_X} {T_OUTER} {F_SRC} "
            f"{NX} {NY} {DEGREE}\\n")
    f.write("Right Right-Mesh Temperature Heat-Flux precice-config.xml\\n")
sys.exit(subprocess.run([DEALII_EXE, "dealii_in.txt",
                         "right_interface.json"]).returncode)
'''


# ── the real `couple_precice` tool, reached the way the server reaches it ───

_TOOL_MANAGER = None


def _tool_manager():
    global _TOOL_MANAGER
    if _TOOL_MANAGER is None:
        from mcp.server.fastmcp import FastMCP
        from core.registry import load_all_backends
        from tools.consolidated import register_consolidated_tools
        m = FastMCP("tier2-precice-four")
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


# ── assertion plumbing ─────────────────────────────────────────────────────

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
    arclength. Recomputed here so the conservation check does not depend on
    any OASiS helper being right."""
    co, fl = export["coordinates"], export["normal_fluxes"]
    ys = [c[1] for c in co]
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    tot = 0.0
    for i, j in zip(order, order[1:]):
        tot += 0.5 * (float(fl[i]) + float(fl[j])) * (ys[j] - ys[i])
    return tot


# ── per-backend environment: interpreter, and what preCICE needs there ─────

def precice_shim(dst: Path) -> Path:
    """A PYTHONPATH entry carrying ONLY pyprecice and what its extension module
    was built against.

    The DUNE payload's recipe — put the WHOLE site-packages of the interpreter
    that has pyprecice on PYTHONPATH — works for DUNE and does NOT work for
    Kratos, for two reasons this fixture found by running it:

      * PYTHONPATH is searched BEFORE the interpreter's own site-packages, so a
        whole-site-packages entry also shadows that interpreter's `KratosMultiphysics`
        with the OASiS venv's broken wheel, which dies at import on `GLIBC_2.32
        not found`.
      * `cyprecice` is a compiled extension built against ONE numpy ABI. Import
        it next to a different numpy and it fails with "numpy.core.multiarray
        failed to import". So numpy has to come from the pyprecice side too.

    A directory of symlinks to exactly precice, cyprecice, numpy and mpi4py
    satisfies both: preCICE gets the numpy it was built against, and the target
    interpreter keeps its own everything else.
    """
    site = Path(sysconfig.get_paths()["purelib"])
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("precice", "cyprecice", "numpy", "numpy.libs", "mpi4py"):
        src = site / name
        if src.exists():
            link = dst / name
            if not link.exists():
                link.symlink_to(src)
    for so in glob.glob(str(site / "cyprecice*.so")):
        link = dst / Path(so).name
        if not link.exists():
            link.symlink_to(so)
    missing = [n for n in ("precice", "numpy") if not (dst / n).exists()]
    if missing:
        raise L.Absent(f"cannot build a preCICE shim: {missing} not in {site}")
    return dst


def kratos_interpreter(shim: Path, lib_dir: str) -> str:
    """The first interpreter on this host that satisfies the WHOLE preCICE
    gate for Kratos, never a hard-coded path taken on faith.

    The gate is `import precice` AND the backend's own modules IN ONE
    INTERPRETER, so that is what is probed — not `import KratosMultiphysics`
    alone. The difference is not academic here: this host has a system Python
    that imports Kratos fine and is 3.8, so it cannot load the cp312 pyprecice
    extension at all. An interpreter chosen on the Kratos half of the gate
    passes the probe and then dies inside the coupling with a numpy C-extension
    error, which is what happened before this probe was widened.
    """
    cands = [os.environ.get("KRATOS_PYTHON"),
             # the default the repo's own scripts use for this host
             "/mnt/kratos-tier2/kv/bin/python",
             sys.executable]
    params = (L.REPO_ROOT / "benchmarks" / "coupling_pairs" /
              "fourc_kratos_cht" / "params.json")
    if params.is_file():
        try:
            p = json.loads(params.read_text()).get("kratos_python")
            if p:
                cands.append(p)
        except (OSError, json.JSONDecodeError):
            pass
    cands += ["/usr/bin/python3", shutil.which("python3") or ""]
    need = ("import precice, KratosMultiphysics, "
            "KratosMultiphysics.ConvectionDiffusionApplication; "
            "print('gate-ok')")
    env = {**os.environ, "PYTHONPATH": str(shim),
           "LD_LIBRARY_PATH": lib_dir + ":" +
           os.environ.get("LD_LIBRARY_PATH", "")}
    tried = []
    for c in cands:
        if not c or not (Path(c).exists() or shutil.which(c)):
            continue
        try:
            r = subprocess.run([c, "-c", need], capture_output=True, text=True,
                               timeout=600, env=env)
        except (OSError, subprocess.SubprocessError) as e:
            tried.append(f"{c}: {e}")
            continue
        if r.returncode == 0 and "gate-ok" in (r.stdout or ""):
            return c
        tried.append(f"{c}: {(r.stderr or '').strip().splitlines()[-1:]}")
    raise L.Absent("no interpreter on this host can import preCICE together "
                   "with Kratos ConvectionDiffusion: " + " | ".join(tried))


def dealii_precice_exe() -> Path:
    """Build the SHIPPED deal.II preCICE participant, once, and cache it.

    This is the claim the deal.II payload makes and nothing tested: a preCICE
    participant for a code with no Python API is a COMPILED EXECUTABLE that
    links libprecice, built through CMake with deal_ii_setup_target plus a
    target_link_libraries against libprecice.
    """
    from core.precice_config import PRECICE_LIB_DIR
    if not shutil.which("cmake"):
        raise L.Absent("cmake is not on PATH; cannot build the deal.II "
                       "preCICE participant")
    cand = [os.environ.get("DEAL_II_DIR", ""),
            str(Path.home() / "dealii" / "build"),
            str(Path.home() / "dealii"), "/usr/local", "/usr"]
    root = next((c for c in cand if c and
                 (Path(c) / "lib/cmake/deal.II/deal.IIConfig.cmake").is_file()),
                None)
    if not root:
        raise L.Absent("no deal.II install tree with "
                       "lib/cmake/deal.II/deal.IIConfig.cmake")
    build = (Path(os.environ.get("TMPDIR", "/tmp")) /
             "oasis_dealii_precice_participant_build")
    exe = build / "precice_heat_dealii"
    if exe.is_file():
        return exe
    build.mkdir(parents=True, exist_ok=True)
    precice_root = str(Path(PRECICE_LIB_DIR).parent)
    for cmd in (["cmake", "-S", str(L.PARTICIPANT_DIR), "-B", str(build),
                 f"-DDEAL_II_DIR={root}", "-DCMAKE_BUILD_TYPE=Release",
                 f"-DPRECICE_ROOT={precice_root}"],
                ["make", "-C", str(build), "-j3", "precice_heat_dealii"]):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            raise L.Absent(
                "the shipped deal.II preCICE participant did not build: "
                + " ".join(cmd) + " -> "
                + (r.stderr or r.stdout or "")[-400:])
    if not exe.is_file():
        raise L.Absent("the deal.II build produced no precice_heat_dealii")
    return exe


def site_of(python: str) -> str:
    """That interpreter's own site-packages — the thing that decides what it
    can import, and therefore what "its own interpreter" means here."""
    r = subprocess.run(
        [python, "-c",
         "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True, timeout=300)
    return (r.stdout or "").strip() or f"<unreadable:{python}>"


def links_libprecice(exe: Path) -> bool:
    """The deal.II participant is only a preCICE participant if it really links
    libprecice — the exact thing a reviewer greps for and did not find."""
    r = subprocess.run(["ldd", str(exe)], capture_output=True, text=True,
                       timeout=120)
    return "libprecice.so" in (r.stdout or "")


# ── one coupling ───────────────────────────────────────────────────────────

def run_pair(tag: str, right_text: str, right_py: str, extra_env: dict | None,
             aux: list[tuple[str, Path]] | None = None) -> None:
    work = L.workroot(f"precice_{tag}")
    print(f"--- {tag}: work_dir={work}")
    left = _SKFEM_LEFT.replace("EXPORT_SIGN = 1.0",
                               f"EXPORT_SIGN = {EXPORT_SIGN}")
    if "EXPORT_SIGN = %s" % EXPORT_SIGN not in left:
        raise AssertionError("the export-sign substitution matched nothing")
    (work / "participant_left.py").write_text(left)
    (work / "participant_right.py").write_text(right_text)
    for name, src in (aux or []):
        shutil.copy(src, work / name)

    py_left = L.interpreter("skfem")
    print(f"{tag}_left_interpreter={py_left}")
    print(f"{tag}_right_interpreter={right_py}")
    # Compared by SITE-PACKAGES, not by resolved path. `Path.resolve()`
    # collapses a venv onto the base interpreter it symlinks to, so it calls
    # two environments with completely different packages "the same" — which is
    # the opposite of what "runs in its own interpreter" means.
    sites = tuple(site_of(p) for p in (py_left, right_py))
    print(f"{tag}_site_packages={sites[0]} | {sites[1]}")
    print(f"{tag}_interpreters_distinct="
          f"{'no' if sites[0] == sites[1] else 'yes'}")

    participants = [
        {"name": "Left", "mesh": "Left-Mesh", "writes": ["Heat-Flux"],
         "reads": ["Temperature"],
         "command": [py_left, "participant_left.py"]},
        {"name": "Right", "mesh": "Right-Mesh", "writes": ["Temperature"],
         "reads": ["Heat-Flux"],
         "command": [right_py, "participant_right.py"]},
    ]
    data = [{"name": "Temperature", "type": "scalar"},
            {"name": "Heat-Flux", "type": "scalar"}]
    # exchanges[0] MUST be the field written by the SECOND participant: OASiS
    # makes it both the convergence measure and the acceleration datum, and
    # preCICE allows only second-to-first data there.
    exchanges = [{"data": "Temperature", "from": "Right", "to": "Left"},
                 {"data": "Heat-Flux", "from": "Left", "to": "Right"}]

    args = {"participants": json.dumps(participants), "data": json.dumps(data),
            "exchanges": json.dumps(exchanges), "work_dir": str(work),
            "scheme": SCHEME, "dimensions": 2, "max_time": MAX_TIME,
            "time_window": TIME_WINDOW, "timeout": 1800,
            "critic_approved": True}
    if extra_env:
        args["extra_env"] = json.dumps(extra_env)
    raw = call_tool("couple_precice", args)
    try:
        res = json.loads(raw)
    except json.JSONDecodeError:
        raise AssertionError(f"couple_precice returned non-JSON: {raw[:400]}")
    check_values(tag, res, work)


def check_values(tag: str, res: dict, work: Path) -> None:
    rcs = res.get("returncodes") or {}
    print(f"{tag}_returncodes={json.dumps(rcs, sort_keys=True)}")
    ok_rc = bool(rcs) and all(int(v) == 0 for v in rcs.values())
    if not verdict(ok_rc, f"{tag}_all_participant_returncodes_zero",
                   str(res.get("error"))[:300]):
        _dump_logs(res)
    if not verdict(bool(res.get("converged")), f"{tag}_coupled_run_converged",
                   str(res.get("error"))[:300]):
        _dump_logs(res)

    # From here on the fixture stops trusting the tool and reads what the two
    # codes actually exchanged.
    ex = {}
    for side, name in (("left", "Left"), ("right", "Right")):
        f = work / f"{side}_interface.json"
        if not f.is_file():
            L.check(False, f"{tag}_{name}_wrote_no_interface_file",
                    f"{f} is missing, so the participant did not finish its "
                    f"coupling loop and there is nothing to check")
            _dump_logs(res)
            return
        ex[side] = json.loads(f.read_text())

    nl, nr = len(ex["left"]["values"]), len(ex["right"]["values"])
    print(f"{tag}_interface_points_left_right={nl}/{nr}")
    verdict(nl != nr, f"{tag}_interface_meshes_non_matching",
            f"both sides used {nl} interface points, so preCICE's mapping "
            f"between NON-matching surfaces was never exercised")

    subs = (int(ex["left"]["sub_iterations"]),
            int(ex["right"]["sub_iterations"]))
    print(f"{tag}_sub_iterations_left_right={subs[0]}/{subs[1]}")
    verdict(min(subs) > N_WINDOWS, f"{tag}_implicit_subiteration_happened",
            f"{subs} solves over {N_WINDOWS} time windows — an implicit "
            f"scheme that never sub-iterates is an explicit one")

    for side in ("left", "right"):
        lo, hi = spans(ex[side]["values"])
        mid = 0.5 * (lo + hi)
        err = abs(mid - T_EXACT)
        print(f"{tag}_{side}_T={mid:.10g} err={err:.3e} span={hi - lo:.3e}")
        verdict(err <= T_ATOL, f"{tag}_{side}_interface_T_matches_closed_form",
                f"|{mid:.10g} - {T_EXACT:.10g}| = {err:.6e} K > {T_ATOL:.1e}")
        verdict(hi - lo <= UNIFORM_ATOL,
                f"{tag}_{side}_interface_T_uniform_in_y",
                f"the exact interface temperature has no y-variation; this one "
                f"spans {hi - lo:.3e} K")

    # The two sides export with respect to their OWN outward normals, which are
    # anti-parallel — so the right side's number must be the NEGATIVE of the
    # left's. A check on magnitudes would pass a sign error.
    for side, sign in (("left", +1.0), ("right", -1.0)):
        lo, hi = spans(ex[side]["normal_fluxes"])
        mid = 0.5 * (lo + hi)
        err = abs(mid - sign * Q_EXACT)
        print(f"{tag}_{side}_q={mid:.10g} err={err:.3e}")
        verdict(err <= Q_ATOL, f"{tag}_{side}_interface_q_matches_closed_form",
                f"|{mid:.10g} - {sign * Q_EXACT:.10g}| = {err:.6e} W/m^2 > "
                f"{Q_ATOL:.1e}")

    net_l, net_r = net_flux(ex["left"]), net_flux(ex["right"])
    rel = abs(net_l + net_r) / max(abs(net_l), abs(net_r), 1e-30)
    print(f"{tag}_flux_balance_rel={rel:.3e}")
    verdict(rel <= BALANCE_RTOL, f"{tag}_interface_flux_balanced",
            f"net(Left)={net_l:.6e} net(Right)={net_r:.6e} — what leaves one "
            f"subdomain must enter the other")


def _dump_logs(res: dict) -> None:
    for name, tail in (res.get("logs") or {}).items():
        print(f"--- participant log tail [{name}] ---")
        print(str(tail)[-1500:])


# ── the fixture body ───────────────────────────────────────────────────────

def body() -> None:
    L.require_available("skfem", "ngsolve", "dune", "dealii")

    from core.precice_config import PRECICE_LIB_DIR, check_precice_available
    if not Path(PRECICE_LIB_DIR).is_dir():
        raise L.Absent(f"no preCICE lib directory at {PRECICE_LIB_DIR}")
    print(f"precice_lib_dir={PRECICE_LIB_DIR}")
    ok, msg = check_precice_available()
    if not ok:
        raise L.Absent(f"preCICE is not usable from this install ({msg[:200]})")

    print(f"closed_form_T_iface={T_EXACT:.10g}")
    print(f"closed_form_q_iface={Q_EXACT:.10g}")

    done = 0

    # 1. NGSolve — the OASiS venv, the same interpreter scikit-fem uses. The
    #    payload's claim is that `import precice` works in the interpreter that
    #    carries NGSolve; what was missing is that a coupling ever RAN.
    run_pair("ngsolve", _RIGHT_NGSOLVE, L.interpreter("ngsolve"), None)
    done += 1

    # 2. DUNE-fem — its own conda env, reached through the PYTHONPATH recipe
    #    its payload prescribes.
    own_site = sysconfig.get_paths()["purelib"]
    run_pair("dune", _RIGHT_DUNE, L.interpreter("dune"),
             {"PYTHONPATH": own_site})
    done += 1

    # 3. deal.II — the COMPILED C++ participant, built here from the shipped
    #    source, and checked to really link libprecice before it is run.
    exe = dealii_precice_exe()
    print(f"dealii_participant_exe={exe}")
    verdict(links_libprecice(exe), "dealii_participant_links_libprecice",
            f"ldd {exe} names no libprecice.so — then it is not a preCICE "
            f"participant, whatever its verdict says")
    run_pair("dealii", _RIGHT_DEALII, L.interpreter("dealii"), None,
             aux=[("precice_heat_dealii", exe)])
    done += 1

    # 4. Kratos — its own interpreter, plus the SUBSET shim that is what
    #    actually makes pyprecice importable there.
    shim = precice_shim(Path(L.workroot("precice_shim")) / "shim")
    print(f"kratos_precice_shim={shim}")
    try:
        kpy = kratos_interpreter(shim, PRECICE_LIB_DIR)
    except L.Absent as e:
        L.check(False, "kratos_precice_and_backend_modules_coexist_violated",
                str(e))
        raise
    print(f"kratos_interpreter={kpy}")
    verdict(True, "kratos_precice_and_backend_modules_coexist")
    run_pair("kratos", _RIGHT_KRATOS, kpy, {"PYTHONPATH": str(shim)})
    done += 1

    print(f"backends_coupled={done}")


L.main(body)
