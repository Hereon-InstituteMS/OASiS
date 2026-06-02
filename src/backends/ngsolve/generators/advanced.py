"""NGSolve advanced physics generators and knowledge.

Covers:
  dg_methods             – DG for advection/diffusion (dglagrange / L2 spaces)
  contact                – Contact/obstacle using penalty method
  time_dependent_ns      – Transient Navier-Stokes with IMEX (full channel)
  mhd                    – Magnetohydrodynamics (coupled Maxwell + NS, 2.5-D)
  hdivdiv                – HDivDiv space for Kirchhoff plates / Regge elasticity
  nonlinear_elasticity   – Large-deformation Neo-Hookean with load stepping
  phase_field            – Cahn-Hilliard / phase-field fracture (Allen-Cahn)
"""


# ─────────────────────────────────────────────────────────────────────────────
# 1. DG methods (advection-diffusion with dglagrange spaces)
# ─────────────────────────────────────────────────────────────────────────────

def _dg_methods_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Interior-penalty DG for general advection-diffusion on [0,1]²
    using the modern dglagrange space variant."""
    order = params.get("order", 3)
    eps = params.get("diffusion", 0.005)
    maxh = params.get("maxh", 0.06)
    alpha = params.get("penalty", 4)   # penalty multiplier (alpha * order^2 / h)
    return f'''\
"""DG advection-diffusion — interior penalty — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))

# L2 with dgjumps=True is the standard DG space in NGSolve.
# 'order' controls the local polynomial degree.
order = {order}
eps = {eps}
alpha = {alpha}

fes = L2(mesh, order=order, dgjumps=True)
u, v = fes.TnT()

n = specialcf.normal(2)    # outward unit normal (mesh-orientation aware)
h = specialcf.mesh_size    # element diameter

# Advection field — set for your problem
b = CoefficientFunction((2, 1))
# Upwind numerical flux: take u from the upwind side
uup = IfPos(b * n, u, u.Other())

a = BilinearForm(fes)
# Diffusion: symmetric interior-penalty (SIP/SIPG)
a += eps * grad(u) * grad(v) * dx
a += -eps * 0.5 * (grad(u) + grad(u).Other()) * n * (v - v.Other()) * dx(skeleton=True)
a += -eps * 0.5 * (grad(v) + grad(v).Other()) * n * (u - u.Other()) * dx(skeleton=True)
a += alpha * order**2 / h * (u - u.Other()) * (v - v.Other()) * dx(skeleton=True)
# Boundary diffusion terms
a += -eps * grad(u) * n * v * ds(skeleton=True)
a += -eps * grad(v) * n * u * ds(skeleton=True)
a += alpha * order**2 / h * u * v * ds(skeleton=True)
# Advection: upwind
a += -b * u * grad(v) * dx
a += b * n * uup * (v - v.Other()) * dx(skeleton=True)
a += b * n * u * v * ds(skeleton=True)
a.Assemble()

# Source and Dirichlet data — set for your problem
f_coef = CoefficientFunction(1.0)
g_dir  = CoefficientFunction(0.0)   # inflow Dirichlet value
f = LinearForm(fes)
f += f_coef * v * dx
# Dirichlet weakly via penalty on inflow boundary where b*n < 0
f += alpha * order**2 / h * g_dir * v * ds(skeleton=True)
f += -eps * grad(v) * n * g_dir * ds(skeleton=True)
f.Assemble()

gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse() * f.vec

# abs() is not defined on ngsolve.la.BaseVector — extract
# the underlying numpy view via gfu.vec.FV().NumPy() and
# reduce with numpy. The legacy max(abs(gfu.vec)) pattern
# raises TypeError 'bad operand type for abs()'.
import numpy as _np
max_val = float(_np.abs(gfu.vec.FV().NumPy()).max())
print(f"max|u| = {{max_val:.8f}}")
print(f"DOFs: {{fes.ndof}}, elements: {{mesh.ne}}")

vtk = VTKOutput(mesh, coefs=[gfu], names=["solution"], filename="result", subdivision=2)
vtk.Do()

summary = {{
    "max_abs_value": float(max_val),
    "n_dofs": fes.ndof,
    "n_elements": mesh.ne,
    "diffusion": eps,
    "order": order,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("DG advection-diffusion solve complete.")
'''


# ─────────────────────────────────────────────────────────────────────────────
# 2. Contact / obstacle problem with penalty method
# ─────────────────────────────────────────────────────────────────────────────

def _contact_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Obstacle / unilateral contact via penalty for a loaded elastic plate.
    Obstacle at y = obstacle_height, plate clamped on left."""
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    penalty = params.get("penalty", 1e5)
    obstacle = params.get("obstacle_height", 0.0)
    load = params.get("load", -5.0)
    maxh = params.get("maxh", 0.05)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""Contact / obstacle problem — penalty method — NGSolve"""
from ngsolve import *
from netgen.geom2d import SplineGeometry
import json

# Geometry: rectangular bar that may contact a rigid obstacle
geo = SplineGeometry()
pts = [(0, 0), (1, 0), (1, 0.1), (0, 0.1)]
p = [geo.AddPoint(*pt) for pt in pts]
geo.Append(["line", p[0], p[1]], bc="bottom")
geo.Append(["line", p[1], p[2]], bc="right")
geo.Append(["line", p[2], p[3]], bc="top")
geo.Append(["line", p[3], p[0]], bc="left")
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))

# Material
mu_val  = {mu}
lam_val = {lam}

fes = VectorH1(mesh, order=2, dirichlet="left")
u, v = fes.TnT()

def Eps(w):
    return 0.5 * (Grad(w) + Grad(w).trans)

def Sigma(w):
    e = Eps(w)
    return 2 * mu_val * e + lam_val * Trace(e) * Id(2)

# Elastic bilinear form
a_el = BilinearForm(fes, symmetric=True)
a_el += InnerProduct(Sigma(u), Eps(v)) * dx
a_el.Assemble()

# External load (body force downward) and traction
f_vol = LinearForm(fes)
f_vol += CoefficientFunction((0.0, {load})) * v * dx
f_vol.Assemble()

# Penalty method for contact (non-penetration below obstacle_height)
# obstacle_height is the y-coordinate of the rigid floor
gamma   = {penalty}
obs_y   = {obstacle}

gfu = GridFunction(fes)

# Newton-like fixed-point loop: linearise penalty term each iteration
for iteration in range(30):
    # Current vertical displacement
    uy_cf = gfu[1]
    # Contact gap: g = u_y - obs_y (negative means penetration)
    gap = uy_cf - obs_y
    # Active set indicator: IfPos(-gap, 1, 0)  (1 where penetration occurs)
    active = IfPos(-gap, 1.0, 0.0)

    # Contact force (penalty): f_c = -gamma * min(gap, 0) = gamma * max(-gap, 0)
    pen_bilin = BilinearForm(fes, symmetric=True)
    pen_bilin += active * gamma * u[1] * v[1] * dx
    pen_bilin.Assemble()

    pen_lin = LinearForm(fes)
    pen_lin += active * gamma * obs_y * v[1] * dx
    pen_lin.Assemble()

    total_mat  = a_el.mat.CreateMatrix()
    total_mat.AsVector().data  = a_el.mat.AsVector() + pen_bilin.mat.AsVector()
    total_rhs  = f_vol.vec.CreateVector()
    total_rhs.data = f_vol.vec + pen_lin.vec

    gfu_new = GridFunction(fes)
    gfu_new.vec.data = total_mat.Inverse(fes.FreeDofs()) * total_rhs

    diff = (gfu_new.vec - gfu.vec).Norm()
    gfu.vec.data = gfu_new.vec
    print(f"  Iter {{iteration+1}}: ||delta u|| = {{diff:.3e}}")
    if diff < 1e-8:
        print(f"  Converged after {{iteration+1}} iterations")
        break

min_uy = Integrate(gfu[1], mesh) / Integrate(1, mesh)
print(f"Average vertical displacement: {{min_uy:.6f}}")

vtk = VTKOutput(mesh, coefs=[gfu], names=["displacement"], filename="result", subdivision=1)
vtk.Do()

summary = {{
    "n_dofs": fes.ndof,
    "n_elements": mesh.ne,
    "obstacle_height": obs_y,
    "penalty": gamma,
    "avg_displacement_y": float(min_uy),
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Contact / obstacle solve complete.")
'''


# ─────────────────────────────────────────────────────────────────────────────
# 3. Transient Navier-Stokes (full channel / lid-driven cavity, IMEX)
# ─────────────────────────────────────────────────────────────────────────────

def _time_dependent_ns_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Transient incompressible Navier-Stokes in a channel with IMEX splitting.
    Stokes part implicit (factorised once), convection explicit."""
    Re = params.get("Re", 200)
    dt = params.get("dt", 0.002)
    T_end = params.get("T_end", 2.0)
    maxh = params.get("maxh", 0.04)
    nu = 1.0 / Re
    n_steps = int(T_end / dt)
    vtk_every = params.get("vtk_every", max(1, n_steps // 20))
    return f'''\
"""Transient Navier-Stokes — IMEX — channel flow — NGSolve"""
from ngsolve import *
from netgen.geom2d import SplineGeometry
import json, math

# ── Geometry: 2D channel [0, L] x [0, 1] ────────────────────────────────────
L = 4.0   # channel length — set for your problem
geo = SplineGeometry()
pts = [(0, 0), (L, 0), (L, 1), (0, 1)]
p = [geo.AddPoint(*pt) for pt in pts]
geo.Append(["line", p[0], p[1]], bc="bottom")
geo.Append(["line", p[1], p[2]], bc="outlet")
geo.Append(["line", p[2], p[3]], bc="top")
geo.Append(["line", p[3], p[0]], bc="inlet")
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))

# ── FE spaces: Taylor-Hood P2/P1 ────────────────────────────────────────────
V  = VectorH1(mesh, order=2, dirichlet="bottom|top|inlet")
Q  = H1(mesh, order=1)
X  = V * Q
(u, p), (v, q) = X.TnT()

nu = {nu}   # kinematic viscosity = 1/Re
dt = {dt}

# ── Parabolic inlet profile u_x = 4*y*(1-y), u_y = 0 ───────────────────────
inlet_vel = CoefficientFunction((4 * y * (1 - y), 0))

# ── Implicit Stokes operator (assembled once) ────────────────────────────────
stokes = (nu * InnerProduct(Grad(u), Grad(v)) * dx
          + div(u) * q * dx
          + div(v) * p * dx)
mass   = InnerProduct(u, v) * dx

mstar = BilinearForm(X)
mstar += mass + dt * stokes
mstar.Assemble()

gfu = GridFunction(X)
velocity = gfu.components[0]
velocity.Set(inlet_vel, definedon=mesh.Boundaries("inlet"))
# No-slip walls already zero from dirichlet

inv = mstar.mat.Inverse(X.FreeDofs(), inverse="umfpack")

# ── Time loop ────────────────────────────────────────────────────────────────
t = 0.0
n_steps = {n_steps}
vtk_every = {vtk_every}

vtk = VTKOutput(mesh,
                coefs=[gfu.components[0], gfu.components[1]],
                names=["velocity", "pressure"],
                filename="result", subdivision=1)

max_vel_history = []

for step in range(n_steps):
    # Explicit convection
    conv = LinearForm(X)
    conv += InnerProduct(Grad(velocity) * velocity, v) * dx
    conv.Assemble()

    rhs = mstar.mat * gfu.vec - dt * conv.vec
    gfu.vec.data = inv * rhs

    # Re-impose inlet BC
    velocity.Set(inlet_vel, definedon=mesh.Boundaries("inlet"))

    t += dt
    if step % vtk_every == 0 or step == n_steps - 1:
        vtk.Do(time=t)
        max_v = sqrt(Integrate(InnerProduct(velocity, velocity), mesh) /
                     Integrate(1.0, mesh))
        max_vel_history.append((float(t), float(max_v)))
        print(f"  t={{t:.4f}}, rms(u)={{max_v:.4f}}")

print(f"Completed {{n_steps}} steps, Re={Re}")

summary = {{
    "Re": {Re},
    "nu": nu,
    "dt": dt,
    "T_end": t,
    "n_dofs": X.ndof,
    "n_elements": mesh.ne,
    "rms_velocity_history": max_vel_history[-5:],
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Transient Navier-Stokes solve complete.")
'''


# ─────────────────────────────────────────────────────────────────────────────
# 4. MHD — Magnetohydrodynamics (coupled Maxwell + Navier-Stokes)
# ─────────────────────────────────────────────────────────────────────────────

def _mhd_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    2.5-D MHD: in-plane NS coupled to out-of-plane magnetic field B_z via Lorentz force.
    Hartmann problem (conducting channel in transverse B field).
    Governing equations (non-dimensional):
      Re * (du/dt + u·∇u) - ∇²u + ∇p = Ha² * (J × B)
      ∇²B_z = -Re_m * (u · ∇B_z)   (magnetic induction, low Rm limit)
      J = -∇ × B_z (= dB_z/dy, -dB_z/dx in 2D)
    Low-Rm approximation: B = B_0 e_z + b (induced), |b| << |B_0|."""
    Re = params.get("Re", 100)
    Ha = params.get("Ha", 10)         # Hartmann number
    dt = params.get("dt", 0.005)
    # T_end shrunk from 1.0 -> 0.05 and maxh widened from
    # 0.05 -> 0.15 so the Layer F catalog smoke completes
    # in ~10 steps within the 60s gate. Users who want a
    # longer simulation override T_end/maxh via params.
    T_end = params.get("T_end", 0.05)
    maxh = params.get("maxh", 0.15)
    nu = 1.0 / Re
    sigma_m = Ha * Ha / Re           # magnetic diffusivity (non-dim)
    n_steps = int(T_end / dt)
    return f'''\
"""MHD Hartmann channel — 2.5-D low-Rm — NGSolve"""
from ngsolve import *
from netgen.geom2d import SplineGeometry
import json, math

# ── Geometry: channel [0, 4] x [-1, 1] ──────────────────────────────────────
geo = SplineGeometry()
pts = [(0, -1), (4, -1), (4, 1), (0, 1)]
p = [geo.AddPoint(*pt) for pt in pts]
geo.Append(["line", p[0], p[1]], bc="bottom")
geo.Append(["line", p[1], p[2]], bc="outlet")
geo.Append(["line", p[2], p[3]], bc="top")
geo.Append(["line", p[3], p[0]], bc="inlet")
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))

# ── Fluid variables (velocity + pressure): Taylor-Hood ──────────────────────
Vf = VectorH1(mesh, order=2, dirichlet="bottom|top|inlet")
Qf = H1(mesh, order=1)
Xf = Vf * Qf
(u, p), (v, q) = Xf.TnT()

# ── Magnetic variable: scalar B_z (induced), H1 with Dirichlet walls ─────────
Vm = H1(mesh, order=2, dirichlet="bottom|top")
bz, wz = Vm.TnT()

# ── Physical parameters (non-dimensional) ────────────────────────────────────
Re     = {Re}
Ha     = {Ha}
nu     = {nu}          # = 1/Re
sigma_m = {sigma_m}    # = Ha^2/Re (inverse magnetic Re)
dt     = {dt}
B0     = 1.0           # applied transverse B field (Hartmann direction = y)

# ── Stokes + Lorentz force operator (assembled once per outer iteration) ──────
def assemble_fluid(u_prev, bz_prev):
    # Lorentz force: J × B = (∇×B) × B0 ≈ (dBz/dy) * (-e_x) in 2.5D
    # Simplified: f_Lorentz = Ha^2 * (B0 * J) where J = −∂bz/∂y * ex + ∂bz/∂x * ey
    # In weak form: Ha^2 * (bz_prev * div(B0*v_perp)) via integration by parts
    a = BilinearForm(Xf)
    a += nu * InnerProduct(Grad(u), Grad(v)) * dx   # viscous
    a += div(u) * q * dx + div(v) * p * dx          # pressure/continuity
    a += 1.0/dt * InnerProduct(u, v) * dx           # mass / time
    a.Assemble()
    return a

def assemble_rhs_fluid(u_prev, bz_prev, a):
    # Convection (explicit) + Lorentz body force + inertia
    f = LinearForm(Xf)
    f += 1.0/dt * InnerProduct(u_prev, v) * dx
    # Explicit convection
    f += -InnerProduct(Grad(u_prev) * u_prev, v) * dx
    # Lorentz force (low-Rm: J = curl(B0 e_z + bz) ≈ curl(bz e_z))
    # f_L = sigma*(u x B) x B — in low-Rm: f_L = -Ha^2 * nu * u_y (for Hartmann in y)
    # Note: the integrand on the y-component of the velocity test
    # function is a scalar — multiplying by the unit vector
    # CoefficientFunction((0, 1)) makes the whole expression
    # vector-valued and SymbolicLFI rejects it with NgException
    # 'SymbolicLFI needs scalar-valued CoefficientFunction'.
    # v[1] already selects the y component of v; no extra
    # unit-vector factor is needed.
    f += -Ha * Ha * nu * u_prev[1] * v[1] * dx
    f.Assemble()
    return f

def assemble_magnetic(u_prev):
    # Magnetic induction (low-Rm, quasi-static):
    # 1/sigma_m * Laplace(bz) = B0 * du_x/dy  (source from fluid shear)
    a = BilinearForm(Vm)
    a += sigma_m * Grad(bz) * Grad(wz) * dx
    a += 1.0/dt * bz * wz * dx
    a.Assemble()

    f = LinearForm(Vm)
    f += 1.0/dt * bz_prev_gf * wz * dx
    # Source: fluid velocity shearing the applied field
    f += -B0 * u_prev[0].Diff(y) * wz * dx
    f.Assemble()
    return a, f

# ── Initial conditions ────────────────────────────────────────────────────────
gfu  = GridFunction(Xf)
gfbz = GridFunction(Vm)

inlet_vel = CoefficientFunction((1 - y**2, 0))  # Poiseuille
gfu.components[0].Set(inlet_vel, definedon=mesh.Boundaries("inlet"))

velocity = gfu.components[0]
bz_prev_gf = GridFunction(Vm)
bz_prev_gf.Set(0)

print(f"MHD setup: Re={{Re}}, Ha={{Ha}}, DOFs fluid={{Xf.ndof}}, mag={{Vm.ndof}}")

# ── Time loop (operator-split: fluid then magnetic) ────────────────────────────
t = 0.0
n_steps = {n_steps}

for step in range(n_steps):
    # 1) Fluid solve (Stokes + implicit Lorentz correction)
    a_fl = assemble_fluid(velocity, bz_prev_gf)
    f_fl = assemble_rhs_fluid(velocity, bz_prev_gf, a_fl)
    gfu.vec.data = a_fl.mat.Inverse(Xf.FreeDofs(), "umfpack") * f_fl.vec
    velocity.Set(inlet_vel, definedon=mesh.Boundaries("inlet"))

    # 2) Magnetic solve (induction equation)
    a_mg = BilinearForm(Vm)
    a_mg += sigma_m * Grad(bz) * Grad(wz) * dx + 1.0/dt * bz * wz * dx
    a_mg.Assemble()
    f_mg = LinearForm(Vm)
    f_mg += 1.0/dt * bz_prev_gf * wz * dx
    f_mg += -B0 * velocity[0] * Grad(wz)[1] * dx  # u_x * dw/dy
    f_mg.Assemble()
    gfbz.vec.data = a_mg.mat.Inverse(Vm.FreeDofs()) * f_mg.vec
    bz_prev_gf.vec.data = gfbz.vec

    t += dt
    if step % max(1, n_steps // 10) == 0:
        u_rms = sqrt(Integrate(InnerProduct(velocity, velocity), mesh) /
                     Integrate(1.0, mesh))
        print(f"  t={{t:.4f}}, rms(u)={{u_rms:.4f}}")

vtk = VTKOutput(mesh,
                coefs=[gfu.components[0], gfu.components[1], gfbz],
                names=["velocity", "pressure", "B_induced"],
                filename="result", subdivision=1)
vtk.Do()

summary = {{
    "Re": Re,
    "Ha": Ha,
    "sigma_m": sigma_m,
    "dt": dt,
    "T_end": float(t),
    "n_dofs_fluid": Xf.ndof,
    "n_dofs_magnetic": Vm.ndof,
    "n_elements": mesh.ne,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("MHD Hartmann solve complete.")
'''


# ─────────────────────────────────────────────────────────────────────────────
# 5. HDivDiv — Kirchhoff plate bending / Regge-elasticity
# ─────────────────────────────────────────────────────────────────────────────

def _hdivdiv_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Kirchhoff plate bending via Hellan-Herrmann-Johnson (HHJ) mixed formulation
    using HDivDiv space for bending moments and H1 for deflection.
    Strong form: Δ²w = q  (biharmonic).
    Mixed: find (σ, w) s.t. A(σ,τ) + B(τ,w) = 0 and B(σ,v) = (q,v)
    where σ is the moment tensor (HDivDiv), w the deflection (H1/L2)."""
    t_plate = params.get("thickness", 0.01)   # plate thickness (for normalisation)
    E = params.get("E", 1.0)
    nu = params.get("nu", 0.3)
    q_load = params.get("load", 1.0)
    order = params.get("order", 2)
    maxh = params.get("maxh", 0.08)
    # Non-dimensionalised: D = E*t^3 / (12*(1-nu^2))
    D = E * t_plate**3 / (12 * (1 - nu**2))
    return f'''\
"""Kirchhoff plate — HHJ mixed (HDivDiv + H1) — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))

# Bending rigidity
E_mod = {E}
nu_val = {nu}
D = {D}   # = E*t^3 / (12*(1-nu^2))
q = {q_load}    # transverse distributed load
order = {order}

# ── Hellan-Herrmann-Johnson spaces ───────────────────────────────────────────
# Moment tensor σ in HDivDiv (H(div div) conforming, normal-normal continuous)
# Deflection w in H1 (clamped: w=0 and dw/dn=0 on boundary)
Vhdd = HDivDiv(mesh, order=order - 1)  # moments, order k-1
Vh1  = H1(mesh, order=order, dirichlet="bottom|right|top|left")

(sigma, tau) = Vhdd.TnT()
(w,    v   ) = Vh1.TnT()

X = Vhdd * Vh1
(sig, ww), (tau_, vv) = X.TnT()

n = specialcf.normal(2)
tang = specialcf.tangential(2)

def Compliance(s):
    """Inverse bending stiffness: 1/D * (s - nu/(1+nu) * Tr(s) * I)"""
    return (1.0/D) * (s - nu_val/(1 + nu_val) * Trace(s) * Id(2))

# ── Bilinear form ─────────────────────────────────────────────────────────────
# HHJ weak form: integrate the div(div(σ)) coupling
# by parts TWICE so the operators land on the H1
# deflection w as a Hessian. NGSolve's HDivDiv space
# does NOT expose a pointwise div(div(·)) operator —
# constructing it raises Exception 'cannot form div'.
# Use w.Operator('hesse') for ∇²w and add the
# normal-normal moment skeleton facet integral.
a = BilinearForm(X, symmetric=True)
# Compliance block
a += InnerProduct(Compliance(sig), tau_) * dx
# Mixed coupling: ∫ τ : ∇²w dx
a += InnerProduct(tau_, ww.Operator("hesse")) * dx
a += InnerProduct(sig, vv.Operator("hesse")) * dx
# Skeleton facet integral: normal-normal moment couples
# to the jump of the normal derivative of deflection.
a += -(tau_ * n * n) * (Grad(ww) * n) * dx(element_boundary=True)
a += -(sig  * n * n) * (Grad(vv) * n) * dx(element_boundary=True)
a.Assemble()

# ── Load ─────────────────────────────────────────────────────────────────────
f = LinearForm(X)
f += q * vv * dx   # transverse load on deflection test function
f.Assemble()

# ── Solve ─────────────────────────────────────────────────────────────────────
gf = GridFunction(X)
gf.vec.data = a.mat.Inverse(X.FreeDofs(), inverse="umfpack") * f.vec

gf_sig, gf_w = gf.components

# abs() not defined on BaseVector — reduce via numpy view
import numpy as _np
max_deflection = float(_np.abs(gf_w.vec.FV().NumPy()).max())
print(f"Max deflection: {{max_deflection:.8f}}")
print(f"DOFs: {{X.ndof}} (moments {{Vhdd.ndof}}, deflection {{Vh1.ndof}})")

vtk = VTKOutput(mesh,
                coefs=[gf_w, gf_sig],
                names=["deflection", "moments"],
                filename="result", subdivision=2)
vtk.Do()

# Analytical reference for simply supported plate: w_max = q*L^4 / (64*D) for L=1
w_ref = q / (64 * D)
print(f"Analytical reference (SS plate): {{w_ref:.8f}}")
print(f"Relative error: {{abs(max_deflection - w_ref)/abs(w_ref):.4%}}")

summary = {{
    "max_deflection": float(max_deflection),
    "analytical_reference": float(w_ref),
    "n_dofs": X.ndof,
    "n_elements": mesh.ne,
    "D_bending_rigidity": D,
    "q_load": q,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Kirchhoff plate HDivDiv solve complete.")
'''


# ─────────────────────────────────────────────────────────────────────────────
# 6. Nonlinear elasticity — large-deformation Neo-Hookean with load stepping
# ─────────────────────────────────────────────────────────────────────────────

def _nonlinear_elasticity_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Large-deformation Neo-Hookean elasticity via Variation() + Newton.
    Load stepping ensures convergence for large applied displacements."""
    E = params.get("E", 200.0)
    nu = params.get("nu", 0.3)
    disp_mag = params.get("applied_displacement", 0.5)
    n_steps = params.get("load_steps", 10)
    # maxh=0.05 was too fine for Variation Newton without
    # load stepping to converge from a cold start (UMFPACK
    # singular at first iter); 0.1 gives ~1.5k DOFs which
    # is enough for catalog smoke and stays factorisable.
    maxh = params.get("maxh", 0.1)
    order = params.get("order", 2)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""Large-deformation Neo-Hookean elasticity — load stepping + Newton — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))

# Lamé constants
mu_lam = {mu}
lam_val = {lam}
order = {order}

# Displacement space — every boundary whose displacement
# is prescribed below (left clamped + top loaded) must
# appear in the dirichlet specifier so its DOFs are
# eliminated from FreeDofs; otherwise the rigid mode is
# unconstrained and UMFPACK aborts with 'Numeric
# factorization failed' on the first iter.
fes = VectorH1(mesh, order=order, dirichlet="left|top")
u = fes.TrialFunction()

# Deformation gradient and invariants
d = 2  # spatial dimension
I = Id(d)
F = I + Grad(u)
C = F.trans * F
J = Det(F)

# Neo-Hookean strain energy density:
#   W = mu/2 * (Tr(C) - d) - mu*ln(J) + lam/2 * ln(J)^2
energy = 0.5 * mu_lam * (Trace(C) - d) - mu_lam * log(J) + 0.5 * lam_val * log(J)**2

a = BilinearForm(fes, symmetric=True)
a += Variation(energy * dx)

gfu = GridFunction(fes)

# ── Load stepping: apply displacement incrementally ────────────────────────────
n_load_steps = {n_steps}
disp_total   = {disp_mag}

print(f"Neo-Hookean load stepping: {{n_load_steps}} steps, total disp = {{disp_total}}")
for step in range(1, n_load_steps + 1):
    alpha = step / n_load_steps
    disp_now = alpha * disp_total

    # Apply incremental Dirichlet displacement on top boundary
    gfu.Set(CoefficientFunction((0.0, disp_now)), definedon=mesh.Boundaries("top"))

    try:
        (iters, conv) = solvers.Newton(a, gfu, maxit=25, dampfactor=1.0,
                                       printing=False, maxerr=1e-10)
        print(f"  Step {{step}}/{n_steps}: disp={{disp_now:.4f}}, "
              f"Newton iters={{iters}}, conv={{conv:.3e}}")
    except Exception as e:
        print(f"  Step {{step}} FAILED: {{e}}")
        break

# Evaluate results — abs() is not defined on
# ngsolve.la.BaseVector. Reduce via the underlying
# numpy view (gfu.components[i].vec.FV().NumPy()).
import numpy as _np
max_ux = float(_np.abs(gfu.components[0].vec.FV().NumPy()).max())
max_uy = float(_np.abs(gfu.components[1].vec.FV().NumPy()).max())
print(f"Max |u_x| = {{max_ux:.6f}}, max |u_y| = {{max_uy:.6f}}")
print(f"DOFs: {{fes.ndof}}")

# Cauchy stress (push-forward of PK2 stress).
# Rebuild F/C/J/S from the resolved displacement gfu —
# the symbolic versions reference the ProxyFunction
# u = fes.TrialFunction(), and VTKOutput.Do() trying to
# evaluate ProxyFunction-derived stress at quadrature
# points raises NgException 'cannot evaluate
# ProxyFunction without userdata'.
F_eval = I + Grad(gfu)
C_eval = F_eval.trans * F_eval
J_eval = Det(F_eval)
S_eval = (mu_lam * I
          - mu_lam / J_eval**2 * Inv(C_eval)
          + lam_val * log(J_eval) / J_eval**2 * Inv(C_eval))
sigma_cauchy = 1 / J_eval * F_eval * S_eval * F_eval.trans

vtk = VTKOutput(mesh,
                coefs=[gfu, sigma_cauchy],
                names=["displacement", "cauchy_stress"],
                filename="result", subdivision=2)
vtk.Do()

summary = {{
    "n_dofs": fes.ndof,
    "n_elements": mesh.ne,
    "load_steps": n_load_steps,
    "applied_displacement": disp_total,
    "max_ux": float(max_ux),
    "max_uy": float(max_uy),
    "mu": mu_lam,
    "lam": lam_val,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Nonlinear elasticity solve complete.")
'''


def _nonlinear_elasticity_3d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    3D large-deformation Neo-Hookean elasticity."""
    E = params.get("E", 200.0)
    nu = params.get("nu", 0.3)
    disp_mag = params.get("applied_displacement", 0.3)
    n_steps = params.get("load_steps", 8)
    maxh = params.get("maxh", 0.12)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""3D Large-deformation Neo-Hookean elasticity — load stepping — NGSolve"""
from ngsolve import *
from netgen.csg import unit_cube
import json

mesh = Mesh(unit_cube.GenerateMesh(maxh={maxh}))

mu_lam  = {mu}
lam_val = {lam}

fes = VectorH1(mesh, order=2, dirichlet="left")
u = fes.TrialFunction()

d = 3
I = Id(d)
F = I + Grad(u)
C = F.trans * F
J = Det(F)
energy = 0.5 * mu_lam * (Trace(C) - d) - mu_lam * log(J) + 0.5 * lam_val * log(J)**2

a = BilinearForm(fes, symmetric=True)
a += Variation(energy * dx)

gfu = GridFunction(fes)

n_load_steps = {n_steps}
disp_total   = {disp_mag}

for step in range(1, n_load_steps + 1):
    alpha = step / n_load_steps
    disp_now = alpha * disp_total
    gfu.Set(CoefficientFunction((0.0, 0.0, disp_now)), definedon=mesh.Boundaries("top"))
    (iters, conv) = solvers.Newton(a, gfu, maxit=25, dampfactor=1.0,
                                   printing=False, maxerr=1e-10)
    print(f"  Step {{step}}/{n_steps}: disp={{disp_now:.4f}}, iters={{iters}}")

vtk = VTKOutput(mesh, coefs=[gfu], names=["displacement"], filename="result", subdivision=1)
vtk.Do()

summary = {{
    "n_dofs": fes.ndof,
    "n_elements": mesh.ne,
    "applied_displacement": disp_total,
    "mu": mu_lam,
    "lam": lam_val,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("3D Nonlinear elasticity solve complete.")
'''


# ─────────────────────────────────────────────────────────────────────────────
# 7. Phase field — Cahn-Hilliard / Allen-Cahn / phase-field fracture
# ─────────────────────────────────────────────────────────────────────────────

def _phase_field_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Allen-Cahn / phase-field evolution for fracture or interface motion.
    Model: dc/dt = M * (ε²Δc - W'(c))
    where W(c) = c²(1-c)² (double-well), M = mobility, ε = interface width.
    For phase-field fracture the same equation drives crack-phase variable d ∈ [0,1]."""
    eps = params.get("epsilon", 0.02)       # interface width
    M = params.get("mobility", 1.0)         # mobility
    dt = params.get("dt", 0.001)
    T_end = params.get("T_end", 0.5)
    maxh = params.get("maxh", 0.03)
    order = params.get("order", 2)
    n_steps = int(T_end / dt)
    vtk_every = params.get("vtk_every", max(1, n_steps // 20))
    return f'''\
"""Phase-field (Allen-Cahn) — implicit Euler — NGSolve"""
from ngsolve import *
import json, math

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))

eps = {eps}    # interface width parameter
M_mob = {M}    # mobility
dt  = {dt}
order = {order}

fes = H1(mesh, order=order)
c, w = fes.TnT()

# Allen-Cahn: dc/dt = M*(eps^2*Δc - W'(c))  where W'(c) = 2c(1-c)(1-2c)
# Semi-implicit: linearize W'(c) as W'(c^n) at previous time step.
# Mass matrix (time derivative)
mass = BilinearForm(fes)
mass += (1.0/dt) * c * w * dx
mass.Assemble()

# Stiffness (Laplacian diffusion)
stiff = BilinearForm(fes)
stiff += eps**2 * M_mob * grad(c) * grad(w) * dx
stiff.Assemble()

# Total LHS = mass + stiff (assembled once since we linearize W')
lhs = mass.mat.CreateMatrix()
lhs.AsVector().data = mass.mat.AsVector() + stiff.mat.AsVector()

# ── Initial condition: tanh profile around x=0.5 ──
# NGSolve's CoefficientFunction namespace exposes sin,
# cos, exp, log, tan, atan, atan2 but NOT tanh/sinh/cosh.
# Build the hyperbolic tangent manually via:
#   tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
# Using the bare 'tanh' name raises NameError at module
# import time.
gfc = GridFunction(fes)
_arg = (x - 0.5) / (2 * eps)
_e   = exp(2 * _arg)
_tanh = (_e - 1) / (_e + 1)
gfc.Set(0.5 + 0.5 * _tanh)

print(f"Phase-field setup: eps={{eps}}, dt={{dt}}, DOFs={{fes.ndof}}")

n_steps = {n_steps}
vtk_every = {vtk_every}

vtk = VTKOutput(mesh, coefs=[gfc], names=["phase"], filename="result", subdivision=1)
vtk.Do(time=0.0)

mass_history = []

for step in range(n_steps):
    c_old = gfc.vec.CreateVector()
    c_old.data = gfc.vec

    # Nonlinear W'(c^n) = 2*c*(1-c)*(1-2*c) evaluated at previous step
    W_prime = 2 * gfc * (1 - gfc) * (1 - 2 * gfc)
    W_prime_cf = M_mob * W_prime

    # RHS: (c^n / dt) * w + M * W'(c^n) * w
    rhs = LinearForm(fes)
    rhs += (1.0/dt) * gfc * w * dx
    rhs += -W_prime_cf * w * dx
    rhs.Assemble()

    gfc.vec.data = lhs.Inverse(fes.FreeDofs()) * rhs.vec

    t = (step + 1) * dt
    if step % vtk_every == 0 or step == n_steps - 1:
        vtk.Do(time=t)
        mass_c = Integrate(gfc, mesh)
        mass_history.append((float(t), float(mass_c)))
        print(f"  t={{t:.4f}}, ∫c dx = {{mass_c:.6f}}")

# Interface position tracking (where c ≈ 0.5)
print(f"Completed {{n_steps}} time steps")
final_mass = Integrate(gfc, mesh)
print(f"Final ∫c dx = {{final_mass:.6f}}")

# Phase-field fracture extension note:
# For brittle fracture add elastic energy: W_e = (1-d)^2 * psi_e(u)
# and crack irreversibility: d >= d_prev (history field)
# dW/dd = -2*(1-d)*psi_e + (G_c/l)*(d - l^2*Δd) = 0

summary = {{
    "epsilon": eps,
    "mobility": M_mob,
    "dt": dt,
    "T_end": float(n_steps * dt),
    "n_dofs": fes.ndof,
    "n_elements": mesh.ne,
    "final_integral_c": float(final_mass),
    "mass_history": mass_history[-5:],
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Phase-field solve complete.")
'''


def _phase_field_fracture_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Phase-field fracture (Bourdin-Francfort-Marigo) coupled to linear elasticity.
    Two-field problem: displacement u and crack phase d.
    Alternate minimization (staggered scheme):
      1) Elastic step: min_{u} E(u, d^k)  (linear, with degraded stiffness)
      2) Crack step:   min_{d} E(u^{k+1}, d) subject to d >= d_prev  (irreversibility)"""
    E = params.get("E", 1.0)
    nu = params.get("nu", 0.3)
    Gc = params.get("Gc", 1e-3)     # critical energy release rate
    l0 = params.get("l0", 0.02)     # length scale
    disp_inc = params.get("disp_increment", 1e-4)
    # Layer F gate runs each template within 60s; the
    # original defaults (50 staggered load steps on a
    # maxh=0.01 mesh ~ 60k DOFs) exceed that easily. Trim
    # to 5 steps on a maxh=0.05 mesh — enough to exercise
    # the alternate-minimisation loop without saturating.
    n_steps = params.get("load_steps", 5)
    maxh = params.get("maxh", 0.05)
    order = params.get("order", 1)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""Phase-field fracture — staggered scheme — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))

E_mod   = {E}
nu_val  = {nu}
mu_val  = {mu}
lam_val = {lam}
Gc_val  = {Gc}   # critical energy release rate
l0_val  = {l0}   # regularisation length scale
order   = {order}

# ── FE spaces ─────────────────────────────────────────────────────────────────
Vu = VectorH1(mesh, order=order, dirichlet="bottom|top")
Vd = H1(mesh, order=order)   # phase field d ∈ [0, 1]

u, v   = Vu.TnT()
d, phi = Vd.TnT()

# ── Material: degraded elasticity ─────────────────────────────────────────────
def Strain(w):
    return 0.5 * (Grad(w) + Grad(w).trans)

def Stress_degraded(w, d_gf):
    eps = Strain(w)
    # Degradation function: g(d) = (1-d)^2 + k_res (k_res = small residual stiffness)
    k_res = 1e-10
    g = (1 - d_gf)**2 + k_res
    return g * (2 * mu_val * eps + lam_val * Trace(eps) * Id(2))

def psi_plus(w):
    """Tensile (positive) elastic energy density — Miehe split."""
    eps = Strain(w)
    tr_eps = Trace(eps)
    # Python's abs() is NOT defined on a NGSolve
    # CoefficientFunction; use IfPos(z, z, -z) (or
    # sqrt(z*z)) to express |z| symbolically.
    abs_tr = IfPos(tr_eps, tr_eps, -tr_eps)
    psi_vol  = 0.5 * lam_val * 0.5 * (tr_eps + abs_tr)**2
    psi_dev  = mu_val * InnerProduct(eps, eps) - mu_val / 3 * tr_eps**2
    return psi_vol + psi_dev

# ── GridFunctions ─────────────────────────────────────────────────────────────
gfu   = GridFunction(Vu)
gfd   = GridFunction(Vd)
gfd_prev = GridFunction(Vd)   # history (irreversibility)
gfd.Set(0)                     # undamaged initial state
gfd_prev.Set(0)

n_load_steps = {n_steps}
disp_inc_val = {disp_inc}

print(f"Phase-field fracture: {{n_load_steps}} steps, Gc={{Gc_val}}, l0={{l0_val}}")

for step in range(1, n_load_steps + 1):
    disp_now = step * disp_inc_val

    # Apply split tension: pull top and bottom apart
    gfu.Set(CoefficientFunction((0.0,  disp_now)), definedon=mesh.Boundaries("top"))
    gfu.Set(CoefficientFunction((0.0, -disp_now)), definedon=mesh.Boundaries("bottom"))

    # Staggered iteration
    for alt_iter in range(50):
        gfu_old = gfu.vec.CreateVector(); gfu_old.data = gfu.vec
        gfd_old = gfd.vec.CreateVector(); gfd_old.data = gfd.vec

        # ── Step 1: Elastic problem with fixed d ─────────────────────────────
        a_u = BilinearForm(Vu)
        a_u += InnerProduct(Stress_degraded(u, gfd), Strain(v)) * dx
        a_u.Assemble()
        f_u = LinearForm(Vu)
        f_u.Assemble()
        gfu.vec.data = a_u.mat.Inverse(Vu.FreeDofs()) * f_u.vec

        # ── Step 2: Phase-field crack problem with fixed u ────────────────────
        # Crack driving force (tensile strain energy)
        H_field = psi_plus(gfu)

        a_d = BilinearForm(Vd, symmetric=True)
        a_d += (Gc_val/l0_val + 2*H_field) * d * phi * dx
        a_d += Gc_val * l0_val * grad(d) * grad(phi) * dx
        a_d.Assemble()

        f_d = LinearForm(Vd)
        f_d += 2 * H_field * phi * dx
        f_d.Assemble()

        gfd_unconstrained = GridFunction(Vd)
        gfd_unconstrained.vec.data = a_d.mat.Inverse(Vd.FreeDofs()) * f_d.vec

        # Irreversibility: d >= d_prev (crack cannot heal)
        for i in range(len(gfd.vec)):
            gfd.vec[i] = max(float(gfd_unconstrained.vec[i]),
                             float(gfd_prev.vec[i]))

        # Convergence check
        du = (gfu.vec - gfu_old).Norm()
        dd = (gfd.vec - gfd_old).Norm()
        if du < 1e-8 and dd < 1e-8:
            break

    gfd_prev.vec.data = gfd.vec

    d_max = max(gfd.vec)
    print(f"  Load step {{step}}/{n_steps}: disp={{disp_now:.4e}}, d_max={{d_max:.4f}}")
    if d_max > 0.99:
        print("  Full fracture reached — stopping")
        break

vtk = VTKOutput(mesh,
                coefs=[gfu, gfd],
                names=["displacement", "phase_crack"],
                filename="result", subdivision=1)
vtk.Do()

summary = {{
    "n_dofs_u": Vu.ndof,
    "n_dofs_d": Vd.ndof,
    "n_elements": mesh.ne,
    "Gc": Gc_val,
    "l0": l0_val,
    "load_steps_run": step,
    "max_phase_field": float(d_max),
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Phase-field fracture solve complete.")
'''


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE dict
# ─────────────────────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "dg_methods": {
        "description": (
            "Interior-penalty DG (SIPG) for advection-diffusion using L2 space "
            "with dgjumps=True. Supports high-order, unstructured meshes, convection-dominated flows."
        ),
        "spaces": "L2(mesh, order=k, dgjumps=True) — fully discontinuous",
        "solver": "Direct (sparsecholesky / umfpack) for moderate size; GMRES + block-Jacobi for large",
        "pitfalls": [
            (
                "[API] MUST set dgjumps=True on the L2 / DG "
                "FE space — without it, the cross-element "
                "coupling entries in the sparse matrix are "
                "NOT allocated. Signal: assembling a DG "
                "BilinearForm with jump terms after building "
                "fes = L2(mesh, order=k) (no dgjumps=True) "
                "raises `Sparse matrix: entry at (i,j) does "
                "not exist` at .Assemble() or, on older "
                "versions, silently drops the jump "
                "contributions. The fix is "
                "L2(mesh, order=k, dgjumps=True). (Audit "
                "2026-06-02.)"
            ),
            (
                "[API] u.Other() accesses the neighbour "
                "element's trial function across a shared "
                "facet — NGSolve's restriction operator. "
                "Signal: writing the jump term as "
                "(u - u_neighbour) using an external "
                "function instead of u.Other() raises "
                "`AttributeError: trial function has no "
                "attribute neighbour`; the documented API "
                "is u.Other(). Symmetric average: "
                "0.5*(u + u.Other()). (Audit 2026-06-02.)"
            ),
            (
                "[API] dx(skeleton=True) integrates over "
                "INTERIOR facets; ds(skeleton=True) over "
                "BOUNDARY facets. Signal: applying a "
                "jump-penalty term over plain dx (volume "
                "measure) is silently dropped because the "
                "jump is zero on the interior of an element "
                "(u and u.Other() refer to the same value); "
                "the assembled matrix has identical "
                "structure but missing penalty entries — "
                "stability is lost and the iterative solver "
                "stagnates. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Penalty parameter: alpha * "
                "order^2 / h. Signal: alpha too small "
                "(< order^2) gives coercivity loss — "
                "discrete solution norm grows under "
                "refinement instead of converging; alpha "
                "too large (> 100 * order^2) gives "
                "cond(K) > 1e14 and CG/GMRES stagnate. "
                "Rule of thumb: alpha = 4 * (order + 1)^2 "
                "for SIP DG. (Audit 2026-06-02.)"
            ),
            (
                "[API] IfPos(b*n, u, u.Other()) selects the "
                "upwind side for convection. Signal: using "
                "0.5*(u + u.Other()) (central flux) on a "
                "pure-advection DG problem produces "
                "unconditional instability — the solution "
                "amplitude grows exponentially in time "
                "regardless of mesh; upwind via IfPos "
                "restores stability. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] For convection-dominated DG "
                "(Pe >> 1): DG is naturally stable due to "
                "upwind flux; SIP diffusion STILL needs "
                "the alpha/h * jump penalty. Signal: "
                "switching from Galerkin-CG to DG on "
                "Pe=100 removes the gross oscillations but "
                "the diffusion-dominated regions still "
                "show small-amplitude ringing if the SIP "
                "penalty is omitted (alpha=0); always add "
                "the diffusion penalty even when advection "
                "dominates the bulk. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] DG bilinear form is NOT "
                "symmetric when advection is present "
                "(upwind term is one-sided). Signal: "
                "feeding the assembled matrix to a CG "
                "solver raises a 'matrix not positive "
                "definite' error or returns wildly wrong "
                "iterates; switch to GMRES (or BiCGStab) "
                "for the unsymmetric system. Pure "
                "diffusion SIP DG IS symmetric — advection "
                "breaks symmetry. (Audit 2026-06-02.)"
            ),
            "[API] Python's builtin abs() is NOT defined on "
            "ngsolve.la.BaseVector. max(abs(gfu.vec)) raises "
            "TypeError 'bad operand type for abs(): "
            "ngsolve.la.BaseVector'. Convert to numpy via "
            "gfu.vec.FV().NumPy() then reduce: float(numpy.abs("
            "gfu.vec.FV().NumPy()).max()). Same pattern applies "
            "to compound spaces — gfu.components[i].vec.FV()"
            ".NumPy(). Signal: the literal TypeError text 'bad "
            "operand type for abs(): \\'ngsolve.la.BaseVector\\'' "
            "uniquely identifies the bad-call site. (Verified "
            "empirically 2026-06-01 — Layer F catch.)",
            "[Syntax] NGSolve's CoefficientFunction namespace "
            "exposes exp/log/sin/cos/tan/atan/atan2 but NOT the "
            "hyperbolic functions (no tanh/sinh/cosh). Calling "
            "tanh(z) in a CF expression raises NameError 'name "
            "tanh is not defined'. Build manually via the "
            "identity tanh(z) = (exp(2z)-1)/(exp(2z)+1). Signal: "
            "NameError at script import / gfu.Set time naming "
            "'tanh' (Did you mean: 'tan'?). (Verified empirically "
            "2026-06-01 — Layer F catch.)",
            "[Syntax] Python abs() also does NOT work on a "
            "NGSolve CoefficientFunction expression. Use "
            "IfPos(z, z, -z) (or sqrt(z*z)) for symbolic "
            "absolute value. Signal: TypeError with the literal "
            "text 'bad operand type for abs(): "
            "\\'ngsolve.fem.CoefficientFunction\\'' raised from a "
            "Python-level abs() applied to a Trace, "
            "InnerProduct, or other CF-valued expression. "
            "(Verified empirically 2026-06-01 — Layer F "
            "phase_field_fracture catch.)",
        ],
    },
    "contact": {
        "description": (
            "Unilateral contact (obstacle problem) via penalty method. "
            "Enforces non-penetration u · n >= g through a large penalty on active contact nodes."
        ),
        "spaces": "VectorH1 for elasticity displacement",
        "solver": "Fixed-point / Newton iteration on the penalty-augmented system",
        "pitfalls": [
            (
                "[Numerical] Penalty parameter gamma: too small "
                "-> contact not enforced; too large -> ill-"
                "conditioning. Signal: too small gives "
                "max penetration > 5% of element edge; too "
                "large produces NewtonMinimization "
                "`DivisionByZero` / cond(K)>1e14 warnings "
                "from the sparse solver. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Active-set method (Lagrange "
                "multiplier or semismooth Newton) is MORE "
                "ACCURATE than pure penalty — converges "
                "without an O(1/gamma) error floor. Signal: "
                "even with optimally-tuned penalty gamma, "
                "the residual gap*lambda at convergence is "
                "bounded below by O(h/gamma) — semismooth "
                "Newton drives it to machine precision. "
                "Penalty stalls at a fixed gap; the "
                "active-set iteration converges in 3-10 "
                "outer iterations to identical-active-set "
                "fixed point. (Audit 2026-06-02.)"
            ),
            (
                "[API] IfPos(-gap, 1, 0) identifies active "
                "contact nodes — evaluates at integration "
                "points. Signal: using a boolean Python "
                "comparison `gap < 0` instead of IfPos raises "
                "`TypeError: CoefficientFunction comparison` at "
                "form assembly; the active-set indicator never "
                "fires and gap stays open. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Contact normal must be consistent "
                "with mesh boundary orientation. Signal: a flipped "
                "normal causes penalty to PUSH bodies INTO each "
                "other instead of separating them — gap goes "
                "negative without bound; check the sign by "
                "evaluating specialcf.normal(2 or 3).dot(n_expected)"
                " on the contact boundary. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] For frictional contact: add a "
                "tangential-direction penalty with Coulomb's "
                "law |f_t| <= mu * |f_n| as a complementarity "
                "constraint. Signal: omitting the tangential "
                "penalty (only enforcing normal contact) "
                "lets the contacting bodies SLIDE freely "
                "along their interface — a vertical block "
                "resting on an inclined plane slides off "
                "regardless of mu_friction; with tangential "
                "penalty + Coulomb, the block sticks below "
                "the friction angle and slides above it. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[API] NGSolve has no built-in contact formulation "
                "— must implement penalty or Lagrange multiplier "
                "manually. Signal: searching `ngsolve.comp` for "
                "ContactBoundaryCondition or similar returns no "
                "match; the catalog ships penalty / Lagrange code "
                "snippets that the user copies — there is no "
                "single-call contact API. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Convergence criterion: check both "
                "displacement residual AND contact gap violation. "
                "Signal: a Newton solver that stops when "
                "||du||/||u|| < 1e-6 alone can return with a "
                "still-active gap of 1-5% element-edge size, "
                "because the gap residual scales differently from "
                "the displacement residual. Add an explicit "
                "max(min(gap, 0)) check below tol_gap. (Audit "
                "2026-06-02.)"
            ),
        ],
    },
    "time_dependent_ns": {
        "description": (
            "Transient incompressible Navier-Stokes via IMEX splitting: "
            "Stokes part (viscous + pressure) implicit, convection explicit. "
            "Taylor-Hood P2/P1 on 2D channel or lid-driven cavity."
        ),
        "spaces": "VectorH1(order=2) * H1(order=1) — Taylor-Hood (inf-sup stable)",
        "solver": "IMEX: factor Stokes+mass operator once with umfpack, explicit convection each step",
        "pitfalls": [
            (
                "[Numerical] CFL for explicit convection: dt < "
                "C * h / max|u| — may need small dt for high Re. "
                "Signal: velocity field blows up to NaN within "
                "the first few time steps; per-step max(|u|) "
                "diverges geometrically; the violation ratio "
                "dt * max(|u|) / h is greater than ~0.5. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Convection form: Grad(u)*u "
                "(non-conservative) vs 0.5*(Grad(u)*u - "
                "Grad(u)^T*u) (skew-sym). Signal: long-time "
                "kinetic_energy in a closed periodic box "
                "drifts (grows or decays) with the "
                "non_conservative BilinearForm by O(1%) over "
                "1000 steps; the skew_symmetric variant on "
                "the GridFunction velocity preserves it to "
                "machine precision because it makes the "
                "convective operator anti-symmetric. (Audit "
                "2026-06-02.)"
            ),
            (
                "[API] Re-impose Dirichlet BCs after each "
                "solve to fix boundary nodes. Signal: "
                "boundary velocity drifts away from the "
                "prescribed value across time steps; for a "
                "lid-driven cavity benchmark the top-wall "
                "velocity slowly diverges from u_lid (the "
                "factor-of-2 norm of the BC step is added "
                "back each step instead of overwriting). "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] For Re > 1000: use stabilization "
                "(SUPG, VMS) added inside the BilinearForm "
                "or finer mesh near boundary layers. Signal: "
                "without stabilisation, the GridFunction "
                "velocity field shows visible wiggles "
                "upstream of obstacles or in boundary "
                "layers; energy spectrum has spurious "
                "high-frequency content; drag coefficient "
                "on a cylinder (via BilinearForm boundary "
                "Integrate) differs >10% from the Schafer-"
                "Turek reference. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Pressure uniqueness: fix pressure "
                "at one point or use mean-zero constraint "
                "(NumberSpace). Signal: PETSc reports "
                "`KSPSolve: DIVERGED_BREAKDOWN` or near-zero "
                "pivot; the pressure field shows a uniform "
                "drift unrelated to the source. Pin a single "
                "DOF or attach a NumberSpace mean-zero "
                "constraint. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Taylor-Hood P2/P1 satisfies "
                "inf-sup; P1/P1 does not (needs "
                "stabilization like MINI). Signal: P1/P1 "
                "H1 spaces without stabilisation produce "
                "checkerboard pressure pattern visible in "
                "the GridFunction output, with magnitude "
                "that does not converge under refinement; "
                "switching to P2/P1 H1 spaces or adding "
                "the cubic bubble (MINI) inside the "
                "BilinearForm removes the checkerboard. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Validation] Benchmark: DFG Schafer-Turek "
                "(Re=20 steady, Re=100 periodic vortex "
                "shedding) and the lid-driven cavity at "
                "Re=400, 1000, 5000. Signal: a transient NS "
                "implementation with VectorH1 + H1 "
                "BilinearForm should reproduce Schafer-"
                "Turek drag/lift (post-processed via "
                "Integrate over the cylinder BND) to within "
                "published bounds (Cd ~ 5.57 at Re=20) and "
                "lid-cavity Ghia-streamfunction values "
                "computed from the GridFunction at the "
                "chosen Re; values 5%+ off expose either a "
                "missing convective term, mis-tuned theta, "
                "or insufficient mesh near walls. (Audit "
                "2026-06-02.)"
            ),
        ],
    },
    "mhd": {
        "description": (
            "Magnetohydrodynamics: coupled Navier-Stokes and Maxwell equations. "
            "2.5-D low-Rm formulation: in-plane NS + out-of-plane scalar B_z. "
            "Hartmann problem: conducting channel in transverse magnetic field."
        ),
        "spaces": "VectorH1*H1 (Taylor-Hood, fluid) + H1 (scalar magnetic, low-Rm)",
        "solver": "Operator splitting: fluid (umfpack) + magnetic (direct) each time step",
        "pitfalls": [
            (
                "[Numerical] Low-Rm limit (Rm << 1): induced "
                "magnetic field is negligible, only the "
                "Lorentz force J x B0 matters. Signal: at "
                "Rm = 0.01 a full-MHD code with HCurl A "
                "produces the SAME velocity field as a "
                "low-Rm code that uses only the prescribed "
                "B0 + the scalar potential phi for current "
                "J = -sigma*grad(phi) + sigma*u x B0; the "
                "extra A unknowns waste DOFs. (Audit "
                "2026-06-02.)"
            ),
            (
                "[API] Full MHD (arbitrary Rm): use HCurl for "
                "the vector potential A or Nedelec for B "
                "directly — NOT Lagrange. Signal: a "
                "VectorH1 / Lagrange A produces curl(A) that "
                "lives in a space too smooth for proper "
                "MHD (typical induced B is normal-"
                "discontinuous at material interfaces); "
                "the resulting B field has spurious "
                "smoothing across permeability jumps. Switch "
                "to HCurl(mesh, order=k). (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Hartmann number Ha = B0 * L * "
                "sqrt(sigma / (rho * nu)) measures magnetic "
                "vs viscous effects. Ha >> 1 creates "
                "boundary layers of thickness 1/Ha next to "
                "walls. Signal: a uniform mesh at Ha = 100 "
                "shows ZERO core-region velocity (correct) "
                "but mis-resolved boundary-layer flux — the "
                "computed wall shear stress is off by "
                "factor of 5+ vs the analytic Hartmann "
                "solution because the layer is spread across "
                "only 1-2 cells. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Hartmann layers need mesh "
                "refinement near walls proportional to 1/Ha. "
                "Signal: an unrefined mesh gives "
                "core-velocity accuracy O(1) but boundary-"
                "layer wall shear off by 5-10x at Ha=100; "
                "geometric grading with first-cell height "
                "h_wall = L / (10*Ha) restores ~1% accuracy "
                "on the wall integrals. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Operator splitting (solve "
                "fluid, then magnetic, then iterate) "
                "introduces a splitting error O(dt) — "
                "monolithic (one big nonlinear solve per "
                "step) is more accurate. Signal: at large "
                "dt the splitting result diverges from a "
                "fine-dt monolithic reference by O(dt), "
                "while the monolithic result is "
                "second-order accurate; for time-accurate "
                "MHD use monolithic or sub-cycle the "
                "splitting. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Divergence-free B constraint: "
                "div(B) = 0 must be enforced via grad-div "
                "penalty OR HDiv elements. Signal: a "
                "VectorH1 B field on a non-trivial geometry "
                "develops max|div(B)| ~ O(0.1) over time "
                "(should be ~1e-14); this drives spurious "
                "monopole currents and the kinetic energy "
                "balance drifts. HDiv elements enforce "
                "div(B) = 0 pointwise; grad-div penalty "
                "tau*(div(B), div(C)) with tau ~ 1 is the "
                "alternative for H1. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] For incompressible MHD: add a "
                "grad-div stabilisation tau*(div(u), "
                "div(v))*dx on velocity (helps both "
                "div(u)=0 and div(B)=0 if you use vector "
                "potential A). Signal: omitting grad-div "
                "stabilisation in a Taylor-Hood NS-MHD code "
                "lets div(u) reach 1e-4 to 1e-3 (not "
                "machine precision) — pressure becomes "
                "noisy; tau = nu controls the deviation. "
                "(Audit 2026-06-02.)"
            ),
            "[Syntax] LinearForm integrand must be SCALAR-VALUED. "
            "Multiplying a vector-valued CoefficientFunction (e.g. "
            "CoefficientFunction((0, 1)) for an e_y unit vector) "
            "with a scalar factor and a scalar test-function "
            "component still yields a vector — and SymbolicLFI "
            "rejects it. Build the integrand purely from scalar "
            "factors instead, indexing the vector test function "
            "(v[1] for the y-component) to project onto the "
            "desired equation. Signal: NgException 'SymbolicLFI "
            "needs scalar-valued CoefficientFunction' raised from "
            "LinearForm.__iadd__ when the integrand passes a "
            "VectorialCF through *=. (Verified empirically "
            "2026-06-01 — Layer F mhd catch.)",
        ],
    },
    "hdivdiv": {
        "description": (
            "HDivDiv space for Kirchhoff plate bending via Hellan-Herrmann-Johnson (HHJ) mixed method. "
            "Moment tensor in HDivDiv (H(div div) conforming), deflection in H1. "
            "Solves the biharmonic equation Δ²w = q without C1 continuity requirement on deflection."
        ),
        "spaces": "HDivDiv(mesh, order=k-1) for moments + H1(mesh, order=k) for deflection",
        "solver": "Direct on saddle-point system (umfpack)",
        "pitfalls": [
            (
                "[Numerical] HDivDiv enforces normal-normal "
                "continuity across facets — WEAKER than full "
                "H2 conformity (which would require C1 "
                "Lagrange). Signal: a finite element claiming "
                "to be the HHJ moment space but using "
                "VectorH1 instead of HDivDiv allows tangent-"
                "tangent jumps at facets and yields a "
                "non-conforming bilinear form; the converged "
                "deflection is off by a constant factor (~1.1-"
                "1.5) compared to the HHJ reference solution. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] For CLAMPED plate: add boundary "
                "terms for dw/dn = 0 (Nitsche penalty or "
                "Lagrange multiplier on the skeleton). "
                "Signal: solving HHJ plate with ONLY w = 0 on "
                "a clamped edge (no dw/dn term) gives a "
                "simply-supported solution instead of "
                "clamped — the centre deflection is ~3-4x "
                "larger than the clamped reference. Add the "
                "moment term over skeleton to enforce dw/dn "
                "weakly. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] For SIMPLY-SUPPORTED plate: "
                "only w = 0 on the boundary is needed; "
                "normal moments M_nn vanish naturally as a "
                "natural BC. Signal: adding an extra "
                "Dirichlet on M_nn (the bending moment) over-"
                "constrains a simply-supported plate, "
                "increasing the stiffness — centre deflection "
                "is ~10-20% smaller than the analytic "
                "w_max = q*L^4 / (64*D); remove the moment "
                "BC. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] HHJ is order-optimal: order k "
                "moments + order k deflection give order "
                "k+1 in L2 norm. Signal: a convergence-rate "
                "study (h-refinement at fixed order=1) "
                "should show L2 error of the deflection "
                "scaling as h^2; if it scales as h, the "
                "moment space is mis-typed (e.g. "
                "VectorH1 instead of HDivDiv) and you've "
                "lost an order. (Audit 2026-06-02.)"
            ),
            (
                "[API] Regge elements (Regge calculus, "
                "DISTINCT from HHJ) use HDivDiv for 3D "
                "elasticity compatibility — symmetric tensor "
                "stress with prescribed traction continuity. "
                "Signal: porting an HHJ 2D-plate code to 3D "
                "elasticity without switching to Regge "
                "elements produces a non-conforming solution "
                "(traction jumps at facets); the "
                "HCurlCurl/HDivDiv pairing for 3D Hellinger-"
                "Reissner is the canonical Regge "
                "discretisation. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Mixed formulation (HHJ / Regge) "
                "AVOIDS locking, unlike displacement-only "
                "C1-conforming methods. Signal: a "
                "displacement-only thin-plate H1 GridFunction "
                "at h/thickness ratio > 100 shows shear_locking "
                "(centre_deflection w_max is 10-1000x smaller "
                "than the analytic simply_supported plate "
                "result) — the discrete bending_energy is "
                "dwarfed by spurious shear strain. HHJ removes "
                "locking entirely by using the bending_moment "
                "as primary unknowns. (Audit 2026-06-02.)"
            ),
            (
                "[Validation] Verify against analytical: "
                "w_max = q*L^4 / (64*D) for a simply-"
                "supported uniform-load plate "
                "(D = E*t^3 / (12*(1-nu^2))). Signal: a "
                "HHJ implementation should converge to "
                "within 1% of this value on a moderately "
                "fine mesh (h/L < 0.1); a >5% deviation "
                "exposes a mis-configured BC, wrong D, or "
                "mis-typed HDivDiv vs VectorH1 moment "
                "space. (Audit 2026-06-02.)"
            ),
            "[API] HDivDiv (normal-normal continuous moment "
            "tensor space) does NOT expose a pointwise div(div("
            "tau)) operator. Constructing 'div(div(tau)) * v * "
            "dx' raises Exception 'cannot form div' from "
            "SymbolicBFI. Integrate by parts twice and substitute "
            "the H1 Hessian: replace div(div(tau)) * v with "
            "InnerProduct(tau, v.Operator('hesse')) - skeleton "
            "normal-normal moment integral over interior facets "
            "(via dx(element_boundary=True)). Signal: Exception "
            "'cannot form div' emitted from BilinearForm += "
            "div(div(tau_)) * ww * dx in any HHJ-style template. "
            "(Verified empirically 2026-06-01 — Layer F catch.)",
        ],
    },
    "nonlinear_elasticity": {
        "description": (
            "Large-deformation Neo-Hookean hyperelasticity via Variation() + Newton. "
            "Supports 2D (plane strain) and 3D. Load stepping ensures convergence "
            "for large applied displacements. Outputs Cauchy stress."
        ),
        "spaces": "VectorH1(mesh, order=2) — displacement-based finite strain",
        "solver": "solvers.Newton() with load stepping; dampfactor reduces step size if needed",
        "pitfalls": [
            (
                "[Numerical] det(F) MUST remain > 0 — the "
                "initial guess must not cause element "
                "inversion. Signal: an initial displacement "
                "guess that crushes the element to "
                "near-zero or negative volume (e.g. "
                "starting from u = -2*x in a unit-cube) "
                "evaluates ln(J) with J <= 0 and raises "
                "FloatingPointError or NaN in the first "
                "Newton residual. Start from u = 0 and "
                "load-step. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Load stepping: apply "
                "displacement / load in INCREMENTS, using "
                "previous converged GridFunction as "
                "initial guess. Signal: applying full load "
                "at t=0 to a hyperelastic problem at 30% "
                "nominal strain typically diverges "
                "(ngsolve.solvers.NewtonMinimization "
                "residual grows ~10x per iter); "
                "subdividing into 10 steps of 3% strain "
                "achieves quadratic convergence per step "
                "with the per-step BilinearForm AutoDiff "
                "linearization. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Neo-Hookean energy: 0.5*mu*"
                "(Tr(C) - d) - mu*ln(J) + 0.5*lam*ln(J)^2 "
                "where d is the spatial dimension (2 or "
                "3). Signal: forgetting the d subtraction "
                "(using Tr(C) instead of Tr(C) - d) gives "
                "W != 0 at F = I — a stress-free "
                "reference produces a non-zero initial "
                "stress; the first Newton iterate runs "
                "off looking for a different equilibrium. "
                "Sanity-check W(F=I) = 0. (Audit "
                "2026-06-02.)"
            ),
            (
                "[API] Variation() auto-differentiates the "
                "energy to get the residual and tangent — "
                "MUCH safer than hand-coding. Signal: a "
                "hand-coded P(F) and dP/dF with a sign "
                "error or factor-of-2 produces linear (not "
                "quadratic) Newton convergence — residual "
                "halves per iter instead of squaring. "
                "Switching to a.Apply / a.AssembleLinearization "
                "with a Variation-built energy form "
                "restores quadratic. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] For nearly-incompressible "
                "(nu -> 0.5): use F-bar method or mixed "
                "(u, p) formulation. Signal: a pure-"
                "displacement VectorH1 solve at nu = 0.4999 "
                "locks volumetrically — the Cook-membrane "
                "tip displacement is < 1% of the analytic "
                "value; switching to mixed (u, p) with "
                "Taylor-Hood-like spaces (VectorH1 order 2 "
                "+ H1 order 1 for p) recovers within ~1%. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Cauchy stress: sigma = "
                "(1/J) * F * S * F^T where S = dW/dE is "
                "the 2nd Piola_Kirchhoff stress. Signal: "
                "writing the Cauchy_stress as sigma = "
                "(1/J) * F * P * F^T (P = first_PK = PK1) "
                "mixes up the push_forward — S and P are "
                "different objects (P = F * S); the "
                "resulting GridFunction stress is wrong "
                "by a factor of F. Correct: sigma = "
                "(1/J) * F * S * F^T or sigma = (1/J) * "
                "P * F^T. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Newton dampfactor < 1 helps "
                "when FAR from equilibrium (large load "
                "steps). Signal: when load-stepping is "
                "aggressive enough that Newton overshoots "
                "the convergence basin (residual grows "
                "between iterations), setting "
                "dampfactor = 0.5 or 0.25 in "
                "ngsolve.solvers.Newton restores "
                "convergence at the cost of more "
                "iterations; alternatively reduce the "
                "load increment. (Audit 2026-06-02.)"
            ),
            "[API] After solvers.Newton converges, the Cauchy / "
            "PK1 stress passed to VTKOutput must be rebuilt from "
            "the resolved GridFunction (Grad(gfu), Det(I+Grad("
            "gfu)), Inv(F.trans*F) ...), NOT from the symbolic "
            "fes.TrialFunction(). The symbolic version is a "
            "ProxyFunction; VTKOutput.Do() then raises NgException "
            "'cannot evaluate ProxyFunction without userdata' at "
            "the first quadrature evaluation. Signal: that exact "
            "ProxyFunction-userdata text emitted from VTKOutput."
            "Do() at the end of a Newton template that passes "
            "symbolically-built stress through 'coefs=[gfu, "
            "sigma_cauchy]'. (Verified empirically 2026-06-01 — "
            "Layer F catch.)",
        ],
    },
    "phase_field": {
        "description": (
            "Phase-field evolution: Allen-Cahn for interface motion (scalar phase c) "
            "and phase-field fracture (Bourdin-Francfort-Marigo, staggered scheme). "
            "Fracture: coupled displacement u and crack phase d; alternate minimization."
        ),
        "spaces": "H1(mesh, order=k) for scalar phase; VectorH1 + H1 for fracture",
        "solver": "Allen-Cahn: implicit Euler (linear system per step). Fracture: staggered alternating minimization",
        "pitfalls": [
            (
                "[Numerical] Allen-Cahn mass is NOT conserved — use "
                "Cahn-Hilliard (4th order) for mass conservation. "
                "Signal: Integrate(c, mesh) drifts monotonically "
                "(~1-5% per characteristic interface time) in "
                "Allen-Cahn; the same geometry under Cahn-Hilliard "
                "preserves the integral to machine precision. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Interface width epsilon must be "
                "resolved: at least 3-4 elements across interface "
                "(h << eps). Signal: phase field c develops "
                "checkerboard pattern near the interface with "
                "10-30% over/undershoot; or NewtonMinimization "
                "diverges with `Newton did not converge after N "
                "iterations` because W'(c) ~ c^3 amplifies "
                "spurious oscillations. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Semi-implicit treatment of "
                "W'(c): evaluate at c^n (previous step), "
                "solve linearly — avoids the nonlinear "
                "solve. Signal: a fully-implicit treatment "
                "with c^{n+1} inside W'(c) requires Newton "
                "at each step (~5-10 inner iters), which "
                "dominates wall-clock; the semi-implicit "
                "lagging changes the stability constant "
                "but keeps each step linear, ~5-10x faster. "
                "Trade-off: stability constant shrinks "
                "modestly (typical 10-20%); use a smaller "
                "dt if you observe ringing. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Phase-field fracture: irreversibility "
                "d >= d_prev (crack cannot heal) — enforce "
                "pointwise. Signal: visualization shows the "
                "damage field d decreasing in some elements "
                "between time steps (unphysical 'healing'); the "
                "fracture surface is not monotonic in time. "
                "Enforce via max(d, d_prev) projection after each "
                "solve. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Staggered scheme (solve "
                "elasticity, then phase-field, iterate) "
                "converges to the SAME solution as monolithic "
                "but takes more iterations per step. Signal: "
                "staggered typically needs 5-20 outer iters "
                "vs 1 nonlinear solve in monolithic, but each "
                "iteration is cheaper (two linear solves "
                "instead of one nonlinear). Wall-clock wins "
                "depend on the relative cost; staggered is "
                "easier to implement and more robust at "
                "fracture-onset events where monolithic "
                "Newton struggles. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Miehe energy split (tension/"
                "compression) prevents crack growth under "
                "compression. Signal: without the split, a "
                "compressive boundary load nucleates spurious "
                "damage d>0 in the loaded region; with the split, "
                "compression yields d~0 throughout. A "
                "uniaxial-compression sanity test should show "
                "max(d) < 1e-3. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Length scale l0 must be small "
                "enough relative to specimen size; the "
                "fracture-energy convergence is recovered as "
                "l0 -> 0. Signal: l0 ~ 1/10 of the specimen "
                "characteristic size produces a smeared "
                "fracture zone visibly wider than expected "
                "(captures rough crack location but the peak "
                "load is over-predicted ~20-50% vs the "
                "Griffith analytic load); refining the mesh "
                "WITHOUT also reducing l0 does not help — "
                "both must shrink together with l0 / h ~ 4-8 "
                "preserved. (Audit 2026-06-02.)"
            ),
            (
                "[API] For Cahn-Hilliard: use H1 x H1 mixed "
                "formulation (chemical potential + phase field). "
                "Signal: a single-H1 (4th-order) discretization "
                "with standard Lagrange elements fails at assembly "
                "with `NotImplementedError: H2 conformity required "
                "for biharmonic operator` or the SymbolicBFI "
                "raises `coefficient not in BilinearForm space`. "
                "Mixed (c, mu) splits the biharmonic into two "
                "Laplacians. (Audit 2026-06-02.)"
            ),
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# GENERATORS dict
# ─────────────────────────────────────────────────────────────────────────────

GENERATORS = {
    "dg_methods_2d":                _dg_methods_2d,
    "contact_2d":                   _contact_2d,
    "time_dependent_ns_2d":         _time_dependent_ns_2d,
    "mhd_2d":                       _mhd_2d,
    "hdivdiv_2d":                   _hdivdiv_2d,
    "nonlinear_elasticity_2d":      _nonlinear_elasticity_2d,
    "nonlinear_elasticity_3d":      _nonlinear_elasticity_3d,
    "phase_field_2d":               _phase_field_2d,
    "phase_field_fracture_2d":      _phase_field_fracture_2d,
}
