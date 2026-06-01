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
a = BilinearForm(X, symmetric=True)
# Compliance block
a += InnerProduct(Compliance(sig), tau_) * dx
# B(tau, w): coupling — div(div(tau)) * w integrated by parts
a += div(div(tau_)) * ww * dx
# B(sigma, v): symmetric
a += div(div(sig)) * vv * dx
# Boundary natural BC: tangential jump for clamped plate
# (edge integrals over skeleton for normal-normal BC)
a += -(tau_ * n) * n * (Grad(ww) * n) * ds(skeleton=True)
a += -(sig  * n) * n * (Grad(vv) * n) * ds(skeleton=True)
a.Assemble()

# ── Load ─────────────────────────────────────────────────────────────────────
f = LinearForm(X)
f += q * vv * dx   # transverse load on deflection test function
f.Assemble()

# ── Solve ─────────────────────────────────────────────────────────────────────
gf = GridFunction(X)
gf.vec.data = a.mat.Inverse(X.FreeDofs(), inverse="umfpack") * f.vec

gf_sig, gf_w = gf.components

max_deflection = abs(max(gf_w.vec))
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
            "MUST set dgjumps=True — without it, cross-element coupling entries are not allocated",
            "u.Other() accesses the neighbor element's trial function across a shared facet",
            "dx(skeleton=True) integrates over interior facets; ds(skeleton=True) over boundary facets",
            "Penalty parameter: alpha * order^2 / h — too small -> unstable, too large -> ill-conditioned",
            "IfPos(b*n, u, u.Other()) selects the upwind side for convection",
            "For convection-dominated (Pe >> 1): DG is naturally stable; SIP diffusion still needs penalty",
            "Bilinear form is not symmetric when advection is present — use GMRES, not CG",
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
            "Penalty parameter gamma: too small -> contact not enforced; too large -> ill-conditioning",
            "Active set method (Lagrange multiplier or semismooth Newton) is more accurate than pure penalty",
            "IfPos(-gap, 1, 0) identifies active contact nodes — evaluates at integration points",
            "Contact normal must be consistent with mesh boundary orientation",
            "For frictional contact: add tangential penalty with Coulomb condition",
            "NGSolve has no built-in contact formulation — must implement penalty or Lagrange multiplier manually",
            "Convergence criterion: check both displacement residual and contact gap violation",
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
            "CFL for explicit convection: dt < C * h / max|u| — may need small dt for high Re",
            "Convection form: Grad(u)*u (non-conservative) vs 0.5*(Grad(u)*u - Grad(u)^T*u) (skew-sym)",
            "Re-impose Dirichlet BCs after each solve to fix boundary nodes",
            "For Re > 1000: use stabilization (SUPG, VMS) or finer mesh near boundary layers",
            "Pressure uniqueness: fix pressure at one point or use mean-zero constraint (NumberSpace)",
            "Taylor-Hood P2/P1 satisfies inf-sup; P1/P1 does not (needs stabilization like MINI)",
            "Benchmark: DFG Schafer-Turek (Re=20 steady, Re=100 periodic) and lid-driven cavity",
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
            "Low-Rm limit (Rm << 1): induced field negligible, only Lorentz force matters",
            "Full MHD (arbitrary Rm): need HCurl for vector A or Nedelec for B directly",
            "Hartmann number Ha = B0*L*sqrt(sigma/rho*nu): Ha >> 1 creates boundary layers of thickness 1/Ha",
            "Hartmann layers need mesh refinement near walls proportional to 1/Ha",
            "Operator splitting introduces splitting error O(dt) — monolithic is more accurate",
            "Divergence-free B constraint: ∇·B = 0 must be enforced (grad-div penalty or HDiv elements)",
            "For incompressible MHD: add grad-div stabilization on velocity",
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
            "HDivDiv enforces normal-normal continuity across facets (weaker than H2 conformity)",
            "For clamped plate: add boundary terms for dw/dn = 0 (Nitsche or Lagrange multiplier on skeleton)",
            "For simply-supported plate: only w = 0 on boundary; normal moments naturally satisfied",
            "HHJ is order-optimal: order k moments + order k deflection -> order k+1 in L2",
            "Regge elements (different from HHJ) use HDivDiv for 3D elasticity compatibility",
            "Mixed formulation avoids locking, unlike displacement-only C1 conforming methods",
            "Verify against analytical: w_max = qL^4/(64D) for simply-supported uniform load",
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
            "Det(F) must remain > 0 — initial guess must not cause element inversion",
            "Load stepping: apply displacement/load in increments, use previous solution as initial guess",
            "Neo-Hookean energy: 0.5*mu*(Tr(C)-d) - mu*ln(J) + 0.5*lam*ln(J)^2 (d=2 or 3)",
            "Variation() auto-differentiates energy to get residual and tangent — very convenient",
            "For nearly-incompressible (nu->0.5): use F-bar method or mixed formulation",
            "Cauchy stress: sigma = (1/J) * F * S * F^T where S = dW/dE (PK2 stress)",
            "Newton dampfactor < 1 helps when far from equilibrium (large load steps)",
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
            "Allen-Cahn mass is NOT conserved — use Cahn-Hilliard (4th order) for mass conservation",
            "Interface width epsilon must be resolved: at least 3-4 elements across interface (h << eps)",
            "Semi-implicit treatment of W'(c): evaluate at c^n, solve linearly — avoids nonlinear solve",
            "Phase-field fracture: irreversibility d >= d_prev (crack cannot heal) — enforce pointwise",
            "Staggered scheme converges to same solution as monolithic but takes more iterations",
            "Miehe energy split (tension/compression) prevents crack growth under compression",
            "Length scale l0 must be small enough relative to specimen size; convergence as l0->0",
            "For Cahn-Hilliard: use H1 x H1 mixed formulation (chemical potential + phase field)",
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
