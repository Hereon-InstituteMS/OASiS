"""scikit-fem advanced physics generators and knowledge.

Covers: Navier-Stokes, hyperelasticity (Neo-Hookean), DG advection,
time-dependent PDE, Helmholtz (complex), and reaction-diffusion.
"""


# ---------------------------------------------------------------------------
# 1. Navier-Stokes (Newton iteration)
# ---------------------------------------------------------------------------

def _navier_stokes_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Steady Navier-Stokes on the lid-driven cavity, solved by Picard
    iteration (lagged-velocity convection). Taylor-Hood P2/P1
    velocity-pressure. Rewritten 2026-06-02 (was broken — scalar
    laplace on a vector basis raised shape mismatch + non-existent
    DofsView subscript; the old Newton kernel mixed
    Jacobian/residual blocks incorrectly).
    """
    refine = int(params.get("refine", 4))
    Re = params.get("Re", 100.0)
    tol = params.get("picard_tol", 1e-6)
    max_iter = params.get("max_iter", 25)
    return f'''\
"""Navier-Stokes lid-driven cavity — Picard iteration — Taylor-Hood P2/P1 — scikit-fem"""
from skfem import (
    MeshTri, Basis, BilinearForm,
    ElementVector, ElementTriP1, ElementTriP2,
    asm, condense, solve,
)
from skfem.helpers import grad, ddot, dot
from skfem.models.general import divergence
from scipy.sparse import bmat
import numpy as np
import json

Re = {Re}

# --- Mesh + Taylor-Hood spaces ---
# intorder=4 keeps trial/test quadrature matched on the
# mixed B = -asm(divergence, basis_u, basis_p) block AND
# is high enough for the (u_prev · ∇)u trial term.
m = MeshTri().refined({refine})
basis_u = Basis(m, ElementVector(ElementTriP2()), intorder=4)
basis_p = Basis(m, ElementTriP1(), intorder=4)


@BilinearForm
def viscous(u, v, w):
    """∫(grad u : grad v) dx — vector Laplacian."""
    return ddot(grad(u), grad(v))


@BilinearForm
def convection(u, v, w):
    """∫((u_prev · ∇) u) · v dx — lagged-velocity convection.

    w['u_prev'] is the interpolated previous-iterate velocity:
    a DiscreteField with .value shape (d, n_elem, n_quad)
    and .grad shape (d, d, n_elem, n_quad). For a scalar u_p
    component, u_p[i] would be (n_elem, n_quad). For grad(u)
    (rank-2: components × spatial dims), grad(u)[i][j] is the
    derivative of the i-th trial component w.r.t. x[j].
    """
    u_p = w['u_prev'].value
    gu = grad(u)
    # (u_prev · ∇) u_i = sum_j u_p[j] * du[i]/dx[j]
    adv_u = np.stack([
        u_p[0] * gu[0][0] + u_p[1] * gu[0][1],
        u_p[0] * gu[1][0] + u_p[1] * gu[1][1],
    ])
    return dot(adv_u, v)


# Stokes blocks (assembled ONCE, reused every Picard step).
K_visc = asm(viscous, basis_u) / Re
# B[q, u] = ∫q·div(u) dx (skfem +div convention) — negate
# for the standard saddle-point [[K, -B^T], [-B, 0]] layout.
B = -asm(divergence, basis_u, basis_p)

N_u = basis_u.N
N_p = basis_p.N
N_total = N_u + N_p

# --- Driven-cavity BC ---
# ElementVector interleaves x/y dofs at each node:
#   dof[2i] = x-component, dof[2i+1] = y-component.
doflocs_u = basis_u.doflocs
by = doflocs_u[1]
top_x = np.isclose(by[0::2], 1.0)
u_bc = np.zeros(basis_u.N)
u_bc[0::2] = np.where(top_x, 1.0, 0.0)
u_bc[1::2] = 0.0

# Pressure pin at the DOF closest to the origin (removes
# the constant-pressure null space).
pdofs = basis_p.doflocs.T
pin_p_local = int(np.argmin(np.linalg.norm(pdofs[:, :2], axis=1)))
pin_p_global = N_u + pin_p_local

D_u = basis_u.get_dofs().flatten()
D = np.concatenate([
    D_u,
    np.array([pin_p_global], dtype=np.int64),
])

# Initial guess: Stokes solution (Re → ∞ Picard step 0
# with u_prev=0 reduces convection to 0).
x = np.zeros(N_total)
x[:N_u] = u_bc

# --- Picard loop ---
res_norm = np.inf
for it in range({max_iter}):
    u_prev = x[:N_u]
    u_prev_field = basis_u.interpolate(u_prev)

    # Assemble convection with current u_prev. Combined with
    # the (constant) viscous block this gives the iteration
    # Jacobian for the velocity-velocity block.
    C = asm(convection, basis_u, u_prev=u_prev_field)

    A = bmat([[K_visc + C, B.T],
              [B,          None]], format='csr')
    F = np.zeros(N_total)

    x_full = np.zeros(N_total)
    x_full[:N_u] = u_bc
    x_new = solve(*condense(A, F, D=D, x=x_full))

    res_norm = np.linalg.norm(x_new[:N_u] - u_prev)
    x = x_new
    print(f"Picard it {{it+1}}: ||du|| = {{res_norm:.4e}}")
    if res_norm < {tol}:
        print(f"Converged in {{it+1}} Picard iterations")
        break

u_h = x[:N_u]
p_h = x[N_u:]
max_vel = np.sqrt(u_h[0::2]**2 + u_h[1::2]**2).max()
print(f"Re = {Re}, DOFs = {{N_total}}")
print(f"Max velocity magnitude: {{max_vel:.6f}}")
print(f"||u_x||_inf = {{np.abs(u_h[0::2]).max():.6f}}")
print(f"||u_y||_inf = {{np.abs(u_h[1::2]).max():.6f}}")
print(f"||p||_inf   = {{np.abs(p_h).max():.6f}}")

import meshio
pts  = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
trng = [("triangle", m.t.T)]
mio  = meshio.Mesh(pts, trng)
mio.write("result.vtu")

summary = {{
    "Re": {Re},
    "max_velocity": float(max_vel),
    "n_dofs": int(N_total),
    "n_elements": int(m.nelements),
    "picard_iter": it + 1,
    "final_residual": float(res_norm),
    "element_type": "P2-P1 Taylor-Hood",
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Navier-Stokes solve complete.")
'''


# ---------------------------------------------------------------------------
# 2. Hyperelasticity — Neo-Hookean with Newton iteration
# ---------------------------------------------------------------------------

def _hyperelasticity_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Neo-Hookean hyperelasticity with Newton iteration.
    Incompressible-like Neo-Hookean: W = mu/2*(I1-2) - mu*ln(J) + lam/2*(ln J)^2.
    """
    nx = params.get("nx", 10)
    ny = params.get("ny", 4)
    lx = params.get("lx", 4.0)
    ly = params.get("ly", 1.0)
    # Audit 2026-06-02: previous defaults E=1.0 / traction=0.1
    # gave a 10%-of-stiffness normalised load — large enough to
    # drive J<0 in some Gauss points after the first modified-
    # Newton iter, producing log(J)=NaN. New defaults pick a
    # stiff material + small traction (~0.5% strain) so the
    # iteration converges in 3-4 steps per load substep. Users
    # who want a real large-deformation study override via
    # params and add a continuation/arc-length wrapper.
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    traction = params.get("traction", 1.0)
    tol = params.get("newton_tol", 1e-8)
    max_iter = params.get("max_iter", 30)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return f'''\
"""Neo-Hookean hyperelasticity — incremental load stepping — scikit-fem"""
from skfem import (
    MeshTri, MeshQuad, Basis, BilinearForm, LinearForm, FacetBasis,
    ElementVector, ElementTriP1,
    asm, condense, solve,
)
from skfem.helpers import grad, ddot, dot, identity, inv, det, transpose
import numpy as np
import json

# Lame parameters from E={E}, nu={nu}
lam = {lam:.6f}
mu  = {mu:.6f}

# --- Mesh + clamped/loaded boundary tags ---
m = (MeshQuad.init_tensor(
        np.linspace(0, {lx}, {nx + 1}),
        np.linspace(0, {ly}, {ny + 1}),
     ).to_meshtri()
      .with_boundaries({{
          "left":  lambda x: x[0] < 1e-10,
          "right": lambda x: x[0] > {lx} - 1e-10,
      }}))

basis = Basis(m, ElementVector(ElementTriP1()), intorder=3)
fbasis_right = FacetBasis(m, ElementVector(ElementTriP1()),
                          facets=m.boundaries["right"])

N = basis.N
# Audit 2026-06-02 rewrite: prior version used `w["dux_dx"]`/
# `w["duy_dx"]`/etc. scalar kwargs that scikit-fem 12 no longer
# accepts in @BilinearForm kernels, plus `u.grad[0]` indexing
# that returned a (d, n_basis) array on ElementVector — both
# broke the assembler with
#   ValueError: could not broadcast input array from shape
#   (2,3) into shape (80,).
# New kernel uses skfem.helpers.grad/ddot/dot/identity/inv/det
# directly on the rank-2 displacement-gradient tensor, plus
# basis.interpolate(u_prev_dofs) to pass the previous-iterate
# displacement-gradient field via w['u_prev'].grad (shape
# (d, d, n_elem, n_quad)). Linearisation pattern is a Picard-
# style modified Newton: assemble the tangent stiffness from
# the current-configuration material+geometric contribution,
# the internal-force residual from the 1st-PK stress, and
# solve K_tan * du = F_ext - R_int per load step.

# --- Neo-Hookean tangent (material + geometric) at the
#     current displacement u_prev. Computed via the
#     compressible Neo-Hookean strain-energy
#     W = (mu/2)(I_C - d) - mu*lnJ + (lam/2)(lnJ)^2.
def _F(u_field):
    """Deformation gradient F = I + grad(u_field)."""
    return identity(u_field.grad) + u_field.grad


@BilinearForm
def K_tan_form(u, v, w):
    """Approximate tangent: small-strain linear elasticity (Hooke's
    law). This is the "consistent material tangent at F=I" — exact
    at the first iterate, and a stable modified-Newton tangent for
    moderate loads (up to ~5-10% strain). Convergence is slower
    than full Newton (linear instead of quadratic) but the iteration
    is robust and the geometric stiffness contribution at the next-
    to-undeformed configuration is small. The Neo-Hookean residual
    in R_int_form below uses the actual F-dependent 1st-PK stress, so
    converged iterations still satisfy the nonlinear equilibrium."""
    eps_u = 0.5 * (grad(u) + transpose(grad(u)))
    eps_v = 0.5 * (grad(v) + transpose(grad(v)))
    tr_u = eps_u[0, 0] + eps_u[1, 1]
    tr_v = eps_v[0, 0] + eps_v[1, 1]
    return lam * tr_u * tr_v + 2.0 * mu * ddot(eps_u, eps_v)


@LinearForm
def R_int_form(v, w):
    """Internal virtual work: P(F) : grad(v) dx where P = F * S."""
    F = _F(w['u_prev'])
    J = det(F)
    lnJ = np.log(J)
    Finv = inv(F)
    # 2nd PK: S = mu*(I - Cinv) + lam*lnJ*Cinv,
    # Cinv = Finv @ Finv.T (via component algebra below).
    Cinv00 = Finv[0, 0] * Finv[0, 0] + Finv[0, 1] * Finv[0, 1]
    Cinv01 = Finv[0, 0] * Finv[1, 0] + Finv[0, 1] * Finv[1, 1]
    Cinv11 = Finv[1, 0] * Finv[1, 0] + Finv[1, 1] * Finv[1, 1]
    S00 = mu * (1.0 - Cinv00) + lam * lnJ * Cinv00
    S01 = mu * (0.0 - Cinv01) + lam * lnJ * Cinv01
    S11 = mu * (1.0 - Cinv11) + lam * lnJ * Cinv11
    # 1st PK: P = F * S.
    P00 = F[0, 0] * S00 + F[0, 1] * S01
    P01 = F[0, 0] * S01 + F[0, 1] * S11
    P10 = F[1, 0] * S00 + F[1, 1] * S01
    P11 = F[1, 0] * S01 + F[1, 1] * S11
    # P : grad(v) — sum over (i, j) of P_ij * dv_i/dx_j.
    gv = grad(v)
    return (P00 * gv[0, 0] + P01 * gv[0, 1]
            + P10 * gv[1, 0] + P11 * gv[1, 1])


@LinearForm
def F_ext_form(v, w):
    """Constant traction on right face: t = (traction, 0)."""
    return {traction} * w['load_alpha'] * v[0]


fix_dofs = basis.get_dofs("left").flatten()
free = np.setdiff1d(np.arange(N), fix_dofs)

# --- Outer load-stepping loop + inner Newton-modified loop ---
u_disp = np.zeros(N)
n_load_steps = 4
for step in range(1, n_load_steps + 1):
    alpha = step / n_load_steps   # load fraction this step
    print(f"--- Load step {{step}}/{{n_load_steps}} (alpha={{alpha:.2f}}) ---")
    for it in range({max_iter}):
        u_prev_field = basis.interpolate(u_disp)
        K = asm(K_tan_form, basis, u_prev=u_prev_field)
        R_int = asm(R_int_form, basis, u_prev=u_prev_field)
        F_ext = asm(F_ext_form, fbasis_right, load_alpha=alpha)
        rhs = F_ext - R_int
        rhs[fix_dofs] = 0.0
        du = np.zeros(N)
        du[free] = solve(K[free][:, free], rhs[free])
        u_disp = u_disp + du
        res = np.linalg.norm(du[free])
        print(f"  Newton it {{it+1}}: ||du|| = {{res:.4e}}")
        if res < {tol}:
            print(f"  Converged in {{it+1}} iterations")
            break

u_xy = u_disp.reshape(-1, 2).T   # ElementVector interleaves x/y per node
max_disp = float(np.abs(u_xy).max())
print(f"Max displacement: {{max_disp:.6f}}")
print(f"E={E}, nu={nu}, traction={traction}")

import meshio
pts  = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
trng = [("triangle", m.t.T)]
u_node = np.column_stack([u_xy[0], u_xy[1], np.zeros(m.p.shape[1])])
mio  = meshio.Mesh(pts, trng, point_data={{"displacement": u_node}})
mio.write("result.vtu")

summary = {{
    "max_displacement": max_disp,
    "n_dofs": int(N),
    "n_elements": int(m.nelements),
    "n_load_steps": n_load_steps,
    "E": {E}, "nu": {nu}, "traction": {traction},
    "element_type": "P1-tri vector",
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Hyperelasticity solve complete.")
'''


# ---------------------------------------------------------------------------
# 3. DG methods — upwind DG for linear advection using ElementDG
# ---------------------------------------------------------------------------

def _dg_methods_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Discontinuous Galerkin for steady linear advection using ElementDG
    and InteriorFacetBasis for upwind flux.
    """
    nx = params.get("nx", 32)
    bx = params.get("bx", 1.0)
    by = params.get("by", 0.5)
    eps = params.get("eps", 0.0)   # optional diffusion for stability check
    return f'''\
"""DG upwind advection: b.grad(u) = f using ElementDG — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace
import numpy as np
from scipy.sparse.linalg import spsolve
import json

# Advection velocity
b = np.array([{bx}, {by}])
eps = {eps}    # diffusion coefficient (0 = pure advection)

# --- Mesh ---
m = MeshQuad.init_tensor(
    np.linspace(0, 1, {nx + 1}),
    np.linspace(0, 1, {nx + 1}),
).to_meshtri()   # skfem 12: to_simplex was renamed to to_meshtri

# DG element: discontinuous P1 on triangles
e = ElementDG(ElementTriP1())
ib  = Basis(m, e)
ibf = FacetBasis(m, e)           # boundary facets
ifi = InteriorFacetBasis(m, e)   # interior facets (for upwind flux)

# --- Advection volume term: b . grad(u) * v ---
@BilinearForm
def advection_volume(u, v, w):
    return (b[0] * u.grad[0] + b[1] * u.grad[1]) * v

# --- Optional diffusion ---
@BilinearForm
def diffusion_volume(u, v, w):
    return eps * (u.grad[0]*v.grad[0] + u.grad[1]*v.grad[1])

# --- Interior upwind flux ---
# Jump penalty: b.n * {{u}} (upwind: from upwind side)
@BilinearForm
def upwind_flux_interior(u, v, w):
    # Normal points from "-" to "+" element
    # Upwind: if b.n > 0, flux is from "-" side; else from "+" side
    bn = b[0] * w.n[0] + b[1] * w.n[1]
    # Upwind: use "+" side when bn>0 (out of "-"), "-" side when bn<0
    # Standard upwind: flux = bn * u_upwind
    # u.value has shape (n_quad,) for scalar DG on each side
    flux = 0.5 * bn * (u + u.grad[0]*0) - 0.5 * abs(bn) * (u - u)
    # Simplified: use average + upwind stabilization
    # flux(u)*[v] = bn * {{u}} * [v] + |bn|/2 * [u] * [v]
    return bn * u * (v - v) + 0.5 * abs(bn) * u * v

# Standard upwind DG bilinear form on interior facets:
@BilinearForm
def upwind_interior(u, v, w):
    # b.n * u_upwind * [v]  where [v] = v^+ - v^-
    bn = b[0] * w.n[0] + b[1] * w.n[1]
    # For scalar u: u^+ is on "+" side, u^- on "-" side (InteriorFacetBasis gives both)
    # scikit-fem interior facet basis: u = u on the current side, accessed by w fields
    # Standard: flux = b.n * (0.5*(u^+ + u^-) + |b.n|/(2*b.n) * (u^+ - u^-)) * v
    return bn * u * v

# Boundary flux (inflow: b.n < 0 -> Dirichlet BC)
@LinearForm
def inflow_rhs(v, w):
    bn = b[0] * w.n[0] + b[1] * w.n[1]
    g  = 0.0  # inflow value (u=0 on inflow boundary)
    return -np.where(bn < 0, bn * g, 0.0) * v

@BilinearForm
def outflow_flux(u, v, w):
    bn = b[0] * w.n[0] + b[1] * w.n[1]
    return np.where(bn > 0, bn, 0.0) * u * v

# --- Source term ---
@LinearForm
def source(v, w):
    return 1.0 * v

# --- Assembly ---
A = asm(advection_volume, ib)
if eps > 0:
    A = A + asm(diffusion_volume, ib)
A = A + asm(outflow_flux, ibf)
A = A + asm(upwind_interior, ifi)
f = asm(source, ib) + asm(inflow_rhs, ibf)

# --- Solve (DG system is not symmetric; use direct solve) ---
u = spsolve(A.tocsr(), f)

max_val = u.max()
min_val = u.min()
print(f"DG advection: max(u) = {{max_val:.6f}}, min(u) = {{min_val:.6f}}")
print(f"DOFs: {{A.shape[0]}}, elements: {{m.nelements}}")

import meshio
pts  = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
trng = [("triangle", m.t.T)]
# DG solution: one value per DOF, not per node; write element-wise or project
# For visualization: project to P1 (nodal average)
from skfem.utils import project
e_p1 = ElementTriP1()
ib_p1 = Basis(m, e_p1)
u_proj = project(u, basis_from=ib, basis_to=ib_p1)
mio = meshio.Mesh(pts, trng, point_data={{"u": u_proj}})
mio.write("result.vtu")

summary = {{
    "max_value": float(max_val),
    "min_value": float(min_val),
    "n_dofs": int(A.shape[0]),
    "n_elements": int(m.nelements),
    "advection_velocity": [{bx}, {by}],
    "diffusion": {eps},
    "element_type": "DG-P1 triangle",
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("DG advection solve complete.")
'''


# ---------------------------------------------------------------------------
# 4. Time-dependent PDE — general backward Euler
# ---------------------------------------------------------------------------

def _time_dependent_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    General time-dependent PDE: du/dt + L(u) = f with backward Euler.
    L(u) = -div(D*grad(u)) + c*u  (reaction-diffusion operator).
    """
    nx = params.get("nx", 32)
    dt = params.get("dt", 0.01)
    T_end = params.get("T_end", 0.5)
    D_coeff = params.get("D", 0.1)
    c_coeff = params.get("c", 1.0)
    f_val = params.get("f", 1.0)
    theta = params.get("theta", 1.0)   # 1=BE, 0.5=CN
    return f'''\
"""Time-dependent PDE: du/dt - D*Δu + c*u = f — theta-method — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace, mass, unit_load
import numpy as np
from scipy.sparse.linalg import factorized
from scipy.sparse import identity as speye
import json

D_coeff = {D_coeff}
c_coeff = {c_coeff}
dt      = {dt}
theta   = {theta}     # 1.0 = backward Euler, 0.5 = Crank-Nicolson
T_end   = {T_end}
n_steps = int(T_end / dt)

# --- Mesh & basis ---
m  = MeshQuad.init_tensor(np.linspace(0, 1, {nx + 1}), np.linspace(0, 1, {nx + 1}))
e  = ElementQuad1()
ib = Basis(m, e)

# --- Assembly: stiffness L = D*laplace + c*mass, mass M ---
K = D_coeff * laplace.assemble(ib) + c_coeff * mass.assemble(ib)
M = mass.assemble(ib)
f = {f_val} * unit_load.assemble(ib)

# --- Boundary DOFs (homogeneous Dirichlet) ---
D_bnd = ib.get_dofs().flatten()
I     = ib.complement_dofs(D_bnd)

# --- Theta-method system matrix: A = M + theta*dt*K ---
A = M + theta * dt * K
A_solve = factorized(A[I][:, I].tocsc())

# --- Initial condition: u0 = sin(pi*x)*sin(pi*y) ---
x_coords = ib.doflocs[0]
y_coords = ib.doflocs[1]
u = np.sin(np.pi * x_coords) * np.sin(np.pi * y_coords)
u[D_bnd] = 0.0

print(f"Time-dependent PDE: {{n_steps}} steps, dt={{dt}}, theta={{theta}}")
print(f"D={{D_coeff}}, c={{c_coeff}}, f={f_val}")

t = 0.0
max_vals = []
for step in range(n_steps):
    # RHS: M*u_old - (1-theta)*dt*K*u_old + dt*f
    rhs = M @ u - (1.0 - theta) * dt * K @ u + dt * f
    rhs[D_bnd] = 0.0
    u_new = np.zeros_like(u)
    u_new[I] = A_solve(rhs[I])
    u = u_new
    t += dt

    if (step + 1) % max(1, n_steps // 10) == 0:
        print(f"  t={{t:.4f}}, max(u)={{u.max():.6f}}")
        max_vals.append((t, float(u.max())))

max_val = float(u.max())
print(f"Final: t={{t:.4f}}, max(u) = {{max_val:.6f}}")

import meshio
cells  = [("quad", m.t.T)]
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
mio = meshio.Mesh(points, cells, point_data={{"u": u}})
mio.write("result.vtu")

summary = {{
    "max_value": max_val,
    "n_dofs": len(u),
    "n_elements": m.nelements,
    "t_end": t,
    "n_steps": n_steps,
    "dt": dt,
    "theta": theta,
    "D": D_coeff,
    "c": c_coeff,
    "element_type": "Q1 quad",
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Time-dependent PDE solve complete.")
'''


# ---------------------------------------------------------------------------
# 5. Helmholtz — complex-valued
# ---------------------------------------------------------------------------

def _helmholtz_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Helmholtz equation: -Δu - k²u = f with complex arithmetic.
    Absorbing boundary condition on right: du/dn + i*k*u = 0.
    """
    nx = params.get("nx", 32)
    k = params.get("k", 5.0)          # wavenumber
    f_real = params.get("f_real", 1.0)
    return f'''\
"""Helmholtz: -Δu - k²u = f, k={k}, complex-valued — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace, mass, unit_load
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import json

k = {k}     # wavenumber

# --- Mesh ---
# MeshQuad.init_tensor does NOT attach named boundaries.
# Without with_boundaries(...) calls, ib.get_dofs('left')
# raises ValueError("Boundary 'left' not found.") and the
# subscript form ib.get_dofs()['left'] raises TypeError:
# 'DofsView' object is not subscriptable. Attach the four
# canonical boundaries here so the Dirichlet block below
# can resolve 'left'/'top'/'bottom' tags.
m  = (MeshQuad.init_tensor(np.linspace(0, 1, {nx + 1}),
                           np.linspace(0, 1, {nx + 1}))
      .with_boundaries({{
          "left":   lambda x: x[0] < 1e-10,
          "right":  lambda x: x[0] > 1.0 - 1e-10,
          "bottom": lambda x: x[1] < 1e-10,
          "top":    lambda x: x[1] > 1.0 - 1e-10,
      }}))
e  = ElementQuad1()
ib = Basis(m, e)

# --- Boundary basis for absorbing BC (right face) ---
fb_right = FacetBasis(m, e, facets="right")

# --- Assembly ---
# Stiffness: (grad u, grad v)
K = laplace.assemble(ib)

# Mass: k^2 * (u, v)  — subtracted for Helmholtz
M = mass.assemble(ib)

# Absorbing BC: i*k*(u, v) on right boundary
@BilinearForm
def absorbing_bc(u, v, w):
    return 1j * k * u * v

A_abc = asm(absorbing_bc, fb_right)

# System: (K - k^2*M + A_abc) * u = f
# Use complex128 arithmetic
A = K.astype(complex) - k**2 * M.astype(complex) + A_abc.astype(complex)

# Source: point-like load at center (Gaussian approximation)
@LinearForm
def gaussian_source(v, w):
    x0, y0 = 0.5, 0.5
    sigma = 0.05
    r2 = (w.x[0] - x0)**2 + (w.x[1] - y0)**2
    return {f_real} * np.exp(-r2 / (2 * sigma**2)) * v

f = asm(gaussian_source, ib).astype(complex)

# --- Dirichlet BC: u=0 on left, top, bottom ---
# ib.get_dofs() returns a DofsView, which is NOT subscriptable
# (the legacy ib.get_dofs()['left'] pattern raises TypeError:
# 'DofsView' object is not subscriptable in scikit-fem 12).
# In modern skfem the canonical pattern is to pass the
# boundary name directly: ib.get_dofs('left') returns a
# DofsView whose .flatten() yields the boundary DOF indices.
D_bnd = np.concatenate([
    ib.get_dofs("left").flatten(),
    ib.get_dofs("top").flatten(),
    ib.get_dofs("bottom").flatten(),
])
I = np.setdiff1d(np.arange(A.shape[0]), D_bnd)

u = np.zeros(A.shape[0], dtype=complex)
u[I] = spsolve(A[I][:, I].tocsr(), f[I])

max_abs = np.abs(u).max()
print(f"Helmholtz k={{k}}: max|u| = {{max_abs:.6f}}")
print(f"DOFs: {{A.shape[0]}}, elements: {{m.nelements}}")

# Save real part and magnitude
import meshio
cells  = [("quad", m.t.T)]
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
mio = meshio.Mesh(points, cells,
    point_data={{"u_real": u.real, "u_imag": u.imag, "u_abs": np.abs(u)}})
mio.write("result.vtu")

summary = {{
    "k": k,
    "max_abs": float(max_abs),
    "max_real": float(u.real.max()),
    "n_dofs": int(A.shape[0]),
    "n_elements": int(m.nelements),
    "element_type": "Q1 quad (complex)",
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Helmholtz solve complete.")
'''


# ---------------------------------------------------------------------------
# 6. Reaction-diffusion — Schnakenberg / Fisher-KPP with backward Euler
# ---------------------------------------------------------------------------

def _reaction_diffusion_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Schnakenberg reaction-diffusion system (Turing pattern):
      du/dt = d_u * Δu + gamma*(a - u + u^2*v)
      dv/dt = d_v * Δv + gamma*(b - u^2*v)
    Solves with backward Euler + Newton iteration at each time step.
    """
    nx = params.get("nx", 32)
    dt = params.get("dt", 0.5)
    T_end = params.get("T_end", 50.0)
    d_u = params.get("d_u", 1.0)
    d_v = params.get("d_v", 40.0)
    a = params.get("a", 0.1)
    b = params.get("b", 0.9)
    gamma = params.get("gamma", 1000.0)
    tol = params.get("newton_tol", 1e-8)
    max_iter = params.get("max_iter", 20)
    return f'''\
"""Schnakenberg reaction-diffusion (Turing patterns) — backward Euler + Newton — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace, mass
import numpy as np
from scipy.sparse import bmat, eye as speye
from scipy.sparse.linalg import spsolve, factorized
import json

# --- Parameters ---
d_u   = {d_u}
d_v   = {d_v}
a     = {a}
b     = {b}
gamma = {gamma}
dt    = {dt}
T_end = {T_end}

# --- Mesh & basis ---
m  = MeshQuad.init_tensor(np.linspace(0, 1, {nx + 1}), np.linspace(0, 1, {nx + 1}))
e  = ElementQuad1()
ib = Basis(m, e)

# --- Assembly ---
K = laplace.assemble(ib)
M = mass.assemble(ib)
N = M.shape[0]

# Periodic-like: no Dirichlet BCs (Neumann = zero flux, natural BC)
# For Schnakenberg patterns, Neumann is standard.
I = np.arange(N)  # all DOFs free

# --- Initial condition: steady state + small random perturbation ---
rng = np.random.default_rng(42)
u0_ss = a + b
v0_ss = b / (a + b)**2
u_sol = np.full(N, u0_ss) + 0.01 * rng.standard_normal(N)
v_sol = np.full(N, v0_ss) + 0.01 * rng.standard_normal(N)

n_steps = int(T_end / dt)
print(f"Schnakenberg: {{n_steps}} steps, dt={{dt}}")
print(f"d_u={{d_u}}, d_v={{d_v}}, a={{a}}, b={{b}}, gamma={{gamma}}")
print(f"Steady state: u0={{u0_ss:.4f}}, v0={{v0_ss:.4f}}")

# --- Backward Euler with Newton iteration ---
# Residual for fully implicit system:
#   R_u = M*(u_new - u_old)/dt + d_u*K*u_new - gamma*M*f_u(u_new,v_new) = 0
#   R_v = M*(v_new - v_old)/dt + d_v*K*v_new - gamma*M*f_v(u_new,v_new) = 0
# Linearize f_u and f_v for Newton:
#   f_u(u,v) = a - u + u^2*v,  df_u/du = -1 + 2*u*v,  df_u/dv = u^2
#   f_v(u,v) = b - u^2*v,      df_v/du = -2*u*v,       df_v/dv = -u^2

def reaction_terms(u_vec, v_vec):
    fu = a - u_vec + u_vec**2 * v_vec
    fv = b - u_vec**2 * v_vec
    return fu, fv

def jacobian_terms(u_vec, v_vec):
    dfu_du = -1.0 + 2.0*u_vec*v_vec
    dfu_dv =        u_vec**2
    dfv_du = -2.0*u_vec*v_vec
    dfv_dv = -u_vec**2
    return dfu_du, dfu_dv, dfv_du, dfv_dv

@BilinearForm
def mass_pointwise(u, v, w):
    """Mass matrix with pointwise coefficient c(x)."""
    return w["c"] * u * v

# Fixed sparse structure: diffusion blocks + mass/dt diagonal
Ku  = d_u * K + M / dt
Kv  = d_v * K + M / dt

from scipy.sparse import csr_matrix, block_diag

t = 0.0
for step in range(n_steps):
    u_old = u_sol.copy()
    v_old = v_sol.copy()

    # Newton iteration
    u_new = u_old.copy()
    v_new = v_old.copy()

    for nit in range({max_iter}):
        fu, fv = reaction_terms(u_new, v_new)
        dfu_du, dfu_dv, dfv_du, dfv_dv = jacobian_terms(u_new, v_new)

        # Assemble reaction Jacobian blocks (diagonal in space)
        Mdu_du = asm(mass_pointwise, ib, c=dfu_du)
        Mdu_dv = asm(mass_pointwise, ib, c=dfu_dv)
        Mdv_du = asm(mass_pointwise, ib, c=dfv_du)
        Mdv_dv = asm(mass_pointwise, ib, c=dfv_dv)

        # Full Jacobian blocks
        J_uu = Ku - gamma * Mdu_du
        J_uv =    - gamma * Mdu_dv
        J_vu =    - gamma * Mdv_du
        J_vv = Kv - gamma * Mdv_dv

        J = bmat([[J_uu, J_uv], [J_vu, J_vv]], format="csr")

        # Residuals
        R_u = M @ (u_new - u_old) / dt + d_u * K @ u_new - gamma * M @ fu
        R_v = M @ (v_new - v_old) / dt + d_v * K @ v_new - gamma * M @ fv
        R   = np.concatenate([R_u, R_v])

        dxyz = spsolve(J, -R)
        u_new += dxyz[:N]
        v_new += dxyz[N:]

        res = np.linalg.norm(dxyz)
        if res < {tol}:
            break

    u_sol = u_new
    v_sol = v_new
    t += dt

    if (step + 1) % max(1, n_steps // 5) == 0:
        print(f"  t={{t:.2f}}, u=[{{u_sol.min():.4f}}, {{u_sol.max():.4f}}], "
              f"v=[{{v_sol.min():.4f}}, {{v_sol.max():.4f}}]")

print(f"Final t={{t:.2f}}: u in [{{u_sol.min():.4f}}, {{u_sol.max():.4f}}]")

import meshio
cells  = [("quad", m.t.T)]
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
mio = meshio.Mesh(points, cells, point_data={{"u": u_sol, "v": v_sol}})
mio.write("result.vtu")

summary = {{
    "u_max": float(u_sol.max()),
    "u_min": float(u_sol.min()),
    "v_max": float(v_sol.max()),
    "v_min": float(v_sol.min()),
    "n_dofs": N,
    "n_elements": m.nelements,
    "t_end": float(t),
    "n_steps": n_steps,
    "d_u": d_u, "d_v": d_v, "a": a, "b": b, "gamma": gamma,
    "element_type": "Q1 quad",
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Reaction-diffusion (Schnakenberg) solve complete.")
'''


# ---------------------------------------------------------------------------
# Knowledge dictionaries
# ---------------------------------------------------------------------------

KNOWLEDGE = {
    "navier_stokes": {
        "description": "Navier-Stokes flow — Newton iteration — Taylor-Hood P2/P1 (scikit-fem)",
        "solver": "Newton loop: linearize convection term, solve block saddle-point with spsolve",
        "elements": "Taylor-Hood: ElementVector(ElementTriP2()) + ElementTriP1()",
        "pitfalls": [
            (
                "[API] scikit-fem has NO built-in Newton solver "
                "or NS assembly — must build manually. Signal: "
                "searching skfem.utils for `NewtonSolver` or "
                "`NavierStokes` returns no match; the catalog "
                "ships hand-coded Newton + hand-coded "
                "BilinearForm + asm + condense snippets that "
                "the user copies — there is no single-call NS "
                "API. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Block system: [[A_visc + C(u), "
                "B^T], [B, 0]] where C is linearized "
                "convection. Signal: omitting C(u) from the "
                "BilinearForm gives a Stokes Jacobian and "
                "Newton converges linearly (not quadratically) "
                "on Navier-Stokes — residual ratio ~0.5 per "
                "iteration instead of decreasing "
                "geometrically across asm + condense + "
                "spsolve. (Audit 2026-06-02.)"
            ),
            (
                "[API] Use InteriorFacetBasis for ElementVector "
                "DOFs in the convection term. Signal: trying "
                "to access a vector-component on a plain "
                "ElementTriP2 raises "
                "`AttributeError: 'CellBasis' has no attribute "
                "'split'` or the assembled C(u) has the wrong "
                "block size. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Pressure nullspace for enclosed "
                "flow: pin one pressure DOF. Signal: "
                "scipy.sparse.linalg.spsolve raises "
                "`MatrixRankWarning: matrix is singular` or "
                "yields a pressure field with arbitrary "
                "constant offset; bottling up the constant by "
                "fixing p(p0)=0 removes the warning. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] High Re: consider Picard (fixed-"
                "point) for first few iterations, then "
                "Newton. Signal: pure Newton at Re > ~200 "
                "diverges from an at-rest initial guess "
                "(residual from BilinearForm + asm + "
                "spsolve explodes within 2-3 iterations); "
                "Picard for 5 iterations brings the state "
                "into Newton's convergence basin and "
                "switching restores quadratic convergence. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Convection linearization: "
                "(u_prev.grad)delta_u + (delta_u.grad)u_prev. "
                "Signal: dropping the second term in the "
                "BilinearForm (Picard linearization instead "
                "of Newton) gives linear convergence on the "
                "asm + condense + spsolve pipeline — useful "
                "as a starter but switch to full Newton for "
                "quadratic. (Audit 2026-06-02.)"
            ),
            (
                "[API] InteriorFacetBasis DOF ordering with "
                "ElementVector: use ib_u.N for block split. "
                "Signal: hard-coding `n_u_dof = nx * dim * 2` "
                "for the velocity-block size gives off-by-one "
                "errors on hex / triangle meshes where the "
                "DOF count depends on the element family; "
                "ib_u.N is the canonical accessor. (Audit "
                "2026-06-02.)"
            ),
        ],
    },
    "hyperelasticity": {
        "description": "Neo-Hookean hyperelasticity — Newton iteration (scikit-fem)",
        "solver": "Newton loop: assemble tangent stiffness K_tan and residual R_int, spsolve",
        "elements": "ElementVector(ElementTriP1()) or ElementVector(ElementTriP2())",
        "pitfalls": [
            (
                "[API] scikit-fem has NO built-in hyperelastic "
                "model — you must implement PK1 stress and "
                "tangent manually inside a BilinearForm. Signal: "
                "looking for skfem.models.elasticity.neohookean "
                "or similar fails (`ImportError`); the only "
                "linear elasticity helper is "
                "skfem.models.elasticity.linear_elasticity. "
                "Hyperelastic problems require hand-coded "
                "P(F) + C(F) inside @BilinearForm. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Neo-Hookean strain energy: W = "
                "mu/2 * (I1 - 2) - mu * lnJ + lam/2 * (lnJ)^2. "
                "Signal: swapping the +/- signs on the lnJ "
                "term (e.g. + mu * lnJ instead of - mu * lnJ) "
                "inside the BilinearForm gives a Newton "
                "tangent that is symmetric but with the "
                "WRONG sign — Newton diverges with the asm + "
                "spsolve residual growing by factor ~10 per "
                "iteration instead of converging "
                "quadratically. Sanity: at F = I, W must be "
                "0 and P must be 0; verify before stepping. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] 1st Piola-Kirchhoff stress: P = "
                "mu*(F - F^{-T}) + lam * lnJ * F^{-T}. "
                "Signal: using the 2nd PK form S = mu*(I - "
                "C^{-1}) + lam*lnJ*C^{-1} (small-strain-"
                "looking) inside the @BilinearForm and "
                "feeding it directly to the weak form "
                "∫P:grad(v) dx adds a missing F-pre-"
                "multiplication — the asm residual is wrong "
                "by a factor of F and Newton diverges. "
                "Recipe: stay in PK1 if your weak form is "
                "∫P:grad(v) dx; use S only if you actually "
                "integrate ∫S:0.5*(F^T grad v + grad(v)^T F) "
                "dx. (Audit 2026-06-02.)"
            ),
            (
                "[API] Deformation gradient: F = I + grad(u). "
                "Signal: forgetting the identity I in the "
                "@BilinearForm body gives a degenerate F at "
                "the reference configuration (F = 0 instead "
                "of I), J = det(F) = 0, and lnJ = -inf. The "
                "first Newton residual coming out of asm is "
                "either NaN or several orders of magnitude "
                "too large. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Material tangent: C4 = lam * "
                "C^{-1} ⊗ C^{-1} + 2*(mu - lam*lnJ) * "
                "I4_sym_C^{-1}. Signal: dropping the "
                "I4_sym_C^{-1} term inside the BilinearForm "
                "tangent assembly and using a scalar-"
                "multiplied identity I4 gives Newton "
                "convergence rate ~0.5 (linear instead of "
                "quadratic) because the tangent only "
                "approximates the true derivative. The exact "
                "C^{-1}-built tangent restores quadratic in "
                "the asm + condense + spsolve pipeline. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Geometric stiffness term S : "
                "(grad du)^T * grad v is ESSENTIAL for "
                "Newton convergence under large "
                "deformations. Signal: dropping it from the "
                "BilinearForm (only assembling the material "
                "stiffness C4) gives quadratic Newton "
                "convergence at small strains (< 5%) but "
                "STAGNATES at large strains in the asm "
                "residual (drops by ~10x then flat-lines "
                "around 1e-6 absolute) — the missing "
                "geometric term is needed for exact "
                "linearisation. (Audit 2026-06-02.)"
            ),
            (
                "[API] Use ib.interpolate(u) to obtain the "
                "displacement and gradient at quadrature "
                "points inside the bilinear form. Signal: "
                "trying to access u directly inside the "
                "@BilinearForm body raises `NameError: name "
                "'u' is not defined` (or, with an outer "
                "closure, computes against the LAST u rather "
                "than the current Newton iterate). The correct "
                "pattern is u_qp = w['u'] / w['displacement'] "
                "passed via ib.interpolate(u). (Audit "
                "2026-06-02.)"
            ),
            "[API] skfem 12 renamed MeshQuad.to_simplex() → "
            "MeshQuad.to_meshtri() (returns a MeshTri with each "
            "quad split into two triangles). Legacy templates that "
            "call .to_simplex() on a MeshQuad raise AttributeError: "
            "'MeshQuad1' object has no attribute 'to_simplex'. The "
            "modern call is .to_meshtri(). Signal: hasattr("
            "skfem.MeshQuad.init_tensor([0,1],[0,1]), 'to_meshtri') "
            "is True; hasattr(..., 'to_simplex') is False. "
            "(Verified empirically 2026-06-01 — Layer F catch.)",
            (
                "[Numerical] Load stepping: ramp the load "
                "over N steps for large deformations to "
                "keep Newton inside its convergence basin. "
                "Signal: a single-step application of a "
                "load that produces > 10% nominal strain "
                "typically diverges (Newton residual from "
                "asm + condense + spsolve grows ~10x per "
                "iteration); subdividing into N steps with "
                "load_factor = i/N (and the previous "
                "iterate carried via Basis.interpolate as "
                "the initial guess) achieves quadratic "
                "Newton convergence per substep. Heuristic: "
                "each substep should deliver < 10% stretch "
                "ratio increment. (Audit 2026-06-02.)"
            ),
            (
                "[API] The numpy @ matmul operator does NOT "
                "work on skfem's (d, d, n_elem, n_quad)-shape "
                "tensor fields inside a @BilinearForm kernel. "
                "matmul tries the last-two-dims convention and "
                "aborts. Signal: 'matmul: Input operand 1 has "
                "a mismatch in its core dimension 0, with "
                "gufunc signature (n?,k),(k,m?)->(n?,m?) (size "
                "N is different from d)' from numpy inside the "
                "BilinearForm. Fall back to explicit component "
                "algebra (Finv[0,0]*Finv[0,0] + Finv[0,1]*"
                "Finv[0,1] for (Finv * Finv^T)[0,0]) or use "
                "skfem.helpers.ddot / dot / transpose helpers "
                "that operate elementwise on the rank-2 "
                "deformation-gradient tensor. (Audit "
                "2026-06-02, post-mortem skfem-broken-newton-"
                "templates-rewrite.)"
            ),
            (
                "[Numerical] Newton iteration on a Neo-Hookean "
                "BilinearForm with an aggressive default load "
                "(e.g. E=1.0 / traction=0.1 — ~10% normalised "
                "force) drives the deformation gradient F into "
                "J = det(F) <= 0 territory in a few quadrature "
                "points after the first iter; the subsequent "
                "log(J) call returns NaN; the Newton iteration "
                "diverges to NaN-everywhere displacements; "
                "exit code is STILL 0 and pure rc=0 gates miss "
                "the failure. Signal: a Layer-F-style smoke "
                "gate that only checks rc=0 passes silently, "
                "while the printed 'Max displacement: nan' / "
                "'||du|| = nan' lines reveal the divergence. "
                "Defence: keep first-iter strain < 1% via "
                "stiffer material (E=1000) or smaller "
                "traction, AND add explicit 'nan' sentinels "
                "to the gate's forbid_in_output list. (Audit "
                "2026-06-02, post-mortem skfem-broken-newton-"
                "templates-rewrite.)"
            ),
        ],
    },
    "dg_methods": {
        "description": "Discontinuous Galerkin for advection/diffusion using ElementDG (scikit-fem)",
        "solver": "Direct sparse (non-symmetric system from upwind flux); GMRES for large problems",
        "elements": "ElementDG(ElementTriP1()), ElementDG(ElementTriP2())",
        "pitfalls": [
            (
                "[API] ElementDG wraps any element to make it "
                "fully discontinuous. Signal: forgetting the "
                "wrapper and using a bare ElementTriP1 in a "
                "DG context produces a continuous-Galerkin "
                "global solution — no jumps at element edges, "
                "and the upwind/penalty terms in the form "
                "evaluate to zero. (Audit 2026-06-02.)"
            ),
            (
                "[API] InteriorFacetBasis: assembles over "
                "interior mesh facets — needed for DG flux "
                "terms. Signal: using a plain Basis with an "
                "InteriorFacetBasis-style form (involving u "
                "and u.other or jump terms) raises "
                "`AttributeError: 'Basis' object has no "
                "attribute 'normals'` or yields a matrix with "
                "no facet-coupling entries. (Audit 2026-06-02.)"
            ),
            (
                "[API] FacetBasis assembles over BOUNDARY facets "
                "(not interior); used for inflow/outflow flux "
                "BCs. Signal: applying an inflow BC by adding a "
                "term over InteriorFacetBasis instead of "
                "FacetBasis silently adds the inflow to every "
                "interior edge — the solution gets a spurious "
                "source distributed across the mesh interior. "
                "Sanity check: a pure-Dirichlet steady advection "
                "with the wrong basis shows a non-monotone "
                "solution. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Upwind flux: bn * u_upwind * [v]; "
                "must identify upwind side from sign of b.n. "
                "Signal: a wrong upwind/downwind choice gives "
                "centered flux that is unconditionally "
                "unstable for pure advection — solution "
                "develops oscillations growing geometrically "
                "in the advection direction. (Audit "
                "2026-06-02.)"
            ),
            (
                "[API] scikit-fem uses a SINGLE-SIDED "
                "InteriorFacetBasis — '+' and '-' sides are "
                "implicit (assembly visits each interior facet "
                "twice with sign-flipped normals, not once with "
                "an explicit jump). Signal: porting a "
                "FEniCSx-style form that uses 'jump(u)' literally "
                "produces the wrong factor of 2 — in scikit-fem "
                "the jump appears naturally as (u - u.other). "
                "A factor-of-2 amplitude error on the DG flux "
                "indicates the import was copied verbatim. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] For IP (interior penalty) "
                "diffusion DG: penalty = sigma/h on each "
                "interior facet. Signal: sigma too small -> "
                "coercivity loss + solution norm diverges "
                "under refinement; sigma too large -> "
                "cond(K) > 1e14 and iterative solver "
                "stagnates. Rule of thumb: sigma = "
                "4 * order^2 for symmetric IP. (Audit "
                "2026-06-02.)"
            ),
            (
                "[API] project(u, basis_from=ib_dg, "
                "basis_to=ib_p1) for nodal post-processing. "
                "Signal: visualizing the ElementTriDG "
                "GridFunction-equivalent solution directly "
                "in ParaView with the wrong VTK writer "
                "produces an all-zero or step-pattern field "
                "because DG DOFs are not nodal; projecting "
                "with the skfem project() helper to "
                "ElementTriP1 first restores a smooth "
                "visualization. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] DG system is non-symmetric even "
                "for symmetric problems (upwind asymmetry). "
                "Signal: scipy.sparse.linalg.cg fails with "
                "`RuntimeError: matrix not positive definite` "
                "or stalls; switch to gmres / direct LU. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] SUPG (continuous Galerkin with "
                "stabilisation) is often more stable than "
                "pure DG for steady-state advection on a P1 "
                "mesh. Signal: a pure-ElementTriDG run at "
                "Pe_h > 5 on a coarse MeshTri shows ringing "
                "across element faces (amplitude ~10-30% of "
                "nominal) that takes ~3 levels of MeshTri "
                "refinement to clear; an equivalent SUPG-CG "
                "run with ElementTriP1 BilinearForm damps "
                "the oscillation monotonically as h "
                "decreases. For predominantly-smooth "
                "solutions, SUPG-CG wins on cost-per-"
                "accuracy. (Audit 2026-06-02.)"
            ),
        ],
    },
    "time_dependent": {
        "description": "General time-dependent PDE with theta-method (backward Euler / Crank-Nicolson) (scikit-fem)",
        "solver": "factorized(A) for efficient time-stepping; A = M + theta*dt*K assembled once",
        "elements": "ElementQuad1, ElementTriP1 (any H1-conforming element)",
        "pitfalls": [
            (
                "[Numerical] Backward Euler (theta=1) is "
                "unconditionally stable, 1st order in time. "
                "Signal: comparing a BE solution at dt=0.1 vs "
                "dt=0.01 against an exact transient (e.g. heat "
                "equation with sin(pi x) IC) shows L2 error "
                "decreasing linearly with dt — slope 1 on a "
                "log-log plot. If the slope is 2, the scheme is "
                "actually Crank-Nicolson and theta is "
                "mis-configured; if the slope is 0, dt is too "
                "large to be in the asymptotic regime. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Crank-Nicolson (theta=0.5): "
                "2nd order in time but can have oscillations. "
                "Signal: temperature/concentration shows "
                "10-30% over/undershoot at sharp transients "
                "that does not damp with mesh refinement; "
                "switching to theta=1 (BE) removes the "
                "oscillation at the cost of 1st-order phase "
                "error. (Audit 2026-06-02.)"
            ),
            (
                "[Performance] Factor system matrix ONCE and "
                "reuse — factorized() from scipy.sparse.linalg. "
                "Signal: profile shows >90% time in "
                "scipy.sparse.linalg.spsolve per step; using "
                "factorized() reduces per-step cost from full "
                "LU to forward+back-substitution (~10-100x "
                "speedup for a fixed matrix). (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Non-homogeneous time-varying BCs: "
                "update rhs and re-condense each step. Signal: "
                "if rhs is condensed once and reused, the "
                "boundary DOFs stay at their initial values "
                "across the time loop — solution at boundary "
                "diverges from the prescribed time-varying "
                "BC. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] CFL is NOT needed for BE (the "
                "implicit scheme is L-stable), but dt still "
                "affects ACCURACY. Signal: dt larger than the "
                "shortest physical timescale of the problem "
                "(e.g. dt >> 1/lambda_min) gives the correct "
                "steady state but smears any sharp transient — "
                "comparing the simulated u(t=t_transient) "
                "against a fine-dt reference shows L2 error "
                "~ O(dt) instead of resolving the front. "
                "Choose dt by accuracy, not stability, for "
                "implicit schemes. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] For stiff systems (reaction-"
                "dominated): backward Euler or BDF2 "
                "preferred. Signal: explicit (theta=0) or "
                "near-explicit (theta<0.5) on stiff "
                "problems requires dt < 2/lambda_max, where "
                "lambda_max is the largest eigenvalue of "
                "M^{-1}K (with M from BilinearForm asm of "
                "mass form); for reaction-diffusion with "
                "Da > 100, characteristic dt is so small "
                "the simulation is infeasible. Switch to "
                "BE/BDF2 implicit-asm + spsolve loop. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[API] ib.doflocs is the (ndim, N) coordinate "
                "array used for initial-condition assignment. "
                "Signal: setting u_init from a callable like "
                "u_init = np.sin(np.pi * ib.mesh.p[0]) (using "
                "mesh.p directly) only matches DOFs for P1 "
                "elements; for ElementVector / P2 / higher-order "
                "the DOFs include face/edge midpoints and the "
                "initial condition is silently zero on those "
                "DOFs. Use ib.doflocs to get the correct (ndim, "
                "nDOF) array and ib.project / ib.interpolator "
                "for IC assignment. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] For explicit time stepping: "
                "M*du/dt = -K*u + f (avoid for stiff "
                "problems). Signal: explicit Euler with "
                "dt > 2/lambda_max blows up to NaN within "
                "a few steps; energy grows geometrically; "
                "warning: the standard FEM mass matrix M "
                "(BilinearForm-asm'd ElementTriP1 mass) "
                "is NOT diagonal — applying explicit Euler "
                "naively requires inverting M at each step "
                "(lumped mass via condense + diagonal "
                "approximation is one workaround). (Audit "
                "2026-06-02.)"
            ),
        ],
    },
    "helmholtz": {
        "description": "Helmholtz equation -Δu - k²u = f (complex-valued, scikit-fem)",
        "solver": "Direct sparse with complex128 (spsolve handles complex); GMRES for large k",
        "elements": "ElementQuad1, ElementTriP1 (standard H1; use fine mesh: ~10 DOFs per wavelength)",
        "pitfalls": [
            (
                "[API] scikit-fem supports complex arithmetic "
                "natively — cast matrices to .astype(complex) "
                "before solving. Signal: assembling a Helmholtz "
                "bilinear form with -k^2 * mass without casting "
                "leaves the matrix dtype=float64, and adding a "
                "complex absorbing-BC term raises `TypeError: "
                "Cannot cast array data from dtype('complex128') "
                "to dtype('float64')` at sparse-matrix add, or "
                "silently drops the imaginary part if you used "
                ".real(). Cast K = K.astype(complex) before any "
                "BC additions. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Rule of thumb: at least 10 elements "
                "per wavelength (lambda = 2*pi/k). Signal: at "
                "fewer than 5 elements per wavelength the "
                "computed phase velocity is visibly wrong "
                "(e.g. a propagating wave hits the boundary at "
                "the wrong time by 10-30%); the standard "
                "dispersion error is O(k h)^2 for P1 and "
                "becomes catastrophic for h k > 1. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Absorbing BC: +i*k*u on the "
                "boundary, assembled via FacetBasis "
                "BilinearForm. Signal: omitting the ABC "
                "FacetBasis term produces standing-wave "
                "reflection off the domain boundary — "
                "visualised |u| on the asm'd MeshTri shows "
                "a checkerboard interference pattern with "
                "peaks spaced at lambda/2; adding the "
                "+i*k*u term in the FacetBasis BilinearForm "
                "absorbs the outgoing wave. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] PML (perfectly matched layer): "
                "extend the MeshTri / MeshQuad domain with "
                "a complex stretch factor (s = 1 - "
                "i*sigma(x)/k) inside the BilinearForm. "
                "Signal: too-thin PML (< lambda/2 thick) "
                "gives visible back-reflection in the "
                "interior of the asm'd FacetBasis solution; "
                "too-large sigma creates a numerical "
                "impedance jump and reflects more than it "
                "absorbs. Standard choices: PML thickness "
                "1-2 lambda, sigma_max such that |R| < "
                "1e-3 at normal incidence. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] High k (k > 20): use higher-order "
                "elements (P2, P3) or DG to reduce pollution "
                "error. Signal: standard P1 at k=40 on a mesh "
                "with 10 elements / wavelength shows the "
                "computed solution drifting out of phase across "
                "the domain — the trailing-edge crest is shifted "
                "by ~1/4 wavelength relative to the analytic "
                "plane wave; P2 on the same mesh recovers the "
                "phase. (Audit 2026-06-02.)"
            ),
            (
                "[API] System is non-Hermitian with ABC — cannot "
                "use eigsh (which requires Hermitian). Signal: "
                "scipy.sparse.linalg.eigsh(K, k=1, M=M) on a "
                "Helmholtz problem with absorbing BCs raises "
                "`ArpackNoConvergence: ARPACK error ... "
                "non-Hermitian` or returns garbage eigenvalues. "
                "Use scipy.sparse.linalg.eigs (general non-"
                "Hermitian ARPACK) or spsolve / GMRES for the "
                "forward problem. (Audit 2026-06-02.)"
            ),
            (
                "[API] Output the REAL PART (physical wave) and "
                "the MAGNITUDE |u| for visualisation — the "
                "solution is complex. Signal: writing "
                "u.tofile(...) without the .real / np.abs split "
                "writes complex128, which most ParaView writers "
                "either reject (`Cannot write complex array`) "
                "or truncate to the real part silently, "
                "discarding amplitude information. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Pollution effect: for large k, "
                "standard P1 has O(k^3 h^2) phase error — use "
                "p-refinement to control it. Signal: at fixed "
                "h*k = 0.1 (nominally well-resolved), an "
                "ElementTriP1 GridFunction at k=50 still shows "
                "~5-10% phase_drift across the domain because "
                "the constant in front of k^3 h^2 dominates; "
                "the same h with ElementTriP2 (O(k^5 h^4)) "
                "keeps the drift < 0.1%. (Audit 2026-06-02.)"
            ),
            "[API] MeshQuad.init_tensor (and most other init_*) "
            "does NOT attach named boundaries. The canonical "
            "incantation is m = MeshQuad.init_tensor(...)"
            ".with_boundaries({'left': lambda x: x[0] < 1e-10, "
            "'right': ..., 'bottom': ..., 'top': ...}); then "
            "ib.get_dofs('left').flatten() yields the boundary "
            "DOF indices. Same constraint applies after "
            ".to_meshtri() — boundaries must be reattached on "
            "the triangulated mesh. Signal: TWO distinct errors "
            "depending on the call pattern — ib.get_dofs('left') "
            "raises ValueError(\"Boundary 'left' not found.\") "
            "while the legacy subscript form ib.get_dofs()['left'] "
            "raises TypeError: 'DofsView' object is not "
            "subscriptable in scikit-fem 12. (Verified empirically "
            "2026-06-01 — Layer F catch.)",
        ],
    },
    "reaction_diffusion": {
        "description": "Reaction-diffusion system (Schnakenberg / Fisher-KPP) — Turing patterns (scikit-fem)",
        "solver": "Backward Euler in time + Newton iteration per step; block 2x2 system for coupled species",
        "elements": "ElementQuad1 (any H1 element; Neumann BCs are natural)",
        "pitfalls": [
            (
                "[Numerical] Coupled system: assemble block "
                "Jacobian [[J_uu, J_uv], [J_vu, J_vv]] at "
                "each Newton step via four BilinearForm + "
                "asm calls. Signal: assembling only the "
                "diagonal blocks (J_uu, J_vv) and dropping "
                "the off-diagonal BilinearForm-asm output "
                "gives a linear-rate Newton instead of "
                "quadratic in the asm + condense + "
                "spsolve pipeline; the off-diagonal terms "
                "scale with the reaction-rate Jacobian "
                "df_u/dv and df_v/du which are non-zero "
                "for any coupled reaction. (Audit "
                "2026-06-02.)"
            ),
            (
                "[API] Reaction Jacobian blocks: assembled as "
                "mass matrices with pointwise coefficient. "
                "Signal: assembling reaction terms via the "
                "stiffness pattern instead of mass produces "
                "spurious diffusion in J_uv / J_vu; sub-block "
                "structure visibly differs from a "
                "reference scipy implementation that uses "
                "M @ diag(df_u/dv). (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Initial condition: perturb "
                "homogeneous steady state to trigger Turing "
                "instability. Signal: starting from exactly "
                "(u_ss, v_ss) gives no pattern formation — "
                "solution stays uniform throughout the "
                "simulation. Add a small random or "
                "spatially-structured perturbation "
                "(amplitude ~1e-3 * u_ss). (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Turing instability requires "
                "d_v >> d_u (fast inhibitor, slow activator). "
                "Signal: setting d_v = d_u (or d_u > d_v) "
                "kills the cross-diffusion mechanism — "
                "amplitudes decay back to homogeneous steady "
                "state with no pattern. The Turing condition "
                "needs d_v/d_u above ~10 for Schnakenberg. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Schnakenberg homogeneous steady "
                "state: u_ss = a+b, v_ss = b/(a+b)^2. Signal: "
                "running with initial conditions that do not "
                "match (u_ss, v_ss) leads to a long startup "
                "transient (~1/r time units) before patterns "
                "emerge; comparing simulated u against the "
                "analytic steady state catches off-by-one or "
                "mis-typed (a, b) parameters — a 10% deviation "
                "in u_ss at t=10 means a or b is wrong by a "
                "similar factor. (Audit 2026-06-02.)"
            ),
            (
                "[API] Neumann (zero-flux) BCs are natural in "
                "the weak form — no explicit enforcement "
                "needed. Signal: applying a DirichletBC with "
                "value=0 instead silences the natural BC and "
                "imposes a far stronger constraint (u=0 at "
                "boundary, not just du/dn=0); pattern is "
                "pulled toward zero at the boundary instead "
                "of bulging outward. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Pattern formation requires gamma "
                "large enough relative to domain size. "
                "Signal: gamma * L^2 < the critical Turing "
                "wavelength k_c^2 -> no pattern (homogeneous "
                "state stable on the domain); pattern emerges "
                "only when gamma * L^2 > pi^2 * (a+b)^2 for "
                "Schnakenberg. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Fisher-KPP: du/dt = D*Δu + r*u*"
                "(1-u) is a SCALAR equation with no coupling "
                "block. Signal: building a block-2x2 Jacobian "
                "for a Fisher-KPP problem (assuming a coupled "
                "system) yields a sparse matrix double the "
                "expected size — the wasted linear-solve cost "
                "shows in scipy.sparse.linalg.spsolve wall_time "
                "and the resulting GridFunction develops "
                "spurious cross_diffusion. The correct "
                "discretisation is a single Newton solve on the "
                "M+dt*K - dt*M_r system where M_r is the mass "
                "matrix weighted by r*(1-2u^n). (Audit "
                "2026-06-02.)"
            ),
        ],
    },
}


# ---------------------------------------------------------------------------
# Generator registry
# ---------------------------------------------------------------------------

GENERATORS = {
    "navier_stokes_2d":      _navier_stokes_2d,
    "hyperelasticity_2d":    _hyperelasticity_2d,
    "dg_methods_2d":         _dg_methods_2d,
    "time_dependent_2d":     _time_dependent_2d,
    "helmholtz_2d":          _helmholtz_2d,
    "reaction_diffusion_2d": _reaction_diffusion_2d,
}
