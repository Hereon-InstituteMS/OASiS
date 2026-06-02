"""scikit-fem advanced physics generators and knowledge.

Covers: Navier-Stokes, hyperelasticity (Neo-Hookean), DG advection,
time-dependent PDE, Helmholtz (complex), and reaction-diffusion.
"""


# ---------------------------------------------------------------------------
# 1. Navier-Stokes (Newton iteration)
# ---------------------------------------------------------------------------

def _navier_stokes_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Navier-Stokes on lid-driven cavity with Newton iteration.
    Velocity: P2 triangle, pressure: P1 triangle (Taylor-Hood).
    """
    nx = params.get("nx", 16)
    Re = params.get("Re", 100.0)
    tol = params.get("newton_tol", 1e-8)
    max_iter = params.get("max_iter", 20)
    return f'''\
"""Navier-Stokes lid-driven cavity — Newton iteration — Taylor-Hood P2/P1 — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace, mass
from skfem.models.general import divergence
import numpy as np
from scipy.sparse import bmat, eye as speye
from scipy.sparse.linalg import spsolve
import json

Re = {Re}

# --- Mesh & bases ---
m = MeshTri.init_sqsymmetric().refined({max(1, int(nx / 8))})
# Refine to get roughly nx cells along each edge
while m.p.max() > 0 and m.nelements < {nx}**2 // 2:
    m = m.refined()

e_u = ElementVector(ElementTriP2())
e_p = ElementTriP1()
ib_u = Basis(m, e_u)
ib_p = Basis(m, e_p)

N_u = ib_u.N
N_p = ib_p.N
N_total = N_u + N_p

# --- Viscous (Laplacian) block ---
A_visc = asm(laplace, ib_u) / Re  # kinematic viscosity 1/Re

# --- Divergence block B (N_p x N_u) ---
B = -asm(divergence, ib_u, ib_p)

# --- Boundary DOFs ---
# Lid (top): u_x = 1, u_y = 0
# Walls (left, right, bottom): u = 0
dofs_u = ib_u.get_dofs()
top_dofs_x  = dofs_u["top"].nodal["u^1"]    # x-component on top
top_dofs_y  = dofs_u["top"].nodal["u^2"]    # y-component on top
wall_dofs    = np.concatenate([
    dofs_u["left"].flatten(),
    dofs_u["right"].flatten(),
    dofs_u["bottom"].flatten(),
    top_dofs_y,
])
all_D_u = np.unique(np.concatenate([top_dofs_x, wall_dofs]))

# Pressure pin: fix pressure at node 0 to remove nullspace
pin_p = np.array([N_u])  # global index of first pressure DOF

# Free DOFs
all_D = np.unique(np.concatenate([all_D_u, pin_p]))
I = np.setdiff1d(np.arange(N_total), all_D)

# --- Initial guess: zero ---
x = np.zeros(N_total)
x[top_dofs_x] = 1.0   # lid BC

# --- Nonlinear convection forms (depend on current velocity) ---
@BilinearForm
def convection_jac(u, v, w):
    """Jacobian contribution: (u_prev . grad) u . v + (u . grad) u_prev . v"""
    up = w["up"]   # interpolated previous velocity, shape (2,)
    # (u_prev . grad) u . v
    adv1 = (up[0] * u.grad[0][0] + up[1] * u.grad[0][1]) * v[0] \
         + (up[0] * u.grad[1][0] + up[1] * u.grad[1][1]) * v[1]
    # (u . grad) u_prev . v  — linearization term
    upg = w["upg"]  # gradient of u_prev at quadrature points, shape (2,2)
    adv2 = (u[0] * upg[0][0] + u[1] * upg[0][1]) * v[0] \
         + (u[0] * upg[1][0] + u[1] * upg[1][1]) * v[1]
    return adv1 + adv2

@LinearForm
def convection_res(v, w):
    """Residual: (u_prev . grad) u_prev . v"""
    up  = w["up"]
    upg = w["upg"]
    return (up[0] * upg[0][0] + up[1] * upg[0][1]) * v[0] \
         + (up[0] * upg[1][0] + up[1] * upg[1][1]) * v[1]

@LinearForm
def divergence_res(v, w):
    """Divergence residual: div(u_prev) * q"""
    upg = w["upg"]
    return (upg[0][0] + upg[1][1]) * v

# Zero block for pressure-pressure part
from scipy.sparse import csr_matrix
Z_pp = csr_matrix((N_p, N_p))

# --- Newton loop ---
for it in range({max_iter}):
    # Extract velocity part of solution
    u_vec = x[:N_u]

    # Interpolate velocity and its gradient at quadrature points
    u_intp  = ib_u.interpolate(u_vec)
    up_val  = u_intp.value      # (2, n_quad_pts) array
    upg_val = u_intp.grad       # (2, 2, n_quad_pts) array

    # Assemble convection Jacobian and residuals
    C = asm(convection_jac, ib_u, up=up_val, upg=upg_val)
    r_u = asm(convection_res, ib_u, up=up_val, upg=upg_val)
    r_p = asm(divergence_res, ib_p, upg=upg_val)

    # Full system Jacobian:  J = [[A_visc + C, B^T], [B, 0]]
    J = bmat([[A_visc + C, B.T], [B, Z_pp]], format="csr")

    # Full residual:  R = J_lin * x + nonlinear_res
    R = np.zeros(N_total)
    R[:N_u] = (A_visc + C) @ u_vec + B.T @ x[N_u:] - r_u
    # Wait — correct residual is: viscous*u + B^T*p - r_conv + B*u - r_div
    # Re-assemble correctly:
    R[:N_u] = A_visc @ u_vec + B.T @ x[N_u:] + r_u
    R[N_u:] = B @ u_vec - r_p

    # Apply BCs
    R[all_D] = 0.0

    dx = np.zeros(N_total)
    dx[I] = spsolve(J[I][:, I], -R[I])

    x += dx
    res_norm = np.linalg.norm(dx[I])
    print(f"Newton it {{it+1}}: ||dx|| = {{res_norm:.4e}}")
    if res_norm < {tol}:
        print(f"Converged in {{it+1}} Newton iterations")
        break

u_sol = x[:N_u].reshape(2, -1)
p_sol = x[N_u:]
max_vel = np.sqrt(u_sol[0]**2 + u_sol[1]**2).max()
print(f"Max velocity magnitude: {{max_vel:.6f}}")
print(f"Re = {Re}, DOFs = {{N_total}}")

import meshio
pts  = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
trng = [("triangle", m.t.T)]
mio  = meshio.Mesh(pts, trng)
mio.write("result.vtu")

summary = {{
    "Re": {Re},
    "max_velocity": float(max_vel),
    "n_dofs": N_total,
    "n_elements": m.nelements,
    "newton_iter": it + 1,
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
    E = params.get("E", 1.0)
    nu = params.get("nu", 0.3)
    traction = params.get("traction", 0.1)
    tol = params.get("newton_tol", 1e-8)
    max_iter = params.get("max_iter", 30)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return f'''\
"""Neo-Hookean hyperelasticity — Newton iteration — scikit-fem"""
from skfem import *
import numpy as np
from scipy.sparse.linalg import spsolve
import json

# Lame parameters from E={E}, nu={nu}
lam = {lam:.6f}
mu  = {mu:.6f}

# --- Mesh ---
# MeshQuad.init_tensor + to_meshtri does NOT auto-attach
# named boundaries — attach them via with_boundaries so
# ib.get_dofs('left') succeeds.
m = (MeshQuad.init_tensor(
        np.linspace(0, {lx}, {nx + 1}),
        np.linspace(0, {ly}, {ny + 1}),
     ).to_meshtri()  # skfem 12: to_simplex was renamed to to_meshtri
      .with_boundaries({{
          "left":   lambda x: x[0] < 1e-10,
          "right":  lambda x: x[0] > {lx} - 1e-10,
          "bottom": lambda x: x[1] < 1e-10,
          "top":    lambda x: x[1] > {ly} - 1e-10,
      }}))
e = ElementVector(ElementTriP1())
ib = Basis(m, e)

N = ib.N          # total DOFs
ndim = 2

# --- Boundary DOFs ---
# ib.get_dofs() returns a DofsView (not subscriptable);
# pass the boundary tag in directly.
fix_dofs = ib.get_dofs("left").flatten()   # clamped left edge

I = np.setdiff1d(np.arange(N), fix_dofs)

# --- Neo-Hookean Piola-Kirchhoff stress and tangent ---
@BilinearForm
def neo_hookean_tangent(u, v, w):
    """Linearized tangent stiffness at current deformation u_prev."""
    # Deformation gradient: F = I + grad(u_prev)
    F00 = 1.0 + w["dux_dx"];  F01 = w["dux_dy"]
    F10 = w["duy_dx"];         F11 = 1.0 + w["duy_dy"]
    J   = F00 * F11 - F01 * F10
    Jm  = 1.0 / J

    # Inverse transpose of F
    Finv00 =  F11 * Jm;  Finv01 = -F01 * Jm
    Finv10 = -F10 * Jm;  Finv11 =  F00 * Jm

    # Tangent modulus components (Voigt-style computation inline):
    # C = lam*(lnJ)*Finv⊗Finv + (mu - lam*lnJ)*I4  (push-forward)
    # Here we use the material (Lagrangian) form for assembly simplicity.
    lnJ = np.log(J)

    # Grad of test/trial increments
    du_dx = u.grad[0];  du_dy = u.grad[1]   # du/dx, du/dy  (scalar component index)
    dv_dx = v.grad[0];  dv_dy = v.grad[1]

    # P2 = dW/dF (1st PK stress) linearization: dP:dF (contracted with grad v)
    # For Neo-Hookean: S = mu*(I - C^{-1}) + lam*lnJ*C^{-1}
    # We assemble the material tangent term: dP(u_prev) : (I ⊗ grad(delta_u)) : grad(v)
    # Simplified planar tangent (isotropic, 2D plane strain):
    c1 = mu
    c2 = lam * lnJ - mu
    c3 = lam

    # Cauchy-Green: C_ij = F^T F
    C00 = F00*F00 + F10*F10
    C01 = F00*F01 + F10*F11
    C11 = F01*F01 + F11*F11

    # C^{-1}
    detC = C00*C11 - C01*C01
    Cinv00 = C11 / detC;  Cinv01 = -C01 / detC;  Cinv11 = C00 / detC

    # S (2nd PK stress)
    S00 = c1 + c2*Cinv00 + c3*lnJ*Cinv00 - c1*Cinv00
    S01 = c2*Cinv01 - c1*Cinv01
    S11 = c1 + c2*Cinv11 + c3*lnJ*Cinv11 - c1*Cinv11
    # More directly: S = (mu - lam*lnJ)*I + lam*lnJ*Cinv  (using standard result)
    alpha = mu - lam * lnJ
    beta  = lam
    S00 = alpha + (alpha + beta*lnJ) * (Cinv00 - 1.0/mu)
    # Restart with standard formula: S = mu*(I - C^{-1}) + lam*lnJ*C^{-1}
    S00 = mu * (1.0 - Cinv00) + lam * lnJ * Cinv00
    S01 = mu * (0.0 - Cinv01) + lam * lnJ * Cinv01
    S11 = mu * (1.0 - Cinv11) + lam * lnJ * Cinv11

    # Geometric stiffness: S : (grad u_delta)^T * grad v
    geom  = (S00 * du_dx * dv_dx
           + S01 * (du_dx * dv_dy + du_dy * dv_dx)
           + S11 * du_dy * dv_dy)

    # Material stiffness: linearized C4 term
    # Cijkl = lam*Cinv_ij*Cinv_kl + 2*(mu-lam*lnJ)*I4sym_Cinv
    # Contraction: F^T grad_delta_u = E_delta (in material frame)
    # We build the 4th-order contraction explicitly for 2D:
    def sym_prod(a, b, c, d):
        """(Cinv_ij * Cinv_kl) contracted with (grad u)_(kl) and (grad v)_(ij)"""
        return (a * c) * (b * d)

    # A simplified but consistent material tangent (plane strain):
    mat_00 = lam * Cinv00 * Cinv00 + 2.0*(mu - lam*lnJ)*(Cinv00*Cinv00)
    mat_01 = lam * Cinv00 * Cinv01
    mat_11 = lam * Cinv11 * Cinv11 + 2.0*(mu - lam*lnJ)*(Cinv01*Cinv01 + Cinv00*Cinv11)

    # Full linearized internal work (material + geometric)
    # Map to spatial frame: via F
    E_delta_00 = F00*du_dx + F10*du_dy   # F^T * grad(delta_u) row 0, col 0
    E_delta_01 = F01*du_dx + F11*du_dy
    E_delta_10 = F00*du_dx + F10*du_dy   # symmetric
    E_delta_11 = F01*du_dy + F11*du_dy   # typo — build properly

    # Use a cleaner pull-back: strain increment dE = sym(F^T grad delta_u)
    dE00 = F00*du_dx + F10*du_dy
    dE11 = F01*du_dx + F11*du_dy
    dE01 = 0.5*(F00*du_dy + F10*du_dx + F01*du_dy + F11*du_dx)  # simplified

    dEv00 = F00*dv_dx + F10*dv_dy
    dEv11 = F01*dv_dx + F11*dv_dy

    # C4:dE : dEv  (Cijkl dE_kl dEv_ij)
    mat_stiff = (lam*(Cinv00*dE00 + Cinv01*(dE01+dE01) + Cinv11*dE11)
                      * (Cinv00*dEv00 + Cinv11*dEv11)
               + 2.0*(mu - lam*lnJ)*(Cinv00*Cinv00*dE00*dEv00
                                   + Cinv11*Cinv11*dE11*dEv11
                                   + Cinv01*Cinv01*(dE01*dEv00 + dE01*dEv11)))

    return geom + mat_stiff

@LinearForm
def internal_forces(v, w):
    """Internal virtual work: P : grad(v)"""
    F00 = 1.0 + w["dux_dx"];  F01 = w["dux_dy"]
    F10 = w["duy_dx"];         F11 = 1.0 + w["duy_dy"]
    J   = F00 * F11 - F01 * F10
    Jm  = 1.0 / J
    lnJ = np.log(J)

    # C^{-1}
    C00 = F00*F00 + F10*F10
    C01 = F00*F01 + F10*F11
    C11 = F01*F01 + F11*F11
    detC = C00*C11 - C01*C01
    Cinv00 = C11/detC;  Cinv01 = -C01/detC;  Cinv11 = C00/detC

    # 2nd PK stress S
    S00 = mu*(1.0 - Cinv00) + lam*lnJ*Cinv00
    S01 = mu*(0.0 - Cinv01) + lam*lnJ*Cinv01
    S11 = mu*(1.0 - Cinv11) + lam*lnJ*Cinv11

    # 1st PK stress P = F*S
    P00 = F00*S00 + F01*S01
    P01 = F00*S01 + F01*S11
    P10 = F10*S00 + F11*S01
    P11 = F10*S01 + F11*S11

    # P : grad(v) = P_ij * dv_i/dx_j
    dv0_dx = v.grad[0];  dv0_dy = v.grad[1]
    dv1_dx = v.grad[2];  dv1_dy = v.grad[3]

    return P00*dv0_dx + P01*dv0_dy + P10*dv1_dx + P11*dv1_dy

@LinearForm
def external_traction(v, w):
    """Neumann BC: traction on right face."""
    return {traction} * v[0]   # x-traction

fb_right = FacetBasis(m, e, facets=m.facets_satisfying(lambda x: x[0] > {lx} - 1e-10))

# --- Initial displacement: zero ---
u_disp = np.zeros(N)

# --- Newton loop ---
for it in range({max_iter}):
    u_intp = ib.interpolate(u_disp)
    # Extract displacement gradient components at quadrature points
    dux_dx = u_intp.grad[0][0]   # du_x/dx
    dux_dy = u_intp.grad[0][1]
    duy_dx = u_intp.grad[1][0]
    duy_dy = u_intp.grad[1][1]

    K_tan = asm(neo_hookean_tangent, ib,
                dux_dx=dux_dx, dux_dy=dux_dy,
                duy_dx=duy_dx, duy_dy=duy_dy)
    R_int = asm(internal_forces, ib,
                dux_dx=dux_dx, dux_dy=dux_dy,
                duy_dx=duy_dx, duy_dy=duy_dy)
    F_ext = asm(external_traction, fb_right)

    # Residual: R = R_int - F_ext
    R = R_int - F_ext
    R[fix_dofs] = 0.0

    du = np.zeros(N)
    du[I] = spsolve(K_tan[I][:, I], -R[I])
    u_disp += du

    res = np.linalg.norm(du[I])
    print(f"Newton it {{it+1}}: ||du|| = {{res:.4e}}")
    if res < {tol}:
        print(f"Converged in {{it+1}} iterations")
        break

u_xy = u_disp.reshape(2, -1)
max_disp = np.abs(u_xy).max()
print(f"Max displacement: {{max_disp:.6f}}")
print(f"E={E}, nu={nu}, traction={traction}")

import meshio
pts  = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
trng = [("triangle", m.t.T)]
u_node = np.column_stack([u_xy[0], u_xy[1], np.zeros(m.p.shape[1])])
mio  = meshio.Mesh(pts, trng, point_data={{"displacement": u_node}})
mio.write("result.vtu")

summary = {{
    "max_displacement": float(max_disp),
    "n_dofs": N,
    "n_elements": m.nelements,
    "newton_iter": it + 1,
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
                "ships hand-coded Newton+block-assembly "
                "snippets that the user copies — there is no "
                "single-call NS API. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Block system: [[A_visc + C(u), "
                "B^T], [B, 0]] where C is linearized "
                "convection. Signal: omitting C(u) gives a "
                "Stokes Jacobian and Newton converges "
                "linearly (not quadratically) on Navier-Stokes "
                "— residual ratio ~0.5 per iteration instead "
                "of decreasing geometrically. (Audit "
                "2026-06-02.)"
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
                "point) for first few iterations, then Newton. "
                "Signal: pure Newton at Re > ~200 diverges "
                "from an at-rest initial guess (residual "
                "explodes within 2-3 iterations); Picard for "
                "5 iterations brings the state into Newton's "
                "convergence basin and switching restores "
                "quadratic convergence. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Convection linearization: "
                "(u_prev.grad)delta_u + (delta_u.grad)u_prev. "
                "Signal: dropping the second term (Picard "
                "linearization instead of Newton) gives "
                "linear convergence — useful as a starter but "
                "switch to full Newton for quadratic. (Audit "
                "2026-06-02.)"
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
            "No built-in hyperelastic model — must implement PK1 stress and tangent manually",
            "Neo-Hookean: W = mu/2*(I1-2) - mu*lnJ + lam/2*(lnJ)^2",
            "1st PK stress: P = mu*(F - F^{-T}) + lam*lnJ*F^{-T}",
            "Deformation gradient: F = I + grad(u_displacement)",
            "Material tangent: C4 = lam*Cinv⊗Cinv + 2*(mu-lam*lnJ)*I4_sym_Cinv",
            "Geometric stiffness: S : (grad delta_u)^T * grad v  (essential for convergence)",
            "Use ib.interpolate(u) to get displacement gradient at quadrature points",
            "[API] skfem 12 renamed MeshQuad.to_simplex() → "
            "MeshQuad.to_meshtri() (returns a MeshTri with each "
            "quad split into two triangles). Legacy templates that "
            "call .to_simplex() on a MeshQuad raise AttributeError: "
            "'MeshQuad1' object has no attribute 'to_simplex'. The "
            "modern call is .to_meshtri(). Signal: hasattr("
            "skfem.MeshQuad.init_tensor([0,1],[0,1]), 'to_meshtri') "
            "is True; hasattr(..., 'to_simplex') is False. "
            "(Verified empirically 2026-06-01 — Layer F catch.)",
            "Load stepping: ramp load for large deformations to keep Newton in convergence basin",
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
            "FacetBasis: assembles over boundary facets — for inflow/outflow BC",
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
            "scikit-fem uses a single-sided InteriorFacetBasis — '+' and '-' sides implicit",
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
                "Signal: visualizing the DG solution directly "
                "in ParaView with the wrong VTK writer "
                "produces an all-zero or step-pattern field "
                "because DG DOFs are not nodal; projecting "
                "to P1 first restores a smooth visualization. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] DG system is non-symmetric even "
                "for symmetric problems (upwind asymmetry). "
                "Signal: scipy.sparse.linalg.cg fails with "
                "`RuntimeError: matrix not positive definite` "
                "or stalls; switch to gmres / direct LU. "
                "(Audit 2026-06-02.)"
            ),
            "SUPG (continuous Galerkin with stabilization) is often more stable than pure DG for advection",
        ],
    },
    "time_dependent": {
        "description": "General time-dependent PDE with theta-method (backward Euler / Crank-Nicolson) (scikit-fem)",
        "solver": "factorized(A) for efficient time-stepping; A = M + theta*dt*K assembled once",
        "elements": "ElementQuad1, ElementTriP1 (any H1-conforming element)",
        "pitfalls": [
            "Backward Euler (theta=1): unconditionally stable, 1st order in time",
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
            "CFL condition: not needed for BE (implicit), but affects accuracy",
            (
                "[Numerical] For stiff systems (reaction-"
                "dominated): backward Euler or BDF2 preferred. "
                "Signal: explicit (theta=0) or near-explicit "
                "(theta<0.5) on stiff problems requires "
                "dt < 2/lambda_max, where lambda_max is the "
                "largest eigenvalue of M^{-1}K; for "
                "reaction-diffusion with Da > 100, "
                "characteristic dt is so small the simulation "
                "is infeasible. Switch to BE/BDF2. (Audit "
                "2026-06-02.)"
            ),
            "doflocs property: ib.doflocs gives (ndim, N) coordinate array for initial conditions",
            (
                "[Numerical] For explicit time stepping: "
                "M*du/dt = -K*u + f (avoid for stiff "
                "problems). Signal: explicit Euler with "
                "dt > 2/lambda_max blows up to NaN within a "
                "few steps; energy grows geometrically; "
                "warning: the standard FEM mass matrix M is "
                "NOT diagonal — applying explicit Euler "
                "naively requires inverting M at each step "
                "(lumped mass is one workaround). (Audit "
                "2026-06-02.)"
            ),
        ],
    },
    "helmholtz": {
        "description": "Helmholtz equation -Δu - k²u = f (complex-valued, scikit-fem)",
        "solver": "Direct sparse with complex128 (spsolve handles complex); GMRES for large k",
        "elements": "ElementQuad1, ElementTriP1 (standard H1; use fine mesh: ~10 DOFs per wavelength)",
        "pitfalls": [
            "scikit-fem supports complex arithmetic natively — cast matrices to .astype(complex)",
            "Rule of thumb: at least 10 elements per wavelength (lambda = 2*pi/k)",
            "Absorbing BC: +i*k*u on boundary via FacetBasis BilinearForm",
            "PML (perfectly matched layer): extend domain with complex stretch factor",
            "High k (k > 20): consider higher-order elements (P2, P3) or DG to reduce pollution error",
            "System is non-Hermitian with ABC — cannot use eigsh; use spsolve or GMRES",
            "Output real part (physical wave) and magnitude |u| for visualization",
            "Pollution effect: for large k, standard P1 has O(k^3 h^2) phase error — use p-refinement",
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
                "Jacobian [[J_uu, J_uv], [J_vu, J_vv]] at each "
                "Newton step. Signal: assembling only the "
                "diagonal blocks (J_uu, J_vv) and dropping the "
                "off-diagonal coupling gives a linear-rate "
                "Newton instead of quadratic; the off-diagonal "
                "terms scale with the reaction-rate Jacobian "
                "df_u/dv and df_v/du which are non-zero for "
                "any coupled reaction. (Audit 2026-06-02.)"
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
            "Schnakenberg steady state: u_ss = a+b, v_ss = b/(a+b)^2",
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
            "For Fisher-KPP: du/dt = D*Δu + r*u*(1-u), scalar equation, no coupling block",
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
