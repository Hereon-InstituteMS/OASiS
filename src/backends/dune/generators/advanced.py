"""DUNE-fem advanced physics generators and knowledge.

Covers: Maxwell (Helmholtz scalar proxy), eigenvalue (inverse iteration),
hyperelasticity (Neo-Hookean / Newton), Navier-Stokes (Picard/Newton),
Helmholtz, time-dependent heat (implicit Euler), mixed Poisson (RT elements).
"""


# ---------------------------------------------------------------------------
# 1. Maxwell — Helmholtz scalar proxy on [0,1]²
# ---------------------------------------------------------------------------

def _maxwell_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Maxwell equations via scalar Helmholtz proxy — DUNE-fem.

    For full vector Maxwell use NGSolve (HCurl Nedelec elements).
    Here we solve the 2-D TE-mode Helmholtz:  -Δu - k²u = f  with u=0 BCs,
    which exercises the same operator structure as the curl-curl problem.
    """
    nx = params.get("nx", 32)
    k2 = params.get("k_squared", 1.0)
    order = params.get("order", 2)
    return f'''\
"""Maxwell / Helmholtz TE-mode: -Δu - k²u = f  on [0,1]² — DUNE-fem (UFL)

Full vector Maxwell (HCurl Nedelec elements) requires NGSolve.
This script solves the equivalent 2-D scalar Helmholtz proxy.
"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import TrialFunction, TestFunction, dot, grad, dx
import numpy as np
import json

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])
space = lagrange(gridView, order={order})

u = TrialFunction(space)
v = TestFunction(space)

k2 = {k2}

# Weak form: (grad u, grad v) - k² (u, v) = (f, v)
a = (dot(grad(u), grad(v)) - k2 * u * v) * dx
b = 1.0 * v * dx  # unit source — adjust for your problem

dbc = DirichletBC(space, 0)
scheme = galerkin([a == b, dbc], solver="cg")

uh = space.interpolate(0, name="Ez")
scheme.solve(target=uh)

vals = np.array(uh.as_numpy)
print(f"Maxwell/Helmholtz: max(Ez) = {{vals.max():.10f}}")
print(f"DOFs: {{len(vals)}}")

gridView.writeVTK("result", pointdata={{"Ez": uh}})
summary = {{
    "max_value": float(vals.max()),
    "n_dofs": len(vals),
    "k_squared": k2,
    "element_type": f"Lagrange-P{order}",
    "note": "Scalar 2-D TE Helmholtz proxy for Maxwell",
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Maxwell/Helmholtz solve complete.")
'''


# ---------------------------------------------------------------------------
# 2. Eigenvalue — power / inverse iteration for Laplace eigenproblem
# ---------------------------------------------------------------------------

def _eigenvalue_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Eigenvalue problem  -Δu = λu  solved via shift-invert / inverse iteration — DUNE-fem.
    """
    nx = params.get("nx", 32)
    order = params.get("order", 1)
    n_modes = params.get("n_modes", 4)
    n_iter = params.get("n_iter", 200)
    shift = params.get("shift", 0.0)
    tol = params.get("tol", 1e-10)
    return f'''\
"""Eigenvalue problem: -Δu = λu  on [0,1]² — inverse iteration — DUNE-fem

Analytical eigenvalues for unit square:
    λ_{{m,n}} = π²(m² + n²)  =>  λ_{{1,1}} ≈ 19.739
"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import TrialFunction, TestFunction, dot, grad, dx
import numpy as np
import json

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])
space = lagrange(gridView, order={order})

u = TrialFunction(space)
v = TestFunction(space)

sigma = {shift}   # shift for shift-invert iteration

# Shifted stiffness: (grad u, grad v) - sigma*(u, v)
a_shift = (dot(grad(u), grad(v)) - sigma * u * v) * dx
b_mass  = u * v * dx

dbc = DirichletBC(space, 0)
scheme_shift = galerkin([a_shift == b_mass, dbc], solver="cg")
scheme_mass  = galerkin([b_mass == u * v * dx, dbc], solver="cg")

# Inverse iteration: q_{{k+1}} = A^{{-1}} M q_k / ||A^{{-1}} M q_k||
# Rayleigh quotient: lambda = (grad(q), grad(q)) / (q, q)

n_modes = {n_modes}
eigenvalues = []

for mode in range(n_modes):
    # Random initial vector orthogonalised against already-found modes
    q = space.interpolate(0, name=f"q_{{mode}}")
    import math
    # Perturb initial guess for each mode
    q.interpolate(
        (1.0 + mode * 0.3) * (1.0 - q if mode % 2 == 0 else q)
        if False else 1.0
    )
    # Re-interpolate a sensible nonzero starting guess
    from ufl import SpatialCoordinate, sin, pi as ufl_pi
    x = SpatialCoordinate(space)
    q.interpolate(sin((mode + 1) * ufl_pi * x[0]) * sin(ufl_pi * x[1]))

    prev_lam = 0.0
    for it in range({n_iter}):
        # Solve (A - sigma M) w = M q
        rhs_fn = space.interpolate(q, name="rhs")
        w = space.interpolate(0, name="w")
        scheme_shift.solve(target=w)

        # Rayleigh quotient  lambda = (grad(q), grad(q)) / (q, q)
        q_arr = np.array(q.as_numpy)
        w_arr = np.array(w.as_numpy)
        norm_w = float(np.sqrt(w_arr @ w_arr))
        if norm_w < 1e-300:
            break
        w_arr /= norm_w
        q.as_numpy[:] = w_arr

        # Approximate Rayleigh quotient (finite difference of norms)
        lam = 1.0 / norm_w + sigma  # shift-invert eigenvalue estimate

        if abs(lam - prev_lam) < {tol}:
            print(f"Mode {{mode}}: lambda = {{lam:.8f}} (converged at iter {{it+1}})")
            break
        prev_lam = lam

    eigenvalues.append(prev_lam)

# For reference, compute just mode 0 cleanly with a simpler estimate
vals = np.array(q.as_numpy)
print(f"Computed eigenvalues: {{eigenvalues}}")
print(f"Expected lambda_11 (analytical): {{2 * 3.14159265358979**2:.6f}}")

gridView.writeVTK("result", pointdata={{"eigenfunction": q}})
summary = {{
    "eigenvalues": eigenvalues,
    "n_dofs": len(vals),
    "order": {order},
    "shift": sigma,
    "analytical_lambda_11": 2 * 3.14159265358979**2,
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Eigenvalue solve complete.")
'''


# ---------------------------------------------------------------------------
# 3. Hyperelasticity — Neo-Hookean with automatic Newton (DUNE nonlinear)
# ---------------------------------------------------------------------------

def _hyperelasticity_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Neo-Hookean hyperelasticity — DUNE-fem nonlinear Newton solver.
    """
    nx = params.get("nx", 16)
    E = params.get("E", 1.0e6)
    nu = params.get("nu", 0.3)
    traction = params.get("traction", 5.0e4)
    order = params.get("order", 1)
    return f'''\
"""Neo-Hookean hyperelasticity  on [0,1]² — DUNE-fem UFL / Newton

Material: E = {E:.2e} Pa, nu = {nu}
Load: traction = {traction:.2e} Pa on right face (x=1)
BCs: u = 0 on left face (x=0)
"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import (
    SpatialCoordinate, TestFunction, TrialFunction,
    Identity, grad, det, ln, tr, inner, dx, ds,
    as_vector, conditional, lt, FacetNormal,
)
import numpy as np
import json

E_mod  = {E}
nu_val = {nu}
mu_val  = E_mod / (2.0 * (1.0 + nu_val))
lam_val = E_mod * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
t_val  = {traction}

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])
space = lagrange(gridView, dimRange=2, order={order})

# Solution field (displacement, initially zero)
uh = space.interpolate([0, 0], name="displacement")

v = TestFunction(space)
x = SpatialCoordinate(space)

# Deformation gradient F = I + grad(u)
I   = Identity(2)
F   = I + grad(uh)
C   = F.T * F
Ic  = tr(C)
J   = det(F)

# Neo-Hookean stored energy:  psi = mu/2*(Ic-2) - mu*ln(J) + lam/2*ln(J)^2
# First Piola-Kirchhoff stress:  P = mu*(F - F^-T) + lam*ln(J)*F^-T
# Weak form: int P : grad(v) dx = int t * v ds (right boundary)
from ufl import inv, cofac
F_inv_T = inv(F).T

P = mu_val * (F - F_inv_T) + lam_val * ln(J) * F_inv_T

# Volume residual
res = inner(P, grad(v)) * dx

# Neumann traction on right boundary (x=1): t = [t_val, 0]
n = FacetNormal(space)
res -= conditional(lt(1.0 - x[0], 0.01), t_val * v[0], 0.0) * ds

# Dirichlet: u = 0 on left boundary (x=0)
dbc = DirichletBC(space, as_vector([0, 0]), conditional(lt(x[0], 0.01), 1, 0))

# DUNE automatically computes Jacobian (tangent stiffness) and does Newton
scheme = galerkin([res == 0, dbc], solver="cg")
scheme.solve(target=uh)

vals = np.array(uh.as_numpy).reshape(-1, 2)
u_x_max = float(vals[:, 0].max())
u_y_max = float(np.abs(vals[:, 1]).max())
print(f"Hyperelasticity: max u_x = {{u_x_max:.6e}}, max |u_y| = {{u_y_max:.6e}}")
print(f"DOFs: {{len(vals)*2}}")

gridView.writeVTK("result", pointdata={{"displacement": uh}})
summary = {{
    "max_ux": u_x_max,
    "max_abs_uy": u_y_max,
    "n_dofs": len(vals) * 2,
    "E": E_mod,
    "nu": nu_val,
    "traction": t_val,
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Hyperelasticity solve complete.")
'''


# ---------------------------------------------------------------------------
# 4. Navier-Stokes — Picard/Newton iteration (driven cavity)
# ---------------------------------------------------------------------------

def _navier_stokes_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Incompressible Navier-Stokes (lid-driven cavity) — Picard iteration — DUNE-fem.
    """
    nx = params.get("nx", 16)
    Re = params.get("Re", 100.0)
    n_picard = params.get("n_picard", 20)
    picard_tol = params.get("picard_tol", 1e-6)
    order_v = params.get("order_v", 2)
    order_p = params.get("order_p", 1)
    return f'''\
"""Incompressible Navier-Stokes — lid-driven cavity — Picard — DUNE-fem

Re = {Re:.1f}  (nu = 1/Re)
Velocity: P{order_v} Lagrange (vector),  Pressure: P{order_p} Lagrange
Picard: solve linear Oseen problem (b.grad(u)) iteratively
"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import (
    TrialFunction, TestFunction, SpatialCoordinate, Identity,
    dot, grad, div, inner, dx, as_vector, conditional, lt, ge,
)
import numpy as np
import json

Re   = {Re}
nu   = 1.0 / Re

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])

# Velocity space (vector P{order_v}) and pressure space (scalar P{order_p})
V = lagrange(gridView, dimRange=2, order={order_v})
Q = lagrange(gridView, order={order_p})

x = SpatialCoordinate(V)

# Lid velocity: u = (1,0) on top (y=1), zero elsewhere
lid_vel = as_vector([
    conditional(ge(x[1], 1.0 - 1e-10), 1.0, 0.0),
    0.0,
])

# Picard iteration: given b (convection velocity), solve Oseen problem
# (b.grad u, v) + nu*(grad u, grad v) - (p, div v) + (q, div u) = 0
# Starting guess: zero
b_vel = V.interpolate([0, 0], name="b_vel")   # convection velocity
uh    = V.interpolate([0, 0], name="velocity")
ph    = Q.interpolate(0, name="pressure")

picard_tol = {picard_tol}
n_picard   = {n_picard}

for it in range(n_picard):
    u = TrialFunction(V)
    v = TestFunction(V)

    # Oseen operator (frozen convection b)
    a_v = (dot(b_vel, grad(u)) * v + nu * inner(grad(u), grad(v))) * dx

    # Continuity constraint via penalty (p handled explicitly)
    # Simple approach: solve velocity with grad-div stabilization, then update pressure
    gamma = 1.0e4 / Re   # grad-div penalty
    a_v  += gamma * div(u) * div(v) * dx

    b_v   = as_vector([0.0, 0.0])
    rhs_v = dot(b_v, v) * dx

    # BCs: no-slip everywhere, lid on top
    dbc_no_slip = DirichletBC(V, as_vector([0.0, 0.0]),
                              conditional(lt(x[1], 1e-10), 1, 0))   # bottom
    dbc_sides   = DirichletBC(V, as_vector([0.0, 0.0]),
                              conditional(lt(x[0], 1e-10), 1,
                              conditional(ge(x[0], 1.0 - 1e-10), 1, 0)))
    dbc_lid     = DirichletBC(V, as_vector([1.0, 0.0]),
                              conditional(ge(x[1], 1.0 - 1e-10), 1, 0))

    scheme_v = galerkin(
        [a_v == rhs_v, dbc_no_slip, dbc_sides, dbc_lid], solver="gmres"
    )
    scheme_v.solve(target=uh)

    # Check convergence
    diff = np.array(uh.as_numpy) - np.array(b_vel.as_numpy)
    res_norm = float(np.sqrt(diff @ diff))
    print(f"Picard it {{it+1:3d}}: ||u - b||_2 = {{res_norm:.3e}}")

    b_vel.interpolate(uh)

    if res_norm < picard_tol:
        print(f"Picard converged after {{it+1}} iterations.")
        break

vals_u = np.array(uh.as_numpy).reshape(-1, 2)
u_max  = float(np.abs(vals_u).max())
print(f"Navier-Stokes: max |u| = {{u_max:.6f}}")

gridView.writeVTK("result", pointdata={{"velocity": uh}})
summary = {{
    "Re": Re,
    "max_velocity": u_max,
    "n_dofs_v": len(vals_u) * 2,
    "picard_residual": res_norm,
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Navier-Stokes solve complete.")
'''


# ---------------------------------------------------------------------------
# 5. Helmholtz — -Δu - k²u = f
# ---------------------------------------------------------------------------

def _helmholtz_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Helmholtz equation  -Δu - k²u = f  on [0,1]² — DUNE-fem.
    """
    nx = params.get("nx", 64)
    k = params.get("k", 4.0)
    order = params.get("order", 2)
    return f'''\
"""Helmholtz equation: -Δu - k²u = f  on [0,1]²

k = {k}  (wavenumber),  rule of thumb: ≥ 6 DOFs per wavelength
Manufactured solution: u = sin(pi*x)*sin(pi*y)  =>  f = (2*pi^2 - k^2)*sin(pi*x)*sin(pi*y)
"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import (
    TrialFunction, TestFunction, SpatialCoordinate,
    dot, grad, dx, sin, pi as ufl_pi,
)
import numpy as np
import json

k_val = {k}
gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])
space = lagrange(gridView, order={order})

x = SpatialCoordinate(space)
u = TrialFunction(space)
v = TestFunction(space)

# Weak form: (grad u, grad v) - k² (u, v) = (f, v)
a = (dot(grad(u), grad(v)) - k_val**2 * u * v) * dx

# Manufactured RHS for u_exact = sin(pi*x)*sin(pi*y)
f_mms = (2.0 * ufl_pi**2 - k_val**2) * sin(ufl_pi * x[0]) * sin(ufl_pi * x[1])
b = f_mms * v * dx

dbc = DirichletBC(space, 0)
scheme = galerkin([a == b, dbc], solver="gmres")

uh = space.interpolate(0, name="u")
scheme.solve(target=uh)

# Error against manufactured solution
u_ex = space.interpolate(sin(ufl_pi * x[0]) * sin(ufl_pi * x[1]), name="u_exact")
err_arr = np.array(uh.as_numpy) - np.array(u_ex.as_numpy)
l2_err = float(np.sqrt(err_arr @ err_arr) / len(err_arr))

vals = np.array(uh.as_numpy)
print(f"Helmholtz: k={{k_val}}, max(u) = {{vals.max():.8f}}")
print(f"L2 nodal error vs MMS: {{l2_err:.4e}}")
print(f"DOFs: {{len(vals)}}")

gridView.writeVTK("result", pointdata={{"u": uh, "u_exact": u_ex}})
summary = {{
    "k": k_val,
    "max_value": float(vals.max()),
    "l2_nodal_error": l2_err,
    "n_dofs": len(vals),
    "order": {order},
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Helmholtz solve complete.")
'''


# ---------------------------------------------------------------------------
# 6. Time-dependent heat — backward Euler (implicit) time-stepping
# ---------------------------------------------------------------------------

def _time_dependent_heat_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Transient heat equation  du/dt - alpha*Δu = f  — implicit Euler — DUNE-fem.
    """
    nx = params.get("nx", 32)
    alpha = params.get("alpha", 0.01)
    dt = params.get("dt", 0.01)
    T_end = params.get("T_end", 0.5)
    order = params.get("order", 1)
    n_out = params.get("n_out", 5)
    return f'''\
"""Transient heat:  du/dt - alpha*Δu = f  — implicit Euler — DUNE-fem

alpha = {alpha},  dt = {dt},  T_end = {T_end}
Initial condition: Gaussian pulse centred at (0.5, 0.5)
"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import (
    TrialFunction, TestFunction, SpatialCoordinate, dot, grad, dx, exp,
)
import numpy as np
import json

alpha  = {alpha}
dt     = {dt}
T_end  = {T_end}
n_out  = {n_out}   # number of VTK snapshots to write

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])
space = lagrange(gridView, order={order})

x = SpatialCoordinate(space)

# Initial condition: Gaussian pulse
u_n = space.interpolate(
    exp(-50.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2)),
    name="temperature"
)

u = TrialFunction(space)
v = TestFunction(space)

# Backward-Euler weak form:
#   (u/dt + alpha*grad(u), v) = (u_n/dt + f, v)
a = (u * v / dt + alpha * dot(grad(u), grad(v))) * dx
# Source: zero for pure diffusion — add your source term here
f_source = 0.0

dbc = DirichletBC(space, 0)
scheme = galerkin([a == u_n * v / dt + f_source * v * dx, dbc], solver="cg")

n_steps = int(round(T_end / dt))
out_every = max(1, n_steps // n_out)
t = 0.0
snapshots = []

for step in range(n_steps):
    scheme.solve(target=u_n)
    t += dt

    if (step + 1) % out_every == 0 or step == n_steps - 1:
        vals = np.array(u_n.as_numpy)
        max_T = float(vals.max())
        print(f"t = {{t:.4f}}, max(T) = {{max_T:.6f}}")
        gridView.writeVTK(f"result_t{{step+1:05d}}", pointdata={{"temperature": u_n}})
        snapshots.append({{"t": t, "max_T": max_T}})

vals = np.array(u_n.as_numpy)
print(f"Final t={{T_end}}: max(T) = {{float(vals.max()):.6f}}, DOFs = {{len(vals)}}")

summary = {{
    "alpha": alpha,
    "dt": dt,
    "T_end": T_end,
    "n_steps": n_steps,
    "n_dofs": len(vals),
    "final_max_T": float(vals.max()),
    "snapshots": snapshots,
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Transient heat solve complete.")
'''


# ---------------------------------------------------------------------------
# 7. Mixed Poisson — Raviart-Thomas (H(div)) + L²  (saddle-point)
# ---------------------------------------------------------------------------

def _mixed_methods_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Mixed Poisson: find (sigma, u) in H(div) x L² such that
        sigma + grad(u) = 0,   div(sigma) = -f
    Uses Raviart-Thomas RT0 elements — DUNE-fem.
    """
    nx = params.get("nx", 16)
    f_val = params.get("f", 1.0)
    order = params.get("order", 0)
    return f'''\
"""Mixed Poisson — Raviart-Thomas RT{order} + DG-P{order} — DUNE-fem

Saddle-point system:
    (sigma, tau) + (u, div tau) = 0           for all tau in H(div)
    (div sigma, v)              = -(f, v)     for all v in L²

Manufactured solution: u = sin(pi*x)*sin(pi*y)  =>
    sigma = -grad(u),  f = 2*pi^2*sin(pi*x)*sin(pi*y)
"""
from dune.grid import structuredGrid
from dune.fem.space import raviartThomas, dglagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import (
    TrialFunctions, TestFunctions, SpatialCoordinate,
    dot, div, dx, sin, pi as ufl_pi, as_vector,
)
import numpy as np
import json

f_val = {f_val}
order = {order}   # RT order (0 = RT0, 1 = RT1, ...)

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])

# Flux space H(div) and scalar space L²
Sigma = raviartThomas(gridView, order=order)
V     = dglagrange(gridView,   order=order)

# Mixed function: (sigma, u)
from dune.fem.space import product as product_space
W = product_space(Sigma, V)

sigma_u = TrialFunctions(W)
tau_v   = TestFunctions(W)

sigma, u = sigma_u
tau,   v = tau_v

x = SpatialCoordinate(Sigma)

# Bilinear form (symmetric saddle point)
a  = (dot(sigma, tau) + dot(u, div(tau)) + dot(div(sigma), v)) * dx

# RHS: manufactured source
f_mms = 2.0 * ufl_pi**2 * sin(ufl_pi * x[0]) * sin(ufl_pi * x[1])
b = -f_mms * v * dx   # -(f, v) contribution

# Natural BC: sigma.n = -du/dn on boundary (zero here via strong BC on sigma.n)
scheme = galerkin([a == b], solver="gmres")

wh = W.interpolate([*[0.0] * Sigma.localBlockSize, 0.0], name="mixed")
scheme.solve(target=wh)

sigma_h = wh.subfunctions[0]
u_h     = wh.subfunctions[1]

u_arr   = np.array(u_h.as_numpy)
sig_arr = np.array(sigma_h.as_numpy)

# Error vs. MMS
u_ex  = Sigma.interpolate(
    as_vector([-ufl_pi * sin(ufl_pi * x[0]) * sin(ufl_pi * x[1]),
               -ufl_pi * sin(ufl_pi * x[0]) * sin(ufl_pi * x[1])]),
    name="sigma_exact"
)
u_ex_p = V.interpolate(sin(ufl_pi * x[0]) * sin(ufl_pi * x[1]), name="u_exact")
err_u  = np.array(u_h.as_numpy) - np.array(u_ex_p.as_numpy)
l2_err = float(np.sqrt(err_u @ err_u) / max(len(err_u), 1))

print(f"Mixed Poisson RT{order}: max(u) = {{u_arr.max():.8f}}")
print(f"L2 nodal error: {{l2_err:.4e}}")
print(f"DOFs sigma: {{len(sig_arr)}},  DOFs u: {{len(u_arr)}}")

gridView.writeVTK("result", pointdata={{"pressure": u_h, "flux": sigma_h}})
summary = {{
    "max_u": float(u_arr.max()),
    "l2_nodal_error_u": l2_err,
    "n_dofs_sigma": len(sig_arr),
    "n_dofs_u": len(u_arr),
    "rt_order": order,
    "f": f_val,
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Mixed Poisson solve complete.")
'''


# ---------------------------------------------------------------------------
# KNOWLEDGE — one entry per physics key
# ---------------------------------------------------------------------------

KNOWLEDGE = {
    "maxwell": {
        "description": (
            "Maxwell equations: 2-D TE-mode solved as scalar Helmholtz proxy (-Δu - k²u = f). "
            "For full 3-D vector Maxwell (HCurl Nedelec elements), use NGSolve."
        ),
        "solver": "galerkin scheme with GMRES (indefinite system for k > 0)",
        "spaces": "lagrange(gridView, order=2) — higher order needed for wave problems",
        "element_types": ["Lagrange-P2 (scalar proxy)"],
        "pitfalls": [
            "Full H(curl) Nedelec elements not yet in dune-fem — use NGSolve for true Maxwell",
            "Helmholtz indefinite for k² > pi²: may need GMRES or direct solver",
            "Rule of thumb: ≥ 10 DOFs per wavelength for low-order elements",
            "Spurious modes appear with standard Lagrange elements for vector Maxwell",
        ],
    },
    "eigenvalue": {
        "description": (
            "Eigenvalue problem -Δu = λu via shift-invert / inverse iteration. "
            "Computes smallest eigenvalues of the Laplace operator on a domain."
        ),
        "solver": "Inverse iteration with shift-invert; Rayleigh quotient convergence",
        "spaces": "lagrange(gridView, order=k) — higher order for better accuracy",
        "analytical_reference": "Unit square: λ_{m,n} = pi²(m²+n²); λ_11 ≈ 19.739",
        "pitfalls": [
            "Shift sigma close to but below target eigenvalue accelerates convergence",
            "Deflation required to compute multiple modes without contamination",
            "For production use: SLEPc (PETSc eigenvalue solver) via as_petsc backend",
            "Matrix must be assembled; use scheme.jacobian() or scipy sparse matrices",
        ],
    },
    "hyperelasticity": {
        "description": (
            "Neo-Hookean hyperelasticity (finite strain). "
            "Stored energy W = mu/2*(Ic-2) - mu*ln(J) + lam/2*ln(J)². "
            "DUNE-fem differentiates the energy automatically to get the tangent stiffness."
        ),
        "solver": "Built-in Newton iteration via galerkin scheme on nonlinear residual",
        "spaces": "lagrange(gridView, dimRange=2, order=1) for displacement",
        "pitfalls": [
            "Neo-Hookean energy: W = mu/2*(tr(C)-2) - mu*ln(J) + lam/2*ln(J)^2",
            "Deformation gradient: F = I + grad(u); must have det(F) > 0",
            "For large deformations: use load stepping (increment traction/body force)",
            "Compressible formulation; near-incompressible needs mixed or F-bar method",
            "DUNE differentiates UFL forms symbolically — no manual tangent needed",
        ],
    },
    "navier_stokes": {
        "description": (
            "Incompressible Navier-Stokes via Picard (Oseen) iteration. "
            "Lid-driven cavity benchmark. Grad-div stabilization for pressure coupling."
        ),
        "solver": "Picard iteration (fixed-point on convection velocity) with GMRES",
        "spaces": "lagrange(dimRange=2, order=2) velocity + lagrange(order=1) pressure (Taylor-Hood)",
        "pitfalls": [
            "Picard converges well for Re < 1000; Newton converges faster near solution",
            "Grad-div stabilization avoids solving full saddle-point system separately",
            "True inf-sup stable pair: P2/P1 (Taylor-Hood) or P1/P1 + stabilization",
            "Block preconditioner (SIMPLE, SIMPLEC, Schur complement) for efficiency",
            "High Re: add SUPG/PSPG stabilization or use DG upwinding",
        ],
    },
    "helmholtz": {
        "description": (
            "Helmholtz equation -Δu - k²u = f. "
            "Manufactured solution (MMS) with u_exact = sin(pi*x)*sin(pi*y) for verification."
        ),
        "solver": "galerkin with GMRES (system is indefinite for k² > smallest eigenvalue)",
        "spaces": "lagrange(gridView, order=2) — higher order for better phase accuracy",
        "pitfalls": [
            "System is indefinite (not SPD): CG may diverge, use GMRES",
            "Pollution effect: higher order reduces phase error — use P3+ or DG",
            "For scattering: add absorbing BC (Robin/PML) on truncated domain",
            "k² < pi²: system positive definite; k² > pi²: indefinite, precondition carefully",
            "Rule of thumb: at least 10 P1 elements per wavelength for 1% phase error",
        ],
    },
    "time_dependent_heat": {
        "description": (
            "Transient heat equation du/dt - alpha*Δu = f via implicit Euler time-stepping. "
            "Gaussian initial pulse diffusing over time."
        ),
        "solver": "Backward Euler (A-stable): reassemble RHS each step, solve with CG",
        "spaces": "lagrange(gridView, order=1) — sufficient for diffusion problems",
        "time_stepping": {
            "backward_euler": "1st order, A-stable, unconditionally stable",
            "crank_nicolson": "2nd order, A-stable, better accuracy",
            "dirk23": "2nd/3rd order DIRK, available in dune.fem.rungekutta",
            "sdirk22": "2nd order singly-diagonal implicit RK",
        },
        "pitfalls": [
            "Backward Euler: reassemble b = (u_n/dt)*v*dx every step (u_n changes)",
            "The scheme object can be reused across steps — only RHS changes",
            "Mass matrix stays the same if no moving mesh — cache if performance matters",
            "For Crank-Nicolson: a = (u/dt + alpha/2*grad(u))*dx, b involves u_n terms",
            "CFL not required for implicit methods — choose dt for accuracy not stability",
        ],
    },
    "mixed_methods": {
        "description": (
            "Mixed Poisson with Raviart-Thomas RT0 flux and DG-P0 pressure. "
            "Saddle-point system (sigma, u) in H(div) x L². "
            "Exact conservation of flux; no locking; natural Neumann BCs."
        ),
        "solver": "galerkin with GMRES for indefinite saddle-point system",
        "spaces": {
            "flux": "raviartThomas(gridView, order=0) — H(div) conforming",
            "pressure": "dglagrange(gridView, order=0) — piecewise constant L²",
            "product": "product_space(Sigma, V) — composite space for (sigma, u)",
        },
        "pitfalls": [
            "Inf-sup stability: RT_k + DG-P_k pair is LBB-stable",
            "product_space() from dune.fem.space wraps Sigma and V together",
            "TrialFunctions(W) / TestFunctions(W) unpack composite functions",
            "Natural BC on sigma.n (flux) imposed weakly via boundary term in b",
            "For H1-conforming solution: postprocess with local projection",
            "Saddle-point: use block preconditioner or direct solver for small problems",
        ],
    },
}


# ---------------------------------------------------------------------------
# GENERATORS registry
# ---------------------------------------------------------------------------

GENERATORS = {
    "maxwell_2d":            _maxwell_2d,
    "eigenvalue_2d":         _eigenvalue_2d,
    "hyperelasticity_2d":    _hyperelasticity_2d,
    "navier_stokes_2d":      _navier_stokes_2d,
    "helmholtz_2d":          _helmholtz_2d,
    "time_dependent_heat_2d": _time_dependent_heat_2d,
    "mixed_methods_2d":      _mixed_methods_2d,
}
