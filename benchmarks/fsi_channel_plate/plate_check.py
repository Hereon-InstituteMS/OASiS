#!/usr/bin/env python3
"""
Independent, converged plane-strain FE solve of the PLATE ALONE under the interface
pressure that 4C itself computed, using scikit-fem with biquadratic elements.

Purpose: separate "is 4C's FSI load right?" from "is 4C's QUAD4 plate too stiff?".
The 4C monolithic FSI run gives an interface pressure p(x); feeding exactly that p(x)
into a well-resolved independent plate solve gives the plate response that a
locking-free discretisation would produce.

  domain  [0,1] x [0.2,0.25], plane strain, E=3e6, nu=0.3
  clamped u=0 on x=0 and x=1
  bottom edge y=0.2 loaded by traction t=(0,+p(x))   (fluid pressure pushing up)
  top edge traction free
"""
import numpy as np
import skfem
from skfem import (MeshQuad, Basis, FacetBasis, ElementQuad2, ElementVector,
                   BilinearForm, LinearForm, asm, condense, solve)
from skfem.helpers import ddot, sym_grad, eye, trace

E, NU = 3.0e6, 0.3
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))     # plane strain
MU = E / (2 * (1 + NU))
Y0, Y1, L = 0.2, 0.25, 1.0

import sys
PFILE = sys.argv[1] if len(sys.argv) > 1 else (
    "/tmp/claude-1001/-home-alexander-4C/b1c8e459-ec06-467a-bad7-474c74f9d0f3/"
    "scratchpad/fourc_fsi/iface_p.npy")
xp, pp = np.load(PFILE)
# the very last node sits on the do-nothing outlet and carries a local BC artifact;
# replace it by linear extrapolation of the clean interior trend
pp = pp.copy()
pp[-1] = pp[-2] + (pp[-2] - pp[-3])


def p_of_x(x):
    return np.interp(x, xp, pp)


def C(T):
    return 2. * MU * T + LAM * eye(trace(T), T.shape[0])


@BilinearForm
def stiffness(u, v, w):
    return ddot(C(sym_grad(u)), sym_grad(v))


def run(nx, ny):
    m = (MeshQuad.init_tensor(np.linspace(0, L, nx + 1), np.linspace(Y0, Y1, ny + 1))
         .with_boundaries({
             "clamped": lambda x: (np.abs(x[0]) < 1e-12) | (np.abs(x[0] - L) < 1e-12),
             "wet": lambda x: np.abs(x[1] - Y0) < 1e-12}))
    e = ElementVector(ElementQuad2())
    basis = Basis(m, e)
    fb = FacetBasis(m, e, facets=m.boundaries["wet"])

    @LinearForm
    def load(v, w):
        return p_of_x(w.x[0]) * v[1]          # traction (0, +p)

    K = asm(stiffness, basis)
    f = asm(load, fb)
    D = basis.get_dofs("clamped")
    u = solve(*condense(K, f, D=D))

    # interface (y = Y0) vertical displacement
    dofs = basis.nodal_dofs                    # (2, n_nodes) for the vertex dofs
    xy = m.p
    sel = np.abs(xy[1] - Y0) < 1e-12
    x = xy[0, sel]
    dy = u[dofs[1, sel]]
    o = np.argsort(x)
    return x[o], dy[o]


if __name__ == "__main__":
    print(f"load: p(0)={pp[0]:.2f}  p(L)={pp[-1]:.2f}  integral={np.trapz(pp, xp):.3f}")
    for nx, ny in [(40, 4), (80, 8), (160, 16), (320, 24)]:
        x, dy = run(nx, ny)
        i = int(np.argmax(np.abs(dy)))
        print(f"  skfem Quad2 {nx:4d}x{ny:3d}: max|dy| = {abs(dy[i]):.6e} at x={x[i]:.4f}")

    # analytical clamped-clamped Euler-Bernoulli under the same load, for orientation
    Ep = E / (1 - NU ** 2)
    I = (Y1 - Y0) ** 3 / 12.0
    n = 20001
    xs = np.linspace(0, L, n)
    q = p_of_x(xs)
    # solve EI w'''' = q with w=w'=0 at both ends, by a spectral (clamped-beam mode) sum
    # -- simpler: finite differences
    h = xs[1] - xs[0]
    N = n - 4                                   # unknowns w_2..w_{n-3}
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    A = diags([1, -4, 6, -4, 1], [-2, -1, 0, 1, 2], shape=(N, N)).tolil()
    # clamped: w_0=w_1=0 and w_{n-1}=w_{n-2}=0  -> rows already consistent
    A = A.tocsr() / h ** 4
    w = spsolve(A, q[2:-2] / (Ep * I))
    print(f"  Euler-Bernoulli (E/(1-nu^2), FD): max|w| = {np.abs(w).max():.6e} "
          f"at x={xs[2:-2][np.argmax(np.abs(w))]:.4f}")
