"""A small P1 finite-element domain-decomposition solver, for MEASUREMENT.

This module never ships to the agent under test and never appears in a task
text.  It exists so that claims about the blind coupled instances can be
*measured* instead of argued:

* every manufactured coupled instance is solved monolithically and its
  convergence order confirmed before it is allowed into the campaign;
* every proposed grading quantity is subjected to **mutation** — take a correct
  partitioned coupling, break one thing in the physics, and see which reported
  numbers move.  A grading quantity that has not been attacked is not known to
  discriminate.

Deliberately dependency-light (numpy + scipy.sparse) and deliberately
structured: the meshes are the ones the tasks prescribe (uniform, h halved per
level, interfaces on mesh lines), so a measured order is the order the campaign
would see and not an artefact of an unrelated discretisation.

Scope
-----
2D scalar diffusion with a constant SPD tensor per subdomain, P1 on a
structured right-triangle grid, homogeneous or manufactured Dirichlet data, and
a partitioned Dirichlet-Neumann iteration across an axis-aligned interface.
That is exactly the class the coupled blind instances live in; anything wider
would be unverified code pretending to be evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

# Dunavant degree-4, 6-point rule on the reference triangle (barycentric).
# Degree 4 integrates the P1 mass/load terms of the polynomial and exponential
# sources used here to well below the discretisation error.
_Q_BARY = np.array([
    [0.108103018168070, 0.445948490915965, 0.445948490915965],
    [0.445948490915965, 0.108103018168070, 0.445948490915965],
    [0.445948490915965, 0.445948490915965, 0.108103018168070],
    [0.816847572980459, 0.091576213509771, 0.091576213509771],
    [0.091576213509771, 0.816847572980459, 0.091576213509771],
    [0.091576213509771, 0.091576213509771, 0.816847572980459],
])
_Q_W = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)


@dataclass
class Grid:
    """Uniform structured P1 triangulation of an axis-aligned rectangle."""

    x0: float
    x1: float
    y0: float
    y1: float
    nx: int
    ny: int
    holes: tuple = ()          # (x0, x1, y0, y1) rectangles removed

    def __post_init__(self):
        xs = np.linspace(self.x0, self.x1, self.nx + 1)
        ys = np.linspace(self.y0, self.y1, self.ny + 1)
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        self.pts = np.column_stack([XX.ravel(), YY.ravel()])
        idx = np.arange((self.nx + 1) * (self.ny + 1)).reshape(self.nx + 1,
                                                              self.ny + 1)
        self.node_index = idx
        tris, live = [], []
        for i in range(self.nx):
            for j in range(self.ny):
                a, b = idx[i, j], idx[i + 1, j]
                c, d = idx[i + 1, j + 1], idx[i, j + 1]
                tris.append((a, b, c))
                tris.append((a, c, d))
                cx = self.x0 + (i + 0.5) * (self.x1 - self.x0) / self.nx
                cy = self.y0 + (j + 0.5) * (self.y1 - self.y0) / self.ny
                inside = any(hx0 < cx < hx1 and hy0 < cy < hy1
                             for hx0, hx1, hy0, hy1 in self.holes)
                live += [not inside, not inside]
        tris = np.array(tris, dtype=np.int64)
        self.all_tris = tris
        self.tris = tris[np.array(live, dtype=bool)]
        self.hx = (self.x1 - self.x0) / self.nx
        self.hy = (self.y1 - self.y0) / self.ny
        used = np.zeros(self.pts.shape[0], dtype=bool)
        used[self.tris.ravel()] = True
        self.active = used
        self.dead = np.flatnonzero(~used)

    @property
    def n(self) -> int:
        return self.pts.shape[0]

    # ── node sets ────────────────────────────────────────────────────
    def face_nodes(self, which: str) -> np.ndarray:
        i = self.node_index
        return {"x0": i[0, :], "x1": i[-1, :],
                "y0": i[:, 0], "y1": i[:, -1]}[which].copy()

    def boundary_nodes(self) -> np.ndarray:
        """Outer faces plus, for a domain with holes, the hole boundaries."""
        b = [self.face_nodes(f) for f in ("x0", "x1", "y0", "y1")]
        if len(self.holes):
            # a node on the rim of a hole belongs to a live and a dead element
            from collections import Counter
            edges = Counter()
            for tri in self.tris:
                for a, c in ((0, 1), (1, 2), (2, 0)):
                    edges[tuple(sorted((tri[a], tri[c])))] += 1
            rim = [n for e, cnt in edges.items() if cnt == 1 for n in e]
            b.append(np.array(sorted(set(rim)), dtype=np.int64))
        out = np.unique(np.concatenate(b))
        return out[self.active[out]]


def assemble(grid: Grid, K, f) -> tuple[sps.csr_matrix, np.ndarray]:
    """Stiffness for ``-div(K grad u)`` and load for ``f`` (callable f(x, y)).

    ``K`` is a constant 2x2 (or scalar) OR a callable ``K(xc, yc)`` returning an
    ``(n_elem, 2, 2)`` array, which is what a piecewise-material domain needs.
    """
    p = grid.pts
    t = grid.tris
    v0, v1, v2 = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    e0 = v1 - v0
    e1 = v2 - v0
    det = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]
    area = 0.5 * np.abs(det)

    # gradients of the three barycentric basis functions, per element
    g = np.empty((t.shape[0], 3, 2))
    g[:, 0, 0] = (v1[:, 1] - v2[:, 1])
    g[:, 0, 1] = (v2[:, 0] - v1[:, 0])
    g[:, 1, 0] = (v2[:, 1] - v0[:, 1])
    g[:, 1, 1] = (v0[:, 0] - v2[:, 0])
    g[:, 2, 0] = (v0[:, 1] - v1[:, 1])
    g[:, 2, 1] = (v1[:, 0] - v0[:, 0])
    g /= det[:, None, None]

    if callable(K):
        cen = (v0 + v1 + v2) / 3.0
        Kel = np.asarray(K(cen[:, 0], cen[:, 1]), dtype=float)
        Kg = np.einsum("eab,ekb->eka", Kel, g)
    else:
        Km = np.asarray(K, dtype=float)
        if Km.ndim == 0:
            Km = float(Km) * np.eye(2)
        Kg = np.einsum("ab,ekb->eka", Km, g)
    ke = area[:, None, None] * np.einsum("eia,eja->eij", g, Kg)

    rows = np.repeat(t, 3, axis=1).ravel()
    cols = np.tile(t, (1, 3)).ravel()
    A = sps.csr_matrix((ke.ravel(), (rows, cols)), shape=(grid.n, grid.n))

    # load vector, degree-4 quadrature
    b = np.zeros(grid.n)
    for w, bary in zip(_Q_W, _Q_BARY):
        xq = bary[0] * v0 + bary[1] * v1 + bary[2] * v2
        fq = f(xq[:, 0], xq[:, 1])
        contrib = (w * area)[:, None] * (fq[:, None] * bary[None, :])
        np.add.at(b, t, contrib)
    return A.tocsr(), b


def solve_with_dirichlet(A: sps.csr_matrix, b: np.ndarray,
                         dof: np.ndarray, val: np.ndarray,
                         dead: np.ndarray | None = None) -> np.ndarray:
    """Solve ``A u = b`` with ``u[dof] = val``, by static condensation.

    ``dead`` are nodes carried by the index grid but belonging to no live
    element (the interior of a removed region); they are pinned to zero so the
    matrix stays non-singular.
    """
    n = A.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[dof] = False
    if dead is not None and len(dead):
        mask[dead] = False
        dof = np.concatenate([dof, dead])
        val = np.concatenate([val, np.zeros(len(dead))])
    u = np.zeros(n)
    u[dof] = val
    rhs = b[mask] - A[mask][:, dof] @ val
    u[mask] = spla.spsolve(A[mask][:, mask].tocsc(), rhs)
    return u


def nodal_flux(A: sps.csr_matrix, b: np.ndarray, u: np.ndarray,
               dof: np.ndarray) -> np.ndarray:
    """Variationally consistent outward flux functional at ``dof``.

    ``(A u - b)_i`` on a constrained node equals ``\\int_\\Gamma (K grad u . n_out)
    phi_i``, the discrete reaction.  This is the quantity a partitioned scheme
    actually exchanges, and it converges one order better than the raw element
    gradient, which is why it is used here rather than differencing the field.
    """
    return (A @ u - b)[dof]


def evaluate(grid: Grid, u: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Evaluate the P1 field at arbitrary points inside the grid's rectangle."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    xi = np.clip(((pts[:, 0] - grid.x0) / grid.hx).astype(int), 0, grid.nx - 1)
    yj = np.clip(((pts[:, 1] - grid.y0) / grid.hy).astype(int), 0, grid.ny - 1)
    lx = (pts[:, 0] - (grid.x0 + xi * grid.hx)) / grid.hx
    ly = (pts[:, 1] - (grid.y0 + yj * grid.hy)) / grid.hy
    a = grid.node_index[xi, yj]
    bnode = grid.node_index[xi + 1, yj]
    c = grid.node_index[xi + 1, yj + 1]
    d = grid.node_index[xi, yj + 1]
    lower = ly <= lx                       # triangle (a, b, c) vs (a, c, d)
    out = np.empty(pts.shape[0])
    # lower triangle: u = ua + (ub-ua)*lx + (uc-ub)*ly
    out[lower] = (u[a[lower]]
                  + (u[bnode[lower]] - u[a[lower]]) * lx[lower]
                  + (u[c[lower]] - u[bnode[lower]]) * ly[lower])
    up = ~lower
    # upper triangle: u = ua + (ud-ua)*ly + (uc-ud)*lx
    out[up] = (u[a[up]]
               + (u[d[up]] - u[a[up]]) * ly[up]
               + (u[c[up]] - u[d[up]]) * lx[up])
    return out


# ──────────────────────────────────────────────────────────────────────
# Partitioned Dirichlet-Neumann coupling across a vertical interface
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Side:
    grid: Grid
    K: np.ndarray
    f: object
    iface_face: str            # which face of THIS grid is the interface
    outer_faces: tuple         # faces carrying the outer Dirichlet datum


@dataclass
class DNResult:
    converged: bool
    iterations: int
    residual: float
    uA: np.ndarray
    uB: np.ndarray
    iface_y: np.ndarray
    trace_A: np.ndarray        # u_A on the interface nodes
    trace_B: np.ndarray        # u_B on the interface nodes
    flux_A: np.ndarray         # consistent OUTWARD flux functional of A
    flux_B: np.ndarray         # consistent OUTWARD flux functional of B
    history: list


def dn_couple(A_side: Side, B_side: Side, *, theta: float = 0.5,
              max_iter: int = 300, tol: float = 1e-11,
              export_scale_A: float = 1.0,
              import_scale_B: float = 1.0,
              flip_map: bool = False,
              g_outer=None) -> DNResult:
    """Partitioned Dirichlet-Neumann iteration, with mutation hooks.

    ``A`` takes the interface **value** and returns its consistent outward flux;
    ``B`` takes that flux as a Neumann datum and returns its interface trace,
    which is relaxed into the next value.  This is the scheme every coupled task
    in the campaign describes.

    The mutation hooks are the point of this function.  They are not options a
    correct run would ever set; they are how a proposed grading quantity is
    attacked:

    ``export_scale_A``  what A exports is multiplied by this.  ``1/kA`` models
        the very common bug of exporting the raw normal derivative instead of
        the conductive flux.
    ``import_scale_B``  what B applies is multiplied by this.  ``kB`` models the
        receiver re-applying its own conductivity to an already-conductive flux.
        The two together (``1/kA`` then ``kB``) are **exactly invisible when
        kA == kB** and wrong otherwise, which is what makes them the right probe
        for a set of coupled problems that used the same material on both sides.
    ``flip_map``  the interface mapping is reversed (s -> 1-s), the classic
        non-matching-mesh connectivity error.
    """
    if g_outer is None:
        def g_outer(x, y):
            return np.zeros_like(x)

    gA, bA = A_side.grid, None
    KA_mat, fA_vec = assemble(A_side.grid, A_side.K, A_side.f)
    KB_mat, fB_vec = assemble(B_side.grid, B_side.K, B_side.f)

    ifA = A_side.grid.face_nodes(A_side.iface_face)
    ifB = B_side.grid.face_nodes(B_side.iface_face)
    yA = A_side.grid.pts[ifA, 1]
    yB = B_side.grid.pts[ifB, 1]
    if len(ifA) != len(ifB) or not np.allclose(yA, yB):
        raise ValueError("this driver assumes matching interface nodes; "
                         "mesh the two sides with the same transverse count")
    perm = np.arange(len(ifB))[::-1] if flip_map else np.arange(len(ifB))

    outA = np.unique(np.concatenate(
        [A_side.grid.face_nodes(f) for f in A_side.outer_faces]))
    outB = np.unique(np.concatenate(
        [B_side.grid.face_nodes(f) for f in B_side.outer_faces]))
    outA = np.setdiff1d(outA, ifA)
    outB = np.setdiff1d(outB, ifB)
    gA_val = g_outer(A_side.grid.pts[outA, 0], A_side.grid.pts[outA, 1])
    gB_val = g_outer(B_side.grid.pts[outB, 0], B_side.grid.pts[outB, 1])

    g = np.zeros(len(ifA))
    hist = []
    uA = uB = None
    for it in range(1, max_iter + 1):
        dofA = np.concatenate([outA, ifA])
        valA = np.concatenate([gA_val, g])
        uA = solve_with_dirichlet(KA_mat, fA_vec, dofA, valA)
        lam = nodal_flux(KA_mat, fA_vec, uA, ifA) * export_scale_A

        rhsB = fB_vec.copy()
        rhsB[ifB[perm]] -= lam * import_scale_B
        uB = solve_with_dirichlet(KB_mat, rhsB, outB, gB_val)
        trace_B = uB[ifB[perm]]

        g_new = theta * trace_B + (1.0 - theta) * g
        scale = max(np.abs(g_new).max(), 1e-30)
        res = np.abs(g_new - g).max() / scale
        hist.append(res)
        g = g_new
        if res < tol:
            break

    fluxA = nodal_flux(KA_mat, fA_vec, uA, ifA)
    fluxB = nodal_flux(KB_mat, fB_vec, uB, ifB)
    return DNResult(converged=res < tol, iterations=it, residual=res,
                    uA=uA, uB=uB, iface_y=yA,
                    trace_A=uA[ifA], trace_B=uB[ifB],
                    flux_A=fluxA, flux_B=fluxB, history=hist)


# ──────────────────────────────────────────────────────────────────────
def rms(v) -> float:
    v = np.asarray(v, dtype=float).ravel()
    return float(np.sqrt(np.mean(v * v))) if v.size else 0.0


def order_from_halving(errs) -> float | None:
    """log2 of successive ratios; the tasks prescribe exact halving."""
    e = [float(v) for v in errs]
    if len(e) < 2 or any(v <= 0 for v in e):
        return None
    r = [np.log2(e[i] / e[i + 1]) for i in range(len(e) - 1)]
    return float(np.mean(r))


def probe_grid_1d(lo: float, hi: float, m: int) -> np.ndarray:
    return lo + (np.arange(m) + 0.5) * (hi - lo) / m


def probe_grid_2d(bx, by, m: int) -> np.ndarray:
    xs = probe_grid_1d(bx[0], bx[1], m)
    ys = probe_grid_1d(by[0], by[1], m)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([XX.ravel(), YY.ravel()])
