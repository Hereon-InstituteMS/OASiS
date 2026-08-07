"""scikit-fem STRUCTURE participant for the OASiS `couple` driver — FSI.

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

Physics: plane-strain linear elasticity on a rectangular wall clamped at both
ends.  The bottom edge is the FSI interface and carries the traction handed in
by the fluid participant as a NEUMANN load.  What is exported back is the
interface displacement.

SIGN CONVENTION — the imported traction is already the load ON THIS BODY
(t = sigma_f . n_s, n_s the structure's outward normal on the interface).  It is
applied DIRECTLY, with NO further sign change.  See the fluid participant's
docstring for the derivation; re-deriving it on this side is how the sign gets
flipped twice.

The interface parametrisation is LAGRANGIAN on both sides: the exported
coordinates are the REFERENCE (undeformed) interface node positions.
"""
import json
import sys
from pathlib import Path

import numpy as np
from skfem import (Basis, FacetBasis, ElementVector, ElementTriP2, MeshTri,
                   asm, condense, solve, LinearForm)
from skfem.models.elasticity import linear_elasticity

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
PARTNER    = "fluid"     # the fluid participant's `name` in your couple(...) call
LX         = 1.0         # wall length
Y0         = 0.2         # the FSI interface (this body's LOWER edge)
HS         = 0.05        # wall thickness
NXS, NYS   = 40, 4       # this body's OWN mesh; need not match the fluid's
E_MOD      = 3.0e6       # Young's modulus
NU         = 0.3         # Poisson ratio
CLAMP_X    = (0.0, 1.0)  # x positions of the clamped ends
T_INIT     = 0.0         # iteration-1 fallback interface traction (both comps)
FEEDBACK   = True        # SET False ONLY to suppress the fluid->structure
                         # direction (freezes the load at T_INIT). A real FSI
                         # run keeps this True.
# ─────────────────────────────────────────────────────────────────────────


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get(PARTNER) or None
    except json.JSONDecodeError:
        return None


def make_sampler(imp, fallback, ncomp=2):
    """Return f(x_array) -> (len(x), ncomp) traction, interpolated along the
    interface parameter x.  The driver does no interpolation: non-matching
    interface meshes are handled here."""
    if not imp or not imp.get("coordinates"):
        return lambda xs: np.full(np.shape(xs) + (ncomp,), float(fallback))
    xs_src = np.asarray(imp["coordinates"], float)[:, 0]
    vals = np.asarray(imp["values"], float).reshape(len(xs_src), -1)
    order = np.argsort(xs_src)
    xs_src, vals = xs_src[order], vals[order]

    def f(xq):
        xq = np.asarray(xq, float)
        out = np.zeros(xq.shape + (ncomp,))
        for c in range(min(ncomp, vals.shape[1])):
            out[..., c] = np.interp(xq, xs_src, vals[:, c])
        return out
    return f


def main():
    lam = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))       # plane strain
    mu = E_MOD / (2.0 * (1.0 + NU))

    m = MeshTri().init_tensor(np.linspace(0.0, LX, NXS + 1),
                              np.linspace(Y0, Y0 + HS, NYS + 1))
    e = ElementVector(ElementTriP2())
    basis = Basis(m, e)

    K = asm(linear_elasticity(lam, mu), basis)

    imp = read_imports() if FEEDBACK else None
    sampler = make_sampler(imp, T_INIT, ncomp=2)

    iface_facets = m.facets_satisfying(lambda x: np.isclose(x[1], Y0))
    fb = FacetBasis(m, e, facets=iface_facets)
    xq = fb.global_coordinates().value          # (dim, nfacets, nqp)
    tq = sampler(xq[0])                         # (nfacets, nqp, 2)

    @LinearForm
    def neumann(v, w):
        return w["tx"] * v[0] + w["ty"] * v[1]

    # skfem passes extra kwargs through as quadrature-point arrays.
    f = asm(neumann, fb, tx=tq[..., 0], ty=tq[..., 1])

    # clamped ends: every dof on x = CLAMP_X
    clamped = basis.get_dofs(
        lambda x: np.isclose(x[0], CLAMP_X[0]) | np.isclose(x[0], CLAMP_X[1]))
    d = solve(*condense(K, f, D=clamped))

    # ── interface displacement at the interface NODES (P1 subset of P2) ─────
    nodal = basis.nodal_dofs                      # (ncomp, nvertices)
    inode = np.where(np.isclose(m.p[1], Y0))[0]
    order = np.argsort(m.p[0, inode])
    inode = inode[order]
    x_if = m.p[0, inode]
    disp = np.column_stack([d[nodal[0, inode]], d[nodal[1, inode]]])
    ref_coords = np.column_stack([x_if, np.full_like(x_if, Y0)])

    # net force actually received on the interface, for the equilibrium check
    fx = float(np.sum(asm(neumann, fb, tx=tq[..., 0], ty=0.0 * tq[..., 1])))
    fy = float(np.sum(asm(neumann, fb, tx=0.0 * tq[..., 0], ty=tq[..., 1])))

    out = {
        "field_name": "interface_displacement",
        "n_points": int(len(x_if)),
        "coordinates": ref_coords.tolist(),
        "values": disp.tolist(),
        "meta": {
            "net_force_received": [fx, fy],
            "feedback": bool(FEEDBACK),
            "max_abs_disp": [float(np.max(np.abs(disp[:, 0]))),
                             float(np.max(np.abs(disp[:, 1])))],
            "n_dofs": int(K.shape[0]),
        },
    }
    Path("exports.json").write_text(json.dumps(out, indent=2))
    print(f"[solid] recv_force=({fx:.6e},{fy:.6e}) "
          f"max|dy|={np.max(np.abs(disp[:, 1])):.6e}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
