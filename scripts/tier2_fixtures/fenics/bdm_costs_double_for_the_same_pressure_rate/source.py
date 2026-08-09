"""Tier-2 for fenics mixed_poisson#5: BDM(k)+DG(k-1) is the alternative H(div)
pair. It is inf-sup stable like RT(k)+DG(k-1) but uses the FULL polynomial
space, so at k=1 in 2D it costs twice the flux dofs, and the order it buys is
in the FLUX, not in the pressure.

The pitfall is the measurement, not the element: someone swaps RT for BDM,
measures only the pressure error, sees the same rate, and concludes BDM is
useless -- or worse, keeps paying for it without knowing what it bought.

This fixture runs the manufactured solution p = sin(pi x) sin(pi y) on 8x8 and
16x16 unit squares with both pairs and reports the observed convergence rate of
the quantity it is looking at. The default (pathological) run looks only at the
pressure; T2_MUTATE=1 looks at the flux instead, which is where the difference
is.

Observed on dolfinx 0.10.0: BDM(1) has exactly twice the flux dofs of RT(1) on
the same mesh, both pairs converge at first order in the pressure, and only in
the flux does BDM show the extra order (RT(1) first order, BDM(1) second).
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import basix.ufl  # noqa: E402
import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

from petsc4py import PETSc  # noqa: E402

MESHES = (8, 16)


def solve(fam: str, N: int):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    S = basix.ufl.element(fam, msh.basix_cell(), 1)
    P = basix.ufl.element("DG", msh.basix_cell(), 0)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([S, P]))
    (sig, p) = ufl.TrialFunctions(W)
    (tau, q) = ufl.TestFunctions(W)
    x = ufl.SpatialCoordinate(msh)
    n = ufl.FacetNormal(msh)
    pex = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    sigex = -ufl.grad(pex)
    f = 2 * ufl.pi**2 * pex
    a = (ufl.inner(sig, tau) * ufl.dx
         - p * ufl.div(tau) * ufl.dx
         + q * ufl.div(sig) * ufl.dx)
    L = f * q * ufl.dx - pex * ufl.dot(tau, n) * ufl.ds

    A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a))
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(L))
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    # Dense LAPACK LU: PETSc's sparse LU does not pivot and stumbles on the
    # zero pressure block of any saddle-point matrix.
    Ad = A.copy().convert("dense")
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(Ad)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.setErrorIfNotConverged(True)
    w = dolfinx.fem.Function(W)
    ksp.solve(b, w.x.petsc_vec)
    w.x.scatter_forward()

    sh, ph = ufl.split(w)
    dxq = ufl.dx(metadata={"quadrature_degree": 4})
    # Only the quantity this run is looking at, so the mutant and the default
    # each compile one error form instead of two.
    err = (ufl.inner(sh - sigex, sh - sigex) if MUTATE
           else (ph - pex) ** 2) * dxq
    e = np.sqrt(dolfinx.fem.assemble_scalar(dolfinx.fem.form(err)))
    V0, _ = W.sub(0).collapse()
    ndof = V0.dofmap.index_map.size_global * V0.dofmap.index_map_bs
    return float(e), ndof


def main() -> int:
    res = {fam: [solve(fam, N) for N in MESHES] for fam in ("RT", "BDM")}
    nrt = res["RT"][0][1]
    nbdm = res["BDM"][0][1]
    print(f"flux_dofs_at_{MESHES[0]}x{MESHES[0]}: RT1={nrt} BDM1={nbdm}")
    double = abs(nbdm - 2 * nrt) == 0
    print(f"bdm_flux_dofs_are_exactly_double_rt={double}")

    label = "flux" if MUTATE else "pressure"
    rates = {}
    for fam in ("RT", "BDM"):
        e0, e1 = res[fam][0][0], res[fam][1][0]
        rates[fam] = float(np.log2(e0 / e1))
        print(f"{fam}1_{label}_error_{MESHES[0]}={e0:.6e} "
              f"{MESHES[1]}={e1:.6e} rate={rates[fam]:.3f}")

    if not MUTATE:
        same = abs(rates["BDM"] - rates["RT"]) < 0.1 and rates["RT"] > 0.9
        print(f"both_pairs_are_first_order_in_the_pressure="
              f"{rates['RT'] > 0.9 and rates['BDM'] > 0.9}")
        print(f"bdm_pressure_rate_matches_rt={same}")
        if same and double:
            print("VERDICT=bdm_doubles_the_flux_dofs_and_the_pressure_rate_"
                  "is_unchanged")
            return 0
        print("VERDICT=bdm_changed_the_pressure_rate")
        return 1

    gain = rates["BDM"] - rates["RT"]
    print(f"bdm_flux_rate_gain_over_rt={gain:.3f}")
    print(f"bdm_buys_one_extra_order_in_the_flux={gain > 0.8}")
    if gain > 0.8:
        print("VERDICT=bdm_extra_order_shows_up_only_in_the_flux")
        return 0
    print("VERDICT=bdm_bought_nothing")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
