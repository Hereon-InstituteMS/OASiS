"""Tier-2 for fenics biharmonic#0: the C0 interior-penalty rule of thumb
alpha = 4*(k+1)^2 is a STABILITY floor, not an accuracy optimum, and an
under-penalised scheme does not diverge under refinement — it converges from a
blown-up constant.

Wrong variant: taking alpha = 4*(k+1)^2 = 36 for P2 as "the" value. Right
variant: alpha = 1, which on this problem is stable and roughly six times more
accurate.

C0-IP MMS on the unit square, u = sin(pi x) sin(pi y), simply supported, P2,
N = 8 / 16 / 32, L2 error.

Observed on dolfinx 0.10.0: alpha = 36 gives 8.4022e-02 -> 2.5985e-02 ->
7.0180e-03 (rates 1.69, 1.89); alpha = 1 gives 1.9212e-02 -> 4.7394e-03 ->
1.1830e-03 (rates 2.02, 2.00) — the same order, 5.9x smaller error at N = 32.
And alpha = 1e-6 does not blow up under refinement: it starts at 2.0848e+02 on
the coarse mesh and then falls at rate ~4.0 (1.2500e+01, 7.6944e-01). The
observable for an under-penalised C0-IP scheme is a huge coarse-mesh error
CONSTANT, not divergence.

Mutation control: T2_MUTATE=1 makes alpha = 1 the primary penalty, so the
primary is no longer beaten by alpha = 1 and the fixture loses its own
expectation.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dolfinx.fem.petsc import LinearProblem  # noqa: E402

DEGREE = 2
RULE_OF_THUMB = 4.0 * (DEGREE + 1) ** 2      # = 36 for P2
MESHES = (8, 16, 32)
_CACHE: dict = {}


def c0ip_error(n: int, alpha: float) -> float:
    key = (n, alpha)
    if key in _CACHE:
        return _CACHE[key]
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", DEGREE))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    f = 4.0 * np.pi ** 4 * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    nrm = ufl.FacetNormal(msh)
    h = ufl.CellDiameter(msh)
    al = dolfinx.fem.Constant(msh, float(alpha))
    a = (ufl.inner(ufl.div(ufl.grad(u)), ufl.div(ufl.grad(v))) * ufl.dx
         - ufl.inner(ufl.avg(ufl.div(ufl.grad(u))),
                     ufl.jump(ufl.grad(v), nrm)) * ufl.dS
         - ufl.inner(ufl.jump(ufl.grad(u), nrm),
                     ufl.avg(ufl.div(ufl.grad(v)))) * ufl.dS
         + al / ufl.avg(h) * ufl.inner(ufl.jump(ufl.grad(u), nrm),
                                       ufl.jump(ufl.grad(v), nrm)) * ufl.dS)
    L = ufl.inner(f, v) * ufl.dx
    bfacets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    bc = dolfinx.fem.dirichletbc(
        dolfinx.fem.Constant(msh, 0.0),
        dolfinx.fem.locate_dofs_topological(V, fdim, bfacets), V)
    problem = LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix="t2_bih0_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})
    uh = problem.solve()
    err = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form((uh - u_exact) ** 2 * ufl.dx))
    _CACHE[key] = float(np.sqrt(abs(err)))
    return _CACHE[key]


def rates(errs):
    return [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]


def main() -> int:
    primary_alpha = 1.0 if MUTATE else RULE_OF_THUMB
    primary = [c0ip_error(n, primary_alpha) for n in MESHES]
    small = [c0ip_error(n, 1.0) for n in MESHES]
    tiny = [c0ip_error(n, 1.0e-6) for n in MESHES]
    print(f"rule_of_thumb_alpha={RULE_OF_THUMB} primary_alpha={primary_alpha}")
    for i, n in enumerate(MESHES):
        print(f"N={n} primary_l2={primary[i]:.4e} alpha1_l2={small[i]:.4e} "
              f"alpha1e-6_l2={tiny[i]:.4e}")
    r_p, r_s, r_t = rates(primary), rates(small), rates(tiny)
    print("primary_rates=" + " ".join(f"{r:.2f}" for r in r_p))
    print("alpha1_rates=" + " ".join(f"{r:.2f}" for r in r_s))
    print("alpha1e-6_rates=" + " ".join(f"{r:.2f}" for r in r_t))

    ratio = primary[-1] / small[-1]
    print(f"primary_over_alpha1_error_ratio={ratio:.3f}")
    beaten = ratio > 3.0
    same_order = abs(r_p[-1] - r_s[-1]) < 0.5
    tiny_coarse_huge = tiny[0] > 10.0
    tiny_converges = all(tiny[i + 1] < tiny[i] for i in range(len(tiny) - 1)) \
        and min(r_t) > 2.0
    print(f"rule_of_thumb_error_exceeds_alpha_one_by_more_than_3x={beaten}")
    print(f"both_penalties_converge_at_the_same_order={same_order}")
    print(f"tiny_alpha_coarse_mesh_error_exceeds_ten={tiny_coarse_huge}")
    print(f"tiny_alpha_still_converges_under_refinement={tiny_converges}")
    if beaten and same_order and tiny_coarse_huge and tiny_converges:
        print("VERDICT=penalty_rule_of_thumb_is_a_stability_floor_not_an_accuracy_optimum")
        return 0
    print("VERDICT=rule_of_thumb_penalty_was_the_best_choice")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
