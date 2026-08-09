"""Tier-2 for fenics thermal_structural#3: in 2D the textbook shorthand
sigma = C:(eps(u) - alpha*dT*I) is wrong when I is spelled ufl.Identity(2) and
the Lame constants are plane strain, because tr(Identity(2)) is 2, not 3.

The thermal modulus silently becomes (2*lam + 2*mu)*alpha instead of
(3*lam + 2*mu)*alpha. The fixture measures the free thermal expansion of a body
held only by symmetry constraints (u_k = 0 on the x_k = 0 face), so the answer is
a pure expansion and can be read off the far corner.

Observed: with plane-strain constants the 2D shorthand returns exactly
alpha*dT -- the plane-STRESS free expansion -- while the stiffness in the same
form is still plane strain; the explicit (3*lam + 2*mu)*alpha spelling returns
the larger plane-strain value. The ratio is (2*lam + 2*mu)/(3*lam + 2*mu),
0.769 at nu = 0.3 and 0.690 at nu = 0.45, so the error grows with nu. The same
comparison in 3D gives bit-identical fields, which is why the mistake survives a
3D test.

Mutation control: T2_MUTATE=1 writes the 2D thermal term explicitly as
(3*lam + 2*mu)*alpha*(T - T_ref), and the two 2D answers then agree.
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

E = 210e9
ALPHA = 1.2e-5
DT = 100.0


def free_expansion(nu: float, dim: int, spelling: str) -> float:
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    if dim == 2:
        msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    else:
        msh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
    d = dim
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (d,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    eps = lambda w: ufl.sym(ufl.grad(w))  # noqa: E731
    a = ufl.inner(2 * mu * eps(u) + lam * ufl.tr(eps(u)) * ufl.Identity(d),
                  eps(v)) * ufl.dx
    if spelling == "shorthand":
        # C : (alpha*dT*Identity(d)) -- tr(Identity(2)) is 2.
        eth = ALPHA * DT * ufl.Identity(d)
        thermal = 2 * mu * eth + lam * ufl.tr(eth) * ufl.Identity(d)
    else:
        thermal = (3 * lam + 2 * mu) * ALPHA * DT * ufl.Identity(d)
    L = ufl.inner(thermal, eps(v)) * ufl.dx

    msh.topology.create_connectivity(d - 1, d)
    bcs = []
    for k in range(d):
        facets = dolfinx.mesh.locate_entities_boundary(
            msh, d - 1, lambda x, k=k: np.isclose(x[k], 0.0))
        Vk, _ = V.sub(k).collapse()
        dofs = dolfinx.fem.locate_dofs_topological(
            (V.sub(k), Vk), d - 1, facets)
        bcs.append(dolfinx.fem.dirichletbc(
            dolfinx.fem.Function(Vk), dofs, V.sub(k)))

    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=bcs, petsc_options_prefix="t2_ts3_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = prob.solve()
    if isinstance(uh, tuple):
        uh = uh[0]
    return float(np.max(uh.x.array))


def main() -> int:
    tested = "explicit" if MUTATE else "shorthand"
    results = {}
    for nu in (0.3, 0.45):
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        wrong = free_expansion(nu, 2, tested)
        right = free_expansion(nu, 2, "explicit")
        ratio = wrong / right
        predicted = (2 * lam + 2 * mu) / (3 * lam + 2 * mu)
        results[nu] = ratio
        print(f"nu={nu} spelling={tested} expansion_2d={wrong:.6e} "
              f"explicit_2d={right:.6e} ratio={ratio:.6f} "
              f"two_lam_plus_two_mu_over_three_lam_plus_two_mu={predicted:.6f}")

    mu = E / (2 * (1 + 0.3))
    lam = E * 0.3 / (1.3 * 0.4)
    predicted_030 = (2 * lam + 2 * mu) / (3 * lam + 2 * mu)
    matches = abs(results[0.3] - predicted_030) < 1e-6
    print(f"ratio_is_two_lam_plus_two_mu_over_three_lam_plus_two_mu={matches}")
    grows = results[0.45] < results[0.3] - 1e-6
    print(f"error_grows_with_nu={grows}")

    plane_stress_value = ALPHA * DT
    shorthand_030 = free_expansion(0.3, 2, tested)
    is_plane_stress = abs(shorthand_030 / plane_stress_value - 1.0) < 1e-9
    print(f"expansion_2d_at_nu030={shorthand_030:.6e} "
          f"alpha_times_dT={plane_stress_value:.6e}")
    print(f"two_d_shorthand_returns_the_plane_stress_expansion={is_plane_stress}")

    s3 = free_expansion(0.3, 3, "shorthand")
    e3 = free_expansion(0.3, 3, "explicit")
    print(f"3d_shorthand={s3:.12e} 3d_explicit={e3:.12e}")
    print(f"in_3d_the_two_spellings_are_bit_identical={s3 == e3}")

    if matches and grows and is_plane_stress and s3 == e3:
        print("VERDICT=identity2_shorthand_loses_lam_times_alpha_in_2d")
        return 0
    print("VERDICT=shorthand_and_explicit_agree_in_2d")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
