"""Tier-2 for fenics fracture#2: without a tension-compression split the model
cannot tell tension from compression at all, and the standard
volumetric-deviatoric split does not fix it. Read the damage extremum straight off
the dolfinx Function with d.x.array.max() at each load step.

Wrong variant: no split (and, barely better, the volumetric-deviatoric Amor
split). Right variant: the spectral (Miehe) eigenvalue split.

Staggered AT2 phase-field on a homogeneous 16x16 unit square, P1 displacement and
P1 damage, E = 210, nu = 0.3, Gc = 2.7e-3, l0 = 2h = 0.125, hybrid formulation
(the displacement problem keeps the isotropically degraded stress, so it stays
linear; only the damage driving force uses the split). Uniaxial loading on
rollers: u_y prescribed on the top and bottom edges, one corner pinned in x, so
the lateral strain is free. Ten steps to +/-6e-2, the same specimen loaded in
tension and in compression.

Observed on dolfinx 0.10.0. With NO split the two load directions produce damage
maxima that agree to every printed digit at every one of the ten steps
(0.277778, 0.606061, ..., 0.974659 in both), and pure compression alone drives
max(d) to 0.974659 -- a full crack under compression. The Amor split changes
almost nothing: compression still reaches 0.964879. The spectral split is the only
one that separates them: at the first load step its compression maximum is 0.047120
against 0.251029 in tension, 5.3 times lower, while the tension case still cracks
(0.971029 at the last step). The compression damage is small but NOT zero, as the
claim says, because the lateral Poisson strains are tensile.

DISCREPANCY worth keeping: the order-of-magnitude gap is a property of the
pre-saturation regime. Driven to 6e-2 of imposed strain the spectral compression
maximum climbs to 0.831793, i.e. d saturates towards one in every model if the
strain is pushed far enough; the ratio, not the bound, is what the split buys.

Mutation control: T2_MUTATE=1 makes the model under test the spectral split, so
the "agrees to every digit" and "compression cracks anyway" tokens go False.
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

from dolfinx import fem, mesh  # noqa: E402

DTYPE = dolfinx.default_scalar_type
E, NU, GC = 210.0, 0.3, 2.7e-3
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))
N, K_RES = 16, 1e-6
NSTEPS, UMAX = 10, 6e-2


def pos(x):
    return ufl.max_value(x, 0.0)


def psi_plus(u, split: str):
    e = ufl.sym(ufl.grad(u))
    if split == "none":
        return 0.5 * LAM * ufl.tr(e) ** 2 + MU * ufl.inner(e, e)
    if split == "amor":
        k2 = LAM + MU
        dev = e - 0.5 * ufl.tr(e) * ufl.Identity(2)
        return 0.5 * k2 * pos(ufl.tr(e)) ** 2 + MU * ufl.inner(dev, dev)
    if split == "spectral":
        tr, det = ufl.tr(e), ufl.det(e)
        disc = ufl.sqrt(pos(tr ** 2 - 4.0 * det))
        e1, e2 = (tr + disc) / 2.0, (tr - disc) / 2.0
        return 0.5 * LAM * pos(tr) ** 2 + MU * (pos(e1) ** 2 + pos(e2) ** 2)
    raise ValueError(split)


def run(split: str, sign: int) -> list[float]:
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N)
    h = 1.0 / N
    l0 = 2.0 * h
    V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    D = fem.functionspace(msh, ("Lagrange", 1))
    Q = fem.functionspace(msh, ("DG", 0))
    u, d = fem.Function(V), fem.Function(D)
    H, Hn = fem.Function(Q), fem.Function(Q)
    v, du = ufl.TestFunction(V), ufl.TrialFunction(V)
    g = (1 - d) ** 2 + K_RES
    e = ufl.sym(ufl.grad(du))
    sig = LAM * ufl.tr(e) * ufl.Identity(2) + 2 * MU * e
    a_u = g * ufl.inner(sig, ufl.sym(ufl.grad(v))) * ufl.dx
    L_u = ufl.inner(fem.Constant(msh, np.zeros(2, dtype=DTYPE)), v) * ufl.dx
    msh.topology.create_connectivity(1, 2)
    msh.topology.create_connectivity(0, 2)
    top = mesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[1], 1.0))
    bot = mesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[1], 0.0))
    Vy = V.sub(1)
    Vy0, _ = Vy.collapse()
    uy_top, uy_bot = fem.Function(Vy0), fem.Function(Vy0)
    bcs = [fem.dirichletbc(uy_top,
                           fem.locate_dofs_topological((Vy, Vy0), 1, top), Vy),
           fem.dirichletbc(uy_bot,
                           fem.locate_dofs_topological((Vy, Vy0), 1, bot), Vy)]
    Vx = V.sub(0)
    Vx0, _ = Vx.collapse()
    corner = mesh.locate_entities_boundary(
        msh, 0, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    ux0 = fem.Function(Vx0)
    bcs.append(fem.dirichletbc(
        ux0, fem.locate_dofs_topological((Vx, Vx0), 0, corner), Vx))
    w, dd = ufl.TestFunction(D), ufl.TrialFunction(D)
    a_d = ((2.0 * H + GC / l0) * dd * w
           + GC * l0 * ufl.dot(ufl.grad(dd), ufl.grad(w))) * ufl.dx
    L_d = 2.0 * H * w * ufl.dx
    pu = dolfinx.fem.petsc.LinearProblem(
        a_u, L_u, bcs=bcs, u=u, petsc_options_prefix=f"t2f2u_{split}_{sign}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})
    pd = dolfinx.fem.petsc.LinearProblem(
        a_d, L_d, bcs=[], u=d, petsc_options_prefix=f"t2f2d_{split}_{sign}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})
    psi_expr = fem.Expression(psi_plus(u, split), Q.element.interpolation_points)
    out = []
    for i in range(NSTEPS):
        uy_top.x.array[:] = sign * UMAX * (i + 1) / NSTEPS
        for _ in range(30):
            before = d.x.array.copy()
            pu.solve()
            Hn.interpolate(psi_expr)
            H.x.array[:] = np.maximum(H.x.array, Hn.x.array)
            pd.solve()
            if np.max(np.abs(d.x.array - before)) < 1e-4:
                break
        out.append(float(d.x.array.max()))
    return out


def main() -> int:
    under_test = "spectral" if MUTATE else "none"
    if MUTATE:
        print("mutation=model_under_test_is_the_spectral_split")
    res = {}
    for split in (under_test, "amor", "spectral"):
        for sign in (1, -1):
            if (split, sign) in res:
                continue
            res[(split, sign)] = run(split, sign)
            print(f"split={split:9s} sign={sign:+d} max_d_per_load_step="
                  f"{['%.6f' % x for x in res[(split, sign)]]}")

    t_u = ["%.6f" % x for x in res[(under_test, 1)]]
    c_u = ["%.6f" % x for x in res[(under_test, -1)]]
    identical = t_u == c_u
    under_test_cracks_in_compression = res[(under_test, -1)][-1] > 0.95
    amor_cracks_in_compression = res[("amor", -1)][-1] > 0.95
    sp_t, sp_c = res[("spectral", 1)], res[("spectral", -1)]
    first_ratio = sp_c[0] / sp_t[0]
    spectral_separates = first_ratio < 0.2
    spectral_tension_cracks = sp_t[-1] > 0.95
    spectral_compression_nonzero = sp_c[0] > 1e-6
    saturates = sp_c[-1] > 0.5
    print(f"model_under_test={under_test}")
    print(f"first_step_spectral_tension={sp_t[0]:.6f} "
          f"first_step_spectral_compression={sp_c[0]:.6f} "
          f"ratio={first_ratio:.4f}")
    print(f"model_under_test_tension_and_compression_agree_to_every_printed_digit="
          f"{identical}")
    print(f"model_under_test_compression_alone_drives_damage_above_0p95="
          f"{under_test_cracks_in_compression}")
    print(f"amor_compression_alone_also_drives_damage_above_0p95="
          f"{amor_cracks_in_compression}")
    print(f"spectral_compression_is_at_least_five_times_lower_at_first_load="
          f"{spectral_separates}")
    print(f"spectral_tension_still_cracks={spectral_tension_cracks}")
    print(f"spectral_compression_damage_is_small_but_not_zero="
          f"{spectral_compression_nonzero}")
    print(f"spectral_compression_damage_saturates_towards_one_at_large_strain="
          f"{saturates}")
    if identical and under_test_cracks_in_compression \
            and amor_cracks_in_compression and spectral_separates \
            and spectral_tension_cracks and spectral_compression_nonzero:
        print("VERDICT=only_the_spectral_split_separates_tension_from_compression")
        return 0
    print("VERDICT=the_model_under_test_already_separated_tension_from_compression")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
