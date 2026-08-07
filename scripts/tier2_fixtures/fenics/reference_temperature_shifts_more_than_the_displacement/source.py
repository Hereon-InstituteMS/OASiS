"""Tier-2 for fenics thermal_structural#6: the reference temperature is real
physics. The thermal strain is alpha*(T - T_ref), so leaving T_ref at 0 in an SI
model sitting at room temperature adds a constant pre-strain of order alpha*300,
and nothing warns about it.

The same square (T = 300 + 50*x K, SI steel) is solved with T_ref = 0 and with
T_ref = 300 K, twice: once held by a minimal constraint set and once with the
left face clamped. Observed:

* minimal constraints -- the deviatoric stress is the same to within round-off
  (relative difference below 1e-12; not literally bit-identical, the two linear
  solves differ in the last digits), and only the displacement changes, by about
  7x: the extra pre-strain is a stress-free uniform expansion;
* clamped face -- the displacement grows about 13x AND the deviatoric stress
  grows about 27x, so the error is not cosmetic;
* the often-repeated "the Newton iteration oscillates" does not reproduce and
  cannot: the one-way thermo-elastic step is linear, so
  fem.petsc.NonlinearProblem/SNES converges in exactly 1 iteration for either
  T_ref, to a field bit-identical (np.array_equal) to the LinearProblem answer.

Mutation control: T2_MUTATE=1 sets the model under test to T_ref = 300 K, the
physically correct value, and the ratios collapse to 1.
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

E, NU, ALPHA = 210e9, 0.3, 1.2e-5
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
BETA = (3 * LAM + 2 * MU) * ALPHA
T_REF_TEST = 300.0 if MUTATE else 0.0


def minimal_bcs(msh, V):
    bcs = []
    for point, comps in (((0.0, 0.0), (0, 1)), ((1.0, 0.0), (1,))):
        for k in comps:
            Vk, _ = V.sub(k).collapse()
            dofs = dolfinx.fem.locate_dofs_geometrical(
                (V.sub(k), Vk),
                lambda x, p=point: np.isclose(x[0], p[0])
                & np.isclose(x[1], p[1]))
            bcs.append(dolfinx.fem.dirichletbc(
                dolfinx.fem.Function(Vk), dofs, V.sub(k)))
    return bcs


def run(t_ref, clamped, prefix, nonlinear=False):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    d = 2
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (d,)))
    S = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    temp = dolfinx.fem.Function(S)
    temp.interpolate(lambda x: 300.0 + 50.0 * x[0])
    t_ref_c = dolfinx.fem.Constant(msh, float(t_ref))
    eps = lambda w: ufl.sym(ufl.grad(w))  # noqa: E731
    msh.topology.create_connectivity(d - 1, d)
    if clamped:
        facets = dolfinx.mesh.locate_entities_boundary(
            msh, d - 1, lambda x: np.isclose(x[0], 0.0))
        bcs = [dolfinx.fem.dirichletbc(
            np.zeros(d),
            dolfinx.fem.locate_dofs_topological(V, d - 1, facets), V)]
    else:
        bcs = minimal_bcs(msh, V)

    if nonlinear:
        uh = dolfinx.fem.Function(V)
        v = ufl.TestFunction(V)
        res = (ufl.inner(2 * MU * eps(uh)
                         + LAM * ufl.tr(eps(uh)) * ufl.Identity(d),
                         eps(v)) * ufl.dx
               - BETA * (temp - t_ref_c) * ufl.div(v) * ufl.dx)
        prob = dolfinx.fem.petsc.NonlinearProblem(
            res, uh, bcs=bcs, petsc_options_prefix=prefix,
            petsc_options={"snes_type": "newtonls", "ksp_type": "preonly",
                           "pc_type": "lu", "snes_rtol": 1e-12})
        prob.solve()
        its = prob.solver.getIterationNumber()
    else:
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        a = ufl.inner(2 * MU * eps(u) + LAM * ufl.tr(eps(u)) * ufl.Identity(d),
                      eps(v)) * ufl.dx
        L = BETA * (temp - t_ref_c) * ufl.div(v) * ufl.dx
        prob = dolfinx.fem.petsc.LinearProblem(
            a, L, bcs=bcs, petsc_options_prefix=prefix,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = prob.solve()
        if isinstance(uh, tuple):
            uh = uh[0]
        its = -1
    sig = (2 * MU * eps(uh) + LAM * ufl.tr(eps(uh)) * ufl.Identity(d)
           - BETA * (temp - t_ref_c) * ufl.Identity(d))
    dev = sig - ufl.tr(sig) / d * ufl.Identity(d)
    dev_norm = float(np.sqrt(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(ufl.inner(dev, dev) * ufl.dx))))
    return (float(np.max(np.abs(uh.x.array))), dev_norm, its,
            np.array(uh.x.array))


def main() -> int:
    print(f"T_ref_of_model_under_test={T_REF_TEST}")
    u_free_w, d_free_w, _, _ = run(T_REF_TEST, False, "t2_ts6a_")
    u_free_r, d_free_r, _, _ = run(300.0, False, "t2_ts6b_")
    dev_rel = abs(d_free_w - d_free_r) / d_free_r
    u_free_ratio = u_free_w / u_free_r
    print(f"minimal_constraints max_abs_u_test={u_free_w:.6e} "
          f"max_abs_u_at_300K={u_free_r:.6e} ratio={u_free_ratio:.3f}")
    print(f"minimal_constraints dev_norm_test={d_free_w:.10e} "
          f"dev_norm_at_300K={d_free_r:.10e} relative_difference={dev_rel:.3e}")
    dev_same = dev_rel < 1e-12
    print(f"minimal_constraints_dev_stress_is_unchanged_by_Tref={dev_same}")
    print(f"minimal_constraints_displacement_shifts={u_free_ratio > 3.0}")

    u_cl_w, d_cl_w, _, _ = run(T_REF_TEST, True, "t2_ts6c_")
    u_cl_r, d_cl_r, _, _ = run(300.0, True, "t2_ts6d_")
    u_ratio = u_cl_w / u_cl_r
    d_ratio = d_cl_w / d_cl_r
    print(f"clamped max_abs_u_test={u_cl_w:.6e} max_abs_u_at_300K={u_cl_r:.6e} "
          f"ratio={u_ratio:.3f}")
    print(f"clamped dev_norm_test={d_cl_w:.6e} dev_norm_at_300K={d_cl_r:.6e} "
          f"ratio={d_ratio:.3f}")
    print(f"clamped_displacement_grows_several_fold={u_ratio > 3.0}")
    print(f"clamped_dev_stress_grows_several_fold={d_ratio > 3.0}")

    its_ok, same_ok = True, True
    for t_ref in (0.0, 300.0):
        _, _, its, arr_n = run(t_ref, True, f"t2_ts6e{int(t_ref)}_",
                               nonlinear=True)
        _, _, _, arr_l = run(t_ref, True, f"t2_ts6f{int(t_ref)}_")
        same = bool(np.array_equal(arr_n, arr_l))
        print(f"snes_at_T_ref={t_ref} iterations={its} "
              f"bit_identical_to_linearproblem={same}")
        its_ok &= its == 1
        same_ok &= same
    print(f"newton_converges_in_exactly_one_iteration={its_ok}")
    print(f"snes_answer_is_bit_identical_to_linearproblem={same_ok}")

    if (dev_same and u_free_ratio > 3.0 and u_ratio > 3.0 and d_ratio > 3.0
            and its_ok and same_ok):
        print("VERDICT=T_ref_zero_is_a_silent_pre_strain")
        return 0
    print("VERDICT=T_ref_did_not_matter")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
