"""Tier-2 for fenics thermal_structural#4: plane strain and plane stress need a
different lambda AND a different thermal modulus, and picking one of the two
wrong pairings raises nothing at all.

Free thermal expansion is measured on a body held by symmetry constraints only,
so the answer is a pure expansion, and it is compared against 3D runs of the same
material: a cube with u_z pinned on both z faces (plane strain) and the same cube
with free z faces (plane stress). Observed at nu = 0.3 and 0.45, SI steel:

* plane strain, lam = E*nu/((1+nu)*(1-2*nu)) with beta = (3*lam+2*mu)*alpha,
  reproduces the u_z-constrained 3D expansion; (3*lam+2*mu)*alpha equals
  E*alpha/(1-2*nu) to the last digit;
* plane stress, lam_ps = 2*lam*mu/(lam+2*mu) with beta_ps = (2*lam_ps+2*mu)*alpha,
  reproduces the free-z 3D expansion, which is just alpha*dT;
  (2*lam_ps+2*mu)*alpha equals E*alpha/(1-nu) to the last digit;
* plane strain gives the larger expansion and the larger deviatoric stress in a
  clamped bar, and both gaps widen as nu grows;
* wrong recipe 1, lam_ps with the 3D modulus (3*lam_ps+2*mu)*alpha, overshoots
  the plane-stress expansion by about 23 % and matches neither 3D reference;
* wrong recipe 2, E' = E/(1-nu**2) put into the plane-strain lambda, lands
  within about 1 % of plain plane strain at nu = 0.3 -- a silently wrong model,
  not an obvious one.

Mutation control: T2_MUTATE=1 gives the plane-stress model its correct modulus
(2*lam_ps+2*mu)*alpha, and the model under test then matches the free-z 3D run.
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

E, ALPHA, DT = 210e9, 1.2e-5, 100.0


def _sym_bcs(msh, V, d, pin_top_z=False):
    msh.topology.create_connectivity(d - 1, d)
    bcs = []
    faces = [(k, 0.0) for k in range(d)]
    if pin_top_z:
        faces.append((2, 1.0))
    for k, at in faces:
        facets = dolfinx.mesh.locate_entities_boundary(
            msh, d - 1, lambda x, k=k, at=at: np.isclose(x[k], at))
        Vk, _ = V.sub(k).collapse()
        dofs = dolfinx.fem.locate_dofs_topological(
            (V.sub(k), Vk), d - 1, facets)
        bcs.append(dolfinx.fem.dirichletbc(
            dolfinx.fem.Function(Vk), dofs, V.sub(k)))
    return bcs


def run2d(lam, mu, beta, clamped=False):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    d = 2
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (d,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    eps = lambda w: ufl.sym(ufl.grad(w))  # noqa: E731
    a = ufl.inner(2 * mu * eps(u) + lam * ufl.tr(eps(u)) * ufl.Identity(d),
                  eps(v)) * ufl.dx
    L = beta * DT * ufl.div(v) * ufl.dx
    if clamped:
        msh.topology.create_connectivity(d - 1, d)
        facets = dolfinx.mesh.locate_entities_boundary(
            msh, d - 1, lambda x: np.isclose(x[0], 0.0))
        bcs = [dolfinx.fem.dirichletbc(
            np.zeros(d),
            dolfinx.fem.locate_dofs_topological(V, d - 1, facets), V)]
    else:
        bcs = _sym_bcs(msh, V, d)
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=bcs, petsc_options_prefix="t2_ts4_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = prob.solve()
    if isinstance(uh, tuple):
        uh = uh[0]
    sig = (2 * mu * eps(uh) + lam * ufl.tr(eps(uh)) * ufl.Identity(d)
           - beta * DT * ufl.Identity(d))
    dev = sig - ufl.tr(sig) / d * ufl.Identity(d)
    dev_norm = float(np.sqrt(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(ufl.inner(dev, dev) * ufl.dx))))
    return float(np.max(uh.x.array)), dev_norm


def run3d(nu, plane_strain):
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    beta = (3 * lam + 2 * mu) * ALPHA
    msh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
    d = 3
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (d,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    eps = lambda w: ufl.sym(ufl.grad(w))  # noqa: E731
    a = ufl.inner(2 * mu * eps(u) + lam * ufl.tr(eps(u)) * ufl.Identity(d),
                  eps(v)) * ufl.dx
    L = beta * DT * ufl.div(v) * ufl.dx
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=_sym_bcs(msh, V, d, pin_top_z=plane_strain),
        petsc_options_prefix="t2_ts4b_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = prob.solve()
    if isinstance(uh, tuple):
        uh = uh[0]
    return float(np.max(uh.x.array.reshape(-1, d)[:, 0]))


def main() -> int:
    free, dev = {}, {}
    for nu in (0.3, 0.45):
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        lam_ps = 2 * lam * mu / (lam + 2 * mu)
        lam_e = (E / (1 - nu ** 2)) * nu / ((1 + nu) * (1 - 2 * nu))
        beta_test = ((2 * lam_ps + 2 * mu) if MUTATE
                     else (3 * lam_ps + 2 * mu)) * ALPHA
        models = {
            "plane_strain": (lam, mu, (3 * lam + 2 * mu) * ALPHA),
            "plane_stress_correct": (lam_ps, mu, (2 * lam_ps + 2 * mu) * ALPHA),
            "plane_stress_model_under_test": (lam_ps, mu, beta_test),
            "Eprime_in_plane_strain_lambda": (lam_e, mu,
                                              (3 * lam_e + 2 * mu) * ALPHA),
        }
        print(f"nu={nu} beta_plane_strain={(3 * lam + 2 * mu) * ALPHA:.6e} "
              f"E_alpha_over_1_minus_2nu={E * ALPHA / (1 - 2 * nu):.6e} "
              f"beta_plane_stress={(2 * lam_ps + 2 * mu) * ALPHA:.6e} "
              f"E_alpha_over_1_minus_nu={E * ALPHA / (1 - nu):.6e}")
        for name, (l_, m_, b_) in models.items():
            f_, _ = run2d(l_, m_, b_)
            free[(nu, name)] = f_
            print(f"  nu={nu} model={name} free_expansion={f_:.6e}")
        for name in ("plane_strain", "plane_stress_correct"):
            l_, m_, b_ = models[name]
            _, dn = run2d(l_, m_, b_, clamped=True)
            dev[(nu, name)] = dn
            print(f"  nu={nu} model={name} clamped_dev_stress_norm={dn:.6e}")

    ref_pe = run3d(0.3, plane_strain=True)
    ref_ps = run3d(0.3, plane_strain=False)
    print(f"3d_uz_constrained_expansion={ref_pe:.6e} "
          f"3d_free_z_expansion={ref_ps:.6e}")

    match_pe = abs(free[(0.3, "plane_strain")] / ref_pe - 1.0) < 1e-9
    match_ps = abs(free[(0.3, "plane_stress_correct")] / ref_ps - 1.0) < 1e-9
    print(f"plane_strain_2d_matches_3d_uz_constrained={match_pe}")
    print(f"plane_stress_2d_matches_3d_free_z={match_ps}")

    bigger = (free[(0.3, "plane_strain")] > free[(0.3, "plane_stress_correct")]
              and dev[(0.3, "plane_strain")] > dev[(0.3, "plane_stress_correct")])
    gap03 = free[(0.3, "plane_strain")] / free[(0.3, "plane_stress_correct")]
    gap045 = free[(0.45, "plane_strain")] / free[(0.45, "plane_stress_correct")]
    dgap03 = dev[(0.3, "plane_strain")] / dev[(0.3, "plane_stress_correct")]
    dgap045 = dev[(0.45, "plane_strain")] / dev[(0.45, "plane_stress_correct")]
    print(f"expansion_gap_nu030={gap03:.4f} expansion_gap_nu045={gap045:.4f}")
    print(f"dev_stress_gap_nu030={dgap03:.4f} dev_stress_gap_nu045={dgap045:.4f}")
    print(f"plane_strain_expands_and_stresses_more={bigger}")
    print(f"both_gaps_widen_with_nu={gap045 > gap03 and dgap045 > dgap03}")

    test = free[(0.3, "plane_stress_model_under_test")]
    over = test / free[(0.3, "plane_stress_correct")] - 1.0
    print(f"model_under_test_over_plane_stress={over:+.4f}")
    overshoots = over > 0.1
    disagrees = abs(test / ref_ps - 1.0) > 0.1
    print(f"plane_stress_lambda_with_three_lam_beta_overshoots={overshoots}")
    print(f"model_under_test_disagrees_with_3d_free_z_reference={disagrees}")

    silent = abs(free[(0.3, "Eprime_in_plane_strain_lambda")]
                 / free[(0.3, "plane_strain")] - 1.0)
    print(f"Eprime_recipe_relative_gap_to_plane_strain={silent:.4f}")
    print(f"Eprime_recipe_is_within_two_percent_of_plane_strain={silent < 0.02}")

    if (match_pe and match_ps and bigger and gap045 > gap03
            and dgap045 > dgap03 and overshoots and disagrees
            and silent < 0.02):
        print("VERDICT=plane_strain_and_plane_stress_need_different_beta")
        return 0
    print("VERDICT=the_two_plane_models_agree")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
