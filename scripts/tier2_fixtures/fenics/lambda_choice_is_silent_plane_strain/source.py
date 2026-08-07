"""Tier-2 for fenics linear_elasticity#5: dolfinx has no plane-strain /
plane-stress switch, so the lambda you type decides it and nothing tells you.

Wrong variant: the 3D Lame lambda in a 2D form. The fixture builds and solves
the same 2D problem with the 3D lambda and with the plane-stress lambda,
captures Python warnings around BOTH runs, and checks that (a) not one warning
was emitted by either, and (b) the two answers nevertheless differ well beyond
solver noise. That pairing — different answer, zero diagnostics — is the claim.

Mutation control: T2_MUTATE=1 uses the plane-stress lambda in both runs, so the
answers agree and "answers differ" fails.
"""
from __future__ import annotations

import os
import tempfile
import warnings

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"
E, NU = 1.0e5, 0.3


def tip(plane_stress: bool) -> float:
    msh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([1.0, 0.1])], [40, 4])
    gdim = msh.geometry.dim
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 2, (gdim,)))
    mu = E / (2.0 * (1.0 + NU))
    lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
    if plane_stress:
        lam = 2.0 * lam * mu / (lam + 2.0 * mu)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(2.0 * mu * eps(u) + lam * ufl.tr(eps(u)) * ufl.Identity(gdim),
                  eps(v)) * ufl.dx
    f = dolfinx.fem.Constant(msh, np.array([0.0, -50.0]))
    L = ufl.dot(f, v) * ufl.dx
    msh.topology.create_connectivity(gdim - 1, gdim)
    left = dolfinx.mesh.locate_entities_boundary(
        msh, gdim - 1, lambda x: np.isclose(x[0], 0.0))
    bc = dolfinx.fem.dirichletbc(
        dolfinx.fem.Function(V),
        dolfinx.fem.locate_dofs_topological(V, gdim - 1, left))
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix="t2_lam_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = prob.solve()
    if isinstance(uh, tuple):
        uh = uh[0]
    return float(np.min(uh.x.array[1::gdim]))


def main() -> int:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wrong = tip(plane_stress=MUTATE)
        right = tip(plane_stress=True)
    print(f"warnings_emitted={len(caught)}")
    print(f"lambda3d_tip={wrong:.6e} plane_stress_tip={right:.6e}")
    rel = abs(wrong - right) / abs(right)
    print(f"relative_difference={rel:.4e}")
    differ = rel > 1e-3
    print(f"answers_differ={differ}")
    print(f"silent={len(caught) == 0}")
    if differ and not caught:
        print("VERDICT=lambda_choice_changes_answer_with_no_diagnostic")
        return 0
    print("VERDICT=not_silent_or_no_difference")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
