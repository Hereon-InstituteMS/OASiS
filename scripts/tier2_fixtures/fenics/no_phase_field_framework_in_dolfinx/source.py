"""Tier-2 for fenics fracture#5: there is no phase-field anything in DOLFINx,
and the third-party frameworks are not installed by default - write the
staggered scheme yourself.

Wrong variant: reach for a ready-made phase-field package (`import phasefieldx`)
or for a DOLFINx phase-field/damage class. Right variant: assemble the staggered
scheme out of the ordinary DOLFINx pieces - fem.functionspace for the three
fields, NonlinearProblem for the displacement step, LinearProblem for the damage
step, and fem.Expression + Function.interpolate for the history update.

Observed on dolfinx 0.10.0: `import phasefieldx` raises
"ModuleNotFoundError: No module named 'phasefieldx'", a scan of dolfinx,
dolfinx.fem, dolfinx.fem.petsc, dolfinx.mesh, dolfinx.io and dolfinx.la for any
name containing phasefield/phase_field/damage/fracture/crack/griffith returns an
empty list, and every one of the five ordinary pieces the claim names is present
and does its job on a 8x8 unit square.

Mutation control: T2_MUTATE=1 skips the third-party import entirely and builds
the staggered step from the DOLFINx pieces only - the ModuleNotFoundError text
then never appears.
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

import dolfinx.io  # noqa: E402
import dolfinx.la  # noqa: E402
import dolfinx.mesh  # noqa: E402
from dolfinx import fem  # noqa: E402

THIRD_PARTY = ["phasefieldx", "fenics_shells", "dolfiny"]
NEEDLES = ("phasefield", "phase_field", "damage", "fracture", "crack",
           "griffith")


def scan_dolfinx_for_phase_field_names() -> list[str]:
    mods = [dolfinx, dolfinx.fem, dolfinx.fem.petsc, dolfinx.mesh, dolfinx.io,
            dolfinx.la]
    hits = []
    for m in mods:
        for n in dir(m):
            if any(k in n.lower() for k in NEEDLES):
                hits.append(f"{m.__name__}.{n}")
    return hits


def build_staggered_step_from_ordinary_pieces() -> dict[str, bool]:
    """The route the claim prescribes: three spaces, a NonlinearProblem for u,
    a LinearProblem for d, fem.Expression + interpolate for the history."""
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    tdim = msh.topology.dim
    V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    W = fem.functionspace(msh, ("Lagrange", 1))
    Q = fem.functionspace(msh, ("DG", 0))
    u, v = fem.Function(V), ufl.TestFunction(V)
    d, q = fem.Function(W), ufl.TestFunction(W)
    dtr = ufl.TrialFunction(W)
    H, Hn = fem.Function(Q), fem.Function(Q)
    mu, lam, Gc, l0, k_res = 80.77, 121.15, 2.7e-3, 0.04, 1e-6
    e = ufl.sym(ufl.grad(u))
    psi_p = 0.5 * lam * ufl.tr(e) ** 2 + mu * ufl.inner(e, e)
    Fu = ufl.derivative(((1 - d) ** 2 + k_res) * psi_p * ufl.dx, u, v)
    a_d = ((2 * H + Gc / l0) * dtr * q
           + Gc * l0 * ufl.inner(ufl.grad(dtr), ufl.grad(q))) * ufl.dx
    L_d = 2 * H * q * ufl.dx
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    bfac = dolfinx.mesh.exterior_facet_indices(msh.topology)
    uD = fem.Function(V)
    uD.sub(1).interpolate(lambda x: np.full_like(x[0], 2.0e-3))
    bcs = [fem.dirichletbc(uD, fem.locate_dofs_topological(V, fdim, bfac))]
    pu = dolfinx.fem.petsc.NonlinearProblem(
        Fu, u, bcs=bcs, petsc_options_prefix="t2_fr5_u_",
        petsc_options={"snes_type": "newtonls", "snes_max_it": 40,
                       "ksp_type": "preonly", "pc_type": "lu"})
    pd = dolfinx.fem.petsc.LinearProblem(
        a_d, L_d, u=d, petsc_options_prefix="t2_fr5_d_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    pu.solve()
    u_ok = pu.solver.getConvergedReason() > 0
    psiE = fem.Expression(psi_p, Q.element.interpolation_points)
    Hn.interpolate(psiE)
    H.x.array[:] = np.maximum(H.x.array, Hn.x.array)
    pd.solve()
    d_ok = (pd.solver.getConvergedReason() > 0
            and bool(np.all(np.isfinite(d.x.array))))
    return {"functionspace": V.dofmap.index_map.size_global > 0,
            "nonlinearproblem_for_u": bool(u_ok),
            "linearproblem_for_d": bool(d_ok),
            "expression_and_interpolate": bool(H.x.array.max() > 0.0)}


def main() -> int:
    missing = []
    if not MUTATE:
        for name in THIRD_PARTY:
            try:
                __import__(name)
                print(f"third_party_import_{name}=IMPORTED")
            except ModuleNotFoundError as exc:
                missing.append(name)
                print(f"third_party_import_{name} -> "
                      f"{type(exc).__name__}: {exc}")
    else:
        print("mutation=writing_the_staggered_scheme_by_hand_no_third_party")

    hits = scan_dolfinx_for_phase_field_names()
    print(f"dolfinx_names_matching_phase_field_or_damage={hits}")

    pieces = build_staggered_step_from_ordinary_pieces()
    for k, ok in pieces.items():
        print(f"ordinary_piece_{k}_works={ok}")

    all_absent = len(missing) == len(THIRD_PARTY)
    scan_empty = hits == []
    hand_written_works = all(pieces.values())
    print(f"every_third_party_phase_field_package_absent={all_absent}")
    print(f"dolfinx_name_scan_is_empty={scan_empty}")
    print(f"hand_written_staggered_scheme_works={hand_written_works}")
    if all_absent and scan_empty and hand_written_works:
        print("VERDICT=no_phase_field_api_write_the_staggered_scheme_yourself")
        return 0
    print("VERDICT=a_phase_field_api_was_available")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
