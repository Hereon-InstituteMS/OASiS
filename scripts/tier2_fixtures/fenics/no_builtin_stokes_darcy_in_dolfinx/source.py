"""Tier-2 for fenics stokes_darcy#0: there is no built-in Stokes-Darcy, Darcy
or Brinkman anything in dolfinx - the whole coupled form has to be written by
hand.

Wrong variant: reach for `dolfinx.fem.StokesDarcy` (or any of the other
plausible spellings). Right variant: the single-mesh Brinkman formulation -
Taylor-Hood over the whole domain plus (mu/K)*inner(u, v)*dx(porous_marker).

Observed on dolfinx 0.10.0: the attribute access raises
"AttributeError: module 'dolfinx.fem' has no attribute 'StokesDarcy'", and a
name scan over dolfinx, dolfinx.fem, dolfinx.fem.petsc, dolfinx.mesh,
dolfinx.io, dolfinx.nls and dolfinx.la for any name containing darcy, stokes,
brinkman, porous or biot returns an empty list. The hand-written Brinkman form
on a 24x12 rectangle solves, KSP reason 4, the mass balance closes to ~1e-16 and
the porous bed is two orders of magnitude slower than the free channel.

Mutation control: T2_MUTATE=1 skips the attribute probes and goes straight to
the hand-written Brinkman form, so the AttributeError text never appears.
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

import basix.ufl  # noqa: E402
import dolfinx.io  # noqa: E402
import dolfinx.la  # noqa: E402
import dolfinx.mesh  # noqa: E402
import dolfinx.nls  # noqa: E402
from dolfinx import fem, mesh  # noqa: E402

NEEDLES = ("darcy", "stokes", "brinkman", "porous", "biot")


def probe_attribute(mod, name: str) -> str:
    try:
        getattr(mod, name)
    except AttributeError as exc:
        return f"{type(exc).__name__}: {exc}"
    return "PRESENT"


def scan() -> list[str]:
    mods = [dolfinx, dolfinx.fem, dolfinx.fem.petsc, dolfinx.mesh, dolfinx.io,
            dolfinx.la, dolfinx.nls]
    hits = []
    for m in mods:
        for n in dir(m):
            if any(k in n.lower() for k in NEEDLES):
                hits.append(f"{m.__name__}.{n}")
    return hits


def hand_written_brinkman():
    """The verified working route: one mesh, Taylor-Hood everywhere, plus a
    Darcy drag term restricted to the porous cells."""
    msh = mesh.create_rectangle(
        MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([2.0, 1.0])],
        [24, 12], mesh.CellType.triangle)
    gdim, tdim = msh.geometry.dim, msh.topology.dim
    por = mesh.locate_entities(msh, tdim, lambda x: x[1] <= 0.5 + 1e-12)
    ids = np.full(msh.topology.index_map(tdim).size_local, 1, dtype=np.int32)
    ids[por] = 2
    ct = mesh.meshtags(msh, tdim, np.arange(ids.size, dtype=np.int32), ids)
    dx = ufl.Measure("dx", domain=msh, subdomain_data=ct)
    fs, ms = [], []
    for tag, fn in ((1, lambda x: np.isclose(x[0], 0.0)),
                    (2, lambda x: np.isclose(x[0], 2.0))):
        f = mesh.locate_entities_boundary(msh, tdim - 1, fn)
        fs.append(f)
        ms.append(np.full(f.size, tag, dtype=np.int32))
    fa, ma = np.concatenate(fs), np.concatenate(ms)
    srt = np.argsort(fa)
    ft = mesh.meshtags(msh, tdim - 1, fa[srt], ma[srt])
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
    Ve = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(gdim,))
    Qe = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = fem.functionspace(msh, basix.ufl.mixed_element([Ve, Qe]))
    (u, p), (v, q) = ufl.TrialFunctions(W), ufl.TestFunctions(W)
    mu = fem.Constant(msh, 1.0)
    K = fem.Constant(msh, 1e-4)
    p_in = fem.Constant(msh, 1.0)
    n = ufl.FacetNormal(msh)
    a = (2.0 * mu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * dx
         - p * ufl.div(v) * dx
         - q * ufl.div(u) * dx
         + (mu / K) * ufl.inner(u, v) * dx(2))
    L = -p_in * ufl.dot(n, v) * ds(1)
    V0, _ = W.sub(0).collapse()
    u_wall = fem.Function(V0)
    walls = mesh.locate_entities_boundary(
        msh, tdim - 1,
        lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    bcs = [fem.dirichletbc(
        u_wall, fem.locate_dofs_topological((W.sub(0), V0), tdim - 1, walls),
        W.sub(0))]
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=bcs, petsc_options_prefix="t2_sd0_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})
    wh = prob.solve()
    reason = prob.solver.getConvergedReason()
    uh = wh.sub(0).collapse()
    q_in = -fem.assemble_scalar(fem.form(ufl.dot(uh, n) * ds(1)))
    q_out = fem.assemble_scalar(fem.form(ufl.dot(uh, n) * ds(2)))
    bal = abs(q_in - q_out) / abs(q_in)
    a_free = fem.assemble_scalar(fem.form(1.0 * dx(1)))
    a_por = fem.assemble_scalar(fem.form(1.0 * dx(2)))
    m_free = fem.assemble_scalar(fem.form(uh[0] * dx(1))) / a_free
    m_por = fem.assemble_scalar(fem.form(uh[0] * dx(2))) / a_por
    return int(reason), float(bal), float(m_free), float(m_por)


def main() -> int:
    probes = {}
    if not MUTATE:
        for mod, name in ((fem, "StokesDarcy"), (fem, "Darcy"),
                          (fem, "Brinkman"), (dolfinx, "StokesDarcy")):
            key = f"{mod.__name__}.{name}"
            probes[key] = probe_attribute(mod, name)
            print(f"probe {key} -> {probes[key]}")
    else:
        print("mutation=skipping_the_attribute_probes_writing_the_form_by_hand")

    hits = scan()
    print(f"dolfinx_names_matching_darcy_stokes_brinkman_porous_biot={hits}")

    reason, bal, m_free, m_por = hand_written_brinkman()
    print(f"brinkman_ksp_reason={reason}")
    print(f"brinkman_mass_balance_rel_err={bal:.3e}")
    print(f"brinkman_mean_ux_free={m_free:.6e} porous={m_por:.6e}")

    none_exist = bool(probes) and all(v.startswith("AttributeError")
                                      for v in probes.values())
    scan_empty = hits == []
    brinkman_ok = (reason > 0 and bal < 1e-10
                   and abs(m_por) < 0.1 * abs(m_free))
    print(f"no_stokes_darcy_attribute_anywhere={none_exist}")
    print(f"dolfinx_name_scan_is_empty={scan_empty}")
    print(f"hand_written_brinkman_route_works={brinkman_ok}")
    if none_exist and scan_empty and brinkman_ok:
        print("VERDICT=no_builtin_stokes_darcy_write_the_brinkman_form")
        return 0
    print("VERDICT=dolfinx_offered_a_stokes_darcy_api")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
