"""Tier-2 for fenics time_dependent_heat#7: a coefficient that varies in SPACE
has to be a `fem.Function` on a DG0 space populated per cell (here from cell
midpoints via `dolfinx.mesh.compute_midpoints`); a single `fem.Constant` cannot
represent a layered conductivity. A coefficient that varies in TIME has to be a
`fem.Constant` updated in place with `c.value = ...`; rebinding a Python name has
no effect at all and nothing is raised either way.

Time part: assemble `c * dx` over the unit square with c a bare Python float
2.0, then rebind the name to 10.0 and assemble the SAME compiled form again.
Space part: 16x16 unit square, T = 1 on the left wall, T = 0 on the right, 60
backward-Euler steps of dt = 0.05 (long enough to reach the steady state), with
k = 1 for x < 0.5 and k = 100 for x > 0.5. The two-layer analytic interface
temperature is 1/101 = 0.009901.

Observed: the float form still assembles to 2.0000 after the rebinding while
`fem.Constant.value = 10.0` gives 10.0000 at once, and neither route raises. The
DG0 conductivity reproduces the analytic interface value (0.009901 against
0.009901), while a single Constant - whether 1, 100 or the arithmetic mean 50.5 -
puts the interface at 0.500000 in every case, i.e. it cannot represent the layers
at all.

Mutation control: T2_MUTATE=1 uses a fem.Constant for the time-varying
coefficient and the DG0 function for the layered conductivity, so both
"the change had no effect" observations are lost.
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

N, DT, NSTEP = 16, 0.05, 60
K_LEFT, K_RIGHT = 1.0, 100.0
ANALYTIC = K_LEFT / (K_LEFT + K_RIGHT)      # 1/101 at the interface


def time_varying(as_constant: bool):
    """Assemble c*dx, 'change' c, assemble again. Returns (first, second, err)."""
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    c = dolfinx.fem.Constant(msh, 2.0) if as_constant else 2.0
    form = dolfinx.fem.form(c * ufl.dx(domain=msh))
    first = float(dolfinx.fem.assemble_scalar(form))
    err = "none"
    try:
        if as_constant:
            c.value = 10.0
        else:
            c = 10.0            # rebinding the Python name
    except Exception as exc:    # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    second = float(dolfinx.fem.assemble_scalar(form))
    return first, second, err


def march(kind: str, tag: str):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    if kind == "dg0":
        Q = dolfinx.fem.functionspace(msh, ("DG", 0))
        k = dolfinx.fem.Function(Q)
        cells = np.arange(msh.topology.index_map(tdim).size_local
                          + msh.topology.index_map(tdim).num_ghosts,
                          dtype=np.int32)
        mid = dolfinx.mesh.compute_midpoints(msh, tdim, cells)
        k.x.array[:] = np.where(mid[:, 0] < 0.5, K_LEFT, K_RIGHT)
    else:
        k = dolfinx.fem.Constant(msh, float(kind))

    T_n = dolfinx.fem.Function(V)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dt = dolfinx.fem.Constant(msh, DT)
    a = (u / dt) * v * ufl.dx + k * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (T_n / dt) * v * ufl.dx
    left = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 0.0))
    right = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 1.0))
    bcs = [dolfinx.fem.dirichletbc(
               dolfinx.fem.Constant(msh, 1.0),
               dolfinx.fem.locate_dofs_topological(V, fdim, left), V),
           dolfinx.fem.dirichletbc(
               dolfinx.fem.Constant(msh, 0.0),
               dolfinx.fem.locate_dofs_topological(V, fdim, right), V)]
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=bcs, u=dolfinx.fem.Function(V),
        petsc_options_prefix=f"t2_tdh7_{tag}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    for _ in range(NSTEP):
        T_h = prob.solve()
        T_n.x.array[:] = T_h.x.array
    xy = V.tabulate_dof_coordinates()
    node = np.where((np.abs(xy[:, 0] - 0.5) < 1e-9)
                    & (np.abs(xy[:, 1] - 0.5) < 1e-9))[0]
    return float(T_h.x.array[node[0]])


def main() -> int:
    f_first, f_second, f_err = time_varying(as_constant=False)
    c_first, c_second, c_err = time_varying(as_constant=True)
    print(f"python_float_form_before={f_first:.4f} after_rebinding={f_second:.4f} "
          f"raised={f_err}")
    print(f"fem_constant_form_before={c_first:.4f} after_value_update={c_second:.4f} "
          f"raised={c_err}")
    print(f"python_float_rebinding_has_no_effect={f_second == f_first}")
    print(f"fem_constant_value_update_is_seen_immediately={c_second == 10.0}")
    print(f"neither_coefficient_route_raised="
          f"{f_err == 'none' and c_err == 'none'}")

    interface = {}
    for kind in ("dg0", str(K_LEFT), str(K_RIGHT), str(0.5 * (K_LEFT + K_RIGHT))):
        interface[kind] = march(kind, kind.replace(".", "p"))
        print(f"conductivity={kind} interface_temperature={interface[kind]:.6f}")
    print(f"two_layer_analytic_interface_temperature={ANALYTIC:.6f}")
    dg_ok = abs(interface["dg0"] - ANALYTIC) < 0.02 * ANALYTIC + 1e-4
    const_flat = all(abs(interface[k] - 0.5) < 1e-6
                     for k in interface if k != "dg0")
    print(f"dg0_layered_matches_the_two_layer_analytic_value={dg_ok}")
    print(f"every_single_constant_puts_the_interface_at_one_half={const_flat}")

    sel_time_no_effect = (c_second == c_first) if MUTATE else (f_second == f_first)
    sel_space_kind = "dg0" if MUTATE else str(0.5 * (K_LEFT + K_RIGHT))
    sel_space_wrong = abs(interface[sel_space_kind] - ANALYTIC) > 0.1
    print(f"selected_time_route={'fem_constant' if MUTATE else 'python_float'} "
          f"selected_space_route={sel_space_kind}")
    print(f"selected_time_change_had_no_effect={sel_time_no_effect}")
    print(f"selected_space_route_misses_the_layered_answer={sel_space_wrong}")

    if (sel_time_no_effect and sel_space_wrong and f_second == f_first
            and c_second == 10.0 and f_err == "none" and c_err == "none"
            and dg_ok and const_flat):
        print("VERDICT=space_varying_needs_dg0_time_varying_needs_a_constant")
        return 0
    print("VERDICT=a_bare_float_and_a_single_constant_were_enough")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
