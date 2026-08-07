"""Tier-2 for fenics multiphase#8: every coefficient you may want to change
between steps (dt, eps, a mobility) has to be a `fem.Constant` updated in place
with `dt.value = new_dt`. A bare Python float is baked into the compiled form,
so rebinding the Python name does nothing at all - no error, no warning.

Allen-Cahn droplet on a 32x32 unit square, 6 backward-Euler steps. Halfway
through the loop the time step is "changed" from 1e-4 to 1e-2. Wrong variant:
dt is a Python float and the change is a rebinding of that name. Right variant:
dt is a `fem.Constant` and the change is `dt.value = 1e-2`. Each variant is run
twice, once with the change and once without, and the two final fields are
compared elementwise.

Observed: the float variant's two fields are bit-identical (max difference
exactly 0.0) while the Constant variant moves by O(1e-1); nothing is raised in
either case and SNES converges in every step of all four runs.

Mutation control: T2_MUTATE=1 selects the `fem.Constant` variant as the checked
one, and the "the change had no effect" observation is then False.
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

N, NSTEP, R, EPS_OVER_H = 32, 6, 0.25, 3.0
DT0, DT1 = 1e-4, 1e-2
OPTS = {"snes_type": "newtonls", "ksp_type": "preonly", "pc_type": "lu"}


def run(kind: str, new_dt: float, tag: str):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    eps = EPS_OVER_H / N
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    phi, phi_n = dolfinx.fem.Function(V), dolfinx.fem.Function(V)

    def ic(x):
        d = R - np.sqrt((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2)
        return np.tanh(d / (eps * np.sqrt(2.0)))

    phi.interpolate(ic)
    phi_n.interpolate(ic)
    v = ufl.TestFunction(V)
    eps_c = dolfinx.fem.Constant(msh, eps)

    dt = DT0 if kind == "float" else dolfinx.fem.Constant(msh, DT0)
    F = ((phi - phi_n) / dt * v * ufl.dx
         + eps_c * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx
         + (1.0 / eps_c) * (phi ** 3 - phi) * v * ufl.dx)
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, phi, petsc_options_prefix=f"t2_mp8_{tag}_", petsc_options=OPTS)

    reasons, raised = [], "none"
    for step in range(NSTEP):
        if step == NSTEP // 2:
            try:
                if kind == "float":
                    dt = new_dt          # rebinding the Python name
                else:
                    dt.value = new_dt    # in-place update of the Constant
            except Exception as exc:     # noqa: BLE001
                raised = f"{type(exc).__name__}: {exc}"
        prob.solve()
        reasons.append(prob.solver.getConvergedReason())
        phi_n.x.array[:] = phi.x.array
    return phi.x.array.copy(), reasons, raised, float(dt) if kind == "float" else float(dt.value)


def main() -> int:
    out = {}
    for kind in ("float", "constant"):
        chg, r1, raised, seen = run(kind, DT1, f"{kind}_chg")
        ref, r2, _, _ = run(kind, DT0, f"{kind}_ref")
        diff = float(np.max(np.abs(chg - ref)))
        out[kind] = (diff, all(x > 0 for x in r1 + r2), raised, seen)
        print(f"kind={kind} python_name_now_holds={seen:g} "
              f"max_field_difference_vs_unchanged_dt={diff:.6e} "
              f"all_steps_converged={out[kind][1]} raised={raised}")

    print(f"float_rebind_changed_nothing={out['float'][0] == 0.0}")
    print(f"constant_update_changed_the_solution={out['constant'][0] > 1e-3}")
    print(f"nothing_was_raised_by_either_route="
          f"{out['float'][2] == 'none' and out['constant'][2] == 'none'}")

    sel = "constant" if MUTATE else "float"
    print(f"selected_kind={sel}")
    print(f"selected_dt_change_had_no_effect={out[sel][0] == 0.0}")
    print(f"selected_all_steps_converged={out[sel][1]}")

    if (out[sel][0] == 0.0 and out[sel][1] and out["float"][0] == 0.0
            and out["constant"][0] > 1e-3 and out["float"][2] == "none"):
        print("VERDICT=python_float_is_baked_in_only_a_constant_can_be_updated")
        return 0
    print("VERDICT=rebinding_the_float_did_change_the_solve")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
