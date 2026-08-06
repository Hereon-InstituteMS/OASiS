"""Tier-2 for fenics multiphase#8: every coefficient you may want to change
between steps (dt, eps, a mobility) must be a `fem.Constant`, updated in place
with `dt.value = new_dt`. A bare Python float is baked into the compiled form, so
rebinding the Python name produces exactly the same numbers as before - no error,
no warning, the new value simply has no effect.

Allen-Cahn droplet, 32x32 unit square, eps = 3h, 4 backward Euler steps. Three
runs share the same first two steps and then differ: (base) keeps dt = 1e-3
throughout; (float) rebinds the Python name dt = 1e-1 after step 2; (constant)
sets dt_c.value = 1e-1 after step 2. The float run is BIT-IDENTICAL to base -
the rebinding did nothing - while the Constant run moves by a large margin.
Nothing is raised in either case.

Mutation control: T2_MUTATE=1 makes the checked run the Constant one, whose
update does take effect, so the "rebinding changed nothing" finding is lost.
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

N, NSTEP, SWITCH, DT0, DT1 = 32, 4, 2, 1e-3, 1e-1
EPS_OVER_H, R = 3.0, 0.25
OPTS = {"snes_type": "newtonls", "ksp_type": "preonly", "pc_type": "lu"}


def run(mode: str):
    """mode: 'base' (never change dt), 'float' (rebind a Python float),
    'constant' (update a fem.Constant in place)."""
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

    dt_float = DT0                              # a bare Python float
    dt_const = dolfinx.fem.Constant(msh, DT0)   # the right way
    dt = dt_const if mode == "constant" else dt_float
    F = ((phi - phi_n) / dt * v * ufl.dx
         + eps_c * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx
         + (1.0 / eps_c) * (phi ** 3 - phi) * v * ufl.dx)
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, phi, petsc_options_prefix=f"t2_mp8_{mode}_", petsc_options=OPTS)

    raised = "none"
    for step in range(NSTEP):
        if step == SWITCH:
            try:
                if mode == "float":
                    dt_float = DT1      # rebinding the name: no effect
                elif mode == "constant":
                    dt_const.value = DT1
            except Exception as exc:  # noqa: BLE001
                raised = f"{type(exc).__name__}: {exc}"
        prob.solve()
        phi_n.x.array[:] = phi.x.array
    del dt_float
    return phi.x.array.copy(), raised


def main() -> int:
    base, _ = run("base")
    flt, raised_f = run("float")
    con, raised_c = run("constant")
    sel, raised = (con, raised_c) if MUTATE else (flt, raised_f)

    same_bits = bool(np.array_equal(sel, base))
    diff = float(np.max(np.abs(sel - base)))
    con_diff = float(np.max(np.abs(con - base)))
    print(f"selected_variant={'constant' if MUTATE else 'float'}")
    print(f"max_abs_difference_from_the_unchanged_run={diff:.3e}")
    print(f"constant_update_difference_from_the_unchanged_run="
          f"{con_diff:.3e}")
    print(f"raised={raised}")
    print(f"selected_run_is_bit_identical_to_the_unchanged_run={same_bits}")
    print(f"nothing_was_raised={raised == 'none'}")
    print(f"constant_update_did_change_the_answer={con_diff > 1e-3}")
    if same_bits and raised == "none" and con_diff > 1e-3:
        print("VERDICT=python_float_is_baked_into_the_compiled_form")
        return 0
    print("VERDICT=the_coefficient_update_took_effect")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
