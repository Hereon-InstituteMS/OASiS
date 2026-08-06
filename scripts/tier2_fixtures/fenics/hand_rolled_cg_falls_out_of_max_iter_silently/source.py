"""Tier-2 for fenics matrix_free_poisson#8: a CG loop that simply falls out of its
`for` after max_iter, without raising or returning a converged flag, is the classic
silent-failure shape -- the caller gets a Function that looks like a solution.
The hand-rolled loop has no PETSc KSP behind it, so there is no
KSPConvergedReason to consult and nothing raises.

Wrong variant: `def cg(...): for k in range(max_iter): ...` with no return value
and no assertion at the call site. Right variant: return (iterations, converged)
and assert the flag, as the minimal_working_example does.

The fixture writes ONE child script and runs it as a subprocess twice, so the
caller-visible symptom -- the process exit code -- is the measurement. Both runs
use the same diverging operator action (the Dirichlet rows are not zeroed).

Observed on dolfinx 0.10.0, 16x16 unit square, P2, 200 iteration cap: the silent
variant leaves the loop normally, its cg() returns None, that return value has no
getConvergedReason to query, it prints a field whose maximum is 1.987e+27, and the
process exits with return code 0. The variant that returns the flag and asserts it
exits non-zero with
"AssertionError: matrix-free CG hit max_iter without reaching rtol"
on exactly the same numbers.

Mutation control: T2_MUTATE=1 makes the run under test the asserting one, so its
return code is no longer 0 and the silent-exit token goes False.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

CHILD = r'''
import os
from mpi4py import MPI
import numpy as np
import dolfinx, ufl
from dolfinx import fem, la, mesh

ASSERT = os.environ["T2_ASSERT"] == "1"
dtype = dolfinx.default_scalar_type
comm = MPI.COMM_WORLD
N, DEG, MAXIT, RTOL = 16, 2, 200, 1e-8

msh = mesh.create_unit_square(comm, N, N)
V = fem.functionspace(msh, ("Lagrange", DEG))
tdim = msh.topology.dim
msh.topology.create_connectivity(tdim - 1, tdim)
facets = mesh.exterior_facet_indices(msh.topology)
bdofs = fem.locate_dofs_topological(V, tdim - 1, facets)
uD = fem.Function(V, dtype=dtype)
uD.interpolate(lambda x: 0.5 * x[0])
bc = fem.dirichletbc(uD, bdofs)
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = 10.0 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L_fem = fem.form(ufl.inner(f, v) * ufl.dx, dtype=dtype)
ui = fem.Function(V, dtype=dtype)
M_fem = fem.form(ufl.action(a, ui), dtype=dtype)

def action_A(xv, yv):
    # NOTE: deliberately missing bc.set(yv.array, alpha=0.0) -- the operator
    # has no fixed point, which is what makes the loop run out of iterations.
    ui.x.array[:] = xv.array
    ui.x.scatter_forward()
    yv.array[:] = 0.0
    fem.assemble_vector(yv.array, M_fem)
    yv.scatter_reverse(la.InsertMode.add)

b = fem.assemble_vector(L_fem)
ui.x.array[:] = 0.0
bc.set(ui.x.array, alpha=-1.0)
fem.assemble_vector(b.array, M_fem)
b.scatter_reverse(la.InsertMode.add)
bc.set(b.array, alpha=0.0)
b.scatter_forward()
nr = b.index_map.size_local

def gdot(v0, v1):
    return comm.allreduce(np.vdot(v0[:nr], v1[:nr]), MPI.SUM)

def cg_silent(xv, bv, max_iter=MAXIT, rtol=RTOL):
    """The silent shape: no return value, no flag, no raise."""
    yv = la.vector(bv.index_map, 1, dtype)
    action_A(xv, yv)
    r = bv.array - yv.array
    p = la.vector(bv.index_map, 1, dtype)
    p.array[:] = r
    rn0 = rn = gdot(r, r)
    for k in range(max_iter):
        action_A(p, yv)
        alpha = rn / gdot(p.array, yv.array)
        xv.array[:] += alpha * p.array
        r -= alpha * yv.array
        rn, rn_old = gdot(r, r), rn
        if rn / rn0 < rtol ** 2:
            break
        p.array[:] = (rn / rn_old) * p.array + r
    xv.scatter_forward()

def cg_checked(xv, bv, max_iter=MAXIT, rtol=RTOL):
    """The shape the minimal_working_example uses: returns the flag."""
    yv = la.vector(bv.index_map, 1, dtype)
    action_A(xv, yv)
    r = bv.array - yv.array
    p = la.vector(bv.index_map, 1, dtype)
    p.array[:] = r
    rn0 = rn = gdot(r, r)
    for k in range(max_iter):
        action_A(p, yv)
        alpha = rn / gdot(p.array, yv.array)
        xv.array[:] += alpha * p.array
        r -= alpha * yv.array
        rn, rn_old = gdot(r, r), rn
        if rn / rn0 < rtol ** 2:
            xv.scatter_forward()
            return k + 1, True
        p.array[:] = (rn / rn_old) * p.array + r
    xv.scatter_forward()
    return max_iter, False

uh = fem.Function(V, dtype=dtype)
if ASSERT:
    its, converged = cg_checked(uh.x, b)
    print(f"child_mode=checked returned={(its, converged)!r}")
else:
    ret = cg_silent(uh.x, b)
    print(f"child_mode=silent cg_return_value={ret!r} "
          f"return_has_getConvergedReason={hasattr(ret, 'getConvergedReason')}")
bc.set(uh.x.array, alpha=1.0)
uh.x.scatter_forward()
print(f"child_solution_min={float(uh.x.array[:nr].min()):.3e} "
      f"child_solution_max={float(uh.x.array[:nr].max()):.3e}")
if ASSERT:
    assert converged, "matrix-free CG hit max_iter without reaching rtol"
print("child_reached_the_end_of_the_script=True")
'''

MUTATE = os.environ.get("T2_MUTATE") == "1"


def run(child: str, assert_flag: bool) -> tuple[int, str]:
    env = os.environ.copy()
    env["T2_ASSERT"] = "1" if assert_flag else "0"
    env.pop("T2_MUTATE", None)
    proc = subprocess.run([sys.executable, child], env=env, timeout=240,
                          capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    tag = "checked" if assert_flag else "silent"
    for line in out.splitlines():
        if line.startswith("child_") or "AssertionError" in line:
            print(f"{tag}> {line}")
    return proc.returncode, out


def parse_max(out: str) -> float:
    for line in out.splitlines():
        if "child_solution_max=" in line:
            return float(line.split("child_solution_max=")[1].split()[0])
    return float("nan")


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="t2mf8_")
    child = os.path.join(tmp, "child_matrix_free.py")
    with open(child, "w") as fh:
        fh.write(CHILD)

    rc_test, out_test = run(child, assert_flag=MUTATE)
    rc_chk, out_chk = run(child, assert_flag=True)
    print(f"run_under_test_returncode={rc_test}")
    print(f"checked_variant_returncode={rc_chk}")
    if MUTATE:
        print("mutation=run_under_test_returns_the_flag_and_asserts_it")

    umax = parse_max(out_test)
    silent_rc_zero = rc_test == 0
    huge = umax > 1e10
    no_reason = "return_has_getConvergedReason=False" in out_test
    checked_fails = rc_chk != 0 and "AssertionError" in out_chk
    print(f"solution_max_of_the_run_under_test={umax:.3e}")
    print(f"silent_loop_exited_with_returncode_zero={silent_rc_zero}")
    print(f"silent_run_still_printed_a_field_above_1e10={huge}")
    print(f"hand_rolled_loop_return_value_has_no_convergedreason={no_reason}")
    print(f"returning_and_asserting_the_flag_makes_the_process_fail="
          f"{checked_fails}")
    if silent_rc_zero and huge and no_reason and checked_fails:
        print("VERDICT=silent_cg_loop_exits_zero_with_a_1e10_plus_solution")
        return 0
    print("VERDICT=silent_cg_loop_did_not_exit_zero")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
