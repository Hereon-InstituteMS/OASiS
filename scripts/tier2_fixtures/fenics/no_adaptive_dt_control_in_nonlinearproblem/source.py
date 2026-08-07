"""Tier-2 for fenics cahn_hilliard#3: dolfinx.fem.petsc.NonlinearProblem does no
adaptive time stepping, so no dt-control message can ever appear; if a step
fails you must catch it and shrink dt yourself in the Python loop.

Wrong variant: a fixed-dt loop that waits for the stack to say something like
"step rejected, reducing dt". Observed on dolfinx 0.10.0 / PETSc 3.24.5:

  * a case-sensitive byte search of the installed libpetsc shared object finds
    0 occurrences of "step rejected", 0 of "Step rejected" and 0 of
    "reducing dt";
  * the only related symbol is DIVERGED_STEP_REJECTED, which belongs to
    PETSc.TS.ConvergedReason -- the time-stepper object NonlinearProblem never
    creates. problem.solver is a petsc4py SNES and PETSc.SNES.ConvergedReason
    has no DIVERGED_STEP_REJECTED member at all;
  * the failing step reports DIVERGED_MAX_IT (-5) or DIVERGED_LINE_SEARCH (-6)
    and prints nothing on stdout, and dt is exactly the value the caller set;
  * shrinking dt in the Python loop after a failure is what actually rescues
    the run.

Mutation control: T2_MUTATE=1 runs the correct pattern -- catch the negative
reason, restore the previous state, quarter dt and retry -- so the loop no
longer dies at the failing step.
"""
from __future__ import annotations

import contextlib
import glob
import os
import sys
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import basix.ufl  # noqa: E402
import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402
from petsc4py import PETSc  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

N, LMBDA, MOB, THETA, DT0, STEPS = 24, 1.0e-2, 1.0, 0.5, 1.0e-4, 4


@contextlib.contextmanager
def capture_fd1():
    tmp = tempfile.TemporaryFile(mode="w+")
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(tmp.fileno(), 1)
    try:
        yield tmp
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
        tmp.seek(0)


def petsc_library_paths() -> list[str]:
    import petsc4py
    roots = []
    try:
        cfg = petsc4py.get_config()
        if cfg.get("PETSC_DIR"):
            roots.append(cfg["PETSC_DIR"])
    except Exception:
        pass
    roots += [sys.prefix, os.path.dirname(os.path.dirname(PETSc.__file__))]
    hits: list[str] = []
    for r in roots:
        hits += glob.glob(os.path.join(r, "lib", "libpetsc.so*"))
    return sorted({h for h in hits if os.path.isfile(h)})


def build():
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    ME = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P1, P1]))
    u, u0 = dolfinx.fem.Function(ME), dolfinx.fem.Function(ME)
    rng = np.random.default_rng(7)
    u.sub(0).interpolate(lambda x: 0.63 + 0.02 * (0.5 - rng.random(x.shape[1])))
    u.sub(1).interpolate(lambda x: np.zeros(x.shape[1]))
    u.x.scatter_forward()
    u0.x.array[:] = u.x.array
    q, v = ufl.TestFunctions(ME)
    c, mu = ufl.split(u)
    c0, mu0 = ufl.split(u0)
    cv = ufl.variable(c)
    dfdc = ufl.diff(100.0 * cv**2 * (1 - cv) ** 2, cv)
    mu_mid = (1.0 - THETA) * mu0 + THETA * mu
    dt = dolfinx.fem.Constant(msh, DT0)
    F = ((c - c0) * q * ufl.dx
         + dt * MOB * ufl.dot(ufl.grad(mu_mid), ufl.grad(q)) * ufl.dx
         + mu * v * ufl.dx - dfdc * v * ufl.dx
         - LMBDA * ufl.dot(ufl.grad(c), ufl.grad(v)) * ufl.dx)
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, u, petsc_options_prefix="t2_ch3_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "snes_max_it": 30})
    return u, u0, prob, dt


def main() -> int:
    # 1) the message does not exist in the installed PETSc
    libs = petsc_library_paths()
    if not libs:
        print("petsc_library_not_found=True")
        print("VERDICT=could_not_inspect_petsc")
        return 1
    lib = libs[-1]
    blob = open(lib, "rb").read()
    counts = {p: blob.count(p.encode())
              for p in ("step rejected", "Step rejected", "reducing dt",
                        "DIVERGED_STEP_REJECTED")}
    print(f"petsc_library={os.path.basename(lib)} bytes={len(blob)}")
    for p, n in counts.items():
        print(f"  occurrences of {p!r}: {n}")
    no_msg = (counts["step rejected"] == 0
              and counts["Step rejected"] == 0
              and counts["reducing dt"] == 0)
    print(f"petsc_library_has_no_step_rejected_message={no_msg}")
    print("petsc_library_does_contain_DIVERGED_STEP_REJECTED="
          f"{counts['DIVERGED_STEP_REJECTED'] > 0}")

    snes_has = hasattr(PETSc.SNES.ConvergedReason, "DIVERGED_STEP_REJECTED")
    ts_has = hasattr(PETSc.TS.ConvergedReason, "DIVERGED_STEP_REJECTED")
    print(f"snes_convergedreason_has_diverged_step_rejected={snes_has}")
    print(f"ts_convergedreason_has_diverged_step_rejected={ts_has}")

    # 2) the solver object is a SNES, not a TS
    u, u0, prob, dt = build()
    print(f"solver_type={type(prob.solver).__name__}")
    is_snes = (isinstance(prob.solver, PETSc.SNES)
               and not isinstance(prob.solver, PETSc.TS))
    print(f"problem_solver_is_a_snes_not_a_ts={is_snes}")

    # 3) march with a dt that fails; nothing reduces it for us
    dt_history, reasons, printed = [], [], ""
    died_at = -1
    for k in range(STEPS):
        u0.x.array[:] = u.x.array
        with capture_fd1() as cap:
            prob.solve()
        printed += cap.read()
        u.x.scatter_forward()
        reason = prob.solver.getConvergedReason()
        reasons.append(reason)
        dt_history.append(float(dt.value))
        if reason > 0:
            continue
        if not MUTATE:
            died_at = k
            break
        # the correct pattern: undo the step, quarter dt, retry
        for _ in range(6):
            u.x.array[:] = u0.x.array
            dt.value = float(dt.value) / 4.0
            prob.solve()
            u.x.scatter_forward()
            reason = prob.solver.getConvergedReason()
            reasons.append(reason)
            dt_history.append(float(dt.value))
            if reason > 0:
                break
        if reason <= 0:
            died_at = k
            break

    print(f"reasons={reasons}")
    print(f"dt_history={['%.3e' % d for d in dt_history]}")
    print(f"captured_solver_stdout_chars={len(printed.strip())}")
    quiet = ("reject" not in printed.lower()
             and "reducing dt" not in printed.lower())
    print(f"failing_solve_printed_no_rejection_message={quiet}")
    bad = [r for r in reasons if r <= 0]
    print(f"failing_reason_is_max_it_or_line_search="
          f"{bool(bad) and bad[0] in (-5, -6)}")
    fixed = len(set(f"{d:.12e}" for d in dt_history)) == 1
    print(f"dt_stayed_exactly_where_the_caller_put_it={fixed}")
    print(f"fixed_dt_loop_died_at_the_failing_step={died_at >= 0}")

    if (no_msg and not snes_has and ts_has and is_snes and bad
            and bad[0] in (-5, -6) and quiet and fixed and died_at >= 0):
        print("VERDICT=nothing_in_the_stack_rejects_a_step_or_shrinks_dt")
        return 0
    print("VERDICT=the_stack_handled_dt_itself")
    return 1



if __name__ == "__main__":
    raise SystemExit(main())
