"""Tier-2: an indefinite Helmholtz system produces NO diagnostic.

  helmholtz#0   dune-fem gives you nothing. At k=10 on a 16x16 P1 grid
                CG raised no exception and reported converged True with
                iterations 0 and linear_iterations 1. The only detector
                is the ANSWER — compare against a manufactured solution
                or a direct solve. The string 'matrix not positive
                definite' occurs nowhere in this install and is never
                emitted.
  helmholtz#3   small k is positive definite and any Krylov method
                works; large k is indefinite and needs care.
  maxwell#1     the same statement, restated for the Maxwell proxy.

k^2 is a dune.ufl.Constant, so sweeping k costs no rebuild; the only
second module is the direct-solver scheme, because the solver choice is
part of the generated code.

FALSIFICATION recorded here: helmholtz#3 says "vanilla ILU stalls".
There is no ILU to stall — 'ilu' is not in the preconditioner
enumeration at all and is rejected at scheme construction. The fixture
asserts that rejection.

Verified by execution against dune-fem 2.12.0.2.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from dune.grid import structuredGrid                            # noqa: E402
from dune.fem.space import lagrange                             # noqa: E402
from dune.fem.scheme import galerkin                            # noqa: E402
from dune.ufl import Constant, DirichletBC                      # noqa: E402
import dune.fem as dfem                                         # noqa: E402
from ufl import (TrialFunction, TestFunction,                    # noqa: E402
                 SpatialCoordinate, dot, grad, dx, sin, pi)


def main() -> int:
    fail: list[str] = []
    gridView = structuredGrid([0, 0], [1, 1], [16, 16])
    space = lagrange(gridView, order=1)
    u, v = TrialFunction(space), TestFunction(space)
    x = SpatialCoordinate(space)

    k2 = Constant(4.0, name="k2")
    # Manufactured: u = sin(pi x) sin(pi y) solves
    # -Laplace(u) - k^2 u = (2 pi^2 - k^2) u with u = 0 on the boundary.
    exact = sin(pi * x[0]) * sin(pi * x[1])
    src = Constant(1.0, name="src")
    a = (dot(grad(u), grad(v)) - k2 * u * v) * dx
    L = src * exact * v * dx
    dbc = DirichletBC(space, 0)

    scheme_cg = galerkin([a == L, dbc], solver="cg")
    scheme_direct = galerkin([a == L, dbc],
                             solver=("suitesparse", "umfpack"))

    def run(scheme, k_value, name):
        k2.value = k_value ** 2
        src.value = 2 * np.pi ** 2 - k_value ** 2
        uh = space.interpolate(0, name=name)
        info = scheme.solve(target=uh)
        err = float(np.sqrt(dfem.integrate(
            (uh - exact) ** 2, gridView=gridView, order=6)))
        return info, err, float(np.abs(np.array(uh.as_numpy)).max())

    # ── helmholtz#3: small k, definite, CG is fine ─────────────────
    info_s, err_s, max_s = run(scheme_cg, 2.0, "small_cg")
    print(f"k2_small_converged={bool(info_s['converged'])}")
    print(f"k2_small_linear_iterations="
          f"{int(info_s['linear_iterations'])}")
    print(f"k2_small_l2_error={err_s:.6e}")
    print(f"k2_small_answer_is_right={err_s < 5e-3}")
    if err_s >= 5e-3:
        fail.append(f"CG at k=2 (k^2=4 < pi^2) gave L2 error "
                    f"{err_s:.6e}; the definite case is supposed to be "
                    f"unproblematic and is the control here")

    # ── helmholtz#0 / maxwell#1: large k, indefinite, no diagnostic ─
    caught = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        info_l, err_l, max_l = run(scheme_cg, 10.0, "large_cg")
        caught = [" ".join(str(m.message).split()) for m in w]
    print(f"k10_raised_nothing=True")
    print(f"k10_warnings={caught}")
    print(f"k10_converged={bool(info_l['converged'])}")
    print(f"k10_iterations={int(info_l['iterations'])}")
    print(f"k10_linear_iterations={int(info_l['linear_iterations'])}")
    print(f"k10_reports_success={bool(info_l['converged'])}")
    if not info_l["converged"]:
        fail.append("CG at k=10 reported converged=False; the claim is "
                    "that it reports SUCCESS on an indefinite system, "
                    "which is what makes the absence of a diagnostic "
                    "dangerous")
    if caught:
        fail.append(f"the indefinite solve emitted warnings {caught}; "
                    f"the claim is that dune-fem says nothing")

    # The solver's own report cannot tell the three cases apart, which
    # is the actual content of "you get NO diagnostic".
    info_d, err_d, max_d = run(scheme_direct, 10.0, "large_direct")
    print(f"k10_cg_l2_error={err_l:.6e}")
    print(f"k10_direct_l2_error={err_d:.6e}")
    print(f"k10_direct_converged={bool(info_d['converged'])}")
    print(f"k10_direct_linear_iterations="
          f"{int(info_d['linear_iterations'])}")
    reports = {
        "definite_cg": (bool(info_s["converged"]),
                        int(info_s["iterations"]),
                        int(info_s["linear_iterations"])),
        "indefinite_cg": (bool(info_l["converged"]),
                          int(info_l["iterations"]),
                          int(info_l["linear_iterations"])),
        "indefinite_direct": (bool(info_d["converged"]),
                              int(info_d["iterations"]),
                              int(info_d["linear_iterations"])),
    }
    print(f"solver_reports={reports}")
    identical = len(set(reports.values())) == 1
    print(f"solver_report_is_identical_for_all_three={identical}")
    if not identical:
        fail.append(f"the info dict distinguishes the definite case, "
                    f"the indefinite case and the direct solve "
                    f"({reports}); the claim is that dune-fem gives NO "
                    f"diagnostic, and a distinguishable report would be "
                    f"one")

    # …so the only thing left is the answer. Here it happens to be
    # FINE — CG landed on the same discretisation error as the direct
    # solve — which is exactly why the absence of a diagnostic matters:
    # nothing in the run tells you whether you were lucky.
    agree = abs(err_l - err_d) <= 1e-6 * max(err_d, 1e-30)
    print(f"k10_cg_matches_direct={agree}")
    print(f"errors_are_discretisation_sized="
          f"{err_l < 1e-2 and err_d < 1e-2}")
    print("only_the_answer_can_detect_it=True")
    if not (err_l < 1e-2 and err_d < 1e-2):
        fail.append(f"neither route reproduced the manufactured "
                    f"solution (cg {err_l:.3e}, direct {err_d:.3e}); "
                    f"the manufactured-solution detector the claim "
                    f"recommends has to work for the claim to stand")

    # ── the retracted string is not in this install ────────────────
    from pathlib import Path
    import dune
    # dune is a NAMESPACE package: dune.__file__ is None, so the search
    # has to walk __path__.
    roots = [Path(p) for p in list(dune.__path__)]
    print(f"dune_package_roots={len(roots)}")
    hits = []
    for p in [q for r in roots for q in r.rglob("*.py")]:
        try:
            if "matrix not positive definite" in p.read_text(
                    errors="ignore"):
                hits.append(str(p))
        except OSError:
            continue
    print(f"matrix_not_positive_definite_occurrences={len(hits)}")
    print(f"retracted_string_is_absent={hits == []}")
    if hits:
        fail.append(f"'matrix not positive definite' occurs in {hits}; "
                    f"the retraction of that Signal would be wrong")

    # ── helmholtz#3: there is no ILU to stall ──────────────────────
    for name in ("ilu", "amg"):
        try:
            galerkin([a == L, dbc], solver="gmres",
                     parameters={"linear.preconditioning.method": name})
            print(f"preconditioner_{name}_accepted=True")
            fail.append(f"'{name}' was accepted as a preconditioner; "
                        f"the enumeration is supposed to reject it")
        except RuntimeError as exc:
            msg = " ".join(str(exc).split())
            print(f"preconditioner_{name}_rejected="
                  f"{type(exc).__name__}")
            if "none, sor, ssor" not in msg:
                fail.append(f"the rejection no longer enumerates the "
                            f"valid names: {msg[:200]}")
    print("ilu_does_not_exist_so_it_cannot_stall=True")

    if not fail:
        print("dune_indefinite_helmholtz_silence_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
