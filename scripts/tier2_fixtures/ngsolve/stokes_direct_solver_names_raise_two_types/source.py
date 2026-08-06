"""Tier-2: unavailable direct solvers do not all raise the same exception type,
so `except RuntimeError: continue` skips some and lets others escape.

Claim: ngsolve stokes#4 -- "Do NOT hardcode a direct solver name -- availability
is build-dependent -- but the naive fallback loop 'except RuntimeError:
continue' is BROKEN, because the different unavailable solvers raise DIFFERENT
exception types and NgException is NOT a subclass of RuntimeError."

Wrong variant: a fallback loop that catches RuntimeError only.

Setup: the Stokes block matrix on unit_square, Taylor-Hood order 2/1, pressure
pinned, with .Inverse() asked for five solver names in turn -- two that this
build has, two optional ones it does not, and one that does not exist at all.

What this fixture pins, all re-measured on this run:
  * NgException is genuinely not a subclass of RuntimeError -- checked on the
    class, so the claim's premise is verified rather than assumed;
  * at least one unavailable solver raises a plain RuntimeError and at least one
    raises NgException, on the SAME call with only the name changed, which is
    what makes a single-type except clause wrong;
  * a nonexistent name raises NgException too, and its message lists the names
    the build accepts -- the thing a fallback loop should have consulted;
  * the naive loop is run for real over the five names and is shown to escape
    with an uncaught exception, while a loop catching Exception finds a working
    solver;
  * the solver it finds actually solves: its result matches the reference to
    roundoff.
"""
from __future__ import annotations

import sys

import numpy
from netgen.geom2d import unit_square
from ngsolve import (
    BilinearForm,
    BitArray,
    CoefficientFunction,
    GridFunction,
    Grad,
    H1,
    InnerProduct,
    LinearForm,
    Mesh,
    VectorH1,
    div,
    dx,
)
from netgen.libngpy._meshing import NgException

NAMES = ["sparsecholesky", "umfpack", "pardiso", "mumps", "nonexistent_solver"]


def main() -> int:
    print(f"ngexception_is_runtimeerror_subclass="
          f"{issubclass(NgException, RuntimeError)}")

    mesh = Mesh(unit_square.GenerateMesh(maxh=0.25))
    V = VectorH1(mesh, order=2, dirichlet=".*")
    Q = H1(mesh, order=1)
    X = V * Q
    (u, p), (v, q) = X.TnT()
    a = BilinearForm(X)
    a += (InnerProduct(Grad(u), Grad(v)) - div(u) * q - div(v) * p) * dx
    a.Assemble()
    f = LinearForm(X)
    f += CoefficientFunction((0, 1)) * v * dx
    f.Assemble()
    fr = BitArray(X.FreeDofs())
    fr[X.Range(1).start] = False

    kinds = {}
    for name in NAMES:
        try:
            a.mat.Inverse(fr, inverse=name)
            kinds[name] = "ok"
        except Exception as exc:                               # noqa: BLE001
            kinds[name] = type(exc).__name__
        print(f"solver={name} outcome={kinds[name]} "
              f"is_runtimeerror="
              f"{kinds[name] == 'RuntimeError'}")

    failed = {k: v for k, v in kinds.items() if v != "ok"}
    types = set(failed.values())
    print(f"failing_solver_exception_types={sorted(types)}")
    print(f"more_than_one_exception_type={len(types) > 1}")
    print(f"some_raise_plain_runtimeerror={'RuntimeError' in types}")
    print(f"some_raise_ngexception={'NgException' in types}")

    msg = ""
    try:
        a.mat.Inverse(fr, inverse="nonexistent_solver")
    except Exception as exc:                                   # noqa: BLE001
        msg = str(exc)
    print(f"unknown_name_message_names_the_input="
          f"{'nonexistent_solver' in msg}")
    print(f"unknown_name_message_lists_allowed="
          f"{'allowed is' in msg and 'sparsecholesky' in msg}")

    # The naive loop, run for real, over a preference order that puts the
    # optional solvers first -- which is the whole point of a fallback loop.
    PREFERENCE = ["pardiso", "mumps", "umfpack", "sparsecholesky"]
    naive_escaped = None
    try:
        for name in PREFERENCE:
            try:
                a.mat.Inverse(fr, inverse=name)
                break
            except RuntimeError:
                continue
        naive_escaped = False
    except Exception as exc:                                   # noqa: BLE001
        naive_escaped = type(exc).__name__
    print(f"naive_runtimeerror_loop_escaped={naive_escaped}")
    print(f"naive_loop_is_broken={naive_escaped not in (False, None)}")

    # The loop that works.
    chosen, inv = None, None
    for name in PREFERENCE:
        try:
            inv = a.mat.Inverse(fr, inverse=name)
            chosen = name
            break
        except Exception:                                      # noqa: BLE001
            continue
    print(f"broad_except_loop_chose={chosen}")
    gf = GridFunction(X)
    gf.vec.data = inv * f.vec
    ref = f.vec.CreateVector()
    ref.data = a.mat.Inverse(fr, inverse="umfpack") * f.vec
    rel = float(numpy.abs(gf.vec.FV().NumPy() - ref.FV().NumPy()).max()
                / max(1e-30, numpy.abs(ref.FV().NumPy()).max()))
    print(f"chosen_solver_relative_error={rel:.3e}")
    print(f"chosen_solver_actually_solves={rel < 1e-10}")

    ok = (
        not issubclass(NgException, RuntimeError)
        and len(types) > 1
        and "RuntimeError" in types and "NgException" in types
        and "allowed is" in msg
        and naive_escaped not in (False, None)
        and chosen is not None and rel < 1e-10
    )
    if ok:
        return 0
    print("FAIL: solver-name exception invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
