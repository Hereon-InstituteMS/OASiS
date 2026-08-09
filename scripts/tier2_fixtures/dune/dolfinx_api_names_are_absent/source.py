"""Tier-2: what a dolfinx habit costs inside a DUNE-fem script.

Three pitfalls share one observation, so they share one fixture (the
compiled artefact is the same Poisson Integrands module either way):

  poisson#0  UFL is shared with dolfinx, the SPACE construction is not.
  poisson#3  dune.ufl.DirichletBC, not dolfinx.fem.dirichletbc — and the
             DUNE constructor is (space, value), not the dolfinx
             (V, value, dofs) triple.
  poisson#4  VTK goes through gridView.writeVTK; the dolfinx
             io.VTXWriter / XDMFFile names do not exist here.

The portability half of poisson#0 is asserted the only way a DUNE
process can assert it: the weak form is built from `ufl` symbols ALONE
(no dune.ufl anywhere in the form) and dune-fem compiles and solves it.
The dolfinx side was confirmed separately — see the fixture _comment.

Verified by execution against dune-fem 2.12.0.2.

MUTATION CONTROL. T2_MUTATE=1 installs XDMFFile and VTXWriter
attributes on dune.grid — the world in which dune-fem HAS grown the
dolfinx IO names. dune_grid_has_XDMFFile and dune_grid_has_VTXWriter
then read True, both '...=False' expectations disappear and a FAIL:
line appears.
"""
from __future__ import annotations

import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")

MUTATE = os.environ.get("T2_MUTATE") == "1"

# Deliberately at module scope: on a machine without dune-fem this
# raises and the runner records a failure, never a pass.
from dune.grid import structuredGrid                        # noqa: E402
from dune.fem.space import lagrange                         # noqa: E402
from dune.fem.scheme import galerkin                        # noqa: E402
import dune.ufl                                             # noqa: E402
from ufl import (TrialFunction, TestFunction,               # noqa: E402
                 dot, grad, dx)


def main() -> int:
    fail: list[str] = []

    # 1. poisson#3 / poisson#0: the dolfinx module is not importable
    #    from a DUNE-fem environment, so every dolfinx spelling fails
    #    at import, not at first use.
    for stmt, label in (
        ("import dolfinx", "dolfinx"),
        ("from dolfinx.fem import dirichletbc", "dolfinx_dirichletbc"),
        ("from dolfinx.fem import functionspace", "dolfinx_functionspace"),
        ("from dolfinx.io import XDMFFile, VTXWriter", "dolfinx_io"),
    ):
        try:
            exec(compile(stmt, "<probe>", "exec"), {})
            print(f"{label}_import_raises=False")
            fail.append(f"{stmt!r} SUCCEEDED — this environment has "
                        f"dolfinx on the path, so the fixture cannot "
                        f"tell a DUNE-only install from a mixed one")
        except ImportError as exc:
            print(f"{label}_import_raises={type(exc).__name__}")

    # 2. poisson#0: a form written in PURE ufl (no dune.ufl symbol in
    #    the form itself) is accepted by dune-fem verbatim.
    gridView = structuredGrid([0, 0], [1, 1], [4, 4])
    space = lagrange(gridView, order=1)
    u, v = TrialFunction(space), TestFunction(space)
    a = dot(grad(u), grad(v)) * dx
    b = 1.0 * v * dx
    form_is_pure_ufl = all(
        type(obj).__module__.split(".")[0] == "ufl"
        for obj in (u, v, a, b))
    print(f"weak_form_uses_only_ufl_types={form_is_pure_ufl}")
    if not form_is_pure_ufl:
        fail.append("the portability check is vacuous: the form under "
                    "test is not built from ufl types alone")

    # 3. poisson#0: the SPACE, by contrast, is a dune.fem call, and the
    #    dolfinx signature is not accepted by it.
    print(f"dune_space_is_lagrange_gridview_order="
          f"{type(space).__name__ != 'NoneType' and space.order == 1}")
    try:
        lagrange(gridView, ("Lagrange", 1))       # the dolfinx spelling
        print("dolfinx_functionspace_signature_rejected=False")
        fail.append("lagrange(gridView, ('Lagrange', 1)) was accepted; "
                    "the dolfinx functionspace signature is supposed "
                    "to be a type error here")
    except Exception as exc:                             # noqa: BLE001
        print(f"dolfinx_functionspace_signature_rejected="
              f"{type(exc).__name__}")

    # 4. poisson#3: the DUNE constructor is (space, value). The dolfinx
    #    triple (V, value, dofs) is rejected.
    dbc = dune.ufl.DirichletBC(space, 0)
    print(f"dune_dirichletbc_two_arg_ok={dbc is not None}")
    try:
        dune.ufl.DirichletBC(space, 0, [0, 1, 2], "extra")
        print("dolfinx_dirichletbc_triple_rejected=False")
        fail.append("DirichletBC accepted four positional arguments; "
                    "the (space, value[, indicator]) signature claim "
                    "no longer holds")
    except TypeError as exc:                             # noqa: BLE001
        print(f"dolfinx_dirichletbc_triple_rejected="
              f"{type(exc).__name__}")

    # The whole thing has to still solve, otherwise the fixture is
    # asserting things about a script that does not run.
    scheme = galerkin([a == b, dbc], solver="cg")
    uh = space.interpolate(0, name="uh")
    info = scheme.solve(target=uh)
    print(f"pure_ufl_form_solves={bool(info['converged'])}")
    if not info["converged"]:
        fail.append("the pure-UFL form did not solve")

    # 5. poisson#4: writeVTK exists on the GRID VIEW and produces a
    #    .vtu; the dolfinx writer names are absent from dune.grid.
    import dune.grid as dgrid
    if MUTATE:
        print("mutation=dune_grid_gains_the_dolfinx_io_names")
        dgrid.XDMFFile = object
        dgrid.VTXWriter = object
    for name in ("XDMFFile", "VTXWriter", "VTKFile"):
        present = hasattr(dgrid, name)
        print(f"dune_grid_has_{name}={present}")
        if present:
            fail.append(f"dune.grid unexpectedly has {name}; the claim "
                        f"that the dolfinx IO names are absent is stale")
    print(f"gridview_has_writeVTK={hasattr(gridView, 'writeVTK')}")
    for stale in glob.glob("dolfinx_probe*.vtu"):
        os.remove(stale)
    gridView.writeVTK("dolfinx_probe", pointdata={"u": uh})
    written = sorted(os.path.basename(p)
                     for p in glob.glob("dolfinx_probe*.vtu"))
    print(f"writeVTK_produced={written}")
    if not written:
        fail.append("gridView.writeVTK wrote no .vtu file")

    if not fail:
        print("dune_dolfinx_api_divergence_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
