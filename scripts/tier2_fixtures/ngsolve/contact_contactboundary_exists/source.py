"""Tier-2: NGSolve does ship ContactBoundary, with two constructor overloads,
and AddIntegrator wants a bare CoefficientFunction.

Claim: ngsolve contact#5 -- "NGSolve DOES ship a built-in contact helper:
ngsolve.comp.ContactBoundary.  The prior catalog claim ('no built-in contact
formulation') is FALSE on 6.2.2604 -- [n for n in dir(ngsolve.comp) if 'ontact'
in n] == ['ContactBoundary'].  Real API: two overloads,
ContactBoundary(master, minion, draw_pairs=False, volume=False,
element_boundary=False) -- the older ContactBoundary(fes, master, minion, ...)
form still constructs but prints 'WARNING: ContactBoundary constructor with
FESpace is deprecated, fes will be set correctly in Update!'.  The object
exposes .gap and .normal as CoefficientFunctions, plus .AddIntegrator(form:
CoefficientFunction, deformed=False), .AddEnergy(...) and .Update(...).  Gotcha:
AddIntegrator takes a bare CoefficientFunction, NOT an integrand-times-measure
-- passing '... * ds' raises TypeError('AddIntegrator(): incompatible function
arguments').  Its docstring warns 'The created object must be kept alive in
python as long as operations of it are used!'"

Wrong variant: AddIntegrator with an integrand multiplied by a measure.

What this fixture pins, all re-measured on this run:
  * the dir() probe the claim gives returns exactly ['ContactBoundary'];
  * both constructor overloads exist in the pybind signature, in the order the
    claim describes -- fes-first is overload 1, master/minion is overload 2 --
    and both actually construct;
  * the deprecation warning text is emitted by the fes-first form, captured
    from the process's own stderr rather than assumed;
  * .gap and .normal are CoefficientFunctions and .AddIntegrator, .AddEnergy,
    .Update are present;
  * AddIntegrator(cf) is accepted and AddIntegrator(cf * ds) raises TypeError
    with the literal message;
  * the keep-alive warning is in the class docstring.
"""
from __future__ import annotations

import os
import tempfile

import sys

import ngsolve.comp
from netgen.geom2d import SplineGeometry
from ngsolve import CoefficientFunction, Mesh, VectorH1, ds


TYPEERROR_TEXT = "AddIntegrator(): incompatible function arguments"
DEPRECATION = ("WARNING: ContactBoundary constructor with FESpace is "
               "deprecated, fes will be set correctly in Update!")
KEEPALIVE = ("The created object must be kept alive in python as long as")


def _capture_fds(fn):
    """Run fn with OS-level stdout+stderr redirected to a temp file."""
    out_fd, err_fd = sys.stdout.fileno(), sys.stderr.fileno()
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(out_fd), os.dup(err_fd)
    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        os.dup2(tmp.fileno(), out_fd)
        os.dup2(tmp.fileno(), err_fd)
        result = fn()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os.dup2(saved_out, out_fd)
        os.dup2(saved_err, err_fd)
        os.close(saved_out)
        os.close(saved_err)
    tmp.seek(0)
    text = tmp.read().decode("utf-8", "replace")
    tmp.close()
    return result, text


def main() -> int:
    matches = [n for n in dir(ngsolve.comp) if "ontact" in n]
    print(f"contact_names_in_ngsolve_comp={matches}")
    print(f"contactboundary_exists={matches == ['ContactBoundary']}")

    CB = ngsolve.comp.ContactBoundary
    init_doc = (CB.__init__.__doc__ or "").replace("\n", " ")
    cls_doc = (CB.__doc__ or "") + init_doc
    print(f"init_doc_is_overloaded={'Overloaded function' in init_doc}")
    i_fes = init_doc.find("fes: ngsolve.comp.FESpace")
    i_master = init_doc.find("master: ngsolve.comp.Region, minion")
    print(f"fes_first_overload_documented={i_fes > 0}")
    print(f"master_minion_overload_documented={i_master > 0}")
    print(f"element_boundary_kwarg_documented="
          f"{'element_boundary: bool' in init_doc}")
    print(f"keepalive_warning_in_docstring={KEEPALIVE in cls_doc}")

    g = SplineGeometry()
    g.AddRectangle((0, 0), (1, 1), bcs=["bot", "right", "top", "left"])
    mesh = Mesh(g.GenerateMesh(maxh=0.3))
    fes = VectorH1(mesh, order=1, dirichlet="bot")
    master = mesh.Boundaries("top")
    minion = mesh.Boundaries("bot")

    # The warning is printed from C++, so it goes out through file descriptors
    # 1 and 2 directly -- contextlib.redirect_stderr only rebinds sys.stderr and
    # would silently miss it.  Capture at the descriptor level instead.
    cb_new, new_out = _capture_fds(
        lambda: ngsolve.comp.ContactBoundary(master, minion))
    print(f"master_minion_form_constructs={cb_new is not None}")
    print(f"master_minion_form_is_quiet={DEPRECATION not in new_out}")

    cb_old, old_out = _capture_fds(
        lambda: ngsolve.comp.ContactBoundary(fes, master, minion))
    print(f"fes_first_form_constructs={cb_old is not None}")
    print(f"fes_first_form_warns_deprecated={DEPRECATION in old_out}")

    print(f"gap_is_coefficientfunction="
          f"{isinstance(cb_new.gap, CoefficientFunction)}")
    print(f"normal_is_coefficientfunction="
          f"{isinstance(cb_new.normal, CoefficientFunction)}")
    members = sorted(n for n in dir(CB) if not n.startswith("_"))
    print(f"public_members={members}")
    print(f"has_addintegrator_addenergy_update="
          f"{all(m in members for m in ('AddIntegrator', 'AddEnergy', 'Update'))}")

    u, v = fes.TnT()
    bare_ok = True
    try:
        cb_new.AddIntegrator(cb_new.gap * v[1])
    except Exception as exc:                                   # noqa: BLE001
        bare_ok = False
        print(f"bare_cf_unexpectedly_rejected={exc}")
    print(f"addintegrator_accepts_bare_cf={bare_ok}")

    msg = ""
    try:
        cb_new.AddIntegrator(cb_new.gap * v[1] * ds)
    except TypeError as exc:
        msg = str(exc)
    print(f"addintegrator_with_measure_raises={bool(msg)}")
    print(f"addintegrator_message_literal={TYPEERROR_TEXT in msg}")

    ok = (
        matches == ["ContactBoundary"]
        and "Overloaded function" in init_doc
        and 0 < i_fes < i_master
        and "element_boundary: bool" in init_doc
        and KEEPALIVE in cls_doc
        and DEPRECATION not in new_out
        and DEPRECATION in old_out
        and isinstance(cb_new.gap, CoefficientFunction)
        and isinstance(cb_new.normal, CoefficientFunction)
        and all(m in members for m in ("AddIntegrator", "AddEnergy", "Update"))
        and bare_ok
        and TYPEERROR_TEXT in msg
    )
    if ok:
        return 0
    print("FAIL: ContactBoundary API invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
