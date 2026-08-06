"""Tier-2: GridFunction.name is read-only — there is no name setter.

Claim: ngsolve thermal_structural#5 — GridFunction.name is a read-only property
in current NGSolve builds; gfu.name = 'displacement' raises AttributeError
"property of 'GridFunction' object has no setter". Pass the name string to the
consumer instead, e.g. VTKOutput(names=['u']).

Wrong variant: assign to the property directly on both GridFunctions the
thermal_structural template builds.

Observed on NGSolve 6.2.2604 (2026-08-03):
  * AttributeError, str() exactly "property of 'GridFunction' object has no
    setter";
  * reading the property is fine and returns the constructor-supplied name;
  * VTKOutput(..., names=[...]) accepts the labels and Do() writes the file, so
    the documented workaround is complete.
"""
from __future__ import annotations

import os
import sys

import ngsolve as ngs
from netgen.geom2d import unit_square

# WRONG: naming a GridFunction by assigning to the read-only property
NAME_VIA_ATTRIBUTE_ASSIGNMENT = True


def try_rename(gf, new_name: str) -> str:
    """Return the exception text, or '' if the assignment went through."""
    try:
        gf.name = new_name
    except Exception as exc:                        # noqa: BLE001
        return str(exc)
    return ""


def main() -> int:
    ok = True
    print(f"ngsolve_version={ngs.__version__}")
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.4))
    gfT = ngs.GridFunction(ngs.H1(mesh, order=2), name="gfT")
    gfu = ngs.GridFunction(ngs.VectorH1(mesh, order=2), name="gfu")

    # reading the property works
    print(f"temperature_name={gfT.name!r} displacement_name={gfu.name!r}")
    print(f"name_is_readable={gfT.name == 'gfT' and gfu.name == 'gfu'}")
    print(f"name_is_a_property="
          f"{isinstance(type(gfu).__dict__.get('name'), property)}")
    print(f"name_property_has_no_setter="
          f"{type(gfu).__dict__.get('name').fset is None}")
    if gfT.name != "gfT" or gfu.name != "gfu":
        print("FAIL: GridFunction.name did not return the constructor name",
              file=sys.stderr)
        ok = False

    # --- WRONG variant: assign to the property ----------------------------
    if NAME_VIA_ATTRIBUTE_ASSIGNMENT:
        msg_u = try_rename(gfu, "displacement")
        msg_t = try_rename(gfT, "temperature")
    else:
        msg_u = msg_t = ""
    print(f"displacement_rename_raises={bool(msg_u)} msg={msg_u!r}")
    print(f"temperature_rename_raises={bool(msg_t)}")
    if "has no setter" not in msg_u or "has no setter" not in msg_t:
        print(f"FAIL: assigning GridFunction.name did not raise the documented "
              f"AttributeError; got {msg_u!r} / {msg_t!r}", file=sys.stderr)
        ok = False
    print(f"name_unchanged_after_failed_assignment={gfu.name == 'gfu'}")
    if gfu.name != "gfu":
        print(f"FAIL: the name changed to {gfu.name!r} despite the exception",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant: give the label to the consumer --------------------
    vtk = ngs.VTKOutput(mesh, coefs=[gfT, gfu],
                        names=["temperature", "displacement"],
                        filename="result_name_setter", subdivision=1)
    vtk.Do()
    written = [f for f in os.listdir(".") if f.startswith("result_name_setter")]
    print(f"vtkoutput_names_accepted=True files_written={len(written) > 0}")
    if not written:
        print("FAIL: VTKOutput(names=[...]) wrote no file", file=sys.stderr)
        ok = False
    for f in written:
        os.remove(f)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
