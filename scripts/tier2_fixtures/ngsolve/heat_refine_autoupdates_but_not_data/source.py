"""Tier-2: mesh.Refine() resizes the GridFunction but does not carry its data.

Claim: NGSolve heat#2 -- mesh.Refine() auto-updates the dependent FESpace and
its GridFunctions; fes.Update() / gfu.Update() are no-ops for the size change.
The claim adds that the resized vector "does not carry a prolongation of the old
solution".

The auto-update half is confirmed. The prolongation half is NOT: measured on
6.2.2604 the old DOF values are preserved entry-for-entry and the new DOFs are
filled by interpolation, not left at zero. The advice to re-solve after every
Refine is still right, for a different reason -- the prolongated function is the
COARSE function re-expressed on the fine mesh, so the representation error is
bit-for-bit unchanged by the refine and only a re-Set (or re-solve) improves it.
That is what this fixture pins.

len(gfu.vec) is read BEFORE fes.ndof after the refine, so the resize cannot be
attributed to the ndof access itself.

Wrong variant: keep using the GridFunction after Refine as if it still held the
solution. Right variant: Set it again.
"""
from __future__ import annotations

import sys

from netgen.geom2d import unit_square
from ngsolve import GridFunction, H1, Integrate, Mesh, x, y


def main() -> int:
    ok = True
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
    fes = H1(mesh, order=1)
    gfu = GridFunction(fes)
    print(f"space_type={fes.type}")

    # A field the coarse P1 space cannot represent, so the representation
    # error is a real number rather than round-off.
    curved = x * x + y * y
    gfu.Set(curved)
    before_curved_err = float(Integrate((gfu - curved) ** 2, mesh)) ** 0.5
    before_vals = [float(val) for val in gfu.vec]
    before_len = len(gfu.vec)
    before_ndof = fes.ndof
    print(f"before_len_vec={before_len}")
    print(f"before_ndof={before_ndof}")
    print(f"curved_field_not_representable_on_coarse_mesh="
          f"{before_curved_err > 1e-3}")

    # --- the refine, with NO Update call of any kind ---------------------
    mesh.Refine()
    after_len = len(gfu.vec)          # read FIRST, before touching fes.ndof
    after_ndof = fes.ndof
    print(f"after_len_vec_read_first={after_len}")
    print(f"after_ndof={after_ndof}")
    resized = after_len > before_len and after_len == after_ndof
    print(f"resized_without_any_update_call={resized}")
    if not resized:
        print(f"FAIL: after Refine the vector is {after_len} and ndof is "
              f"{after_ndof} (before: {before_len} / {before_ndof})",
              file=sys.stderr)
        ok = False

    # --- What actually happens to the DATA -------------------------------
    # The claim says the resized vector "does not carry a prolongation of the
    # old solution". Measured, it DOES: the pre-refine DOF values are preserved
    # entry-for-entry and the new DOFs are filled by interpolation, not zeros.
    after_vals = [float(val) for val in gfu.vec]
    preserved = all(abs(a - b) < 1e-14
                    for a, b in zip(before_vals, after_vals[:before_len]))
    new_vals = after_vals[before_len:]
    new_all_zero = all(abs(val) < 1e-14 for val in new_vals)
    print(f"old_dof_values_preserved={preserved}")
    print(f"new_dofs_all_zero={new_all_zero}")
    if not preserved or new_all_zero:
        print(f"FAIL: the prolongation behaviour changed -- preserved="
              f"{preserved}, new_all_zero={new_all_zero}", file=sys.stderr)
        ok = False

    # The operative hazard, stated correctly: the prolongated function is the
    # COARSE function re-expressed on the fine mesh, so refining buys no
    # accuracy. On a field the coarse space cannot represent, the error after
    # the refine is the SAME as before it -- you must re-solve, but not for the
    # reason the claim gives.
    after_err = float(Integrate((gfu - curved) ** 2, mesh)) ** 0.5
    unchanged = abs(after_err - before_curved_err) < 1e-12
    print(f"curved_error_before_refine={before_curved_err:.6e}")
    print(f"curved_error_after_refine={after_err:.6e}")
    print(f"refine_alone_buys_no_accuracy={unchanged}")
    if not unchanged:
        print(f"FAIL: the refine changed the representation error "
              f"({before_curved_err!r} -> {after_err!r})", file=sys.stderr)
        ok = False

    # --- RIGHT variant: re-Set on the fine mesh --------------------------
    gfu.Set(curved)
    reset_err = float(Integrate((gfu - curved) ** 2, mesh)) ** 0.5
    improved = reset_err < 0.5 * before_curved_err
    print(f"curved_error_after_reset={reset_err:.6e}")
    print(f"reset_after_refine_actually_improves={improved}")
    if not improved:
        print(f"FAIL: re-Set on the fine mesh did not improve the error "
              f"({before_curved_err!r} -> {reset_err!r})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
