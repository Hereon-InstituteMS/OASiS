"""Tier-2: skfem.models.poisson.mass IS a hand-written u*v BilinearForm.

Claim: skfem heat_transient#3 -- "Mass matrix M comes from
skfem.models.poisson.mass -- re-implementing u*v as a BilinearForm works but
is slow.  Signal: user-implemented mass matrix M_assemble takes 10-100x longer
than skfem.models.poisson.mass."

Measured on skfem 12.0.1 by reading the shipped source with inspect and
comparing the assembled matrices, NOT by timing anything -- a wall-clock
assertion would go red on a busy machine and would not be evidence either way.

  * THE CLAIM IS FALSE, and the source settles it: skfem/models/poisson.py
    defines the shipped helper as exactly

        @BilinearForm
        def mass(u, v, _):
            return u * v

    so a user who "re-implements u*v as a BilinearForm" has written the same
    thing.  There is no compiled fast path to miss.
  * the two assembled matrices are BIT-IDENTICAL: same shape, same non-zero
    count, same sparsity pattern, and a difference matrix with zero stored
    entries after eliminate_zeros().
  * the shipped object's own type is BilinearForm, the same class the user's
    decorator produces.

What IS worth knowing, and is structural rather than temporal, is that
assembling the mass matrix twice costs twice as much work whichever form you
use -- so the saving in a time loop comes from assembling ONCE, not from
picking the shipped helper.
"""
from __future__ import annotations

import inspect
import sys

import numpy as np
import skfem.models.poisson as poisson
from skfem import Basis, BilinearForm, ElementTriP1, MeshTri


@BilinearForm
def my_mass(u, v, w):
    return u * v


def main() -> int:
    ok = True
    src = inspect.getsource(poisson)
    # BilinearForm is not a function object, so getsource() on it raises;
    # the shipped definition has to be read out of the module text.
    shipped = "def mass" + src.split("def mass", 1)[1].split("\n\n")[0]
    print(f"shipped_mass_type={type(poisson.mass).__name__}")
    print(f"user_mass_type={type(my_mass).__name__}")
    print(f"same_class={type(poisson.mass) is type(my_mass)}")
    print(f"shipped_source={shipped.strip()!r}")
    print(f"shipped_is_a_bilinearform_decorator="
          f"{src.split('def mass')[0].rstrip().endswith('@BilinearForm')}")
    getsource_err = None
    try:
        inspect.getsource(poisson.mass)
    except TypeError as e:
        getsource_err = e
    print(f"getsource_on_the_form_raises={getsource_err is not None}")
    print(f"getsource_message={str(getsource_err)[-30:]!r}")
    print(f"shipped_body_is_u_times_v={'return u * v' in shipped}")
    print(f"shipped_has_no_compiled_fast_path="
          f"{'import' not in shipped and 'cython' not in shipped.lower()}")
    if type(poisson.mass) is not type(my_mass):
        print("FAIL: the shipped mass is not the same class as a user form",
              file=sys.stderr)
        ok = False
    if "return u * v" not in shipped:
        print("FAIL: the shipped mass is not a plain u*v form",
              file=sys.stderr)
        ok = False

    m = MeshTri().refined(5)
    ib = Basis(m, ElementTriP1())
    A = poisson.mass.assemble(ib).tocsr()
    B = my_mass.assemble(ib).tocsr()
    diff = (A - B)
    diff.eliminate_zeros()
    print(f"dofs_N={ib.N}")
    print(f"shipped_shape={A.shape} shipped_nnz={A.nnz}")
    print(f"user_shape={B.shape} user_nnz={B.nnz}")
    print(f"same_shape={A.shape == B.shape}")
    print(f"same_nnz={A.nnz == B.nnz}")
    print(f"same_sparsity_pattern="
          f"{bool(np.array_equal(A.indices, B.indices)
                  and np.array_equal(A.indptr, B.indptr))}")
    print(f"difference_stored_entries={diff.nnz}")
    print(f"matrices_are_bit_identical={diff.nnz == 0}")
    print(f"total_mass={float(A.sum()):.12f}")
    print(f"total_mass_is_domain_area={abs(float(A.sum()) - 1.0) < 1e-12}")
    if diff.nnz != 0:
        print("FAIL: the two mass matrices differ", file=sys.stderr)
        ok = False
    if A.nnz != B.nnz or A.shape != B.shape:
        print("FAIL: shape or nnz differ", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
