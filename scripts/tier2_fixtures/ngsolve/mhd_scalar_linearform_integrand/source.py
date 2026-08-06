"""Tier-2: SymbolicLFI rejects a vector-valued integrand, and it does so at the
LinearForm's __iadd__.

Claim: ngsolve mhd#7 -- "LinearForm integrand must be SCALAR-VALUED.  Multiplying
a vector-valued CoefficientFunction (e.g. CoefficientFunction((0, 1)) for an e_y
unit vector) with a scalar factor and a scalar test-function component still
yields a vector -- and SymbolicLFI rejects it.  Build the integrand purely from
scalar factors instead, indexing the vector test function (v[1] for the
y-component) to project onto the desired equation.  Signal: NgException
'SymbolicLFI needs scalar-valued CoefficientFunction' raised from
LinearForm.__iadd__ when the integrand passes a VectorialCF through *=."

Wrong variant: the Lorentz-force source written as e_y * J * v[1].

What this fixture pins, all re-measured on this run:
  * the literal NgException text, raised from LinearForm.__iadd__ -- recorded by
    the stage the failure is caught at, since the claim is specific about it;
  * the integrand really is vector-valued at that point -- its .dim is 2, not 1,
    so the diagnosis in the claim is the actual cause;
  * the scalar spelling the claim recommends assembles;
  * InnerProduct(e_y, v) is a second working spelling and gives the SAME vector
    as indexing v[1], so the fix is not unique and both are correct -- checked
    entry by entry rather than asserted.

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology site
-- the vector-valued factor e_y is dropped from the offending integrand, which
becomes the purely scalar 1.0 * J * v[1] the claim recommends. SymbolicLFI then
accepts it, so `integrand_is_vector_valued=True`, `vector_integrand_raised=
True`, `message_literal=True` and `raised_at_iadd=True` all disappear from the
output and the run prints "FAIL:", which the fixture forbids.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy
from netgen.geom2d import unit_square
from ngsolve import (
    CoefficientFunction,
    InnerProduct,
    LinearForm,
    Mesh,
    VectorH1,
    dx,
)

MESSAGE = "SymbolicLFI needs scalar-valued CoefficientFunction"
J = 3.0
MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    fes = VectorH1(mesh, order=1)
    u, v = fes.TnT()
    e_y = CoefficientFunction((0, 1))

    # The pathology is the vector-valued e_y factor in front of the already
    # projected scalar J*v[1]. T2_MUTATE=1 replaces it by the scalar 1.0,
    # which is the fix the claim recommends.
    lead = 1.0 if MUTATE else e_y
    integrand = lead * J * v[1]
    print(f"integrand_dim={integrand.dim}")
    print(f"integrand_is_vector_valued={integrand.dim > 1}")

    f_bad = LinearForm(fes)
    stage = "form_constructed"
    msg = ""
    try:
        f_bad += integrand * dx
        stage = "iadd_accepted"
        f_bad.Assemble()
        stage = "assembled"
    except Exception as exc:                                   # noqa: BLE001
        msg = str(exc)
    print(f"vector_integrand_stage={stage}")
    print(f"vector_integrand_raised={bool(msg)}")
    print(f"message_literal={MESSAGE in msg}")
    print(f"raised_at_iadd={stage == 'form_constructed'}")

    f_idx = LinearForm(fes)
    f_idx += J * v[1] * dx
    f_idx.Assemble()
    print(f"scalar_indexed_spelling_assembles=True")

    f_ip = LinearForm(fes)
    f_ip += InnerProduct(e_y * J, v) * dx
    f_ip.Assemble()
    print(f"innerproduct_spelling_assembles=True")

    a = f_idx.vec.FV().NumPy()
    b = f_ip.vec.FV().NumPy()
    same = float(numpy.abs(a - b).max())
    print(f"two_spellings_max_difference={same:.3e}")
    print(f"two_spellings_agree={same < 1e-14}")
    print(f"assembled_vector_is_nonzero={float(numpy.abs(a).max()) > 1e-12}")

    ok = (
        integrand.dim > 1
        and stage == "form_constructed"
        and MESSAGE in msg
        and same < 1e-14
        and float(numpy.abs(a).max()) > 1e-12
    )
    if ok:
        return 0
    print("FAIL: SymbolicLFI scalar invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
