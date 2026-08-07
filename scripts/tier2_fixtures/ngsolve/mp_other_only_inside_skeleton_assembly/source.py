"""Tier-2: .Other() works in a skeleton BilinearForm, not in Integrate.

Claim: NGSolve mixed_poisson#1 -- the HDiv normal component is continuous, so a
normal-flux jump term is mathematically zero. In practice a standalone
Integrate of the jump does not return zero; it raises
NgException('other mir not set, pls report to developers').

Wrong variant: Integrate((gfu*n - gfu.Other()*n)**2*dx(skeleton=True), mesh).
Right variant: assemble the same integrand as a skeleton BilinearForm -- which
needs dgjumps=True on the space, or the assembly itself raises
NgException('SparseMatrixTM::AddElementMatrix: illegal dnums') -- and read the
energy off the assembled operator. It is machine zero for HDiv and decidedly
non-zero for a VectorL2 field with arbitrary DOF values, so the measure is shown
to discriminate rather than merely reporting a small number.

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology site --
the probe that is wrapped in the try/except stops being the standalone
Integrate(... .Other() ...) and becomes the skeleton-BilinearForm route
(jump_energy(HDiv(dgjumps=True))), which is exactly what the prose tells the user
to write instead. Nothing then raises, so the expectations 'other mir not set'
and 'standalone_integrate_raises=True' go missing and the fixture goes red.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

from netgen.geom2d import unit_square
from ngsolve import (
    BilinearForm,
    CoefficientFunction,
    GridFunction,
    HDiv,
    Integrate,
    Mesh,
    VectorL2,
    dx,
    specialcf,
    x,
    y,
)

# Mutation control: under T2_MUTATE=1 the wrong variant below is replaced by the
# documented fix (skeleton BilinearForm instead of a standalone Integrate).
MUTATE = os.environ.get("T2_MUTATE") == "1"


def jump_energy(space, field=None, randomise=False) -> float:
    """Assemble the normal-jump form as a skeleton BilinearForm and evaluate it
    on gf. `dgjumps=True` is required on the space: without it the assembly
    raises NgException('SparseMatrixTM::AddElementMatrix: illegal dnums')."""
    gf = GridFunction(space)
    if randomise:
        import random
        random.seed(7)
        for i in range(len(gf.vec)):
            gf.vec[i] = random.uniform(-1.0, 1.0)
    else:
        gf.Set(field)
    n = specialcf.normal(2)
    u, v = space.TnT()
    a = BilinearForm(space)
    a += ((u * n - u.Other() * n) * (v * n - v.Other() * n)
          * dx(skeleton=True))
    a.Assemble()
    return abs(float(gf.vec.InnerProduct(a.mat * gf.vec)))


def main() -> int:
    ok = True
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    field = CoefficientFunction((x * x, y))
    hdiv = HDiv(mesh, order=2)
    print(f"hdiv_space_type={hdiv.type}")

    # --- WRONG variant: .Other() outside a skeleton assembly -------------
    gf = GridFunction(hdiv)
    gf.Set(field)
    n = specialcf.normal(2)
    msg = ""
    try:
        if MUTATE:
            # the documented fix, applied at the pathology site: the same
            # integrand routed through a skeleton BilinearForm
            jump_energy(HDiv(mesh, order=2, dgjumps=True), field)
        else:
            Integrate((gf * n - gf.Other() * n) ** 2 * dx(skeleton=True), mesh)
    except Exception as exc:                      # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
    print(f"standalone_integrate_raises={bool(msg)}")
    print(f"standalone_integrate_msg={msg[:120]!r}")
    if "other mir not set" not in msg:
        print(f"FAIL: the standalone Integrate did not raise the expected "
              f"NgException; got {msg!r}", file=sys.stderr)
        ok = False

    # --- RIGHT variant: the same integrand through skeleton assembly -----
    hdiv_energy = jump_energy(HDiv(mesh, order=2, dgjumps=True), field)
    # A VectorL2 field with arbitrary DOF values genuinely jumps across facets;
    # it is here to prove the measure is sensitive, so that HDiv's machine zero
    # means conformity rather than a form that measures nothing.
    l2_energy = jump_energy(VectorL2(mesh, order=2, dgjumps=True),
                            randomise=True)
    print(f"hdiv_jump_energy={hdiv_energy:.3e}")
    print(f"discontinuous_field_jump_energy={l2_energy:.3e}")
    conforming = hdiv_energy < 1e-12
    discriminates = l2_energy > 1e6 * max(hdiv_energy, 1e-30)
    print(f"skeleton_bilinearform_jump_energy_at_machine_zero={conforming}")
    print(f"discontinuous_field_has_nonzero_normal_jump={discriminates}")
    if not conforming:
        print(f"FAIL: HDiv normal jump energy is {hdiv_energy!r}, not machine "
              f"zero", file=sys.stderr)
        ok = False
    if not discriminates:
        print(f"FAIL: the measure does not discriminate -- the discontinuous "
              f"field gave {l2_energy!r} against HDiv's {hdiv_energy!r}",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
