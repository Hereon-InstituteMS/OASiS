"""Tier-2: porting a 2D HHJ plate to 3D — HDivDiv is nn-, Regge/HCurlCurl is tt-conforming.

Claim: ngsolve hdivdiv#4 — Regge elements are DISTINCT from HHJ; porting an HHJ
2D-plate code to 3D elasticity without switching elements produces a
non-conforming solution (traction jumps at facets), and the HCurlCurl/HDivDiv
pairing is the canonical Regge discretisation.

Wrong variant: take the 2D plate form verbatim (specialcf.normal(2), Id(2)) and
build it on a 3D mesh.

Observed on NGSolve 6.2.2604 (2026-08-03) on unit_cube maxh=0.5
(nv=26 ne=53 nedge=96 nface=124):
  * the ported 2D form does not even assemble: NgException "Dimensions don't
    match, op = -"; with only the normal wrong it is "Matrix dimensions don't
    fit: mat is 3 x 3, vec is 2".
  * HDivDiv(3D, order=1): relative nn facet jump 6.8e-17, relative FULL TRACTION
    (sigma.n) jump 1.57 -> nn-conforming but NOT traction-conforming, so the
    catalog's "prescribed traction continuity" is wrong while its "traction jumps
    at facets" is right.
  * HCurlCurl(3D, order=1) (= Regge): relative tt jump 1.9e-17, relative nn jump
    1.40 -> the complementary space, and HCurlCurl(order=0).ndof == mesh.nedge
    exactly (one dof per edge, the defining property of lowest-order Regge).

Mutation control: ``T2_MUTATE=1 python source.py`` applies the documented fix at
the pathology site -- PORTED_FORM_DIM becomes 3, so specialcf.normal(3) and
Id(3) are used and the form is dimensionally consistent with the 3D mesh.  It
then assembles without raising, so ``ported_2d_form_on_3d_raises=True`` and the
literal expectation ``Dimensions don't match`` disappear and the fixture goes
red.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import ngsolve as ngs
from netgen.csg import unit_cube

MUTATE = os.environ.get("T2_MUTATE") == "1"

# WRONG: the spatial dimension baked into the ported 2D plate form.
# Under T2_MUTATE the form is ported properly and uses the mesh dimension.
PORTED_FORM_DIM = 3 if MUTATE else 2


def rel_jumps(V: ngs.FESpace, dim: int, seed: int = 11) -> dict[str, float]:
    """Relative squared facet jumps of nn, full traction and tt components."""
    gf = ngs.GridFunction(V)
    gf.vec.FV().NumPy()[:] = np.random.default_rng(seed).standard_normal(V.ndof)
    u, v = V.TnT()
    n = ngs.specialcf.normal(dim)

    def nn(w):
        return ngs.InnerProduct(w * n, n)

    def trac(w):
        return w * n

    def tt(w):
        # w projected onto the facet tangent plane
        return (w - ngs.OuterProduct(w * n, n) - ngs.OuterProduct(n, w * n)
                + ngs.InnerProduct(w * n, n) * ngs.OuterProduct(n, n))

    def quad(f, jump: bool) -> float:
        b = ngs.BilinearForm(V, symmetric=True)
        lhs = f(u) - f(u.Other()) if jump else f(u)
        rhs = f(v) - f(v.Other()) if jump else f(v)
        b += ngs.InnerProduct(lhs, rhs) * ngs.dx(skeleton=True)
        b.Assemble()
        return abs(float(ngs.InnerProduct(gf.vec, b.mat * gf.vec)))

    return {"nn": quad(nn, True) / quad(nn, False),
            "traction": quad(trac, True) / quad(trac, False),
            "tt": quad(tt, True) / quad(tt, False)}


def ported_2d_plate_form_on_3d(mesh: ngs.Mesh, dim: int) -> str:
    """Build the 2D HHJ compliance + moment skeleton term with `dim` hard-coded."""
    Vm = ngs.HDivDiv(mesh, order=1, dirichlet=".*")
    Vw = ngs.H1(mesh, order=2, dirichlet=".*")
    X = Vm * Vw
    (sig, ww), (tau, vv) = X.TnT()
    n = ngs.specialcf.normal(dim)
    a = ngs.BilinearForm(X, symmetric=True)
    a += ngs.InnerProduct(sig - 0.3 * ngs.Trace(sig) * ngs.Id(dim), tau) * ngs.dx
    a += -(tau * n * n) * (ngs.Grad(ww) * n) * ngs.dx(element_boundary=True)
    a.Assemble()
    return "assembled"


def main() -> int:
    ok = True
    print(f"ngsolve_version={ngs.__version__}")
    mesh = ngs.Mesh(unit_cube.GenerateMesh(maxh=0.5))
    print(f"mesh3d nv={mesh.nv} ne={mesh.ne} nedge={mesh.nedge} "
          f"nface={mesh.nface} dim={mesh.dim}")

    # --- WRONG variant: 2D plate form on a 3D mesh ------------------------
    msg = ""
    try:
        ported_2d_plate_form_on_3d(mesh, PORTED_FORM_DIM)
    except Exception as exc:                        # noqa: BLE001
        msg = str(exc)
    print(f"ported_2d_form_on_3d_raises={bool(msg)}")
    print(f"ported_2d_form_msg={(msg.splitlines() or [''])[0].strip()!r}")
    if "Dimensions don't match" not in msg:
        print(f"FAIL: the ported 2D plate form built on a 3D mesh without the "
              f"documented dimension error; got {msg!r}", file=sys.stderr)
        ok = False

    # --- what HDivDiv actually guarantees in 3D ---------------------------
    hdd = rel_jumps(ngs.HDivDiv(mesh, order=1, dgjumps=True), 3)
    print(f"hdivdiv3d_rel_nn_jump={hdd['nn']:.3e} "
          f"rel_traction_jump={hdd['traction']:.3e} rel_tt_jump={hdd['tt']:.3e}")
    print(f"hdivdiv3d_nn_conforming={hdd['nn'] < 1e-10}")
    print(f"hdivdiv3d_traction_jumps_at_facets={hdd['traction'] > 0.1}")
    if not hdd["nn"] < 1e-10:
        print("FAIL: HDivDiv in 3D is not normal-normal conforming",
              file=sys.stderr)
        ok = False
    if not hdd["traction"] > 0.1:
        print("FAIL: HDivDiv in 3D shows no traction jump, so the ported plate "
              "stress space would be traction-conforming after all",
              file=sys.stderr)
        ok = False

    # --- the Regge space HCurlCurl ----------------------------------------
    hcc = rel_jumps(ngs.HCurlCurl(mesh, order=1, dgjumps=True), 3)
    print(f"hcurlcurl3d_rel_nn_jump={hcc['nn']:.3e} rel_tt_jump={hcc['tt']:.3e}")
    print(f"regge_hcurlcurl_tt_conforming={hcc['tt'] < 1e-10}")
    print(f"regge_hcurlcurl_nn_jumps={hcc['nn'] > 0.1}")
    n0 = ngs.HCurlCurl(mesh, order=0).ndof
    print(f"hcurlcurl_order0_ndof={n0} nedge={mesh.nedge}")
    print(f"regge_order0_ndof_equals_nedge={n0 == mesh.nedge}")
    print(f"hdivdiv_and_hcurlcurl_are_distinct_spaces="
          f"{hdd['nn'] < 1e-10 < hcc['nn'] and hcc['tt'] < 1e-10 < hdd['tt']}")
    if not hcc["tt"] < 1e-10:
        print("FAIL: HCurlCurl (Regge) is not tangential-tangential conforming",
              file=sys.stderr)
        ok = False
    if n0 != mesh.nedge:
        print(f"FAIL: lowest-order Regge has {n0} dofs, not one per edge "
              f"({mesh.nedge})", file=sys.stderr)
        ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
