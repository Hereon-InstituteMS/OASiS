"""Tier-2: refinement and adaptivity — one silent no-op and four API
traps.

  poisson#13             dune.fem.globalRefine(level, uh) — the call
                         form that should refine AND prolong — is a
                         SILENT no-op on a YaspGrid: cell count, space
                         size and dof count all come back unchanged,
                         with no exception and no warning. It works on
                         adaptiveLeafGridView(aluConformGrid(...)).
                         globalRefine(level, gridView.hierarchicalGrid)
                         refines everywhere but does not prolong.
  adaptive_poisson#0     structuredGrid has no .mark at all, so
                         gridView.mark(...) dies on the PYTHON attribute
                         lookup. .mark IS present on
                         adaptiveLeafGridView(yasp) and on
                         .hierarchicalGrid, so its presence is not
                         evidence that local refinement will happen. The
                         ALUGrid factories are camelCase.
  adaptive_poisson#2     there is no space.update().
  adaptive_poisson#3     dune.fem.mark(..., gridView=gv) raises
                         AttributeError "'GridMarker' object has no
                         attribute 'gridView'" unconditionally;
                         markNeighbors forwards the same kwarg and
                         fails identically.
  adaptive_poisson#4     dune.fem.mark returns the marking STATISTICS,
                         not a marker; passing that tuple to adapt
                         raises AssertionError about "only one list of
                         discrete functions".
  adaptive_poisson#5     adapt on a plain ALUGrid leaf view raises
                         AssertionError about grid views supporting
                         adaptivity; the wrapped view has canAdapt True.

Grouped because they are all properties of the same four grid/view
objects, and because the working mark/adapt cycle at the end is the
control that makes the six failures meaningful.

Verified by execution against dune-fem 2.12.0.2.

MUTATION CONTROL. T2_MUTATE=1 runs the poisson#13 probe — the
globalRefine(level, uh) call — on adaptiveLeafGridView(aluConformGrid)
instead of on the YaspGrid, i.e. on the view where that call is not a
no-op. The (cells, space size, dof count) triple then changes, so
'yasp_globalRefine_is_a_silent_noop=True' is no longer printed and a
FAIL: line appears. The ALU view is one the fixture already builds.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dune.grid import structuredGrid, cartesianDomain            # noqa: E402
from dune.fem.space import lagrange, finiteVolume                # noqa: E402
from dune.fem.view import adaptiveLeafGridView                   # noqa: E402
import dune.alugrid as dalu                                      # noqa: E402
import dune.fem as dfem                                          # noqa: E402
from ufl import SpatialCoordinate                                # noqa: E402


def state(gv, space, uh):
    return gv.size(0), space.size, len(np.array(uh.as_numpy))


def main() -> int:
    fail: list[str] = []

    # ── the ALUGrid factory names are camelCase ────────────────────
    for good in ("aluConformGrid", "aluCubeGrid", "aluSimplexGrid"):
        print(f"alugrid_has_{good}={hasattr(dalu, good)}")
        if not hasattr(dalu, good):
            fail.append(f"dune.alugrid.{good} is missing")
    for bad in ("alusimplexGrid", "alucubeGrid", "aluconformGrid"):
        print(f"alugrid_has_{bad}={hasattr(dalu, bad)}")
        if hasattr(dalu, bad):
            fail.append(f"dune.alugrid.{bad} exists; the claim is that "
                        f"the factories are camelCase only")

    # ── four grid/view combinations ────────────────────────────────
    yasp = structuredGrid([0, 0], [1, 1], [4, 4])
    yasp_ad = adaptiveLeafGridView(structuredGrid([0, 0], [1, 1], [4, 4]))
    alu = dalu.aluConformGrid(cartesianDomain([0, 0], [1, 1], [4, 4]))
    alu_ad = adaptiveLeafGridView(
        dalu.aluConformGrid(cartesianDomain([0, 0], [1, 1], [4, 4])))

    # adaptive_poisson#0: .mark is absent on the plain structured view,
    # present on the wrapped view and on the hierarchical grid — so its
    # presence proves nothing.
    print(f"yasp_has_mark={hasattr(yasp, 'mark')}")
    print(f"yasp_adaptive_has_mark={hasattr(yasp_ad, 'mark')}")
    print(f"yasp_hierarchical_has_mark="
          f"{hasattr(yasp.hierarchicalGrid, 'mark')}")
    if hasattr(yasp, "mark"):
        fail.append("structuredGrid(...) has a .mark attribute; the "
                    "claim is that gridView.mark dies on the Python "
                    "attribute lookup")
    if not (hasattr(yasp_ad, "mark")
            and hasattr(yasp.hierarchicalGrid, "mark")):
        fail.append("the .mark attribute is no longer present on the "
                    "adaptive view / hierarchical grid, so the 'its "
                    "presence is not evidence' warning is stale")

    # adaptive_poisson#5: canAdapt
    print(f"alu_leaf_canAdapt_present={hasattr(alu, 'canAdapt')}")
    print(f"alu_adaptive_canAdapt={getattr(alu_ad, 'canAdapt', None)}")
    if getattr(alu_ad, "canAdapt", None) is not True:
        fail.append("adaptiveLeafGridView(aluConformGrid(...)).canAdapt "
                    "is not True")

    # ── adaptive_poisson#2: there is no space.update() ─────────────
    sp_yasp = lagrange(yasp, order=1)
    print(f"space_has_update={hasattr(sp_yasp, 'update')}")
    if hasattr(sp_yasp, "update"):
        fail.append("space.update() exists; the claim is that it does "
                    "not and that adapt() resizes the space for you")

    # ── poisson#13: globalRefine(level, uh) on a YaspGrid ──────────
    if MUTATE:
        # The pathology removed: run the same globalRefine(level, uh)
        # call on the adaptively wrapped ALU view, where it is not a
        # no-op.
        print("mutation=the_globalrefine_probe_runs_on_the_adaptive_"
              "alu_view")
        refine_view = alu_ad
    else:
        refine_view = yasp
    sp_yasp = lagrange(refine_view, order=1)
    x = SpatialCoordinate(sp_yasp)
    uh_yasp = sp_yasp.interpolate(x[0], name="uh_yasp")
    before = state(refine_view, sp_yasp, uh_yasp)
    dfem.globalRefine(1, uh_yasp)
    after = state(refine_view, sp_yasp, uh_yasp)
    print(f"yasp_before={before}")
    print(f"yasp_after_globalRefine_uh={after}")
    print(f"yasp_globalRefine_is_a_silent_noop={before == after}")
    if before != after:
        fail.append(f"globalRefine(1, uh) on a YaspGrid changed "
                    f"{before} -> {after}; the claim is that it is a "
                    f"silent no-op there")

    # …while refining the HIERARCHICAL grid does work. It gets its own
    # grid object: calling globalRefine(level, uh) above REGISTERS a
    # discrete function living on a non-adaptive space, and a later
    # hierarchical refine of the same grid then dies trying to prolong
    # it (measured: RuntimeError "NotImplemented [numBlocks:...]:
    # Method numBlocks() called on non-adaptive block mapper"). That
    # interaction is recorded in the fixture _comment; it is not what
    # poisson#13 is about.
    yasp2 = structuredGrid([0, 0], [1, 1], [4, 4])
    cells_before2 = yasp2.size(0)
    dfem.globalRefine(1, yasp2.hierarchicalGrid)
    cells_after = yasp2.size(0)
    print(f"yasp_cells_before_hierarchical_refine={cells_before2}")
    print(f"yasp_cells_after_hierarchical_refine={cells_after}")
    print(f"hierarchical_refine_works={cells_after == 4 * cells_before2}")
    if cells_after != 4 * cells_before2:
        fail.append(f"globalRefine(1, hierarchicalGrid) took the cell "
                    f"count {cells_before2} -> {cells_after}; the claim "
                    f"is a factor of four in 2D")

    # …and on the wrapped ALU view the prolonging form DOES work
    sp_alu = lagrange(alu_ad, order=1)
    uh_alu = sp_alu.interpolate(SpatialCoordinate(sp_alu)[0],
                                name="uh_alu")
    before_alu = state(alu_ad, sp_alu, uh_alu)
    dfem.globalRefine(1, uh_alu)
    after_alu = state(alu_ad, sp_alu, uh_alu)
    print(f"alu_adaptive_before={before_alu}")
    print(f"alu_adaptive_after={after_alu}")
    print(f"alu_adaptive_globalRefine_works={after_alu != before_alu}")
    if after_alu == before_alu:
        fail.append(f"globalRefine(1, uh) on the wrapped ALU view also "
                    f"did nothing ({before_alu}); without that control "
                    f"the YaspGrid no-op is not attributable to the "
                    f"grid")

    # ── adaptive_poisson#3: the gridView kwarg ─────────────────────
    ind_space = finiteVolume(alu_ad)
    indicator = ind_space.interpolate(SpatialCoordinate(sp_alu)[0],
                                      name="eta")
    for fn_name in ("mark", "markNeighbors"):
        fn = getattr(dfem, fn_name)
        try:
            fn(indicator, 0.5, gridView=alu_ad)
            print(f"{fn_name}_gridView_kwarg_raises=False")
            fail.append(f"dune.fem.{fn_name}(..., gridView=...) was "
                        f"accepted; the claim is an unconditional "
                        f"AttributeError from GridMarker.__init__")
        except AttributeError as exc:
            msg = " ".join(str(exc).split())
            print(f"{fn_name}_gridView_kwarg_raises="
                  f"{type(exc).__name__}")
            print(f"{fn_name}_gridView_kwarg_message={msg[:140]}")
            if "gridView" not in msg:
                fail.append(f"the {fn_name} AttributeError no longer "
                            f"names 'gridView': {msg[:140]}")

    # ── adaptive_poisson#4: mark returns statistics ────────────────
    ret = dfem.mark(indicator, 0.5)
    print(f"mark_return_type={type(ret).__name__}")
    print(f"mark_return_value={ret}")
    print(f"mark_returns_a_tuple_not_a_marker="
          f"{isinstance(ret, tuple)}")
    print(f"mark_return_is_callable={callable(ret)}")
    if not isinstance(ret, tuple) or callable(ret):
        fail.append(f"dune.fem.mark returned {type(ret).__name__}; the "
                    f"claim is a (nRefined, nCoarsened) tuple that is "
                    f"NOT callable")
    ret_stats = dfem.mark(indicator, 0.5, statistics=True)
    print(f"mark_with_statistics={ret_stats}")

    try:
        dfem.adapt(ret, [uh_alu])
        print("adapt_with_marker_first_raises=False")
        fail.append("dune.fem.adapt(<mark return value>, [uh]) was "
                    "accepted; the claim is an AssertionError")
    except AssertionError as exc:
        msg = " ".join(str(exc).split())
        print(f"adapt_with_marker_first_raises={type(exc).__name__}")
        print(f"adapt_with_marker_first_message={msg[:160]}")
        if "one list of discrete functions" not in msg:
            fail.append(f"the AssertionError no longer mentions 'one "
                        f"list of discrete functions': {msg[:160]}")

    # ── adaptive_poisson#5: adapt needs an ADAPTIVE view ───────────
    sp_plain_alu = lagrange(alu, order=1)
    uh_plain = sp_plain_alu.interpolate(0, name="uh_plain")
    try:
        dfem.adapt([uh_plain])
        print("adapt_on_plain_alu_raises=False")
        fail.append("dune.fem.adapt([uh]) on a plain ALUGrid leaf view "
                    "was accepted; the claim is an AssertionError about "
                    "grid views supporting adaptivity")
    except AssertionError as exc:
        msg = " ".join(str(exc).split())
        print(f"adapt_on_plain_alu_raises={type(exc).__name__}")
        print(f"adapt_on_plain_alu_message={msg[:160]}")
        if "adaptivity" not in msg:
            fail.append(f"the AssertionError no longer mentions "
                        f"adaptivity: {msg[:160]}")

    # ── the control: the documented cycle actually adapts ──────────
    cells0, size0, _ = state(alu_ad, sp_alu, uh_alu)
    dfem.mark(indicator, 0.5)
    dfem.adapt([uh_alu])
    cells1, size1, dofs1 = state(alu_ad, sp_alu, uh_alu)
    print(f"cycle_before={cells0},{size0}")
    print(f"cycle_after={cells1},{size1},{dofs1}")
    print(f"mark_then_adapt_refines={cells1 > cells0}")
    print(f"adapt_resized_the_space={size1 == dofs1 and size1 > size0}")
    if not (cells1 > cells0 and size1 > size0 and size1 == dofs1):
        fail.append(f"the documented mark/adapt cycle did not refine "
                    f"and prolong: cells {cells0}->{cells1}, space "
                    f"{size0}->{size1}, dofs {dofs1}")

    if not fail:
        print("dune_refinement_and_adaptivity_api_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
