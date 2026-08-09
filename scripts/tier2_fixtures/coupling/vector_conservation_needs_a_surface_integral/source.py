"""Conservation of a VECTOR flux across an interface that is a SURFACE.

THE DEFECT THIS EXISTS FOR. `check_interface_balance` has to compare the net
flux the two participants exchange, and the two sample the same interface at
different points, so the two sums are not comparable — only the two INTEGRALS
are. It formed that integral by sorting the exported points lexicographically,
taking the distance between consecutive points and applying a trapezoid rule.
On a LINE that is exactly right. On a SURFACE the lexicographic order snakes
row by row through the point cloud, and the "arclength" it accumulates is the
length of that snake: a quantity that grows with the mesh resolution and has
nothing to do with the area. Multiplying a traction by it produces a "net flux"
whose size is set by the discretisation, and two such numbers agree only when
both sides happen to snake the same way.

Nothing caught it, because every coupling fixture in this tree runs on a
straight interface in 2-D, where the point cloud IS a line and the snake IS the
arclength.

WHAT IS CHECKED HERE, all of it arithmetic on synthetic exports so it runs in
seconds and needs no solver:

  1. the quadrature reproduces the true measure of an interface that is a line
     in 2-D, a tilted line in 3-D, and a tilted plane patch in 3-D;
  2. the snake the old rule would have used on that patch, for the size of the
     error it was making;
  3. two participants sampling the SAME 3-D surface at DIFFERENT resolutions
     and exchanging a conservative, spatially varying vector traction are
     reported as balanced — while their plain SUMS differ by a large factor,
     which is what the old rule was comparing;
  4. a genuinely non-conservative exchange is still caught, componentwise: a
     +x imbalance cancelled by a -y one sums to zero across components and must
     NOT read as conserved;
  5. an interface the quadrature cannot handle — points filling a volume — is
     reported as UNCHECKED rather than silently summed. "Conservation was not
     evaluated" and "conservation was evaluated and is fine" must never be the
     same output.

T2_MUTATE=1 replaces the surface quadrature with the path-length rule the old
implementation used, and nothing else. Check 3 must then fail.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                     # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"


def _snake_integral(coords, values):
    """The OLD rule: lexicographic order, distance between consecutive points,
    trapezoid. Kept here so the fixture can show what it was computing."""
    c = np.asarray(coords, float)
    v = np.asarray(values, float)
    order = np.lexsort(tuple(c[:, k] for k in range(c.shape[1] - 1, -1, -1)))
    cs, vs = c[order], v[order]
    ds = np.linalg.norm(np.diff(cs, axis=0), axis=1)
    return np.sum(0.5 * (vs[:-1] + vs[1:]) * ds[:, None], axis=0), float(ds.sum())


def _patch(nu_, nv_):
    """A tilted plane patch in 3-D: 2 x 3 in its own plane, tilted so its true
    area is 6*sqrt(1 + 0.5^2 + 0.25^2)."""
    u, v = np.meshgrid(np.linspace(0.0, 2.0, nu_), np.linspace(0.0, 3.0, nv_))
    u, v = u.ravel(), v.ravel()
    return np.column_stack([u, v, 0.5 * u + 0.25 * v])


def _traction(pts):
    """A conservative, spatially VARYING vector traction on the patch. Constant
    would make every quadrature agree and prove nothing."""
    u, v = pts[:, 0], pts[:, 1]
    return np.column_stack([1.0 + 0.30 * u - 0.20 * v,
                            0.4 - 0.15 * u + 0.35 * v])


def body() -> None:
    from core.quality_checks import (check_interface_balance,
                                     interface_nodal_weights)
    exact_area = 6.0 * float(np.sqrt(1.0 + 0.25 + 0.0625))

    # ── 1. the quadrature reproduces the true measure ──────────────────────
    line2 = np.column_stack([np.full(11, 0.5), np.linspace(0.0, 1.0, 11)])
    t = np.linspace(0.0, 1.0, 7)
    line3 = np.column_stack([t, 2.0 * t, 3.0 * t])
    patch = _patch(9, 7)
    for tag, co, want, dim_want in (("line2d", line2, 1.0, 1),
                                    ("line3d", line3, float(np.sqrt(14.0)), 1),
                                    ("surface", patch, exact_area, 2)):
        w, dim, detail = interface_nodal_weights(co)
        L.check(w is not None, f"{tag}_no_quadrature", str(detail))
        got = float(np.sum(w)) if w is not None else float("nan")
        print(f"{tag}_dim={dim} measure={got:.10g} exact={want:.10g}")
        L.check(dim == dim_want, f"{tag}_wrong_dimension", f"got {dim}")
        L.close(got, want, 1e-9 * want, f"{tag}_measure_err")
    print("quadrature_reproduces_the_measure=True")

    # ── 2. what the old rule was computing on that surface ─────────────────
    _, snake = _snake_integral(patch, _traction(patch))
    print(f"surface_snake_length={snake:.6g} true_area={exact_area:.6g}")
    print(f"surface_snake_overstates_by={snake / exact_area:.4g}x")
    L.check(snake / exact_area > 3.0, "snake_was_harmless",
            "the path-length rule happened to land near the area on this "
            "surface, so this fixture does not demonstrate the defect")

    # ── 3. two resolutions of the same surface, conservative exchange ──────
    pa, pb = _patch(9, 7), _patch(13, 11)
    ta, tb = _traction(pa), -_traction(pb)      # anti-parallel normals
    ea = {"coordinates": pa.tolist(), "normal_fluxes": ta.tolist()}
    eb = {"coordinates": pb.tolist(), "normal_fluxes": tb.tolist()}
    print(f"surface_points={len(pa)}/{len(pb)}")
    sum_a, sum_b = ta.sum(axis=0), tb.sum(axis=0)
    print(f"plain_sum_ratio_x={abs(sum_b[0] / sum_a[0]):.4g}")
    L.check(abs(abs(sum_b[0] / sum_a[0]) - 1.0) > 0.5, "plain_sum_was_fine",
            "the two plain sums happen to agree, so this case does not show "
            "why an integral is needed")
    if MUTATE:
        # THE MUTATION: integrate with the path-length rule instead. Everything
        # else — the exports, the check, the tolerance — is untouched.
        ia, _ = _snake_integral(pa, ta)
        ib, _ = _snake_integral(pb, tb)
        w = check_interface_balance({"normal_fluxes": ia.tolist()},
                                    {"normal_fluxes": ib.tolist()}, "A", "B")
    else:
        w = check_interface_balance(ea, eb, "A", "B")
    print(f"surface_balance_findings={len(w)}")
    if w:
        print(f"surface_balance_detail={'; '.join(w)[:220]}")
    ok_s = L.check(not w, "surface_exchange_not_balanced", "; ".join(w)[:300])
    print(f"surface_vector_exchange_balances={bool(ok_s)}")

    # ── 4. a componentwise imbalance is still caught ───────────────────────
    bad = -_traction(pb)[:, ::-1]               # x and y swapped: sums alike,
    eb_bad = {"coordinates": pb.tolist(),       # each component wrong
              "normal_fluxes": bad.tolist()}
    wb = check_interface_balance(ea, eb_bad, "A", "B")
    print(f"swapped_components_findings={len(wb)}")
    L.check(bool(wb), "component_swap_not_caught",
            "swapping the two traction components leaves the total unchanged; "
            "only a componentwise balance sees it")
    print(f"component_swap_is_caught={bool(wb)}")

    # ── 5. an interface the quadrature cannot handle says so ───────────────
    rng = np.random.default_rng(0)
    vol = rng.normal(size=(60, 3))
    w3, dim3, why3 = interface_nodal_weights(vol)
    print(f"volume_dim={dim3} quadrature={'none' if w3 is None else 'formed'}")
    L.check(w3 is None and dim3 == 3, "volume_was_integrated", str(why3))
    wv = check_interface_balance(
        {"coordinates": vol.tolist(),
         "normal_fluxes": np.ones((60, 2)).tolist()},
        {"coordinates": vol[:40].tolist(),
         "normal_fluxes": (-np.ones((40, 2))).tolist()}, "A", "B")
    print(f"volume_reported_unchecked="
          f"{bool(wv and 'could NOT be evaluated' in wv[0])}")
    L.check(bool(wv) and "could NOT be evaluated" in wv[0],
            "volume_silently_summed",
            "an interface with no surface integral must be reported as "
            "unchecked, not summed")
    print(f"mutated={MUTATE}")


L.main(body)
