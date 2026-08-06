"""The conservation check has to fire on a wrong interface and stay silent on a
right one. Both directions, because only one of them is usually tested.

THE CLAIM UNDER TEST, from the interface-flux section of the coupling knowledge:

  "That is what OASiS's conservation check tests. Export both sides with the
   same sign and a CORRECT coupling fails it ... a genuinely near-zero net flux
   (a symmetric profile) is NOT reported as unbalanced — the comparison is
   floored by the flux magnitudes, so float noise on two cancelling integrals
   cannot manufacture a failure."

  "if EITHER side omits `normal_fluxes` it reports `Interface conservation was
   NOT CHECKED` rather than passing silently."

A validator is two claims, and the second one is the one that rots. Everybody
tests that a detector fires. Almost nobody tests that it stays quiet, and a
detector that cries wolf is worse than none: it trains the reader to skip the
line where the real finding will appear. The specific case that motivated this
is a FRICTIONLESS contact interface — the normal tractions cancel exactly and
the tangential ones are two different roundoff crumbs — which a per-component
comparison without a magnitude floor reports as ~92% unbalanced on a physically
perfect interface.

This fixture is pure arithmetic on hand-built exports: no solver, no coupling,
and therefore no way for a slow backend to hide a regression in the validator.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _couplinglib as L                                    # noqa: E402


def iface(fluxes, n=None, y0=0.0, y1=0.4, values=None):
    """An InterfaceData-shaped dict on a straight interface at x = 0.6."""
    n = n if n is not None else len(fluxes)
    ys = [y0 + (y1 - y0) * i / (n - 1) for i in range(n)]
    return {"field_name": "traction",
            "coordinates": [[0.6, y] for y in ys],
            "values": values if values is not None else [0.0] * n,
            "normal_fluxes": fluxes,
            "n_points": n}


def balance(a, b, rtol=0.05):
    from core.quality_checks import check_interface_balance
    return check_interface_balance(a, b, "A", "B", rtol=rtol)


def body() -> None:
    n = 9

    # ── (1) THE QUIET DIRECTION: a correct frictionless interface ──────────
    # Normal traction cancels exactly; the tangential component is zero on both
    # sides up to roundoff, and the two crumbs are DIFFERENT crumbs. Nothing
    # here is non-conservative: a frictionless interface transmits no shear.
    p = 1.0e5
    a = iface([[+p, +3.0e-17] for _ in range(n)], n)
    b = iface([[-p, -1.0e-17] for _ in range(n)], n)
    w = balance(a, b)
    print(f"frictionless_warnings={len(w)}")
    if w:
        print(f"frictionless_text={w[0][:200]}")
    L.check(not w, "frictionless_interface_reported_unbalanced",
            f"a frictionless interface — normal tractions cancelling exactly, "
            f"tangential at roundoff — must not be reported as an imbalance: "
            f"{w}")

    # ── (2) a symmetric profile whose NET is genuinely ~0 on both sides ────
    # Both integrals are float noise. Without a magnitude floor this is a 100%
    # imbalance on a physically perfect interface.
    sym = [(-1.0) ** i * 1.0e4 for i in range(n)]
    w = balance(iface(sym, n), iface([-v for v in sym], n))
    print(f"symmetric_profile_warnings={len(w)}")
    L.check(not w, "symmetric_profile_reported_unbalanced", str(w)[:300])

    # ── (3) THE LOUD DIRECTION: genuinely non-conservative ─────────────────
    # Half the flux arrives. This is what the check exists for.
    w = balance(iface([+p] * n, n), iface([-0.5 * p] * n, n))
    print(f"half_flux_warnings={len(w)}")
    L.check(len(w) == 1 and "NOT balanced" in w[0],
            "genuine_imbalance_not_caught",
            f"half the flux arriving must be reported: {w}")
    if w:
        print(f"half_flux_says_not_balanced={'NOT balanced' in w[0]}")

    # ── (4) the SIGN-CONVENTION case, which is the common cause ────────────
    # Same magnitude, same sign. The knowledge says naming this is the point:
    # "the most common cause of this warning is not a non-conservative coupling
    # but both sides exporting their flux with the SAME sign".
    w = balance(iface([+p] * n, n), iface([+p] * n, n))
    print(f"same_sign_warnings={len(w)}")
    L.check(len(w) == 1, "same_sign_not_caught", str(w)[:200])
    if w:
        named = "SIGN-CONVENTION" in w[0]
        print(f"same_sign_names_the_convention={named}")
        L.check(named, "same_sign_not_named_as_a_convention_error",
                f"a same-sign export must be named as a convention error, not "
                f"left as a bare imbalance that sends the reader hunting for a "
                f"physics bug: {w[0][:200]}")

    # ── (5) NOT CHECKED is not the same as passed ──────────────────────────
    missing = iface([+p] * n, n)
    del missing["normal_fluxes"]
    w = balance(missing, iface([-p] * n, n))
    print(f"missing_flux_warnings={len(w)}")
    L.check(len(w) == 1 and "NOT CHECKED" in w[0],
            "missing_flux_passed_silently",
            f"a run with no conservation evidence must say so rather than "
            f"being stamped like one that passed: {w}")

    # ── (6) and the quiet direction must not be quiet about EVERYTHING ─────
    # A check that never fires would pass (1) and (2) too. This is the guard on
    # the guard: the same frictionless shape, but with the normal component
    # genuinely wrong, has to be caught.
    w = balance(iface([[+p, +3.0e-17] for _ in range(n)], n),
                iface([[-0.4 * p, -1.0e-17] for _ in range(n)], n))
    print(f"frictionless_but_wrong_warnings={len(w)}")
    L.check(len(w) == 1 and "NOT balanced" in w[0],
            "wrong_normal_component_missed_under_a_frictionless_shape",
            f"the tangential roundoff must not buy silence for a wrong normal "
            f"traction: {w}")

    print("balance_cases_checked=6")


L.main(body)
