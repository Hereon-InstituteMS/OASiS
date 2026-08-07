"""Tier-2: dune.fem.function.integrate is deprecated — on CALL.

poisson_mms#0 says the DeprecationWarning fires when you IMPORT from
dune.fem.function. Measured, it does not: the import is silent and the
warning is emitted by the call. This fixture asserts both halves —
silence at import, warning at call — so the claim's wording cannot be
restored without the fixture failing.

It also pins the argument orders, which is the part that actually bites:
the DEPRECATED function keeps the old (gridView, expr, order) signature
and rejects gridView= as a keyword, while the replacement
dune.fem.integrate takes (expr, gridView, order). Both return the same
number.

Verified by execution against dune-fem 2.12.0.2.
"""
from __future__ import annotations

import subprocess
import sys
import warnings

from dune.grid import structuredGrid                            # noqa: E402
from dune.fem.space import lagrange                             # noqa: E402
import dune.fem as dfem                                         # noqa: E402
from ufl import SpatialCoordinate                               # noqa: E402

_IMPORT_PROBE = """
import warnings, sys
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    from dune.fem.function import integrate
    texts = [str(w.message) for w in caught]
print("IMPORT_WARNINGS=%d" % len(texts))
print("IMPORT_TEXTS=%r" % (texts,))
"""


def main() -> int:
    fail: list[str] = []

    # ── the import, in a FRESH process (a warning already triggered
    #    in this one would not fire twice) ────────────────────────────
    proc = subprocess.run([sys.executable, "-c", _IMPORT_PROBE],
                          capture_output=True, text=True, timeout=300)
    out = (proc.stdout or "") + (proc.stderr or "")
    n_import_warnings = None
    for line in out.splitlines():
        if line.startswith("IMPORT_WARNINGS="):
            n_import_warnings = int(line.split("=", 1)[1])
    print(f"import_warning_count={n_import_warnings}")
    print(f"import_is_silent={n_import_warnings == 0}")
    if n_import_warnings is None:
        fail.append(f"the import probe produced no verdict: {out[-300:]}")
    elif n_import_warnings != 0:
        fail.append(f"importing dune.fem.function.integrate emitted "
                    f"{n_import_warnings} warning(s); this fixture "
                    f"records that the import is silent and the CALL is "
                    f"what warns")

    # ── the call, which is where the warning really lives ───────────
    gridView = structuredGrid([0, 0], [1, 1], [4, 4])
    space = lagrange(gridView, order=1)
    x = SpatialCoordinate(space)
    uh = space.interpolate(x[0] * x[0], name="uh")

    import dune.fem.function as dfunc
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old_value = dfunc.integrate(gridView, uh, order=3)
        texts = [" ".join(str(w.message).split()) for w in caught]
    blob = " || ".join(texts)
    print(f"call_warning_count={len(texts)}")
    print(f"call_warning_text={blob[:260]}")
    print(f"call_warns_deprecated="
          f"{any('deprecated' in t for t in texts)}")
    print(f"call_names_the_replacement="
          f"{any('dune.fem.integrate' in t for t in texts)}")
    print(f"call_states_the_new_signature="
          f"{any('(expr, gridView, order)' in t for t in texts)}")
    if not any("deprecated" in t for t in texts):
        fail.append(f"calling dune.fem.function.integrate emitted no "
                    f"deprecation warning; warnings were {texts}")
    for needle in ("dune.fem.integrate", "(expr, gridView, order)"):
        if not any(needle in t for t in texts):
            fail.append(f"the deprecation warning no longer contains "
                        f"{needle!r}: {blob[:260]}")

    # ── the signatures are NOT interchangeable ─────────────────────
    try:
        dfunc.integrate(uh, gridView=gridView, order=3)
        print("deprecated_accepts_new_signature=True")
        fail.append("the deprecated function accepted the NEW argument "
                    "order; the trap this fixture records is that it "
                    "does not")
    except TypeError as exc:
        print(f"deprecated_rejects_new_signature={type(exc).__name__}")
        print(f"deprecated_signature_error="
              f"{' '.join(str(exc).split())[:140]}")

    new_value = dfem.integrate(uh, gridView=gridView, order=3)
    print(f"deprecated_value={old_value}")
    print(f"replacement_value={new_value}")
    print(f"both_routes_agree={abs(old_value - new_value) < 1e-14}")
    if abs(old_value - new_value) >= 1e-14:
        fail.append(f"the two routes disagree ({old_value} vs "
                    f"{new_value}), so the deprecation is not a pure "
                    f"rename and the knowledge understates the change")

    if not fail:
        print("dune_integrate_deprecation_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
