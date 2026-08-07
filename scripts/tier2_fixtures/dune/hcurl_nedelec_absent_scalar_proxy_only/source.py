"""Tier-2: there is no H(curl) / Nedelec space in dune-fem (maxwell#0).

The claim is that looking for one raises ImportError and that the
shipped 'maxwell' template is a SCALAR proxy. Both halves are checked:
every plausible spelling is probed against the real space registry, and
the space families that DO exist are enumerated from the module so a
future addition of Nedelec elements makes this fixture fail rather than
quietly stay true.

Nothing here builds a weak form, so nothing JIT-compiles.

Verified by execution against dune-fem 2.12.0.2.
"""
from __future__ import annotations

import importlib
import sys
import warnings

warnings.filterwarnings("ignore")

from dune.grid import structuredGrid                           # noqa: E402
import dune.fem.space as dspace                                # noqa: E402

# Names an agent reaching for vector Maxwell would try.
_HCURL_SPELLINGS = ("nedelec", "Nedelec", "hcurl", "hCurl", "HCurl",
                    "curl", "edge", "whitney")


def main() -> int:
    fail: list[str] = []

    for name in _HCURL_SPELLINGS:
        present = hasattr(dspace, name)
        print(f"space_has_{name}={present}")
        if present:
            fail.append(f"dune.fem.space.{name} exists; the claim that "
                        f"H(curl) Nedelec elements are absent is stale "
                        f"and the maxwell knowledge needs rewriting")

    for mod in ("dune.fem.space.nedelec", "dune.fem.space.hcurl"):
        try:
            importlib.import_module(mod)
            print(f"import_{mod.split('.')[-1]}_raises=False")
            fail.append(f"{mod} is importable")
        except ImportError as exc:
            print(f"import_{mod.split('.')[-1]}_raises="
                  f"{type(exc).__name__}")

    # What DOES exist: the H(div) families and the scalar/DG families.
    factories = sorted(
        n for n in dir(dspace)
        if not n.startswith("_") and callable(getattr(dspace, n))
        and n in {"lagrange", "lagrangehp", "dglagrange", "dgonb",
                  "dglegendre", "dganisotropic", "finiteVolume",
                  "raviartThomas", "bdm", "bdfm", "p1Bubble",
                  "rannacherTurek", "combined", "composite", "product"})
    print(f"available_space_factories={factories}")
    for needed in ("lagrange", "raviartThomas", "bdm"):
        if needed not in factories:
            fail.append(f"the space registry lost {needed!r}, so the "
                        f"enumeration this fixture compares against is "
                        f"no longer trustworthy")

    # The proxy the maxwell template actually uses is a SCALAR Lagrange
    # space — dimRange 1, not a vector-valued edge element.
    gridView = structuredGrid([0, 0], [1, 1], [4, 4])
    proxy = dspace.lagrange(gridView, order=2)
    print(f"maxwell_proxy_dimRange={proxy.dimRange}")
    print(f"maxwell_proxy_is_scalar={proxy.dimRange == 1}")
    if proxy.dimRange != 1:
        fail.append("the scalar proxy is no longer scalar")

    # And the honest limit of the proxy: a vector-valued Lagrange space
    # exists, but it is a Lagrange space, so it cannot represent the
    # tangential-continuity-only conformity H(curl) needs. That is
    # checkable structurally: the vector space's element is still
    # Lagrange.
    vec = dspace.lagrange(gridView, order=2, dimRange=2)
    elem = str(vec.ufl_element())
    print(f"vector_space_element={elem}")
    print(f"vector_space_is_lagrange_not_edge="
          f"{'agrange' in elem}")
    if "agrange" not in elem:
        fail.append(f"the vector space's UFL element is {elem!r}; the "
                    f"claim that the only vector option is a Lagrange "
                    f"proxy needs re-checking")

    if not fail:
        print("dune_no_hcurl_nedelec_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
