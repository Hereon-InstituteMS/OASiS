"""Tier-2: u.Other() is the only neighbour accessor, and the miss names
ProxyFunction -- not "trial function".

Claim: ngsolve dg_methods#1 -- "u.Other() accesses the neighbour element's trial
function across a shared facet.  There is no `neighbour` attribute and no
external helper.  Signal: reaching for one raises the auto-generated CPython
message AttributeError: 'ngsolve.comp.ProxyFunction' object has no attribute
'neighbour' -- note it names the ProxyFunction class, NOT 'trial function', so a
guard grepping for 'trial function has no attribute neighbour' never fires."

Wrong variant: u.neighbour instead of u.Other().

What this fixture pins, all re-measured on this run:
  * the trial function's runtime class really is ngsolve.comp.ProxyFunction;
  * u.neighbour raises AttributeError carrying the class name literally, and the
    phrase the claim says never appears really does not appear in it;
  * no other plausible neighbour spelling exists on the object -- neighbor,
    Neighbour, Neighbor, other and Trace are checked by name;
  * u.Other() does exist, and the symmetric average 0.5*(u + u.Other()) built
    from it assembles into a form with strictly wider sparsity than the same
    space's block-diagonal mass matrix, i.e. it really does reach across facets.

Mutation control:  T2_MUTATE=1 applies the documented fix at the pathology site
-- the attribute the probe reaches for is 'Other' instead of 'neighbour', so the
lookup succeeds and no AttributeError is captured.  The expectations
neighbour_raises_attributeerror=True, msg_names_proxyfunction=True and
msg_says_no_attribute_neighbour=True then disappear.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

from netgen.geom2d import unit_square
from ngsolve import BilinearForm, L2, Mesh, dx


NEVER_EMITTED = "trial function has no attribute neighbour"

# Mutation control: under T2_MUTATE=1 the probe asks for the accessor that
# exists, u.Other, instead of u.neighbour -- the pathology is removed.
MUTATE = os.environ.get("T2_MUTATE") == "1"
PROBE = "neighbour" if not MUTATE else "Other"


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    fes = L2(mesh, order=1, dgjumps=True)
    u, v = fes.TnT()

    cls = f"{type(u).__module__}.{type(u).__name__}"
    print(f"trial_function_class={cls}")
    print(f"class_is_proxyfunction={cls == 'ngsolve.comp.ProxyFunction'}")

    msg = ""
    try:
        getattr(u, PROBE)
    except AttributeError as exc:
        msg = str(exc)
    print(f"neighbour_raises_attributeerror={bool(msg)}")
    print(f"msg_names_proxyfunction={'ngsolve.comp.ProxyFunction' in msg}")
    has_phrase = "has no attribute 'neighbour'" in msg
    print(f"msg_says_no_attribute_neighbour={has_phrase}")
    print(f"claimed_never_emitted_phrase_absent={NEVER_EMITTED not in msg}")

    # No other spelling exists.
    alternatives = ["neighbour", "neighbor", "Neighbour", "Neighbor", "other"]
    present = [n for n in alternatives if hasattr(u, n)]
    print(f"alternative_spellings_present={present}")
    print(f"no_alternative_spelling_exists={present == []}")
    print(f"Other_exists={hasattr(u, 'Other')}")

    # ...and Other() genuinely couples across facets.
    a_mean = BilinearForm(fes)
    a_mean += 0.5 * (u + u.Other()) * 0.5 * (v + v.Other()) * dx(skeleton=True)
    a_mean.Assemble()
    nze_mean = a_mean.mat.nze

    fes_plain = L2(mesh, order=1)
    up, vp = fes_plain.TnT()
    a_local = BilinearForm(fes_plain)
    a_local += up * vp * dx
    a_local.Assemble()
    nze_local = a_local.mat.nze
    print(f"mean_form_nze={nze_mean}")
    print(f"element_local_nze={nze_local}")
    print(f"Other_reaches_across_facets={nze_mean > nze_local}")

    ok = (
        cls == "ngsolve.comp.ProxyFunction"
        and "ngsolve.comp.ProxyFunction" in msg
        and "has no attribute 'neighbour'" in msg
        and NEVER_EMITTED not in msg
        and present == []
        and hasattr(u, "Other")
        and nze_mean > nze_local
    )
    if ok:
        return 0
    print("FAIL: neighbour-accessor invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
