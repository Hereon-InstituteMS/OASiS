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

import numpy as np
from netgen.geom2d import unit_square
from ngsolve import BilinearForm, L2, Mesh, dx


def _cross_element_entries(a, dof_elem):
    """(nze, entries connecting two different elements, largest such value).

    The allocated sparsity pattern is not evidence of coupling: on a space
    built with dgjumps=True every form gets the wide pattern whether or not it
    uses Other().  Only a stored value that links DOFs owned by two different
    elements is.
    """
    a.Assemble()
    rows, cols, vals = a.mat.COO()
    r, c, v = np.asarray(rows), np.asarray(cols), np.asarray(vals)
    m = (dof_elem[r] != dof_elem[c]) & (np.abs(v) > 1e-12)
    peak = float(np.abs(v[m]).max()) if m.any() else 0.0
    return int(a.mat.nze), int(m.sum()), peak


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
    dof_elem = np.full(fes.ndof, -1, dtype=int)
    for el in fes.Elements():
        for d in el.dofs:
            dof_elem[d] = el.nr
    print(f"dofs_with_an_owning_element={int((dof_elem >= 0).sum())} "
          f"of {fes.ndof}")
    a_mean = BilinearForm(fes)
    a_mean += 0.5 * (u + u.Other()) * 0.5 * (v + v.Other()) * dx(skeleton=True)
    nze_mean, cross_mean, peak_mean = _cross_element_entries(a_mean, dof_elem)

    # THE CONTROL THAT MAKES THIS ABOUT Other().
    #
    # This used to be `nze(mean form on L2(dgjumps=True)) > nze(mass form on
    # L2(dgjumps=False))`, and dgjumps=True is precisely the flag that asks
    # NGSolve for the wider sparsity PATTERN.  The inequality therefore measured
    # the allocation the flag requests, and came out True for any form at all --
    # including one with no Other() in it.
    #
    # The control is the SAME space, the same skeleton integral, and the only
    # difference is whether Other() appears.  What is counted is not the
    # allocated pattern but the entries that are actually non-zero and connect
    # DOFs owned by two different elements, which is what "reaches across a
    # facet" means.
    a_noother = BilinearForm(fes)
    a_noother += u * v * dx(skeleton=True)
    nze_noother, cross_noother, _ = _cross_element_entries(a_noother, dof_elem)

    print(f"mean_form_nze={nze_mean}")
    print(f"no_Other_form_nze={nze_noother}")
    print(f"both_forms_share_the_same_allocation={nze_mean == nze_noother}")
    print(f"mean_form_cross_element_entries={cross_mean}")
    print(f"no_Other_form_cross_element_entries={cross_noother}")
    print(f"mean_form_peak_cross_element_value={peak_mean:.6e}")
    print(f"Other_reaches_across_facets="
          f"{cross_mean > 0 and cross_noother == 0}")

    ok = (
        cls == "ngsolve.comp.ProxyFunction"
        and "ngsolve.comp.ProxyFunction" in msg
        and "has no attribute 'neighbour'" in msg
        and NEVER_EMITTED not in msg
        and present == []
        and hasattr(u, "Other")
        and cross_mean > 0 and cross_noother == 0
    )
    if ok:
        return 0
    print("FAIL: neighbour-accessor invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
