"""Tier-2: `kinematic growth` is a TOP-LEVEL material with two required
children, and each way of getting it wrong fails differently.

Verifies febio::growth_remodeling#0 — four distinct failures from one
claim, which is why they belong in one fixture:

  * wrapped in a `solid mixture`: READS successfully, then dies at the
    first increment — but NOT the way the claim said. See below.
  * <growth> or <elastic> deleted: caught at read time, each named,
  * an UNCOUPLED elastic child: caught at initialisation with
    `Elastic material should not be of type uncoupled`,
  * an unregistered type name: `tag "material" ... invalid value`.

Four different stages of the pipeline; a wrapper that watches only one of
them sees three of these as silence.

FALSIFICATION. The claim says the mixture form dies with
`N negative jacobians detected.`. It does not. Executed on the shipped
template: the run reads SUCCESS, the Newton loop hits
`Max nr of iterations reached.` and reforms the stiffness repeatedly, then
`------- failed to converge at time`, `Number of time steps completed
.... : 0` and `E R R O R  T E R M I N A T I O N`, exit 1. No element
inverts and the string `negative jacobians detected.` never appears. The
fixture asserts its ABSENCE, so the wrong Signal cannot be re-introduced.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    # MUTATION CONTROL. The pathology this fixture reproduces is the
    # EDIT that turns a correct deck into the broken one, so removing
    # the pathology means not making that edit. Neutralising L.swap /
    # L.drop does exactly that: every deck built below is the correct
    # one, the pitfall is never triggered, the diagnostic must not
    # appear and the verdict token flips to not_reproduced.
    print("mutation=the_deck_edits_that_introduce_the_pitfall_are_"
          "neutralised")
    L.swap = lambda deck, old, new, **kw: deck
    L.drop = lambda deck, fragment: deck


ELASTIC = ('      <elastic type="neo-Hookean">\n'
           "        <density>1.0</density>\n"
           "        <E>1000.0</E>\n"
           "        <v>0.3</v>\n"
           "      </elastic>\n")
GROWTH = ('      <growth type="volume growth">\n'
          '        <multiplier lc="1">1.0</multiplier>\n'
          "      </growth>\n")
UNCOUPLED = ('      <elastic type="Mooney-Rivlin">\n'
             "        <density>1.0</density>\n"
             "        <c1>1.0</c1>\n"
             "        <c2>0.0</c2><k>1000.0</k>\n"
             "      </elastic>\n")


def main() -> int:
    base = L.template("growth_remodeling_3d_isotropic")
    r = L.run(base, timeout=400)

    named = 0
    for prop, block in (("growth", GROWTH), ("elastic", ELASTIC)):
        w = L.run(L.drop(base, block))
        hit = w.has(f'Component "Material1" needs to have property '
                    f'"{prop}" defined')
        print(f"missing_{prop}: rc={w.rc} read_failed={int(w.read_failed)} "
              f"named={int(hit)}")
        if hit and w.read_failed and w.rc != 0:
            named += 1

    unc = L.run(L.swap(base, ELASTIC, UNCOUPLED), timeout=400)
    unc_ok = (unc.has("Elastic material should not be of type uncoupled")
              and unc.has("Model initialization failed")
              and unc.read_success and unc.rc != 0)
    print(f"uncoupled_elastic_child: rc={unc.rc} "
          f"read_success={int(unc.read_success)} "
          f"initialisation_message={int(unc_ok)}")

    bad = L.run(L.swap(base, 'type="kinematic growth"', 'type="growth"'))
    bad_ok = ('tag "material"' in bad.text
              and 'invalid value for attribute "type"' in bad.text
              and bad.read_failed and bad.rc != 0)
    print(f"unregistered_type: rc={bad.rc} "
          f"read_failed={int(bad.read_failed)} invalid_type={int(bad_ok)}")

    inner = ELASTIC + GROWTH
    mixture = L.swap(
        base,
        '    <material id="1" name="Material1" type="kinematic growth">\n'
        "      <density>1.0</density>\n" + inner,
        '    <material id="1" name="Material1" type="solid mixture">\n'
        "      <density>1.0</density>\n"
        '      <solid type="kinematic growth">\n' + inner + "      </solid>\n")
    mx = L.run(mixture, timeout=600)
    mx_ok = (mx.read_success
             and "negative jacobians detected." not in mx.text
             and mx.has("Max nr of iterations reached.")
             and "------- failed to converge at time" in mx.text
             and mx.error_termination
             and mx.steps_completed == 0 and mx.rc != 0)
    print(f"wrapped_in_solid_mixture: rc={mx.rc} "
          f"read_success={int(mx.read_success)} "
          f"max_iterations_reached="
          f"{int(mx.has('Max nr of iterations reached.'))} "
          f"failed_to_converge="
          f"{int('------- failed to converge at time' in mx.text)} "
          f"error_termination={int(mx.error_termination)} "
          f"steps={mx.steps_completed} "
          f"negative_jacobians_ABSENT="
          f"{int('negative jacobians detected.' not in mx.text)}")
    print(f"control: rc={r.rc} normal={int(r.normal_termination)} "
          f"steps={r.steps_completed}")
    print(f"properties_named={named} of 2")
    good = (named == 2 and unc_ok and bad_ok and mx_ok
            and r.rc == 0 and r.normal_termination)
    if not good:
        print(mx.text[:1200])
    return L.report(good, "kinematic_growth_shape", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
