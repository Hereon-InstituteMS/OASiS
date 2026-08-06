"""Tier-2: `elastic damage` needs THREE properties and each missing one is
reported by its own name.

Verifies febio::damage#1. Five variants in one run:

  * type="damage" — there is no such material,
  * <elastic>, <damage>, <criterion> each deleted in turn, each reported
    as `Component "Material1" needs to have property "<name>" defined`,
  * elastic parameters written flat at the top level —
    `tag "E" ... unrecognized tag`.

The three property names are what make the message actionable, so the
fixture requires the quoted name to CHANGE with the deleted property
rather than merely requiring the message to appear.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

ELASTIC = ('      <elastic type="neo-Hookean">\n'
           "        <density>1.0</density>\n"
           "        <E>1000.0</E>\n"
           "        <v>0.3</v>\n"
           "      </elastic>\n")
DAMAGE = ('      <damage type="CDF Weibull">\n'
          "        <Dmax>0.9</Dmax>\n"
          "        <alpha>2.0</alpha>\n"
          "        <mu>0.5</mu>\n"
          "      </damage>\n")
CRITERION = '      <criterion type="DC strain energy density"/>\n'


def main() -> int:
    base = L.template("damage_3d_cycle")
    r = L.run(base, timeout=600)
    bad_type = L.run(L.swap(base, 'type="elastic damage"', 'type="damage"'))
    t_ok = ('tag "material"' in bad_type.text
            and 'invalid value for attribute "type"' in bad_type.text)
    print(f"unregistered_type_damage: rc={bad_type.rc} "
          f"read_failed={int(bad_type.read_failed)} invalid_type={int(t_ok)}")

    named = 0
    for prop, block in (("elastic", ELASTIC), ("damage", DAMAGE),
                        ("criterion", CRITERION)):
        w = L.run(L.drop(base, block))
        this = w.has(f'Component "Material1" needs to have property '
                     f'"{prop}" defined')
        # The quoted name must be THIS property, not another one.
        others = [p for p in ("elastic", "damage", "criterion") if p != prop]
        only_this = not any(
            w.has(f'needs to have property "{o}" defined') for o in others)
        print(f"missing_{prop}: rc={w.rc} read_failed={int(w.read_failed)} "
              f"names_this_property={int(this)} "
              f"names_no_other={int(only_this)}")
        if this and only_this and w.read_failed and w.rc != 0:
            named += 1

    flat = L.run(L.swap(base, "      <density>1.0</density>\n      <elastic",
                        "      <density>1.0</density>\n      <E>10</E>\n"
                        "      <elastic"))
    f_ok = 'tag "E"' in flat.text and "unrecognized tag" in flat.text
    print(f"flat_parameters: rc={flat.rc} "
          f"read_failed={int(flat.read_failed)} unrecognized_tag={int(f_ok)}")
    print(f"control: rc={r.rc} normal={int(r.normal_termination)} "
          f"steps={r.steps_completed}")
    print(f"properties_named_individually={named} of 3")
    good = (t_ok and bad_type.rc != 0 and named == 3 and f_ok
            and flat.rc != 0 and r.rc == 0 and r.normal_termination)
    return L.report(good, "elastic_damage_properties", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
