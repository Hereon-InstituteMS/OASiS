"""Tier-2: <SurfacePair> takes <primary>/<secondary>, and the 2.x contact
names are gone.

Verifies febio::rigid_body#5 — four distinct rejections, each with its own
message, on the shipped contact template:

  * <master>/<slave> — `tag "master" ... unrecognized tag`,
  * a 2.x type string and a mis-hyphenated one —
    `tag "contact" ... invalid value for attribute "type"`,
  * a surface_pair naming a pair that does not exist —
    `tag "contact" ... invalid value for attribute "surface_pair"`,
  * <contact ...></contact> with an empty body —
    `tag "contact" ... unrecognized tag`.

The last two share the tag name and differ only in the rest of the
message, which is why the fixture matches the whole message rather than
the tag.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

CONTACT = ('<contact type="sliding-elastic" surface_pair="PunchOnBlock">\n'
           "      <laugon>PENALTY</laugon>\n"
           "      <penalty>10.0</penalty>\n"
           "      <auto_penalty>1</auto_penalty>\n"
           "      <two_pass>1</two_pass>\n"
           "      <tolerance>0.1</tolerance>\n"
           "      <search_radius>1.0</search_radius>\n"
           "    </contact>")


def main() -> int:
    base = L.template("rigid_contact_3d_indentation")
    cases = {
        "master_slave": (
            L.swap(base,
                   "<primary>BlockTop</primary>\n"
                   "      <secondary>PunchBottom</secondary>",
                   "<master>BlockTop</master>\n"
                   "      <slave>PunchBottom</slave>"),
            'tag "master"', "unrecognized tag"),
        "space_not_hyphen": (
            L.swap(base, 'type="sliding-elastic"',
                   'type="sliding elastic"'),
            'tag "contact"', 'invalid value for attribute "type"'),
        "febio2_type_string": (
            L.swap(base, 'type="sliding-elastic"',
                   'type="facet-to-facet sliding"'),
            'tag "contact"', 'invalid value for attribute "type"'),
        "unknown_surface_pair": (
            L.swap(base, 'surface_pair="PunchOnBlock"',
                   'surface_pair="NOPE"'),
            'tag "contact"', 'invalid value for attribute "surface_pair"'),
        "empty_contact_body": (
            L.swap(base, CONTACT,
                   '<contact type="sliding-elastic" '
                   'surface_pair="PunchOnBlock"></contact>'),
            'tag "contact"', "unrecognized tag"),
    }
    hit = 0
    for name, (deck, tag, rest) in cases.items():
        w = L.run(deck, timeout=900)
        ok = tag in w.text and rest in w.text
        print(f"{name}: rc={w.rc} read_failed={int(w.read_failed)} "
              f"message={int(ok)}")
        if ok and w.read_failed and w.rc != 0:
            hit += 1
    r = L.run(base, timeout=900)
    print(f"control: rc={r.rc} normal={int(r.normal_termination)} "
          f"steps={r.steps_completed}")
    print(f"rejected_with_the_quoted_message={hit} of {len(cases)}")
    good = (hit == len(cases) and r.rc == 0 and r.normal_termination
            and 'tag "contact"' not in r.text)
    return L.report(good, "contact_syntax", "reproduced", "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
