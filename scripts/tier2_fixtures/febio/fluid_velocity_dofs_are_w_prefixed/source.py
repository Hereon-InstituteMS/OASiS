"""Tier-2: fluid velocity DOFs are wx/wy/wz, and a solid-style name is
indistinguishable from an invented one.

Verifies febio::fluid#1. The claim worth executing is not that <y_dof> is
rejected — it is that the message for a REAL-but-wrong name and for a
name nobody has ever used are the same shape, so the diagnostic gives no
hint that a w prefix was wanted. The fixture runs both and requires the
two messages to be identical once the tag name is factored out.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

import re


def message(run) -> str | None:
    m = re.search(r'tag "([^"]+)" \(line \d+\) : (.+?)\s*\*', run.text)
    return None if m is None else f"{m.group(1)}|{m.group(2).strip()}"


def main() -> int:
    right = L.template("fluid_3d_channel")
    solidish = L.swap(right, "<wy_dof>1</wy_dof>", "<y_dof>1</y_dof>")
    invented = L.swap(right, "<wy_dof>1</wy_dof>", "<qq_dof>1</qq_dof>")
    ws = L.run(solidish)
    wi = L.run(invented)
    r = L.run(right)
    ms, mi = message(ws), message(wi)
    print(f"solid_style_y_dof: rc={ws.rc} read_failed={int(ws.read_failed)} "
          f"message={ms}")
    print(f"invented_qq_dof:   rc={wi.rc} read_failed={int(wi.read_failed)} "
          f"message={mi}")
    print(f"right: rc={r.rc} normal={int(r.normal_termination)} "
          f"steps={r.steps_completed}")
    both = (ms is not None and mi is not None
            and ms.split("|", 1)[0] == "y_dof"
            and mi.split("|", 1)[0] == "qq_dof"
            and ms.split("|", 1)[1] == mi.split("|", 1)[1]
            == "unrecognized tag")
    print(f"same_message_modulo_tag_name={int(both)}")
    good = (both and ws.read_failed and wi.read_failed
            and ws.rc != 0 and wi.rc != 0
            and r.rc == 0 and r.normal_termination
            and "unrecognized tag" not in r.text)
    return L.report(good, "fluid_dof_prefix", "reproduced", "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
