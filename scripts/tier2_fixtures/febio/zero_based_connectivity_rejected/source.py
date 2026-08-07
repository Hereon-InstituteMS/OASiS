"""Tier-2: a 0-based hex8 connectivity list — what actually happens.

Verifies febio::linear_elasticity#6, and FALSIFIES half of it. The
pitfall says a 0-based connectivity list (what meshio / PyVista export
without an explicit +1) "reaches N O R M A L  T E R M I N A T I O N with
exit 0 and a single WARNING box reading `1 isolated vertex removed.`; the
stresses are wrong".

Executed on structured hex8 unit cubes at n = 1, 2, 3 and 4, every entry
of every element shifted down by one:

  * the WARNING box is real — `1 isolated vertex removed.` appears, and
    that is the last node, which nothing now references,
  * but the run does NOT reach normal termination. It reads SUCCESS,
    prints the warning, then an ERROR box reading `Negative jacobian
    detected during mesh initialization.`, then `Model initialization
    failed`, and exits 1.

The shift is a translation of every element by one node in the lattice
ordering, so the elements that wrap a row are twisted, which is what the
negative jacobian is. So this is NOT a silent wrong answer on a
structured mesh: it is caught before the first step. The pitfall's
advice (assert min(connectivity) == min(node ids) before writing the
deck) stands; its Signal did not.

The fixture pins what was executed, on all four meshes, and requires the
positive control to run clean on each.

MUTATION CONTROL. T2_MUTATE=1 leaves the connectivity 1-BASED in the
"wrong" slot — the pathology removed. No isolated-vertex warning, no
negative jacobian, no failed initialisation, so
'zero_based_connectivity=rejected_at_init' is no longer printed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

import re


def shift_down(mesh: str) -> str:
    out = re.sub(r'(<elem id="\d+">)([\d,]+)</elem>',
                 lambda m: m.group(1) + ",".join(
                     str(int(x) - 1) for x in m.group(2).split(","))
                 + "</elem>", mesh)
    if out == mesh:
        L.die("connectivity shift did not apply — the mesh builder "
              "changed shape, so nothing was triggered")
    return out


def main() -> int:
    rows = []
    for n in (1, 2, 3, 4):
        mesh, _info = L.hex8_box(n)
        if MUTATE and n == 1:
            print("mutation=the_wrong_slot_keeps_a_1_based_"
                  "connectivity")
        w = L.run(L.solid_deck(
            mesh=mesh if MUTATE else shift_down(mesh), n=n))
        r = L.run(L.solid_deck(mesh=mesh, n=n))
        warn = "1 isolated vertex removed." in w.text
        neg = "Negative jacobian detected during mesh initialization." in w.text
        init = "Model initialization failed" in w.text
        rows.append((n, w.read_success, warn, neg, init, w.rc,
                     w.normal_termination, r.rc, r.normal_termination))
        print(f"n={n} wrong: read_success={int(w.read_success)} "
              f"isolated_vertex_warning={int(warn)} "
              f"negative_jacobian_at_init={int(neg)} "
              f"model_init_failed={int(init)} rc={w.rc} "
              f"normal_termination={int(w.normal_termination)}")
        print(f"n={n} right: rc={r.rc} "
              f"normal_termination={int(r.normal_termination)}")
    good = all(rs and warn and neg and init and rc != 0 and not nt
               and rrc == 0 and rnt
               for _n, rs, warn, neg, init, rc, nt, rrc, rnt in rows)
    print("falsified_claim=normal_termination_with_exit_0 "
          f"observed_on_any_mesh={int(any(nt for *_x, nt, _a, _b in rows))}")
    return L.report(good, "zero_based_connectivity", "rejected_at_init",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
