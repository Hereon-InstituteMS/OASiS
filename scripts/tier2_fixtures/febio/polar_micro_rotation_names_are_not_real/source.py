"""Tier-2: `micro_rotation` is not a plot variable and `zero
micro-rotation` is not a BC — and the angular-velocity field cannot be
logged at all on this build.

Verifies febio::polar_fluid#0. Three executed observations:

  * <var type="micro_rotation"/> reads SUCCESS and then aborts with
    `FATAL ERROR: Output variable "micro_rotation" is not defined`,
  * <bc type="zero micro-rotation"> is a parse error,
  * and — the part that limits the claim — the micro-rotation DOFs cannot
    be written to a logfile: <node_data data="gvx"/> is refused with
    `"gvx" is not a valid field variable name`. So the pitfall's advice to
    "read the field" means reading the .xplt, not a CSV.

What the fixture CAN measure about the wall BC is the flow field:
deleting the `zero fluid angular velocity` BC leaves the logged element
data almost unchanged — far below the level at which it would signal a
different solution — which is why a quiet result is not evidence the BC
is right.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

OUT = ("  <Output>\n    <logfile>\n"
       '      <element_data data="fJ;fp;fsxx" delim="," file="e.csv"/>\n'
       "    </logfile>\n  </Output>\n")
WALL_BC = ('    <bc name="no_microrot" type="zero fluid angular velocity" '
           'node_set="walls">\n'
           "      <gx_dof>1</gx_dof><gy_dof>1</gy_dof>"
           "<gz_dof>1</gz_dof>\n"
           "    </bc>\n")


def with_log(deck: str) -> str:
    return L.swap(deck, "</febio_spec>", OUT + "</febio_spec>")


def series(run):
    out = []
    for _t, rows in L.parse_log_csv(run.files.get("e.csv") or ""):
        for nid in sorted(rows):
            out.extend(rows[nid])
    return out


def main() -> int:
    base = L.template("polar_fluid_3d_channel")

    var = L.run(L.swap(base, '<var type="polar fluid angular velocity"/>',
                       '<var type="micro_rotation"/>'), timeout=900)
    var_ok = (var.read_success and not var.read_failed and var.rc != 0
              and var.has('FATAL ERROR: Output variable "micro_rotation" '
                          "is not defined"))
    bc = L.run(L.swap(base, 'type="zero fluid angular velocity"',
                      'type="zero micro-rotation"'), timeout=900)
    bc_ok = ('tag "bc"' in bc.text
             and 'invalid value for attribute "type"' in bc.text
             and bc.read_failed and bc.rc != 0)
    gv = L.run(L.swap(with_log(base), '<element_data data="fJ;fp;fsxx"',
                      '<node_data data="gvx"'), timeout=900)
    gv_ok = gv.has('"gvx" is not a valid field variable name')

    print(f"plot_variable_micro_rotation: rc={var.rc} "
          f"read_success={int(var.read_success)} "
          f"fatal_after_read={int(var_ok)}")
    print(f"bc_zero_micro_rotation: rc={bc.rc} "
          f"read_failed={int(bc.read_failed)} parse_error={int(bc_ok)}")
    print(f"logfile_gvx: rc={gv.rc} "
          f"not_a_valid_field_variable={int(gv_ok)}")

    a = L.run(with_log(base), collect=("e.csv",), timeout=900)
    b = L.run(with_log(base), collect=("e.csv",), timeout=900)
    c = L.run(with_log(L.drop(base, WALL_BC)), collect=("e.csv",),
              timeout=900)
    sa, sb, sc = series(a), series(b), series(c)

    def dev(x, y):
        if not x or len(x) != len(y):
            return float("inf")
        return max(abs(p - q) / max(abs(p), abs(q), 1e-12)
                   for p, q in zip(x, y))

    noise = dev(sa, sb)
    without = dev(sa, sc)
    print(f"identical_runs_max_rel_dev={noise:.3e} "
          f"bit_identical={int(sa == sb)}")
    print(f"wall_bc_deleted_max_rel_dev={without:.3e} "
          f"rc={c.rc} normal={int(c.normal_termination)}")
    barely_moves = without < 1e-4
    print(f"deck_barely_notices_the_wall_bc={int(barely_moves)}")
    good = (var_ok and bc_ok and gv_ok and barely_moves
            and a.rc == 0 and a.normal_termination
            and c.rc == 0 and c.normal_termination)
    return L.report(good, "polar_micro_rotation_names", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
