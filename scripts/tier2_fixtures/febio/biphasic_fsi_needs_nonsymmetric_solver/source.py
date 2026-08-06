"""Tier-2: <solver type="fluid-FSI"/> on its own cannot store the matrix,
and `schur` will not even construct.

Verifies febio::biphasic_fsi#3. Two different failures from the same
section:

  * the bare solver inherits the skyline default, the deck READS
    successfully, and the run aborts with the matrix-format ERROR box and
    zero completed steps,
  * <linear_solver type="schur"/> is refused at parse with
    `Component "linear_solver" needs to have property "A_solver" defined`.

Matched on the short fragment for the first, because the ERROR box wraps
at 71 columns.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

SOLVER = ('    <solver type="fluid-FSI">\n'
          "      <symmetric_stiffness>non-symmetric"
          "</symmetric_stiffness>\n"
          '      <linear_solver type="bicgstab"/>\n'
          "    </solver>\n")


def main() -> int:
    right = L.template("biphasic_fsi_3d_block")
    bare = L.swap(right, SOLVER, '    <solver type="fluid-FSI"/>\n')
    schur = L.swap(right, '<linear_solver type="bicgstab"/>',
                   '<linear_solver type="schur"/>')
    wb = L.run(bare)
    ws = L.run(schur)
    r = L.run(right)
    fmt = (wb.has("does not support the requested")
           and wb.read_success and wb.steps_completed == 0
           and wb.error_termination)
    a_solver = ws.has('Component "linear_solver" needs to have property '
                      '"A_solver" defined')
    print(f"bare_solver: rc={wb.rc} read_success={int(wb.read_success)} "
          f"matrix_format_error={int(wb.has('does not support the requested'))} "
          f"steps={wb.steps_completed} "
          f"error_termination={int(wb.error_termination)}")
    print(f"schur: rc={ws.rc} read_failed={int(ws.read_failed)} "
          f"needs_A_solver={int(a_solver)}")
    print(f"right: rc={r.rc} normal={int(r.normal_termination)} "
          f"steps={r.steps_completed}")
    good = (fmt and wb.rc != 0 and a_solver and ws.rc != 0
            and r.rc == 0 and r.normal_termination
            and not r.has("does not support the requested"))
    return L.report(good, "biphasic_fsi_solver", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
