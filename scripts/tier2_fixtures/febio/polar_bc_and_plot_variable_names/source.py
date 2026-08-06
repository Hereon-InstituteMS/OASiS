"""Tier-2: `zero fluid angular velocity` and `polar fluid angular
velocity` — and the two wrong names fail DIFFERENTLY.

Verifies febio::polar_fluid#3. The point of the claim is that the two
mistakes are caught at different stages, which changes how a wrapper has
to watch for them:

  * a made-up BC type is a PARSE error —
    `tag "bc" (line N) : invalid value for attribute "type"`,
  * a made-up plot variable is caught only AFTER the reader says SUCCESS,
    with `FATAL ERROR: Output variable "micro rotation" is not defined`.

The fixture asserts the reader said SUCCESS on the second one, which is
the half a reader would not predict.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

def main() -> int:
    right = L.template("polar_fluid_3d_channel")
    bad_bc = L.swap(right, 'type="zero fluid angular velocity"',
                    'type="zero micro-rotation"')
    bad_var = L.swap(right, '<var type="polar fluid angular velocity"/>',
                     '<var type="micro rotation"/>')
    wb = L.run(bad_bc)
    wv = L.run(bad_var)
    r = L.run(right)

    bc_msg = ('tag "bc"' in wb.text
              and 'invalid value for attribute "type"' in wb.text)
    var_msg = ('FATAL ERROR: Output variable "micro rotation" is not '
               'defined') in wv.text
    print(f"bad_bc_type: rc={wb.rc} read_failed={int(wb.read_failed)} "
          f"parse_error={int(bc_msg)}")
    print(f"bad_plot_var: rc={wv.rc} read_success={int(wv.read_success)} "
          f"read_failed={int(wv.read_failed)} fatal_after_read={int(var_msg)}")
    print(f"right: rc={r.rc} normal={int(r.normal_termination)} "
          f"steps={r.steps_completed}")
    good = (bc_msg and wb.read_failed and wb.rc != 0
            and var_msg and wv.read_success and not wv.read_failed
            and wv.rc != 0
            and r.rc == 0 and r.normal_termination
            and "FATAL ERROR" not in r.text)
    return L.report(good, "polar_names", "reproduced", "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
