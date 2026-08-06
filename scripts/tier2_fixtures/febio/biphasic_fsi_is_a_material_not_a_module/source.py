"""Tier-2: <Module type="biphasic-FSI"/> SEGFAULTS with no diagnostic.

Verifies febio::biphasic_fsi#0. `biphasic-FSI` is registered as a MATERIAL
inside the fluid-FSI module, and naming it as a module hands an unknown
string to FEModelBuilder::SetActiveModule() with no existence check.

The observable is the process signal and the ABSENCE of everything else:
no `SUCCESS!`, no `FAILED!`, no ERROR box, no .log file. The fixture
asserts every one of those absences, because a wrapper that greps for the
word error sees a completely silent failure. The positive control is the
shipped template, which differs only in the module string.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

def main() -> int:
    right = L.template("biphasic_fsi_3d_block")
    wrong = L.swap(right, '<Module type="fluid-FSI"/>',
                   '<Module type="biphasic-FSI"/>')
    w = L.run(wrong)
    r = L.run(right)
    print(f"wrong: rc={w.rc} sigsegv={int(w.segfault)} "
          f"read_success={int(w.read_success)} "
          f"read_failed={int(w.read_failed)} "
          f"error_box={int('ERROR' in w.out)} "
          f"log_written={int(bool(w.log))} "
          f"stops_mid_line={int('Reading file in.feb ...' in w.out)}")
    print(f"right: rc={r.rc} read_success={int(r.read_success)} "
          f"normal={int(r.normal_termination)} steps={r.steps_completed}")
    silent = (w.segfault and not w.read_success and not w.read_failed
              and "ERROR" not in w.out and not w.log
              and "Reading file in.feb ..." in w.out)
    good = (silent and r.rc == 0 and r.read_success
            and r.normal_termination)
    if not good:
        print(w.out[:900])
    return L.report(good, "biphasic_fsi_module_segfault", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
