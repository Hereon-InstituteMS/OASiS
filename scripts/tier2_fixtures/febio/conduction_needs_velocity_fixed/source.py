"""Tier-2: a pure-conduction thermo-fluid deck must fix every fluid
velocity DOF.

Verifies febio::heat#4, and CORRECTS one clause of it. thermo-fluid
solves momentum and energy together, so leaving wx/wy/wz free leaves the
momentum problem unconstrained and Newton never converges: the deck reads
SUCCESS, then `------- failed to converge at time : 1`, zero completed
steps, `E R R O R  T E R M I N A T I O N`, exit 1. All of that
reproduces.

What does NOT reproduce is the pitfall's closing clause, "the logged
temperature field is all zeros". The failed run still writes its logfile
CSV, but it contains only the `*Step = 0` block, and that block holds the
INITIAL CONDITION — 300 everywhere on the shipped deck, not zeros. The
field is zeros only if the `<Initial>` block is also removed. The fixture
executes both variants and pins each, because the difference decides
whether "all zeros" is a usable detection rule (it is not).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

VEL = ('    <bc name="at_rest" type="zero fluid velocity" '
       'node_set="all_nodes">\n'
       "      <wx_dof>1</wx_dof><wy_dof>1</wy_dof><wz_dof>1</wz_dof>\n"
       "    </bc>\n")
INIT = ("  <Initial>\n"
        '    <ic name="T0" type="initial fluid temperature" '
        'node_set="all_nodes">\n'
        "      <value>300.0</value>\n"
        "    </ic>\n"
        "  </Initial>\n")
CSV = "heat_bar_T.csv"


def field(run) -> list:
    blocks = L.parse_log_csv(run.files.get(CSV) or "")
    return [v[0] for v in blocks[-1][1].values()] if blocks else []


def main() -> int:
    right = L.template("heat_3d_bar", n=2)
    r = L.run(right, collect=(CSV,))
    w = L.run(L.drop(right, VEL), collect=(CSV,))
    w2 = L.run(L.drop(L.drop(right, VEL), INIT), collect=(CSV,))

    diverged = ("------- failed to converge at time : 1" in w.text
                and w.steps_completed == 0 and w.error_termination
                and w.read_success and w.rc != 0)
    blocks = L.parse_log_csv(w.files.get(CSV) or "")
    with_init = field(w)
    without_init = field(w2)
    print(f"free_velocity: rc={w.rc} read_success={int(w.read_success)} "
          f"failed_to_converge={int('------- failed to converge at time : 1' in w.text)} "
          f"steps={w.steps_completed} "
          f"error_termination={int(w.error_termination)}")
    print(f"csv_blocks_written={len(blocks)} "
          f"field_with_Initial={sorted(set(with_init))} "
          f"field_without_Initial={sorted(set(without_init))}")
    print(f"control: rc={r.rc} normal={int(r.normal_termination)} "
          f"steps={r.steps_completed} "
          f"field={sorted(set(field(r)))[:4]}")

    # The corrected claim: with <Initial> present the logged field is the
    # initial condition, NOT zeros; only with <Initial> removed is it zeros.
    init_held = bool(with_init) and set(with_init) == {300.0}
    zeros_only_without_init = bool(without_init) and set(without_init) == {0.0}
    print(f"all_zeros_claim_holds_only_without_Initial="
          f"{int(init_held and zeros_only_without_init)}")

    good = (diverged and init_held and zeros_only_without_init
            and w2.rc != 0 and r.rc == 0 and r.normal_termination
            and "failed to converge" not in r.text)
    return L.report(good, "thermofluid_free_velocity",
                    "reproduced_with_correction", "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
