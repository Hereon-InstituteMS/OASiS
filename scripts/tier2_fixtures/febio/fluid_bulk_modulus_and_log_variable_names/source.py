"""Tier-2: <k> does not set the volume ratio in a dilatation-driven deck,
and the log-variable names are per RECORD TYPE.

Verifies febio::fluid#2, and corrects one detail. Two halves:

  * a sweep of the `fluid` material's <k> over four decades on the shipped
    channel deck leaves the logged volume ratio fJ and dilatation fd
    BIT-IDENTICAL — the inlet and outlet PRESCRIBE the dilatation, so k
    controls nothing there. Only the pressure fp moves. Reading an
    unchanged volume ratio as "k is ignored" is the mistake,
  * the rejection of a log variable is per record type, not per name:
    <element_data data="ef"/> is refused while <node_data data="ef"/> is
    ACCEPTED and runs to normal termination, because `ef` is the nodal
    dilatation DOF.

CORRECTION. The pitfall says the element_data rejection comes "after the
deck has already read successfully". It does not: the run prints
`Reading file ...FAILED!` and the message
`"ef" is not a valid field variable name (line N)` is part of the read
failure. The asymmetry between record types is real; the stage is not.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

CSV = "fluid_elem.csv"
SCALES = ("1e0", "1e2", "1e3", "1e4")


def volume_and_dilatation(run):
    """(fJ, fd) per element at the last step; fp deliberately excluded."""
    blocks = L.parse_log_csv(run.files.get(CSV) or "")
    if not blocks:
        return None, None
    rows = blocks[-1][1]
    return ([rows[k][0] for k in sorted(rows)],
            [rows[k][1] for k in sorted(rows)])


def pressure(run):
    blocks = L.parse_log_csv(run.files.get(CSV) or "")
    if not blocks:
        return None
    rows = blocks[-1][1]
    return [rows[k][2] for k in sorted(rows)]


def main() -> int:
    base = L.template("fluid_3d_channel")
    ref_J = ref_d = None
    pressures = {}
    invariant = True
    for k in SCALES:
        r = L.run(L.swap(base, "<k>1e3</k>", f"<k>{k}</k>"),
                  collect=(CSV,), timeout=900)
        fJ, fd = volume_and_dilatation(r)
        fp = pressure(r)
        if ref_J is None:
            ref_J, ref_d = fJ, fd
        same = (fJ == ref_J and fd == ref_d)
        invariant = invariant and same and r.rc == 0
        pressures[k] = fp[0] if fp else None
        print(f"k={k}: rc={r.rc} normal={int(r.normal_termination)} "
              f"fJ={fJ[0] if fJ else None} fd={fd[0] if fd else None} "
              f"fp={pressures[k]} volume_ratio_unchanged={int(same)}")
    moved = len({round(v, 12) for v in pressures.values()
                 if v is not None}) == len(pressures)
    print(f"volume_ratio_invariant_over_four_decades={int(invariant)}")
    print(f"pressure_does_move_with_k={int(moved)}")

    elem = L.run(L.swap(base, '<element_data data="fJ;fd;fp"',
                        '<element_data data="ef"'), timeout=900)
    node = L.run(L.swap(base, '<node_data data="nfvx;nfvy;nfvz"',
                        '<node_data data="ef"'), timeout=900)
    elem_msg = elem.has('"ef" is not a valid field variable name')
    print(f"element_data_ef: rc={elem.rc} "
          f"read_success={int(elem.read_success)} "
          f"read_failed={int(elem.read_failed)} "
          f"not_a_valid_field_variable={int(elem_msg)} "
          f"(claim said this comes AFTER a successful read)")
    print(f"node_data_ef: rc={node.rc} "
          f"normal={int(node.normal_termination)} "
          f"steps={node.steps_completed} accepted={int(node.rc == 0)}")
    good = (invariant and moved and elem_msg and elem.rc != 0
            and elem.read_failed and not elem.read_success
            and node.rc == 0 and node.normal_termination)
    return L.report(good, "fluid_k_and_log_names",
                    "reproduced_with_correction", "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
