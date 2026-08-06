"""Tier-2: omitting <mat_axis> on an HGO material is silently the global
frame.

Verifies febio::fiber_reinforced#0. Same shape of evidence as the
active-contraction default: the deck with <mat_axis> deleted agrees with
the deck that writes a = (1,0,0), d = (0,1,0) explicitly to within the
measured solver noise, and a frame aligned with the load gives stresses
that differ by orders of magnitude more.
No warning, no log line, normal termination in every case.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

def with_log(deck: str) -> str:
    """Add the standard position + stress logfile rows to a deck.

    Merges into the template's existing <Output> block — appending a
    second one leaves a deck that does not run, and the fixture would
    then compare two empty lists and call them identical.
    """
    return L.add_logfile(deck,
                         ("node_data", "z", "p.csv"),
                         ("element_data", "sx;sy;sz;J", "e.csv"))


def series(run, name="e.csv"):
    """Every logged value of one file, flattened in file order."""
    out = []
    for _t, rows in L.parse_log_csv(run.files.get(name) or ""):
        for nid in sorted(rows):
            out.extend(rows[nid])
    return out


def max_rel_dev(a, b) -> float:
    if not a or len(a) != len(b):
        return float("inf")
    return max(abs(x - y) / max(abs(x), abs(y), 1e-12)
               for x, y in zip(a, b))


def noise_floor(deck: str, timeout=900):
    """Max relative deviation between two IDENTICAL runs of one deck.

    FEBio on this build is not bit-reproducible in general — even a
    direct-solver deck moves in the last few significant digits between
    identical runs. An identity assertion therefore has to be made
    against a measured floor; asserting exact equality is either flaky
    or, worse, an apparent "effect" that is only round-off.
    """
    a = L.run(deck, collect=("e.csv", "p.csv"), timeout=timeout)
    b = L.run(deck, collect=("e.csv", "p.csv"), timeout=timeout)
    return max_rel_dev(series(a), series(b)), a


MAT_AXIS_X = ('      <mat_axis type="vector">\n'
              "        <a>1,0,0</a>\n"
              "        <d>0,1,0</d>\n"
              "      </mat_axis>\n")
MAT_AXIS_Z = ('      <mat_axis type="vector">\n'
              "        <a>0,0,1</a>\n"
              "        <d>1,0,0</d>\n"
              "      </mat_axis>\n")


def main() -> int:
    base = L.template("fiber_reinforced_3d_hgo")
    decks = {
        "explicit_global_x": base,
        "mat_axis_omitted": L.drop(base, MAT_AXIS_X),
        "aligned_with_load": L.swap(base, MAT_AXIS_X, MAT_AXIS_Z),
    }
    data = {}
    for tag, deck in decks.items():
        r = L.run(with_log(deck), collect=("e.csv", "p.csv"), timeout=900)
        data[tag] = (r, series(r))
        print(f"{tag}: rc={r.rc} normal={int(r.normal_termination)} "
              f"steps={r.steps_completed} "
              f"warnings={int('WARNING' in r.text)}")
    noise, _ = noise_floor(with_log(decks["explicit_global_x"]))
    tol = max(1e-9, 100 * noise)
    omitted = data["mat_axis_omitted"][1]
    explicit = data["explicit_global_x"][1]
    aligned = data["aligned_with_load"][1]
    d_same = max_rel_dev(omitted, explicit)
    d_diff = max_rel_dev(omitted, aligned)
    identical = bool(omitted) and d_same < tol
    differs = d_diff > 1e-3
    print(f"solver_noise_floor={noise:.3e} identity_tolerance={tol:.3e}")
    print(f"omitted_equals_global_x: max_rel_dev={d_same:.3e} "
          f"within_noise={int(identical)}")
    print(f"load_aligned_frame_differs={int(differs)} "
          f"max_rel_dev={d_diff:.3e}")
    all_clean = all(r.rc == 0 and r.normal_termination
                    for r, _s in data.values())
    good = identical and differs and all_clean
    return L.report(good, "hgo_mat_axis_default", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
