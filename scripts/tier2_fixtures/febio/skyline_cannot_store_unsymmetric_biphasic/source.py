"""Tier-2: the registered linear solvers, sorted by what they actually do
to an unsymmetric biphasic system.

Verifies febio::biphasic#3, and FALSIFIES one entry in its list. The
claim sweeps the FELINEARSOLVER_ID factories on a USE_MKL=OFF build and
sorts them into buckets. Re-executed on the shipped confined-compression
deck, every bucket reproduces except one:

  * matrix-format error: skyline, fgmres, cg, superlu_mt, accelerate,
    diagonal, hypre_gmres, hypre_pcg_amg, boomeramg, ichol,
    pardiso-project, bipn,
  * `Linear solver failed to find solution`: LU,
  * `Fatal error in factorization of stiffness matrix`: ilu0, ilut,
  * `An error occurred during preprocessing of linear solver`: block,
  * not a registered type string at all: pardiso,
  * runs: bicgstab and test (the null solver).

`strategy` is listed in the claim among the solvers that emit the
matrix-format message. It does not. It is refused at PARSE with
`Component "linear_solver" needs to have property "solver1" defined` —
it is a composite solver that has to be given its sub-solvers, so it
never reaches the matrix at all.

The fixture asserts the bucket for every name, so any future change of
outcome for any of them turns it red.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

BUCKETS = {
    "matrix_format": ("skyline", "fgmres", "cg", "superlu_mt", "accelerate",
                      "diagonal", "hypre_gmres", "hypre_pcg_amg",
                      "boomeramg", "ichol", "pardiso-project", "bipn"),
    "failed_to_find": ("LU",),
    "factorization": ("ilu0", "ilut"),
    "preprocessing": ("block",),
    "not_a_registered_type": ("pardiso",),
    "needs_sub_solver": ("strategy",),
    "needs_A_solver": ("schur",),
    "runs": ("bicgstab", "test"),
}


def classify(run) -> str:
    if run.has("does not support the requested"):
        return "matrix_format"
    if run.has("Linear solver failed to find solution"):
        return "failed_to_find"
    if run.has("Fatal error in factorization"):
        return "factorization"
    if run.has("An error occurred during preprocessing"):
        return "preprocessing"
    if run.has('invalid value for attribute "type"'):
        return "not_a_registered_type"
    if run.has('needs to have property "solver1" defined'):
        return "needs_sub_solver"
    if run.has('needs to have property "A_solver" defined'):
        return "needs_A_solver"
    if run.rc == 0 and run.normal_termination:
        return "runs"
    return f"unclassified(rc={run.rc})"


def main() -> int:
    base = L.template("biphasic_3d_confined")
    wrong = 0
    total = 0
    for expected, names in BUCKETS.items():
        for name in names:
            deck = L.swap(base, '<linear_solver type="bicgstab"/>',
                          f'<linear_solver type="{name}"/>')
            got = classify(L.run(deck, timeout=300))
            total += 1
            mark = "ok" if got == expected else "MISMATCH"
            print(f"solver={name:16s} expected={expected:22s} "
                  f"got={got:22s} {mark}")
            if got != expected:
                wrong += 1
    print(f"solvers_in_their_recorded_bucket={total - wrong} of {total}")
    return L.report(wrong == 0, "biphasic_solver_sweep", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
