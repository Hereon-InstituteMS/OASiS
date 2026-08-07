"""The independent references the coupling fixtures grade against must not be
reachable through any tool an agent can call.

WHY THIS IS A COUPLING FIXTURE. Every pair fixture checks the converged
interface state against an answer computed OUTSIDE the coupling — a closed form
and an un-split monolithic re-solve in the fixture harness, and, under
`benchmarks/coupling_pairs/`, per-pair reference runs with their own analytic
answers. That evidence is only worth anything while the answers stay out of
reach: knowledge may describe how the code behaves, never what the answer is.

WHAT WAS WRONG. `developer(action='files', solver=..., keyword=...)` passes
`keyword` straight to `Path.rglob`, and pathlib treats `..` in a glob pattern as
an ordinary path component. `keyword='../../../benchmarks/coupling_pairs/*/
run_pair.py'` therefore walked out of the backend directory and enumerated the
reference tree. A listing is not a file's contents, but "which reference files
exist, and how big they are" is still a channel out of the harness, and it is
the only one that existed.

WHAT IS CHECKED HERE, all through the REGISTERED tool:
  * the reference tree really is there, so the fixture is not passing because
    there is nothing to find;
  * every traversal pattern is REFUSED, with a message that says why;
  * an ordinary relative pattern still works, so the refusal is a guard and not
    a broken tool;
  * no output of the tool, on any of these calls, names a reference file.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                     # noqa: E402

# Patterns that must not resolve. Each is a real way out of the backend tree.
TRAVERSALS = [
    "../../../benchmarks/coupling_pairs/*/run_pair.py",
    "../../../benchmarks/coupling_pairs/*/*.py",
    "../../../scripts/tier2_fixtures/coupling/_lib/*.py",
    "../../*.py",
    "/etc/*",
]

# A pattern that must keep working: refusing everything is not a fix.
LEGITIMATE = "*.py"

# Filenames that must never appear in any answer above.
LEAKS = ["run_pair.py", "couplinglib.py", "params.json"]


def files(keyword: str) -> str:
    return L.call_tool("developer", {"action": "files", "solver": "fenics",
                                     "keyword": keyword})


def body() -> None:
    refs = L.REPO_ROOT / "benchmarks" / "coupling_pairs"
    present = sorted(p.name for p in refs.glob("*/run_pair.py")) if refs.is_dir() else []
    print(f"reference_pairs_on_disk={len(present)}")
    L.check(len(present) >= 2, "no_reference_tree_to_protect",
            f"benchmarks/coupling_pairs holds {len(present)} run_pair.py — with "
            f"nothing there this fixture would pass vacuously")

    refused = 0
    for pat in TRAVERSALS:
        out = files(pat)
        ok = "searches WITHIN one backend's source directory" in out
        if ok:
            refused += 1
        else:
            L.check(False, "traversal_was_not_refused",
                    f"{pat!r} returned: {out[:200]}")
        leaked = [n for n in LEAKS if n in out]
        L.check(not leaked, "traversal_leaked_a_reference_file",
                f"{pat!r} named {leaked}")
    print(f"traversals_refused={refused}/{len(TRAVERSALS)}")
    print(f"every_traversal_refused={refused == len(TRAVERSALS)}")

    good = files(LEGITIMATE)
    works = ".py" in good and "searches WITHIN" not in good
    print(f"ordinary_pattern_still_works={works}")
    L.check(works, "guard_broke_the_tool",
            f"a plain {LEGITIMATE!r} must still list that backend's files, got: "
            f"{good[:200]}")
    leaked = [n for n in LEAKS if n in good]
    L.check(not leaked, "ordinary_pattern_leaked_a_reference_file", str(leaked))
    print(f"no_reference_file_named_in_any_answer={not leaked}")
    print("claims_checked=4")


L.main(body)
