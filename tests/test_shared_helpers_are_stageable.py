"""A shared helper must survive being copied into the mutation scratch tree.

WHY, and this has now bitten three times
----------------------------------------
The mutation harness copies a fixture into a temporary directory and runs it
there, so a fixture that reaches outside its own directory only works if the
harness brings that thing along. It stages siblings whose name begins with `_`:

    scripts/mutate_tier2_fixtures.py:83
        if sib.is_dir() and sib.name.startswith("_"):

**Directories only.** A shared helper written as a bare `_helper.py` is never
copied, the staged fixture cannot import it, and the run dies before the
mutation is even applied. The harness then reports VACUOUS_BASELINE — "I could
not evaluate this" — which is easy to skim past as "not red".

That is the whole danger. The mutation verdict is the ONLY evidence that a
fixture detects its pathology rather than passing for some unrelated reason, so
a harness that cannot evaluate is indistinguishable from a fixture that has no
evidence, and both look calm in a summary.

Three separate instances, all found by execution rather than review:

  * 4C — 174 fixtures sourced `../_lib/preamble.sh`, which the harness did not
    stage at all. Sampled 12 and 9 were vacuous: the UNMUTATED copy already
    failed, so every mutation "KILLED" for the wrong reason. Fixed by staging
    the parent layout, plus a baseline precheck that emits VACUOUS_BASELINE
    instead of a verdict.
  * coupling — the flagship's 18 fixtures shared a bare `_couplinglib.py`. All
    18 would have been vacuous. Fixed by making it `_lib/`.
  * febio (70 fixtures) and sparta (27) still share `_febio_lib.py` and
    `_spahelp.py` as bare files. 97 fixtures whose mutation evidence would mean
    nothing.

The fix is trivial — make the helper a package directory — which is exactly why
it should be enforced rather than remembered.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "scripts" / "tier2_fixtures"

BACKENDS = sorted(p.name for p in FIXTURES.iterdir() if p.is_dir()) \
    if FIXTURES.is_dir() else []


@pytest.mark.parametrize("backend", BACKENDS)
def test_shared_helpers_are_directories(backend):
    """A `_`-prefixed sibling must be a directory, or it is never staged."""
    be = FIXTURES / backend
    if not be.is_dir():
        pytest.skip(f"{backend} has no fixtures")

    bare = []
    for item in sorted(be.iterdir()):
        name = item.name
        if not name.startswith("_") or name.startswith("__"):
            continue  # __pycache__ and friends are not helpers
        if item.is_file():
            # Count how many fixtures would be affected, so the message says
            # how much evidence is at stake rather than just naming a file.
            stem = item.stem
            users = 0
            for d in be.iterdir():
                if not d.is_dir():
                    continue
                for f in d.iterdir():
                    if f.is_file() and f.suffix in (".py", ".sh"):
                        try:
                            if stem in f.read_text(errors="ignore"):
                                users += 1
                                break
                        except OSError:
                            pass
            bare.append(f"{name} — imported by ~{users} fixtures")

    assert not bare, (
        f"{backend}: shared helpers are bare FILES, which the mutation harness "
        f"never stages (it copies `_`-prefixed DIRECTORIES only, "
        f"mutate_tier2_fixtures.py:83):\n  " + "\n  ".join(bare)
        + "\n\nEvery fixture importing one of these dies in the scratch tree "
          "before its mutation is applied, and the harness reports "
          "VACUOUS_BASELINE — which reads as 'not red' rather than 'no "
          "evidence'. Since the mutation verdict is the only proof a fixture "
          "detects its pathology, that silently voids the evidence for all of "
          "them.\n\n"
          "Fix: make the helper a package directory — `_lib/__init__.py` plus "
          "the module — as coupling, fourc and dealii already do. Then re-run "
          "the mutation harness for that backend and confirm the verdicts are "
          "KILLED rather than VACUOUS_BASELINE.")
