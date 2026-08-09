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


def test_the_runner_tells_a_staged_fixture_where_the_checkout_is():
    """The fourth instance of the same failure class, and its fix.

    A fixture that audits the CATALOG rather than a solver has to find the
    checkout. In place it walks up from `__file__`; staged into the scratch
    tree there is no such ancestor, so it aborts with
    FIXTURE_ABORT=no_oasis_checkout and the harness scores VACUOUS_BASELINE —
    "this verdict would mean nothing" — on every run.

    Measured: 7 of the 11 Kratos fixtures that ship a `_mutation` block sat in
    that state, so the ledger carried no machine discrimination evidence for
    any of them, even though each had been proved KILLED by hand with
    OASIS_REPO exported on the command line. With the runner exporting it,
    all 11 are KILLED and 0 are vacuous.

    The runner knows where the checkout is and the staged fixture cannot, so
    saying it is the runner's job. Asserted on the environment the runner
    builds, not on a source grep.
    """
    import os
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    import run_tier2_fixtures as runner  # noqa: E402

    seen: dict[str, str] = {}
    real_run = runner.subprocess.run

    def spy(*args, **kwargs):
        env = kwargs.get("env")
        if env and "OASIS_REPO" in env:
            seen["OASIS_REPO"] = env["OASIS_REPO"]
        return real_run(*args, **kwargs)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "probe"
        d.mkdir()
        (d / "source.py").write_text(
            "import os\nprint('checkout=' + os.environ.get('OASIS_REPO', ''))\n")
        meta = {"backend": "skfem", "physics": "poisson", "pitfall_index": 0,
                "mode": "python", "expect_in_output": ["checkout="]}
        runner.subprocess.run = spy
        try:
            result = runner._eval_fixture(d, meta)
        finally:
            runner.subprocess.run = real_run

    if result.status == "skipped":
        pytest.skip(f"no interpreter to run the probe here: {result.notes}")

    assert seen.get("OASIS_REPO"), (
        "the fixture runner did not export OASIS_REPO, so a fixture staged "
        "into the mutation scratch tree cannot find the checkout it audits. "
        "Every such fixture reports VACUOUS_BASELINE and its mutation evidence "
        "silently disappears. See run_tier2_fixtures._eval_fixture.")
    assert Path(seen["OASIS_REPO"], "src", "backends").is_dir(), (
        f"OASIS_REPO={seen['OASIS_REPO']} does not look like an OASiS "
        f"checkout; a wrong pin is worse than none, because the fixture then "
        f"audits somebody else's catalog and says nothing about this one.")
    assert os.environ.get("OASIS_REPO", seen["OASIS_REPO"]) == seen["OASIS_REPO"], (
        "the runner overrode an OASIS_REPO the caller had already set")
