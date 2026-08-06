"""A fixture must not pass on a machine that cannot run it.

OASiS is meant to be cloned, run and picked apart by other people. That makes
"green on this workstation" worth very little on its own: what matters is
whether a stranger's run tells them the truth.

The defect this guards against was found twice. A DUNE fixture exited 0 with the
catalog tree absent. Five FEBio fixtures do this:

    fixture.json:  expect_in_output = ["degenerate_hex8="]
                   forbid_in_output = ["Traceback", "FAIL:"]
    source.py:     print("degenerate_hex8=skipped_no_binary")

With no FEBio installed, the fixture prints its key, the expectation matches,
neither forbidden string appears, and it PASSES — having verified nothing. The
same five raised the tier-2 floor from 108 to 113, so the floor now certifies
five phantom passes on any host without the binary.

Skipping when a backend is missing is correct and necessary. Reporting a PASS
for it is not. A fixture must either fail loudly or be recorded as skipped, and
the difference has to be visible in the run's own output — otherwise absence of
a backend is indistinguishable from presence of a working one, which is the same
class of silent-wrong the fixtures exist to catch.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = REPO / "scripts" / "tier2_fixtures"

# Markers a fixture prints when it declines to do the real work.
_ABSENT_MARKER = re.compile(
    r"=\s*(skipped\w*|no_binary|not_available|unavailable|missing_\w+)\b",
    re.IGNORECASE)


def _fixture_dirs():
    if not FIXTURES.exists():
        return []
    return sorted(p.parent for p in FIXTURES.glob("*/*/fixture.json"))


def _sources(d: pathlib.Path) -> str:
    text = ""
    for name in ("source.py", "cmd.sh", "run.sh"):
        p = d / name
        if p.exists():
            text += p.read_text(errors="ignore") + "\n"
    return text


def _vacuous(d: pathlib.Path) -> list[str]:
    """Lines where this fixture prints an EXPECTED key with an absent-marker
    value, and nothing in forbid_in_output catches it."""
    meta = json.loads((d / "fixture.json").read_text())
    expect = [str(s) for s in meta.get("expect_in_output", [])]
    forbid = [str(s).lower() for s in meta.get("forbid_in_output", [])]
    if not expect:
        return []
    # Does any forbidden string catch a skip? Then the fixture is guarded.
    if any(re.search(r"skip|no_binary|unavailable|not_available", f)
           for f in forbid):
        return []
    bad = []
    for line in _sources(d).splitlines():
        m = _ABSENT_MARKER.search(line)
        if not m:
            continue
        # The printed key must be one the expectation would match.
        prefix = line[:m.start() + 1]
        if any(e.rstrip() and e.rstrip() in prefix + "=" or
               (e.endswith("=") and e[:-1] in prefix)
               for e in expect):
            bad.append(line.strip()[:120])
    return bad


@pytest.mark.parametrize("d", _fixture_dirs(),
                         ids=lambda p: f"{p.parent.name}/{p.name}")
def test_fixture_cannot_report_a_pass_without_running(d):
    bad = _vacuous(d)
    assert not bad, (
        f"{d.parent.name}/{d.name} satisfies its own expectation while "
        f"declining to run: it prints an expected key with a skip marker, and "
        f"nothing in forbid_in_output catches that. On a machine without this "
        f"backend the fixture PASSES having verified nothing.\n  "
        + "\n  ".join(bad)
        + "\n\nFix by one of: add the skip marker to forbid_in_output; expect "
          "a substantive value rather than a bare key; or make the absent path "
          "exit non-zero so the runner records a failure instead of a pass.")


def test_the_guard_itself_detects_the_known_case():
    """Proof the predicate discriminates, using the shape of the real defect
    rather than trusting that a green run means anything."""
    import tempfile

    d = pathlib.Path(tempfile.mkdtemp()) / "backend" / "fx"
    d.mkdir(parents=True)
    (d / "fixture.json").write_text(json.dumps({
        "expect_in_output": ["degenerate_hex8="],
        "forbid_in_output": ["Traceback", "FAIL:"],
    }))
    (d / "source.py").write_text('print("degenerate_hex8=skipped_no_binary")\n')
    assert _vacuous(d), "the guard fails to catch the defect it was written for"

    # …and does not fire once the skip is forbidden.
    (d / "fixture.json").write_text(json.dumps({
        "expect_in_output": ["degenerate_hex8="],
        "forbid_in_output": ["Traceback", "FAIL:", "skipped_no_binary"],
    }))
    assert not _vacuous(d), "the guard fires on a properly guarded fixture"
