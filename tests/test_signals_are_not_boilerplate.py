"""A Signal shared by many entries distinguishes none of them.

WHAT THIS PREVENTS
------------------
The `Signal:` clause is what `knowledge(topic='pitfalls', signal=...)` matches
against, so it is the entry's index entry: the thing that separates this pitfall
from its neighbours when an agent pastes an error. When several entries carry
the same Signal, a symptom query returns all of them and ranks none — the
entries are present and unfindable at once, which is the same end state as an
entry nobody can reach, arrived at from a different direction.

Measured when this was first looked at: Kratos had **63 entries sharing one
string**, a generic "solver reports 'Convergence is not achieved' / 'iteration
count exceeded' / oscillating residual", plus three more groups of 12, 9 and 7.
91 of its 204 entries — 45% — carried boilerplate. Every other backend was at
zero, which is what made it a defect rather than a convention.

Fixing it was not a rewrite of the Signals alone. Of the 94 boilerplate entries,
41 had a real observable nobody had written down (a registry rejection, an
AttributeError at attribute access, a ModuleNotFoundError on an inner import
line, a silently wrong value) and 53 were genuine design advice with no failure
mode at all — those moved to a `guidance` list rather than keeping a Signal they
could not deliver. That is the honest fix: an entry must not promise an
observable it does not have.

WHERE THE THRESHOLD COMES FROM
------------------------------
Not zero. Some sharing is correct: two entries can genuinely produce the same
message, and a container-type error that fires for two different mistakes is one
string honestly reused. Measured across all ten surfaces after the Kratos work,
the worst reuse anywhere is 2 and no group reaches 4. So 4 is comfortably above
what legitimate sharing produces and far below the 63 that made the corpus
unsearchable.
"""
from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BACKENDS_DIR = REPO / "src" / "backends"

# A Signal may be shared by up to this many entries. See the module docstring:
# measured legitimate reuse peaks at 2; the defect that motivated this was 63.
MAX_SHARING = 3

_SIGNAL_RE = re.compile(r"Signal:\s*(.*)", re.IGNORECASE | re.DOTALL)
_PROVENANCE_RE = re.compile(
    r"\((?:verified|audit|checked|measured|confirmed)\b[^()]*\)\s*$",
    re.IGNORECASE)


def _signal_of(entry: str) -> str:
    m = _SIGNAL_RE.search(entry)
    if not m:
        return ""
    return _PROVENANCE_RE.sub("", m.group(1)).strip()


def _entries(backend: str) -> set[str]:
    """Distinct pitfall texts. Deduplicated by value: a sub-dict attached by
    reference to several physics is one entry, and counting it repeatedly once
    inflated a DUNE total from 111 to 127."""
    be_dir = BACKENDS_DIR / backend
    found: set[str] = set()
    if not be_dir.is_dir():
        return found
    for py in sorted(be_dir.rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(errors="ignore"))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and "Signal:" in node.value):
                found.add(node.value)
    return found


BACKENDS = sorted(p.name for p in BACKENDS_DIR.iterdir()
                  if p.is_dir() and not p.name.startswith("_")) \
    if BACKENDS_DIR.is_dir() else []


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_signal_is_shared_by_too_many_entries(backend):
    """Boilerplate makes an entry present and unfindable at the same time."""
    texts = _entries(backend)
    if not texts:
        pytest.skip(f"{backend} has no pitfall entries in this tree")

    owners: dict[str, list[str]] = defaultdict(list)
    for text in texts:
        sig = _signal_of(text)
        if sig:
            owners[sig].append(text[:70])

    counts = Counter({sig: len(v) for sig, v in owners.items()})
    over = {sig: n for sig, n in counts.items() if n > MAX_SHARING}

    assert not over, (
        f"{backend}: {len(over)} Signal clauses are shared by more than "
        f"{MAX_SHARING} entries, so a symptom query returns all of them and "
        f"ranks none:\n  "
        + "\n  ".join(f"x{n}: {sig[:100]}" for sig, n in
                      sorted(over.items(), key=lambda kv: -kv[1])[:5])
        + "\n\nGive each entry the observable that distinguishes IT. If the "
          "entry has no failure mode — if it is design advice like a "
          "recommended parameter range — it should not carry a Signal clause "
          "at all; move it to a guidance list rather than stapling on a "
          "symptom it cannot deliver. That is how 53 Kratos entries were "
          "resolved; the other 41 turned out to have a real observable nobody "
          "had written down.")
