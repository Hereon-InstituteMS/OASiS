"""Tier-2: DUNE raviartThomas is camelCase, not lowercase.

The catalog had:

  src/backends/dune/generators/advanced.py L592, L608:
    from dune.fem.space import raviartthomas, dglagrange
    Sigma = raviartthomas(gridView, order=order)
  src/backends/dune/generators/__init__.py L40:
    "raviartthomas": "H(div) conforming"

The real DUNE function is raviartThomas (camelCase),
defined in dune-fem's python/dune/fem/space/_spaces.py:

  def raviartThomas(gridView, order=0, dimRange=None, ...)

Note: the underlying C++ header IS lowercase
('dune/fem/space/raviartthomas.hh') — that's the source
of catalog confusion. But the Python factory function
exposed by 'from dune.fem.space import ...' is
camelCase.

PORTABILITY FIX (2026-08-03). The first version of this
fixture hard-coded an absolute path into one developer's
DUNE source checkout (/home/hermann/Schreibtisch/
dune-src/...) plus an absolute path to the catalog. On
any other machine it printed "FAIL: <path> not found"
and counted as failed, even though the claim it verifies
is true. It now resolves both locations from the running
process:

  * the space module through `import dune.fem.space._spaces`
    (falling back to a dune-src checkout if one exists),
    so it inspects the DUNE that is actually INSTALLED
    rather than a checkout that may not match;
  * the catalog root relative to this file, so the fixture
    works from a git worktree.

It asserts:
  * 'def raviartThomas(' is present in _spaces.py
  * 'def raviartthomas(' (lowercase) is ABSENT from the
    same file (regression catches any future alias add)
  * the '.hh' include for the C++ header IS lowercase
    (confirms why the catalog confused it)
  * the catalog contains no bare lowercase spelling

Executed against dune-fem 2.12.0.2 on 2026-08-03:
camelCase def present, lowercase def absent,
'raviartthomas.hh' present at _spaces.py:922,
catalog_lowercase_count=0.

MUTATION CONTROL. T2_MUTATE=1 appends a synthetic
`def raviartthomas(` line to the _spaces.py text being scanned — the
pathology removed, i.e. the world in which dune-fem ships a lowercase
alias after all. raviartthomas_lowercase_def_present then reads True,
that expectation disappears and a FAIL: line appears. The file on disk
is never touched; only the in-memory copy is.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

MUTATE = os.environ.get("T2_MUTATE") == "1"


def _find_spaces_py():
    """Locate dune-fem's python/dune/fem/space/_spaces.py.

    Preference order: the INSTALLED package this interpreter can
    import (the ground truth for the version under test), then a
    developer source checkout, then nothing.
    """
    try:
        import dune.fem.space._spaces as mod  # type: ignore
        p = Path(mod.__file__)
        if p.is_file():
            return p
    except Exception:
        pass
    for base in (Path.home() / "Schreibtisch" / "dune-src",
                 Path.home() / "dune-src"):
        cand = (base / "dune-fem" / "python" / "dune" / "fem"
                / "space" / "_spaces.py")
        if cand.is_file():
            return cand
    return None


def _catalog_root() -> Path:
    # .../<repo>/scripts/tier2_fixtures/dune/<fixture>/source.py
    repo = Path(__file__).resolve().parents[4]
    return repo / "src" / "backends" / "dune"


def main() -> int:
    spaces_py = _find_spaces_py()
    if spaces_py is None:
        print("FAIL: could not locate dune/fem/space/_spaces.py "
              "(neither importable nor in a known checkout)",
              file=sys.stderr)
        return 2
    print(f"spaces_py_source={spaces_py}")
    text = spaces_py.read_text()
    if MUTATE:
        print("mutation=the_scanned_spaces_py_text_gains_a_lowercase_"
              "raviartthomas_alias")
        text = text + "\ndef raviartthomas(gridView, order=0):\n    pass\n"
    camel = re.search(r"^def\s+raviartThomas\s*\(", text, re.MULTILINE)
    lowercase = re.search(r"^def\s+raviartthomas\s*\(", text,
                          re.MULTILINE)
    hh_include = "raviartthomas.hh" in text
    print(f"raviartThomas_camelcase_def_present={bool(camel)}")
    print(f"raviartthomas_lowercase_def_present={bool(lowercase)}")
    print(f"raviartthomas_hh_include_present_in_text={hh_include}")

    # And the catalog (under audit) must NOT present the bare
    # lowercase form as an API name anywhere after the fix.
    #
    # A mention is benign ONLY when the matched text is physically
    # inside one of the exact literals below — the C++ header name, or
    # the dotted path / quoted name the catalog uses to document that
    # the Python spelling does NOT exist. A nearby word such as
    # "lowercase" is NOT enough: any re-introduced API use would sit
    # next to explanatory prose containing that word, which would make
    # this counter unfalsifiable (audit finding 2026-08-03).
    benign_literals = (
        "raviartthomas.hh",                    # the C++ header
        "dune.fem.space.raviartthomas",        # documented-absent path
        "'raviartthomas'",                     # hasattr(...) prose
        '"raviartthomas"',
    )
    catalog_root = _catalog_root()
    # A missing catalog tree must FAIL, not pass vacuously. The
    # predecessor of this fixture hard-coded one developer's absolute
    # paths; replacing them with a relative walk fixed the paths but
    # left the same hole — rglob over a nonexistent directory yields
    # nothing, catalog_lowercase stays 0, and the check reports OK.
    # Proved by execution during the adversarial audit 2026-08-03:
    # with src/backends/dune renamed away the fixture still exited 0.
    catalog_files = sorted(catalog_root.rglob("*.py"))
    if not catalog_files:
        print(f"FAIL: catalog tree {catalog_root} has no .py files — "
              f"the lowercase-regression half of this fixture would "
              f"pass vacuously", file=sys.stderr)
        return 2
    catalog_lowercase = 0
    for p in catalog_files:
        body = p.read_text(encoding="utf-8")
        benign_spans = []
        for lit in benign_literals:
            start = body.find(lit)
            while start != -1:
                benign_spans.append((start, start + len(lit)))
                start = body.find(lit, start + 1)
        for m in re.finditer(r"\braviartthomas\b", body):
            if any(lo <= m.start() and m.end() <= hi
                   for lo, hi in benign_spans):
                continue
            catalog_lowercase += 1
    print(f"catalog_lowercase_count={catalog_lowercase}")

    ok = (camel is not None
          and lowercase is None
          and hh_include
          and catalog_lowercase == 0)
    if ok:
        return 0
    print("FAIL: DUNE raviartThomas casing regression",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
