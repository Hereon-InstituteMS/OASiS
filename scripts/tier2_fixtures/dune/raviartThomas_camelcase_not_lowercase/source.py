"""Tier-2: DUNE raviartThomas is camelCase, not lowercase.

The catalog had:

  src/backends/dune/generators/advanced.py L592, L608:
    from dune.fem.space import raviartthomas, dglagrange
    Sigma = raviartthomas(gridView, order=order)
  src/backends/dune/generators/__init__.py L40:
    "raviartthomas": "H(div) conforming"

The real DUNE function is raviartThomas (camelCase),
defined in dune-fem/python/dune/fem/space/_spaces.py:

  def raviartThomas(gridView, order=0, dimRange=None, ...)

Note: the underlying C++ header IS lowercase
('dune/fem/space/raviartthomas.hh') — that's the source
of catalog confusion. But the Python factory function
exposed by 'from dune.fem.space import ...' is
camelCase.

This fixture walks the installed DUNE Python source tree
directly (DUNE runtime import is broken by a conda
libibverbs ABI mismatch — task #15 — so we cannot use
ngsolve-style runtime introspection here).

It asserts:
  * 'def raviartThomas(' is present in
    dune-fem/python/dune/fem/space/_spaces.py
  * 'def raviartthomas(' (lowercase) is ABSENT from the
    same file (regression catches any future alias add)
  * The literal '.hh' include for the C++ header IS
    lowercase (confirms why catalog confused it)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


SPACES_PY = (Path("/home/hermann/Schreibtisch/dune-src"
                  "/dune-fem/python/dune/fem/space/"
                  "_spaces.py"))


def main() -> int:
    if not SPACES_PY.is_file():
        print(f"FAIL: {SPACES_PY} not found",
              file=sys.stderr)
        return 2
    text = SPACES_PY.read_text()
    camel = re.search(r"^def\s+raviartThomas\s*\(",
                      text, re.MULTILINE)
    lowercase = re.search(r"^def\s+raviartthomas\s*\(",
                          text, re.MULTILINE)
    hh_include = "raviartthomas.hh" in text
    print(f"raviartThomas_camelcase_def_present="
          f"{bool(camel)}")
    print(f"raviartthomas_lowercase_def_present="
          f"{bool(lowercase)}")
    print(f"raviartthomas_hh_include_present_in_text="
          f"{hh_include}")

    # And the catalog (under audit) must NOT contain the
    # lowercase form anywhere after the fix.
    catalog_root = (Path("/home/hermann/Schreibtisch"
                         "/Open-FEM-agent/src/backends/dune"))
    catalog_lowercase = 0
    for p in catalog_root.rglob("*.py"):
        text2 = p.read_text()
        catalog_lowercase += len(re.findall(
            r"\braviartthomas\b", text2))
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
