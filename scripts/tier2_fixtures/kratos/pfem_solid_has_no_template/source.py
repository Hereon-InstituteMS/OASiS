"""Tier-2: the stub generator this claim describes no longer exists.

The claim describes a template that was removed. What is true
today is that the physics is documented but has no generator, so
the catalog refuses rather than emitting a silent no-op. This
fixture holds that line.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

# Imported unconditionally so the fixture dies rather than passes when
# Kratos is absent.
import KratosMultiphysics as KM

print(f"kratos_version_present={bool(KM.__file__)}")

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[4]
sys.path.insert(0, str(_REPO / "src"))

from backends.kratos.generators import GENERATORS, KNOWLEDGE  # noqa: E402

PHYSICS = "pfem_solid"


def main() -> int:
    bad = 0
    in_knowledge = PHYSICS in KNOWLEDGE
    gens = sorted(k for k in GENERATORS if k.startswith(PHYSICS + "_"))
    print(f"in_knowledge[{PHYSICS}]={in_knowledge}")
    print(f"generators_for[{PHYSICS}]={gens}")
    if not in_knowledge:
        print(f"FAIL: {PHYSICS} is not a KNOWLEDGE key at all", file=sys.stderr)
        bad += 1
    if gens:
        print(f"FAIL: a generator still exists for {PHYSICS} — the claim that "
              f"the catalog emits an availability-probe stub may be live again",
              file=sys.stderr)
        bad += 1
    print(f"stub_template_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
