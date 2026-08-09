"""Tier-2: the stub generator this claim describes no longer exists.

The claim describes a template that was removed. What is true
today is that the physics is documented but has no generator, so
the catalog refuses rather than emitting a silent no-op. This
fixture holds that line.

Mutation control: the pathology is a fact about the shipped catalog -- pfem_solid is
documented but has no generator -- so it cannot be removed by editing a physics
row. T2_MUTATE=1 therefore changes what the fixture ASKS about: it repoints the
same two probes at linear_elasticity, which is documented AND ships
linear_elasticity_2d plus linear_elasticity_2d_nonlinear. Both probes still run
for real against backends.kratos.generators; the output becomes
in_knowledge[linear_elasticity]=True with a NON-empty generators_for list, so
in_knowledge[pfem_solid]=True, generators_for[pfem_solid]=[] and
stub_template_mismatches=0 all disappear and the fixture goes red.
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

MUTATE = os.environ.get("T2_MUTATE") == "1"

_HERE = Path(__file__).resolve()
# In place the checkout is four levels up. The mutation harness stages a copy
# of this fixture into a scratch tree that has no such ancestor, so the search
# walks up looking for the catalog itself and only then falls back to
# $OASIS_REPO. If neither resolves, abort loudly: a fixture that cannot find
# the catalog it audits must never report a pass.
_REPO = next((p for p in _HERE.parents
              if (p / "src" / "backends" / "kratos" / "generators").is_dir()),
             None)
if _REPO is None:
    _REPO = Path(os.environ.get("OASIS_REPO") or "/nonexistent")
    if not (_REPO / "src" / "backends" / "kratos" / "generators").is_dir():
        print("FIXTURE_ABORT=no_oasis_checkout: set OASIS_REPO to the checkout "
              "whose Kratos catalog is under audit", file=sys.stderr)
        raise SystemExit(2)
sys.path.insert(0, str(_REPO / "src"))

from backends.kratos.generators import GENERATORS, KNOWLEDGE  # noqa: E402

# Mutation control. The pathology is a fact about the SHIPPED catalog — this
# physics is documented but has no generator — so it cannot be removed by
# editing a physics row from outside. The honest antidote is to change what
# the fixture ASKS about: T2_MUTATE=1 repoints the same two probes at
# linear_elasticity, which is documented AND ships two working generators, so
# the whole "documented but no template" assertion has to disappear.
PHYSICS = "pfem_solid"
if MUTATE:
    print("mutation=probe_repointed_to_a_physics_that_has_generators")
    PHYSICS = "linear_elasticity"


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
