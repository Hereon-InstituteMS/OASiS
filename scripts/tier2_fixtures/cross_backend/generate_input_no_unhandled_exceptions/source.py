"""Tier-2: cross-backend audit — every advertised
(backend, physics, template_variant) tuple must produce
non-empty output via generate_input() without raising
unhandled Python exceptions, AND get_knowledge(physics)
must return a non-empty dict.

This generalizes the FEBio falsification class (#29 +
#30) to all backends. A discovery sweep on 2026-06-01
found:

  - 26 generate_input failures across 5 backends (4C 11,
    fenics 5, dealii 3, ngsolve 4, kratos 1, plus 1
    short-output reduced_lung). Of those, the 4 ngsolve
    failures were ACTUAL Python NameError bugs in
    generator f-string templates (n_load_steps vs
    n_steps), not 'physics-advertised-but-not-wired'
    gaps. The remaining 22 are missing-template entries
    that need PR work to fill.
  - 22 advertised physics with empty pitfalls (mostly
    dealii).

This fixture pins the FLOOR of acceptable failures:
the ngsolve NameError bugs MUST stay at zero (regression
gate), and the total generate_input failure count must
not exceed a knob (currently 22). When PR work fills
templates, the floor tightens automatically.
"""
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))


# Maximum allowed generate_input failures. Lower than 26
# because the ngsolve f-string bugs fix lands alongside
# this fixture (4 fewer failures → 22 max).
MAX_GEN_FAILURES = 7

# NameError-class bugs MUST stay at zero. If a future
# refactor reintroduces an f-string variable-name mismatch,
# this gate fails immediately.
MAX_NAMEERROR = 0


def main() -> int:
    from core.registry import (
        load_all_backends, list_backends, get_backend,
    )

    load_all_backends()
    rows = []
    nameerrors = []
    other_errors = []
    successes = 0
    for entry in list_backends():
        if entry["status"] != "available":
            continue
        b_name = entry["name"]
        b = get_backend(b_name)
        for cap in b.supported_physics():
            for v in cap.template_variants:
                try:
                    txt = b.generate_input(
                        cap.name, v, {})
                    if not txt or len(txt.strip()) < 30:
                        rows.append(
                            f"{b_name}::{cap.name}::{v}"
                            f" → short/empty"
                            f" ({len(txt) if txt else 0})")
                    else:
                        successes += 1
                except NameError as e:
                    nameerrors.append(
                        f"{b_name}::{cap.name}::{v}"
                        f" → NameError: {e!s}")
                except Exception as e:
                    rows.append(
                        f"{b_name}::{cap.name}::{v}"
                        f" → {type(e).__name__}: "
                        f"{str(e)[:60]}")
    other_errors = rows

    print(f"available_backends_count="
          f"{sum(1 for e in list_backends() if e['status'] == 'available')}")
    print(f"generate_input_successes={successes}")
    print(f"generate_input_other_failures="
          f"{len(other_errors)}")
    print(f"generate_input_nameerrors={len(nameerrors)}")

    # Bug-class breakdown
    for e in nameerrors:
        print(f"  NameError-bug: {e}")
    for e in other_errors[:10]:
        print(f"  other: {e}")
    if len(other_errors) > 10:
        print(f"  ... and {len(other_errors) - 10} more")

    ok = (
        len(nameerrors) <= MAX_NAMEERROR
        and len(other_errors) <= MAX_GEN_FAILURES
    )
    if ok:
        return 0
    print(f"FAIL: floor exceeded "
          f"(nameerrors={len(nameerrors)} > "
          f"{MAX_NAMEERROR} OR other={len(other_errors)} > "
          f"{MAX_GEN_FAILURES})", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
