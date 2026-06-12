"""Regression: per-backend supported_physics() count cannot drop
below the documented floor without an explicit floor update.

This locks in coverage gains the user-stated goal demands. Each
backend has a current floor that represents the minimum acceptable
catalog size. If a refactor or migration deletes a physics, the
gate fires.

The floors are set conservatively at the actual live count at the
time the floor was first armored. To deliberately remove a
physics (e.g. it's a stub-only entry the team decided not to ship
end-to-end), lower the floor in this file AND document the
removal in the commit message — the gate forces deliberate
shrinking, never accidental.

Audit-driven physics shipped this session (skfem +6, fenics +1)
are protected here; they were grounded in real upstream demos
and verified end-to-end. Future audit closures should bump the
floor in lock-step.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


# Per-backend floor: minimum acceptable supported_physics() count.
# When you ADD a physics, bump the floor.
# When you REMOVE a physics, lower the floor + document why.
_FLOORS: dict[str, int] = {
    "fenics":   25,   # 24 baseline + matrix_free_poisson
    "dealii":   32,
    "ngsolve":  21,
    "skfem":    22,   # 16 baseline + wave, adaptive_poisson, point_source,
                      #              schrodinger, contact, hydraulic_resistance
    "kratos":   39,
    "dune":     15,
    "fourc":    47,
    "febio":    16,
}


class TestCoverageFloor(unittest.TestCase):
    def test_each_backend_meets_minimum_coverage(self) -> None:
        from core.registry import load_all_backends, all_backends
        load_all_backends()
        backends = {b.name(): b for b in all_backends()}
        below: list[tuple[str, int, int]] = []
        for be_name, floor in _FLOORS.items():
            b = backends.get(be_name)
            if b is None:
                self.fail(
                    f"_FLOORS lists {be_name!r} but no such "
                    f"backend is registered. Either backend was "
                    f"renamed/removed (update this file) or the "
                    f"floor entry is wrong.")
            actual = len(b.supported_physics())
            if actual < floor:
                below.append((be_name, actual, floor))
        if below:
            details = "\n".join(
                f"  {be}: {actual} physics (floor: {floor}, "
                f"deficit {floor - actual})"
                for be, actual, floor in below)
            self.fail(
                f"{len(below)} backend(s) below the documented "
                f"coverage floor:\n{details}\n\n"
                f"This gate locks in the audit-driven coverage "
                f"wins. To DELIBERATELY shrink, lower the floor "
                f"in tests/test_coverage_floor.py "
                f"AND document why in the commit message. To "
                f"FIX, restore the missing PhysicsCapability + "
                f"generator wiring + KNOWLEDGE entry. See "
                f"src/backends/<be>/backend.py "
                f"supported_physics() and "
                f"src/backends/<be>/generators/__init__.py.")


if __name__ == "__main__":
    unittest.main()
