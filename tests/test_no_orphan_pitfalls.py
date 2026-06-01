"""Regression: every advertised physics has at least one pitfall.

The catalog audit on 2026-06-01 found 7 kratos entries that
advertised themselves in supported_physics() with full
description+application+capabilities metadata but ZERO pitfalls.
prepare_simulation then renders these as confident solver
descriptions, and the LLM has no warning that the template is an
availability-probe stub rather than a real solver.

This test pins the post-batch-16 contract: each backend×physics
pair that gets registered MUST own at least one pitfall entry in
its KNOWLEDGE block. New physics added without one will fail this
test, forcing the author to either add a pitfall or remove the
advertisement.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


class TestNoOrphanPitfalls(unittest.TestCase):
    """Every (backend, physics) row in supported_physics() must
    expose at least one pitfall in its KNOWLEDGE dict."""

    # Backends that follow the "supported_physics() returns
    # PhysicsCapability list" pattern. deal.II / dune / febio
    # do not use the same KNOWLEDGE-dict surface and are
    # audited separately.
    BACKENDS = ("fenics", "ngsolve", "skfem", "kratos")

    def test_every_physics_has_a_pitfall(self) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        offenders: list[tuple[str, str, str]] = []
        for be in self.BACKENDS:
            backend = get_backend(be)
            if backend is None:
                continue
            for cap in backend.supported_physics():
                k = backend.get_knowledge(cap.name)
                if not k:
                    offenders.append((be, cap.name, "NO-KNOWLEDGE"))
                    continue
                pitfalls = k.get("pitfalls", []) if isinstance(k, dict) else []
                if not pitfalls:
                    offenders.append((be, cap.name, "NO-PITFALLS"))
        if offenders:
            lines = "\n".join(f"  {be}::{ph} - {why}"
                              for be, ph, why in offenders)
            self.fail(
                f"{len(offenders)} advertised physics without pitfalls:\n"
                f"{lines}\n\n"
                "Either add a pitfall to the KNOWLEDGE dict (even just "
                "an '[Integration] this is a stub probe' entry) or "
                "remove the physics from supported_physics().")


if __name__ == "__main__":
    unittest.main()
