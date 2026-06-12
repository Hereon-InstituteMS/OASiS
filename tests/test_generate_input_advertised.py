"""Regression: every (backend, physics, variant) advertised in
supported_physics() must produce a non-empty template via
generate_input() WITHOUT raising.

Catalog audit on 2026-06-01: dune::helmholtz::2d raised
NameError: name 'k_val' is not defined at f-string evaluation
time because the generator embedded an UNESCAPED {k_val}
reference inside its own f-string (k_val is a variable in the
generated script, not in the generator scope). The generator
crashed before any of the existing Tier-2 fixtures could even
run.

Pin this contract: walking supported_physics() x
template_variants and calling generate_input(name, variant, {})
for each MUST return a string ≥ 30 chars. A future generator
that f-string-escapes incorrectly, or a supported_physics()
entry whose template_variants drift from the registered
generator names, fails this test.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


class TestEveryAdvertisedTemplateGenerates(unittest.TestCase):
    """generate_input must succeed for every advertised
    (backend, physics, variant) row."""

    BACKENDS = ("fenics", "ngsolve", "skfem", "kratos",
                "dealii", "fourc", "dune", "febio")

    def test_no_advertised_template_raises(self) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        failures: list[tuple[str, str, str, str]] = []
        for be in self.BACKENDS:
            backend = get_backend(be)
            if backend is None:
                continue
            for cap in backend.supported_physics():
                for variant in cap.template_variants:
                    try:
                        src = backend.generate_input(
                            cap.name, variant, {})
                    except Exception as e:  # noqa: BLE001
                        failures.append((be, cap.name, variant,
                                         f"{type(e).__name__}: "
                                         f"{str(e)[:120]}"))
                        continue
                    if not isinstance(src, str) or len(src) < 30:
                        failures.append((be, cap.name, variant,
                                         "non-string or "
                                         "< 30 chars"))
        if failures:
            lines = "\n".join(
                f"  {be}::{ph}::{var} - {why}"
                for be, ph, var, why in failures)
            self.fail(
                f"{len(failures)} advertised templates raised "
                f"or returned empty:\n{lines}\n\n"
                "Either fix the generator (most likely an "
                "f-string brace-escape error like {k_val} -> "
                "{{k_val}} where k_val is a runtime variable) "
                "or remove the variant from "
                "PhysicsCapability.template_variants.")


if __name__ == "__main__":
    unittest.main()
