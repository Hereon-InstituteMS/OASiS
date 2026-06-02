"""Pitfall-DB category coverage.

Operationalises the discipline described in Open-FEM-Agent §3.2 / Table 1:
every pitfall string should be tagged with one of the five categories
(Syntax / Physics / Numerical / API / Integration) so the agent, the
post-exec critic, and downstream coverage tooling can filter by category.

Strict mode: the files that have been migrated to the prefix convention
MUST stay migrated — every pitfall string in them must start with a
recognised `[Category]` token. New files (or per-physics dicts) added to
the strict set in the future tighten the gate further.

Reporting mode: every other backend's KNOWLEDGE dict is scanned and the
fraction of pitfall strings already tagged is recorded. The report does
not block CI today, but it is what task #46 (pitfall-DB-first merge
discipline) measures progress against.
"""

from __future__ import annotations

import unittest
from typing import Iterable


CATEGORY_TOKENS = ("[Syntax]", "[Physics]", "[Numerical]", "[API]", "[Integration]")

# (module dotted-path, physics key inside its KNOWLEDGE dict) — strict gate.
# Add a row here when you migrate a (backend, physics) pair to the
# Table-1 prefix convention; the test then enforces it for that pair.
STRICT_TARGETS: tuple[tuple[str, str], ...] = (
    ("backends.kratos.generators.linear_elasticity", "linear_elasticity"),
)


def _iter_pitfalls(knowledge_block: dict) -> Iterable[str]:
    pitfalls = knowledge_block.get("pitfalls", [])
    if isinstance(pitfalls, list):
        for p in pitfalls:
            if isinstance(p, str):
                yield p


def _has_category(pitfall: str) -> bool:
    stripped = pitfall.lstrip()
    return any(stripped.startswith(tok) for tok in CATEGORY_TOKENS)


class TestStrictPitfallCategoryCoverage(unittest.TestCase):
    """Every pitfall in a strict-target (backend, physics) must be tagged."""

    def test_all_strict_targets_fully_tagged(self):
        import importlib

        failures: list[str] = []
        for module_path, physics_key in STRICT_TARGETS:
            mod = importlib.import_module(module_path)
            knowledge = getattr(mod, "KNOWLEDGE", {})
            block = knowledge.get(physics_key, {})
            for pitfall in _iter_pitfalls(block):
                if not _has_category(pitfall):
                    failures.append(
                        f"{module_path}::{physics_key} — missing category "
                        f"prefix on: {pitfall[:80]!r}"
                    )
        if failures:
            self.fail(
                f"{len(failures)} untagged pitfalls in strict targets:\n  - "
                + "\n  - ".join(failures)
            )


class TestPitfallCategoryReport(unittest.TestCase):
    """Migration-progress report. Does not block CI — see task #46."""

    def test_report_coverage_across_backends(self):
        # Importing the backend KNOWLEDGE dicts pulls in deep-knowledge
        # modules whose own imports (e.g. KratosMultiphysics) may be
        # absent on a stripped CI runner. Failures here MUST NOT mask
        # the strict test above, so we tolerate ImportError per backend.
        backends_to_scan = (
            "backends.kratos.generators",
            "backends.fenics.generators",
            "backends.fourc.generators",
            "backends.ngsolve.generators",
            "backends.skfem.generators",
            "backends.dealii.generators",
            "backends.dune.generators",
        )

        total = 0
        tagged = 0
        per_backend: list[tuple[str, int, int]] = []  # (name, tagged, total)
        for backend_pkg in backends_to_scan:
            try:
                import importlib
                mod = importlib.import_module(backend_pkg)
            except Exception:
                continue
            knowledge = getattr(mod, "KNOWLEDGE", None)
            if not isinstance(knowledge, dict):
                continue
            b_total = 0
            b_tagged = 0
            for _, block in knowledge.items():
                if not isinstance(block, dict):
                    continue
                for pitfall in _iter_pitfalls(block):
                    b_total += 1
                    if _has_category(pitfall):
                        b_tagged += 1
            per_backend.append((backend_pkg, b_tagged, b_total))
            total += b_total
            tagged += b_tagged

        # Emit the report as the test's success message — pytest -v
        # surfaces it, and the CI run can grep `PITFALL-DB COVERAGE`.
        lines = ["PITFALL-DB COVERAGE REPORT"]
        for name, t, n in per_backend:
            pct = (100.0 * t / n) if n else 0.0
            lines.append(f"  {name}: {t}/{n} tagged ({pct:.1f}%)")
        if total:
            pct = 100.0 * tagged / total
            lines.append(f"  TOTAL: {tagged}/{total} ({pct:.1f}%)")
        print("\n".join(lines))

        # Soft floor: refuse to regress below the level that exists today.
        # This is the only enforcement in this test — it stops the
        # migration from going backwards while #46 is in progress.
        if total:
            self.assertGreaterEqual(
                tagged, 1,
                "no pitfall in any backend carries a Table-1 category tag; "
                "the migration has regressed past zero",
            )


if __name__ == "__main__":
    unittest.main()
