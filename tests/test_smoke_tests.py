"""Tests for solver smoke tests."""

import unittest
from core.smoke_tests import (
    SmokeResult, smoke_ngsolve, smoke_skfem, smoke_kratos,
    smoke_dune, smoke_fourc, run_all_smoke_tests,
)


class TestSmokeResults(unittest.TestCase):

    def test_ngsolve_smoke(self):
        r = smoke_ngsolve()
        self.assertTrue(r.passed, f"NGSolve smoke failed: {r.error}")
        self.assertLess(r.duration_ms, 5000)

    def test_skfem_smoke(self):
        r = smoke_skfem()
        self.assertTrue(r.passed, f"scikit-fem smoke failed: {r.error}")
        self.assertLess(r.duration_ms, 5000)

    def test_kratos_smoke(self):
        r = smoke_kratos()
        self.assertTrue(r.passed, f"Kratos smoke failed: {r.error}")

    def test_dune_smoke(self):
        """Hard gate only where dune is actually installed.

        On this machine dune is genuinely absent: conda-forge has no
        dune-fem package (Run-1 finding, FINDINGS.md 2026-06-12) and
        the local source build at ~/Schreibtisch/dune-src is
        ABI-broken (RPATH into the ofa-4c-build env). An
        unconditional assert kept this test red on every sweep,
        training everyone to ignore the smoke suite. Skip with the
        real reason instead — machines WITH a working dune still get
        the hard assertion."""
        r = smoke_dune()
        if not r.passed and r.error and (
                "No module named" in str(r.error)
                or "not found" in str(r.error).lower()
                or "ImportError" in str(r.error)):
            self.skipTest(f"dune not installed/usable here: "
                          f"{str(r.error)[:200]}")
        self.assertTrue(r.passed, f"DUNE smoke failed: {r.error}")

    def test_fourc_smoke(self):
        r = smoke_fourc()
        # May or may not be found depending on system
        self.assertIsInstance(r, SmokeResult)

    def test_run_all(self):
        results = run_all_smoke_tests(["skfem", "ngsolve"])
        self.assertIn("skfem", results)
        self.assertIn("ngsolve", results)
        self.assertTrue(results["skfem"].passed)
        self.assertTrue(results["ngsolve"].passed)

    def test_smoke_result_to_dict(self):
        r = SmokeResult("test", True, version="1.0", duration_ms=100)
        d = r.to_dict()
        self.assertEqual(d["solver"], "test")
        self.assertTrue(d["passed"])
        self.assertNotIn("error", d)  # None should be dropped


if __name__ == "__main__":
    unittest.main()
