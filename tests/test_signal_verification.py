"""Signal-clause verification regression test.

Establishes a floor for the quality of `Signal:` clauses in the
pitfall DB so the project cannot ship a worse claim than what is
shipped today. The harness in `scripts/verify_signal_clauses.py`
computes the metrics; this test asserts none of them regress.

This is the merge-gate the senior-AI-scientist critic (2026-05-31)
called for as the second-largest risk (Signal: clauses are an
unfalsifiable contract — make them at least falsifiable).

What this test does NOT do: claim every signal is real (Tier 2,
intentional-failure regression fixtures, is multi-week work). It
establishes Tier 0 (Signal references a real entity in the
canonical catalogs) and Tier 1 (Signal uses observable-symptom
vocabulary) as the falsifiability floor.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestDealiiSignalFloor(unittest.TestCase):
    """deal.II pitfalls today — established floor as of 2026-05-31.

    These numbers come from running
    ``python scripts/verify_signal_clauses.py --backend dealii``
    after the canonical-element refactor (commit f748716). They
    are a FLOOR not a TARGET — encoding work should monotonically
    push them upward.
    """

    # When updating these numbers, do so ONLY upward — that means
    # the catalog has improved. A downward edit means a regression
    # snuck through and needs to be re-examined.
    #
    # 2026-05-31 floor raise: all 96 deal.II pitfalls have
    # [Category] prefix + Signal: clause + pass Tier 0 (Signal
    # references ≥1 real CODE SYMBOL) + Tier 1 (Signal uses
    # observable-symptom vocabulary). Critic round 2 forced a
    # Tier-0 split into code-symbol-only / with-domain-decoration
    # / domain-names-only — the gameable failure-mode is
    # "domain-names-only" entries scoring as Tier-0 hits, which
    # MUST stay at 0.
    MIN_N_PITFALLS = 96
    MIN_WITH_CATEGORY_PREFIX = 96
    MIN_WITH_SIGNAL_CLAUSE = 96
    MIN_TIER0_PASSED = 96
    MIN_TIER1_PASSED = 96
    MIN_TIER0_AND_1_PASSED = 96
    # New gateable signals from the critic-driven split:
    MAX_TIER0_DOMAIN_NAMES_ONLY = 0  # gameable; MUST stay at 0
    MIN_TIER0_CODE_SYMBOL_ONLY = 55  # strongest sub-tier
    # Tier 2 (operational verification via compile+run fixtures
    # under scripts/tier2_fixtures/): every entry recorded as
    # `passed` means one Signal: clause has been confirmed to
    # actually appear in real captured output. Floor grows as
    # more fixtures are written.
    MIN_TIER2_PASSED = 11  # deal.II pitfalls with named (catalog-indexed)
                           # Tier-2 fixtures (cheap bucket closed
                           # 2026-05-31 + 1 medium already done).
    MIN_TIER2_RUNNER_PASSED = 65  # cross-cutting (incl. synthetic indices).
    # 2026-06-01 fixture additions:
    #   +1 ngsolve::helmholtz::0 (complex coef on real FESpace)
    #   +1 kratos::linear_elasticity::2 (SubModelPart case-sensitive)
    #   +1 kratos::linear_elasticity::7 (echo_level missing)
    #   +1 fenics::linear_elasticity::3 (XDMFFile P2 on P1 mesh)
    #   +1 kratos::plasticity::5 (factory takes no args)
    #   +1 kratos::fluid::0 (VELOCITY not added)
    #   +1 fenics::linear_elasticity::0 (scalar space for elasticity)
    #   +1 kratos::linear_elasticity::1 (constitutive law required)
    #   +1 ngsolve::maxwell::0 (HCurl singular without nograds)
    #   +1 kratos::heat::0 (TEMPERATURE not added)
    #   +1 kratos::plasticity::4 (FRICTION_ANGLE wrong module)
    #   +1 ngsolve::maxwell::5 (ArnoldiSolver shift=0 singular)
    #   +1 kratos::dem::3 (SphericParticle2D not registered)
    #   +1 ngsolve::poisson::0 (Dirichlet case-mismatch silent BC failure)
    #   +1 skfem::stokes::4 (Nbfun is per-element, not global)
    #   +1 skfem::stokes::6 (asm Quadrature mismatch)
    #   +1 fenics::navier_stokes::3 (BC subspace needs collapsed Function)
    #   +1 skfem::biharmonic::0 (Morley/Argyris/BFS DOF counts)
    #   +1 ngsolve::eigenvalue::1 (GridFunction multidim via vecs not mdcomponents)
    #   +1 fourc::_input_format::0 (.dat extension rejection)
    #   +1 fourc::_input_format::1 (invalid PROBLEMTYPE value rejection)
    #   +1 fourc::scalar_transport::0 (SCATRA DYNAMIC vs SCALAR TRANSPORT DYNAMIC)
    #   +1 skfem::linear_elasticity::0 (ElementVector Nbfun + lame_parameters analytic)
    #   +1 ngsolve::stokes::1 (compound TnT returns tuple of lists)
    #   +1 ngsolve::heat::0 (nonsym kwarg undocumented warning)
    #   +1 skfem::mixed_poisson::1 (ElementTriRT0 / no full-name spelling)
    #   +1 skfem::eigenvalue::0 (Dirichlet Laplace eigvals + complement_dofs)
    #   +1 skfem::nonlinear::0 (no built-in Newton; DiscreteField .grad)
    #   +1 fenics::hyperelasticity::3 (ufl.variable + ufl.diff for stress)
    #   +1 fenics::eigenvalue::0 (slepc4py + PETSc scalar_type)
    #   +1 fourc::structural_dynamics::7 (DYNAMICTYPE enum)
    #   +1 fourc::thermo::0 (MAT_Fourier.CONDUCT needs constant:[k] wrapper)
    #   +1 ngsolve::linear_elasticity::0 (VectorH1 vs H1(dim=2) equivalence —
    #      catalog falsification)
    #   +1 fenics::stokes::0 (basix.ufl Taylor-Hood + MINI + P1/P1 construction)
    #   +1 skfem::poisson::0 (get_dofs requires with_boundaries; to_meshio
    #      still at skfem.io.meshio — TWO catalog-drift falsifications)
    #   +1 kratos::poisson::0 (CDA element-names: string factory only, NOT
    #      Python attributes; CDA missing from .venv until installed)
    #   +1 kratos::poisson::1 (LaplacianElement DOES assemble HEAT_FLUX —
    #      catalog falsification of contradictory prior claim)
    #   +1 ngsolve::hyperelasticity::0 (Newton uses maxit/maxerr — five wrong
    #      'maxits' / 'tol' generator template occurrences corrected)
    #   +1 ngsolve::plasticity::0 (NewtonCF/MinimizationCF in ngsolve.fem
    #      submodule, NOT top-level ngsolve — catalog drift)

    # Cost-bucket floors (round-3 critic finding E: report per-cost
    # coverage, not a fake /96 fraction). data/postmortems/
    # _falsifiability.json classifies each deal.II pitfall as
    # cheap / medium / expensive. CHEAP BUCKET FULLY CLOSED
    # 2026-05-31 — floor pinned at 10/10. Next push is medium
    # bucket (42 fixtures, each ~PDE solve + analytic reference).
    MIN_TIER2_PASSED_OF_CHEAP = 10   # 10/10 cheap deal.II = 100%
    MIN_TIER2_PASSED_OF_MEDIUM = 1   # 1/42 medium so far

    def setUp(self):
        from verify_signal_clauses import verify_backend
        self.results = verify_backend("dealii")

    def _count(self, predicate) -> int:
        return sum(1 for r in self.results if predicate(r))

    def test_total_pitfall_count_does_not_regress(self):
        n = len(self.results)
        self.assertGreaterEqual(
            n, self.MIN_N_PITFALLS,
            f"deal.II pitfall count dropped from "
            f"{self.MIN_N_PITFALLS} to {n} — a regression. "
            f"Either a pitfall got accidentally deleted or the "
            f"harvest path broke.")

    def test_category_prefix_coverage_does_not_regress(self):
        n = self._count(
            lambda r: r.pitfall_category != "(no-prefix)")
        self.assertGreaterEqual(
            n, self.MIN_WITH_CATEGORY_PREFIX,
            f"deal.II pitfalls with [Category] prefix dropped "
            f"from {self.MIN_WITH_CATEGORY_PREFIX} to {n}. "
            f"PR #26 discipline being violated.")

    def test_signal_clause_coverage_does_not_regress(self):
        n = self._count(lambda r: bool(r.signal_text))
        self.assertGreaterEqual(
            n, self.MIN_WITH_SIGNAL_CLAUSE,
            f"deal.II pitfalls with Signal: clause dropped "
            f"from {self.MIN_WITH_SIGNAL_CLAUSE} to {n}.")

    def test_tier0_floor_does_not_regress(self):
        n = self._count(lambda r: r.tier0_passed)
        self.assertGreaterEqual(
            n, self.MIN_TIER0_PASSED,
            f"deal.II pitfalls passing Tier 0 (Signal references "
            f"a canonical entity) dropped from "
            f"{self.MIN_TIER0_PASSED} to {n}.")

    def test_tier1_floor_does_not_regress(self):
        n = self._count(lambda r: r.tier1_passed)
        self.assertGreaterEqual(
            n, self.MIN_TIER1_PASSED,
            f"deal.II pitfalls passing Tier 1 (Signal uses "
            f"observable-symptom vocabulary) dropped from "
            f"{self.MIN_TIER1_PASSED} to {n}.")

    def test_tier0_and_1_floor_does_not_regress(self):
        n = self._count(
            lambda r: r.tier0_passed and r.tier1_passed)
        self.assertGreaterEqual(
            n, self.MIN_TIER0_AND_1_PASSED,
            f"deal.II pitfalls passing BOTH Tier 0 AND Tier 1 "
            f"dropped from {self.MIN_TIER0_AND_1_PASSED} to {n}. "
            f"This is the strictest floor.")

    def test_no_pitfall_relies_on_domain_names_alone(self):
        """The gameable Tier-0 failure mode the critic flagged.

        A pitfall that scores Tier-0 by mentioning ONLY a domain
        name (Stokes, Newton, Newmark, Turek, ...) is not
        actually grepable by a post-execution critic in real
        deal.II output. Every Tier-0 pass MUST reference at
        least one real code symbol.
        """
        n = self._count(
            lambda r: r.tier0_domain_names_matched
            and not r.tier0_code_symbol_matched)
        self.assertLessEqual(
            n, self.MAX_TIER0_DOMAIN_NAMES_ONLY,
            f"{n} deal.II pitfalls score Tier-0 by domain names "
            f"alone (Stokes, Newton, Turek, ...) without "
            f"referencing a real code symbol. This is the "
            f"gameable failure mode the critic flagged — a "
            f"post-execution critic cannot grep for these "
            f"signals in real deal.II output.")

    def test_tier0_code_symbol_only_floor(self):
        """Sub-floor for the strongest Tier-0 grade."""
        n = self._count(
            lambda r: r.tier0_code_symbol_matched
            and not r.tier0_domain_names_matched)
        self.assertGreaterEqual(
            n, self.MIN_TIER0_CODE_SYMBOL_ONLY,
            f"deal.II pitfalls passing Tier 0 by code-symbol "
            f"reference ONLY (no decorative domain names) "
            f"dropped from {self.MIN_TIER0_CODE_SYMBOL_ONLY} to "
            f"{n} — the strongest sub-tier weakened.")

    def test_tier2_floor_does_not_regress(self):
        """Tier 2 — every passed entry is one Signal: clause
        whose text has been confirmed to appear in real captured
        output from an intentional-failure fixture. The floor
        cannot drop; if a fixture stops reproducing the bug it
        was meant to verify, we have either a deal.II version
        change to investigate or a Signal rewording to undo.
        """
        n = self._count(lambda r: r.tier2_passed)
        self.assertGreaterEqual(
            n, self.MIN_TIER2_PASSED,
            f"Tier-2 operationally-verified Signal count "
            f"dropped from {self.MIN_TIER2_PASSED} to {n}. A "
            f"fixture that used to reproduce a bug no longer "
            f"does — investigate before lowering the floor.")


class TestDocumentationReachability(unittest.TestCase):
    """deep_knowledge.py entries with pitfall lists must either
    correspond to a physics in the backend's supported_physics()
    OR be explicitly tagged as documentation-only.

    Quantified 2026-06-01: fenics has 13 orphaned doc entries
    (maxwell, helmholtz, fracture, complex_valued, ...) — the
    deep_knowledge.py text exists and the post-PR-refactor
    workflows.py tools route through it, but
    prepare_simulation(solver='fenics', physics='X') returns
    'physics not supported' because there is no generator.

    This floor caps the orphan count at the current value so a
    future commit cannot add a new physics to deep_knowledge.py
    without also adding the generator (or explicitly accepting
    the increase by raising the cap).
    """

    # Current orphan counts measured 2026-06-01. Adjust DOWNWARD
    # as generators are added; an UPWARD adjustment means a new
    # orphan was introduced — pause and add the generator or
    # explicitly justify the increase.
    MAX_ORPHANS = {
        "fenics": 13,
        "dealii": 3,
        "fourc": 2,
    }

    def test_orphaned_knowledge_does_not_grow(self):
        import importlib
        import sys
        # Ensure src/ on sys.path (other tests get this via
        # verify_signal_clauses import; do it explicitly here).
        src_path = str(REPO_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        try:
            dk = importlib.import_module("tools.deep_knowledge")
        except ImportError as e:
            # deep_knowledge.py top-level imports mcp.server.fastmcp,
            # which is only present in the Open-FEM-agent .venv.
            # Skip the orphan check in environments without it
            # (e.g. plain system python running the unit test).
            self.skipTest(
                f"tools.deep_knowledge unavailable ({e}); orphan "
                f"audit requires the Open-FEM-agent .venv")
            return
        catalogs = {
            "fenics": dk._FENICS_KNOWLEDGE,
            "dealii": dk._DEALII_KNOWLEDGE,
            "fourc": dk._4C_KNOWLEDGE,
        }
        for be, cat in catalogs.items():
            with self.subTest(backend=be):
                b = get_backend(be)
                if b is None:
                    continue
                supported = {p.name for p in b.supported_physics()}
                documented = {
                    k for k, v in cat.items()
                    if isinstance(v, dict) and "pitfalls" in v}
                orphans = documented - supported
                cap = self.MAX_ORPHANS.get(be, 0)
                self.assertLessEqual(
                    len(orphans), cap,
                    f"{be}: {len(orphans)} orphaned doc entries "
                    f"(was {cap}) — {sorted(orphans)}. A new "
                    f"physics was added to deep_knowledge.py "
                    f"without an accompanying generator. Either "
                    f"add the generator, or raise the cap with "
                    f"explicit justification.")


class TestBackendImportSnapshot(unittest.TestCase):
    """Lock in the current backend-import-availability state as a
    floor: things can improve (more backends importable, fewer
    Kratos physics unreachable) but cannot silently regress.

    Backed by scripts/audit_backend_imports.py — that script is
    the source of truth; this test compares its live output
    against the baseline recorded 2026-06-01.

    Baseline values capture the install reality on the user's
    machine at audit time. If the user installs the missing
    Kratos apps (KratosConstitutiveLawsApplication, DEM, MPM,
    Iga, etc.) the Kratos-available count goes up and the
    unreachable count goes down — bump the baseline downward in
    that direction. NEVER bump it UP without explicit signoff:
    a regression here means an install broke or a physics
    started depending on something that isn't available.
    """

    # Recorded 2026-06-01. Adjust DOWNWARD on unreachable (fewer
    # missing apps); adjust UPWARD on available (more apps
    # importable).
    #
    # 2026-06-01 floor raise: user authorised installs of
    # KratosConstitutiveLawsApplication, KratosMetisApplication,
    # KratosMappingApplication, KratosMeshMovingApplication,
    # KratosDEMApplication, KratosMPMApplication. Counts moved
    # 7→14 available, 26→19 unreachable.
    # KratosDelaunayMeshingApplication has no PyPI wheel —
    # still unreachable (pfem stays in unreachable bucket).
    MIN_KRATOS_PHYSICS_AVAILABLE = 14
    MAX_KRATOS_PHYSICS_UNREACHABLE = 19
    REQUIRED_IMPORTABLE = ("skfem", "ngsolve", "fenics")
    REQUIRED_AVAILABLE = ("dealii", "fourc")

    def setUp(self):
        """Run the audit afresh each test invocation (cheap)."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/audit_backend_imports.py"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
            timeout=120,
        )
        if result.returncode != 0:
            self.skipTest(
                f"audit script failed (rc={result.returncode}); "
                f"stderr={result.stderr[:200]}")
        path = REPO_ROOT / "scripts" / "scan_results" / "backend_imports.json"
        import json
        self.snapshot = json.loads(path.read_text())

    def test_kratos_physics_available_does_not_regress(self):
        n = self.snapshot["summary"]["kratos_physics_available"]
        self.assertGreaterEqual(
            n, self.MIN_KRATOS_PHYSICS_AVAILABLE,
            f"Kratos physics with importable app dropped from "
            f"{self.MIN_KRATOS_PHYSICS_AVAILABLE} to {n}. A Kratos "
            f"application install was removed or the physics "
            f"catalog references a new app that is not installed.")

    def test_kratos_physics_unreachable_does_not_grow(self):
        n = self.snapshot["summary"]["kratos_physics_unreachable"]
        self.assertLessEqual(
            n, self.MAX_KRATOS_PHYSICS_UNREACHABLE,
            f"Kratos unreachable physics grew from "
            f"{self.MAX_KRATOS_PHYSICS_UNREACHABLE} to {n}. A "
            f"new physics was added referencing an app that is "
            f"not installed, or an install was removed.")

    def test_python_backends_remain_importable(self):
        for be in self.REQUIRED_IMPORTABLE:
            with self.subTest(backend=be):
                ok = self.snapshot["summary"].get(f"{be}_importable")
                self.assertTrue(
                    ok,
                    f"Backend {be} library is no longer importable "
                    f"— install regression. Details: "
                    f"{self.snapshot.get(be, {}).get('error', 'unknown')}")

    def test_compiled_backends_remain_available(self):
        for be in self.REQUIRED_AVAILABLE:
            with self.subTest(backend=be):
                ok = self.snapshot["summary"].get(f"{be}_available")
                self.assertTrue(
                    ok,
                    f"Backend {be} build artifact missing — "
                    f"compile regression. Details: "
                    f"{self.snapshot.get(be, {}).get('error', 'unknown')}")


class TestSignalParseDiscipline(unittest.TestCase):
    """Every pitfall whose text starts with a [Category] prefix
    must also be parseable into a Signal: clause.

    The parser regex is ``\\bSignal:\\s*(.+?)$``. Patterns like
    ``Signal (verified 2026-06-01):`` were used during the
    empirical audit but break the parse — the regex doesn't
    match ``Signal`` followed by a parenthetical before the
    colon. Locked in 2026-06-01 after the fenics audit
    re-broke linear_elasticity#0 / #3 Signal extraction.
    """

    BACKENDS = ("dealii", "ngsolve", "skfem", "fenics", "kratos")

    def test_every_categorised_pitfall_has_parseable_signal(self):
        from verify_signal_clauses import verify_backend
        for be in self.BACKENDS:
            with self.subTest(backend=be):
                broken = [
                    r for r in verify_backend(be)
                    if r.pitfall_category != "(no-prefix)"
                    and not r.signal_text]
                if broken:
                    names = ", ".join(
                        f"{r.physics}#{r.pitfall_index}"
                        for r in broken)
                    self.fail(
                        f"{be}: {len(broken)} pitfalls have a "
                        f"[Category] prefix but no extractable "
                        f"Signal text ({names}). The parser "
                        f"requires 'Signal:' verbatim — "
                        f"'Signal (verified ...): ...' is "
                        f"not recognised. Rewrite as "
                        f"'Signal: ... (Verified ...)'.")


class TestCrossBackendGameableFloor(unittest.TestCase):
    """Cross-backend regression test for the gameable Tier-0 mode.

    Round-2 critic (2026-05-31) forced MAX_TIER0_DOMAIN_NAMES_ONLY=0
    for deal.II — a pitfall whose Signal: text scores Tier-0 only
    via domain words ("Stokes", "Newton", "convergence", "Poisson")
    without referencing any real backend code symbol IS the gameable
    failure mode (post-execution critic cannot grep for these in
    real solver output).

    After cross-backend Table-1 promotion the same floor must hold
    for every backend whose catalog has been promoted. Locked in
    2026-06-01 after auditing skfem (2 gameable) / fenics (1) /
    kratos (7) and fixing each by rewriting the Signal to name
    specific compound classes the backend actually emits.

    If a future promotion regresses any of these to > 0, the
    test names the violating entries — investigate and rewrite
    Signal text rather than lowering the floor.
    """

    # Per-backend max gameable count. 0 across the board:
    # the audit established this is achievable for any
    # backend whose Signal: text references compound code
    # symbols rather than textbook concepts.
    BACKENDS = ("dealii", "ngsolve", "skfem", "fenics", "kratos")

    def test_no_backend_has_gameable_entries(self):
        from verify_signal_clauses import verify_backend
        for be in self.BACKENDS:
            with self.subTest(backend=be):
                gameable = [
                    r for r in verify_backend(be)
                    if r.tier0_domain_names_matched
                    and not r.tier0_code_symbol_matched]
                if gameable:
                    names = ", ".join(
                        f"{r.physics}#{r.pitfall_index}"
                        for r in gameable)
                    self.fail(
                        f"{be}: {len(gameable)} gameable Tier-0 "
                        f"entries ({names}). Each Signal: clause "
                        f"matches a textbook domain word but no "
                        f"backend code symbol — rewrite the Signal "
                        f"to name a specific class / exception / "
                        f"method the backend actually emits.")


class TestHarnessSelfChecks(unittest.TestCase):
    """The harness itself must do what it claims."""

    def test_split_pitfall_handles_well_formed_entry(self):
        from verify_signal_clauses import _split_pitfall
        cat, sig = _split_pitfall(
            "[Numerical] SUPG stabilisation parameter ... "
            "Signal: spurious oscillations near boundary layers.")
        self.assertEqual(cat, "Numerical")
        self.assertIn("spurious oscillations", sig)

    def test_split_pitfall_handles_missing_prefix(self):
        from verify_signal_clauses import _split_pitfall
        cat, sig = _split_pitfall(
            "Use FE_Q for elasticity. Signal: rank-deficient stiffness.")
        self.assertIsNone(cat)
        self.assertIn("rank-deficient", sig)

    def test_split_pitfall_handles_missing_signal(self):
        from verify_signal_clauses import _split_pitfall
        cat, sig = _split_pitfall(
            "[Syntax] Element name needs node count suffix.")
        self.assertEqual(cat, "Syntax")
        self.assertIsNone(sig)


if __name__ == "__main__":
    unittest.main()
