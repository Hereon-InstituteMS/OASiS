"""Regression gate for the EXECUTION-VERIFIED DUNE-fem API surface.

`src/backends/dune/generators/verified_api.py` holds the part of the
DUNE-fem catalog that was established by RUNNING scripts against the
installed dune-fem (2.12.0.2) rather than by reading documentation.
These tests pin that content so it cannot silently rot:

  * the executed sections are actually reachable through
    backend.get_knowledge("_general") and
    backend.get_knowledge("poisson");
  * every executed pitfall keeps its [Category] prefix, its Signal:
    clause and its provenance date — dune sits at the 100%
    Signal-coverage floor and a bare pitfall would regress the
    verify_signal_clauses gate;
  * the measured numbers that the Tier-2 fixtures assert against are
    still recorded in the knowledge (a fixture without the matching
    claim, or a claim without the matching fixture, is a half-finished
    verification);
  * the Tier-2 fixture pitfall_index values still point at the pitfall
    they were written for. Indices are POSITIONAL, so inserting a
    pitfall in the middle of the list silently re-points every fixture
    after it — this test is the tripwire;
  * claims FALSIFIED by execution stay falsified. Three catalog
    statements were contradicted by the installed package and must not
    creep back: the ~/.dune JIT cache path, the
    dune.fem.space.product_space factory, and the lowercase
    raviartthomas spelling.

These tests do NOT require dune.fem — they pin the catalog only. The
live gate is the Tier-2 fixture set under
scripts/tier2_fixtures/dune/, which re-runs the measurements.
"""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from backends.dune.generators.verified_api import (  # noqa: E402
    EXECUTED_API, EXECUTED_PITFALLS)

_FIXTURES = _REPO / "scripts" / "tier2_fixtures" / "dune"

# Sections that must survive any future edit — each is a distinct
# execution campaign, not a restatement of the others.
_REQUIRED_SECTIONS = (
    "install_under_test",
    "grid_cell_types_measured",
    "ufl_cell_is_always_a_simplex",
    "space_construction_measured",
    "dirichlet_bc_measured",
    "silent_wrong_traps_measured",
    "solver_control_measured",
    "integrate_measured",
    "jit_compilation_measured",
    "convergence_behaviour_2d_poisson",
    "runtime_constants_measured",
    "vtk_output_measured",
    "phantom_apis_checked",
    "module_inventory_2_12",
    # added by the 2026-08-03 knowledge expansion
    "natural_bc_measured",
    "assemble_measured",
    "companion_modules_measured",
)


class TestVerifiedApiModule(unittest.TestCase):

    def test_all_required_sections_present(self) -> None:
        missing = [s for s in _REQUIRED_SECTIONS if s not in EXECUTED_API]
        self.assertEqual(
            missing, [],
            f"verified_api.EXECUTED_API lost sections {missing}. Each "
            f"section is a separate execution campaign; deleting one "
            f"discards evidence that cost a live run to obtain.")

    def test_install_fingerprint_is_recorded(self) -> None:
        # Without the version the measurements are unattributable.
        self.assertIn("2.12.0.2", EXECUTED_API["install_under_test"])
        self.assertIn("dune-fem", EXECUTED_API["install_under_test"])

    def test_every_section_carries_a_provenance_date(self) -> None:
        for name in _REQUIRED_SECTIONS:
            blob = json.dumps(EXECUTED_API[name])
            self.assertRegex(
                blob, r"20\d\d-\d\d-\d\d",
                f"section {name!r} has no provenance date — an "
                f"undated measurement cannot be re-checked")

    def test_sections_with_a_signal_use_the_category_prefix(self) -> None:
        for name, body in EXECUTED_API.items():
            if not isinstance(body, dict) or "Signal" not in body:
                continue
            sig = body["Signal"]
            self.assertRegex(
                sig, r"^\[(API|Numerical|Performance|Syntax|Input|"
                     r"Integration)\]",
                f"_general section {name!r} Signal lacks a "
                f"[Category] prefix")


class TestExecutedPitfalls(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import get_backend, load_all_backends
        load_all_backends()
        cls.backend = get_backend("dune")
        if cls.backend is None:                     # pragma: no cover
            raise unittest.SkipTest("dune backend not registered")
        cls.poisson = cls.backend.get_knowledge("poisson")["pitfalls"]

    def test_executed_pitfalls_are_appended_to_poisson(self) -> None:
        for pit in EXECUTED_PITFALLS:
            self.assertIn(
                pit, self.poisson,
                "an executed pitfall is no longer reachable through "
                "get_knowledge('poisson') — knowledge(topic='pitfalls') "
                "walks supported_physics() and never sees _general, so "
                "dropping it from poisson hides it from every model")

    def test_every_executed_pitfall_is_well_formed(self) -> None:
        for i, pit in enumerate(EXECUTED_PITFALLS):
            with self.subTest(i=i):
                self.assertRegex(
                    pit, r"^\[(API|Numerical|Performance|Syntax|Input|"
                         r"Integration)\] ",
                    "missing [Category] prefix")
                self.assertIn("Signal:", pit, "missing Signal: clause")
                self.assertRegex(
                    pit, r"Executed 20\d\d-\d\d-\d\d",
                    "missing an 'Executed <date>' provenance note — "
                    "this list is for claims proven by running code")

    def test_general_knowledge_exposes_the_executed_sections(self) -> None:
        general = self.backend.get_knowledge("_general")
        for name in _REQUIRED_SECTIONS:
            self.assertIn(
                name, general,
                f"_general lost {name!r}; generators/__init__.py must "
                f"keep merging verified_api.EXECUTED_API")


class TestMeasuredNumbersSurvive(unittest.TestCase):
    """The specific numbers the Tier-2 fixtures re-measure.

    If a future edit prettifies the prose and drops the numbers, the
    claims become unfalsifiable again — which is the exact failure mode
    this whole verification effort exists to prevent.
    """

    def _blob(self) -> str:
        return json.dumps(EXECUTED_API) + "\n".join(EXECUTED_PITFALLS)

    def test_silent_bc_trap_numbers_recorded(self) -> None:
        blob = self._blob()
        for needle in ("23935", "7.510512e+14", "1.899705e-03"):
            self.assertIn(
                needle, blob,
                f"the measured value {needle} from the "
                f"2026-08-03 silent-Dirichlet run is gone")

    def test_solver_enumeration_recorded(self) -> None:
        blob = self._blob()
        self.assertIn("fem.solver.linear.method", blob)
        for name in ("gmres", "cg", "bicgstab"):
            self.assertIn(name, blob)

    def test_grid_cell_counts_recorded(self) -> None:
        blob = self._blob()
        # 4x4 structuredGrid -> 16 quadrilaterals; aluConformGrid -> 32
        # triangles. Both counts are what the fixture asserts.
        self.assertIn("quadrilateral", blob)
        self.assertIn("hexahedron", blob)
        self.assertIn("tetrahedron", blob)

    def test_quadrature_measurements_recorded(self) -> None:
        blob = self._blob()
        self.assertIn("0.405284734569", blob)
        self.assertIn("5.3e-02", blob)

    def test_no_measured_convergence_orders_in_reachable_knowledge(self):
        """The inverse of what this test used to assert.

        An earlier revision PINNED the measured MMS orders and the
        finest absolute errors into EXECUTED_API, which meant the
        knowledge an agent reaches through
        knowledge(topic="overview", solver="dune") contained the answer
        to a convergence study — and that a future purge could not
        remove it without failing this file. Adversarial audit
        2026-08-03 inverted it: the numbers belong to the Tier-2 gate
        that re-measures them, not to the catalog. This is now the
        guard that keeps them out.
        """
        from backends.dune.generators import KNOWLEDGE  # noqa: E402
        blob = json.dumps(EXECUTED_API, default=str) + json.dumps(
            KNOWLEDGE, default=str)
        forbidden = (
            # per-order observed EOCs
            "1.989", "1.997", "2.979", "2.995", "3.985", "3.996",
            "1.894", "1.972", "1.993", "2.981",
            # finest absolute L2 errors the fixture floors derive from
            "4.556318e-04", "3.846536e-06", "2.180413e-08",
            "1.312335e-03", "8.602468e-06",
            # the 3D variable-coefficient family's own rate sequences,
            # which survived the 2026-08-03 audit inside
            # KNOWLEDGE["poisson_mms"]["expected_order"] because that
            # audit only swept verified_api.py. Widening the guard to
            # the whole KNOWLEDGE dict is what caught them.
            "1.984/1.996", "1.028/1.007", "1.628(pre-asymptotic)",
            "1.059(pre-asymptotic)",
        )
        leaked = [n for n in forbidden if n in blob]
        self.assertEqual(
            leaked, [],
            f"measured convergence results leaked back into "
            f"agent-reachable DUNE knowledge: {leaked}. Knowledge "
            f"states how the code behaves; a measured order is the "
            f"ANSWER to a study the agent may be asked to run. Keep "
            f"them in scripts/tier2_fixtures/dune/"
            f"poisson_mms_convergence/ instead.")

    def test_convergence_section_still_carries_its_guidance(self) -> None:
        """Stripping the numbers must not strip the diagnostic value."""
        conv = EXECUTED_API["convergence_behaviour_2d_poisson"]
        blob = json.dumps(conv, default=str)
        self.assertIn("Signal", conv)
        for needle in ("k+1", "globalRefine", "linear.tolerance"):
            self.assertIn(needle, blob,
                          f"convergence guidance lost {needle!r}")


class TestFalsifiedClaimsStayFalsified(unittest.TestCase):
    """Three catalog statements were contradicted by the install."""

    def _dune_catalog_text(self) -> str:
        root = _REPO / "src" / "backends" / "dune"
        return "\n".join(p.read_text(encoding="utf-8")
                         for p in sorted(root.rglob("*.py")))

    def test_no_home_dune_jit_cache_claim(self) -> None:
        text = self._dune_catalog_text()
        # The literal path ~/.dune/dune-py does not exist on dune-fem
        # 2.12; getDunePyDir() yields <sys.prefix>/.cache/dune-py or
        # ~/.cache/dune-py. Mentioning it as a NEGATIVE ("NOT ~/.dune")
        # is fine; asserting it as the cache location is not.
        bad = re.findall(r"~/\.dune/dune-py/python", text)
        self.assertEqual(
            bad, [],
            "the falsified ~/.dune/dune-py/python cache path is back "
            "in the dune catalog; the measured location is "
            "<sys.prefix>/.cache/dune-py/python/dune/generated")
        self.assertIn(".cache/dune-py", text)

    # Markers that turn a `product_space` mention into a DENIAL rather
    # than a claim: the template's own import alias, or prose saying
    # the name is an alias / absent / falsified.
    _PRODUCT_SPACE_DENIALS = (
        "import product as product_space",
        "NO dune.fem.space.product_space",
        "alias",
        "ABSENT",
        "FALSIFIED",
    )

    def test_product_space_is_not_claimed_as_an_api(self) -> None:
        text = self._dune_catalog_text()
        for m in re.finditer(r"product_space", text):
            window = text[max(0, m.start() - 300):m.start() + 160]
            self.assertTrue(
                any(d in window for d in self._PRODUCT_SPACE_DENIALS),
                "product_space is being presented as a real "
                "dune.fem.space API again; hasattr() is False on "
                "dune-fem 2.12.0.2 and the string does not occur "
                "anywhere in the installed package. Context:\n"
                + window)

    # A lowercase mention is benign only when the matched text sits
    # INSIDE one of these exact literals. A nearby word like
    # "lowercase" is not enough — any re-introduced API use would sit
    # beside explanatory prose containing it, which would make the
    # check unfalsifiable (audit finding 2026-08-03).
    _RAVIART_BENIGN = (
        "raviartthomas.hh",
        "dune.fem.space.raviartthomas",
        "'raviartthomas'",
        '"raviartthomas"',
    )

    def test_lowercase_raviartthomas_absent(self) -> None:
        text = self._dune_catalog_text()
        spans = []
        for lit in self._RAVIART_BENIGN:
            start = text.find(lit)
            while start != -1:
                spans.append((start, start + len(lit)))
                start = text.find(lit, start + 1)
        for m in re.finditer(r"\braviartthomas\b", text):
            self.assertTrue(
                any(lo <= m.start() and m.end() <= hi
                    for lo, hi in spans),
                "the lowercase raviartthomas spelling is back as an "
                "API name; the Python factory is raviartThomas. "
                "Context:\n"
                + text[max(0, m.start() - 200):m.start() + 160])


class TestTier2FixtureWiring(unittest.TestCase):
    """The fixtures and the pitfalls they verify must stay aligned."""

    # fixture directory -> the substring that must appear in the
    # pitfall sitting at that fixture's pitfall_index.
    _EXPECTED = {
        "dirichletbc_not_in_scheme_list_silent": "IN THE LIST",
        "solver_name_not_validated": "fem.solver.linear.method",
        "ufl_cell_reports_simplex_on_cube_grid": "SIMPLEX UFL cell",
        "boundary_ids_are_geometric_and_one_based": "ds(id) works",
        "companion_modules_not_installed": "dune.femdg",
    }

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import get_backend, load_all_backends
        load_all_backends()
        cls.backend = get_backend("dune")
        if cls.backend is None:                     # pragma: no cover
            raise unittest.SkipTest("dune backend not registered")

    def test_fixtures_exist_and_are_parseable(self) -> None:
        self.assertTrue(_FIXTURES.is_dir())
        for name in (*self._EXPECTED,
                     "poisson_mms_convergence",
                     "raviartThomas_camelcase_not_lowercase",
                     "preconditioner_enumeration_is_short",
                     "componentwise_bc_corner_drops_a_constraint"):
            d = _FIXTURES / name
            self.assertTrue((d / "fixture.json").is_file(),
                            f"missing fixture.json for {name}")
            meta = json.loads((d / "fixture.json").read_text())
            self.assertEqual(meta["backend"], "dune")
            self.assertTrue(meta["expect_in_output"],
                            f"{name} has no expect_in_output")

    def test_fixture_indices_point_at_the_intended_pitfall(self) -> None:
        pitfalls = self.backend.get_knowledge("poisson")["pitfalls"]
        for name, needle in self._EXPECTED.items():
            with self.subTest(name):
                meta = json.loads(
                    (_FIXTURES / name / "fixture.json").read_text())
                self.assertEqual(meta["physics"], "poisson")
                idx = int(meta["pitfall_index"])
                self.assertLess(
                    idx, len(pitfalls),
                    f"{name}: pitfall_index {idx} is past the end of "
                    f"the poisson pitfall list ({len(pitfalls)})")
                self.assertIn(
                    needle, pitfalls[idx],
                    f"{name}: pitfall_index {idx} no longer points at "
                    f"the pitfall this fixture verifies. Pitfall "
                    f"indices are POSITIONAL — inserting an entry "
                    f"earlier in the list re-points every fixture "
                    f"after it. Fix the index in fixture.json (and "
                    f"check the other dune fixtures too).")

    def test_fixture_keys_are_unique(self) -> None:
        seen: dict[tuple, str] = {}
        for d in sorted(_FIXTURES.iterdir()):
            if not (d / "fixture.json").is_file():
                continue
            meta = json.loads((d / "fixture.json").read_text())
            key = (meta["backend"], meta["physics"],
                   int(meta["pitfall_index"]))
            self.assertNotIn(
                key, seen,
                f"fixture key collision {key}: {d.name} vs "
                f"{seen.get(key)} — run_tier2_fixtures.py stores "
                f"results per key, so the second would overwrite the "
                f"first")
            seen[key] = d.name

    def test_mms_fixture_records_its_measured_orders(self) -> None:
        meta = json.loads(
            (_FIXTURES / "poisson_mms_convergence"
             / "fixture.json").read_text())
        comment = meta["_comment"]
        for needle in ("1.989", "2.979", "3.985", "2.12.0.2"):
            self.assertIn(
                needle, comment,
                "the MMS fixture must record the orders it was "
                "calibrated against, so a future drift is legible")


if __name__ == "__main__":
    unittest.main()


class TestEveryPhysicsEntryIsSelfSufficient(unittest.TestCase):
    """Structural gate, not a content gate.

    Project-owner requirement (2026-08-03): the DUNE knowledge has to be
    usable by a SMALL model that will not infer, will not hunt for a
    second payload and will not recover from a partial answer. Three
    properties make that possible and are pinned here:

      * every physics entry opens with a COMPLETE runnable script, not
        a fragment — before this gate the scripts existed only inside
        the GENERATORS callables, which knowledge(topic="physics") does
        not reach;
      * the load-bearing keys come FIRST in the entry, because a model
        that stops reading early must still have the answer;
      * every entry names the other lookups, so nothing depends on the
        reader already knowing a topic= string.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from backends.dune.generators import GENERATORS, KNOWLEDGE
        cls.GENERATORS = GENERATORS
        cls.KNOWLEDGE = KNOWLEDGE
        cls.physics = [k for k in KNOWLEDGE if k != "_general"]

    def test_every_physics_entry_leads_with_a_runnable_example(self):
        for name in self.physics:
            with self.subTest(name):
                entry = self.KNOWLEDGE[name]
                keys = list(entry)
                self.assertEqual(
                    keys[0], "READ_THIS_FIRST",
                    f"{name}: the orientation line must come first")
                self.assertEqual(
                    keys[1], "minimal_working_example",
                    f"{name}: the complete script must be the second "
                    f"key; a model that stops reading after two keys "
                    f"has to already have something it can run")
                script = entry["minimal_working_example"]
                self.assertIsInstance(script, str)
                for needle in ("import", "print("):
                    self.assertIn(needle, script,
                                  f"{name}: example is not a script")
                self.assertIn(
                    "DUNE_TEMPLATE_COMPLETE", script,
                    f"{name}: the example must print the terminal "
                    f"sentinel — a DUNE run can exit 134 after "
                    f"printing every correct result, so the sentinel "
                    f"is what distinguishes 'finished' from 'died'")

    def test_every_physics_entry_names_the_other_lookups(self):
        for name in self.physics:
            with self.subTest(name):
                where = self.KNOWLEDGE[name].get("where_else_to_look")
                self.assertIsInstance(
                    where, dict,
                    f"{name}: nothing tells the reader how to reach "
                    f"the rest of the catalog")
                blob = json.dumps(where)
                for needle in ("topic='overview'", "topic='pitfalls'",
                               "topic='physics'"):
                    self.assertIn(needle, blob,
                                  f"{name}: {needle} not advertised")

    def test_general_leads_with_start_here(self):
        general = self.KNOWLEDGE["_general"]
        self.assertEqual(list(general)[0], "START_HERE")
        start = general["START_HERE"]
        self.assertIn("complete_runnable_poisson", start)
        script = start["complete_runnable_poisson"]
        for needle in ("from dune.grid import structuredGrid",
                       "from dune.fem.scheme import galerkin",
                       "scheme.solve(target=uh)"):
            self.assertIn(needle, script)
        self.assertIn("REQUIRED_imports", start)
        self.assertIn("OPTIONAL_imports", start)

    def test_every_generator_emits_the_terminal_sentinel(self):
        for variant, gen in self.GENERATORS.items():
            with self.subTest(variant):
                script = gen({})
                self.assertIn(
                    "DUNE_TEMPLATE_COMPLETE", script,
                    f"{variant} does not print the terminal sentinel")
                self.assertTrue(
                    script.rstrip().endswith(
                        'print("DUNE_TEMPLATE_COMPLETE")'),
                    f"{variant}: the sentinel must be the LAST "
                    f"statement, otherwise it does not prove the run "
                    f"reached the end")


class TestAdvertisedButAbsentStaysRetired(unittest.TestCase):
    """dune-fem-dg and dune-vem are not importable on a dune-fem install.

    The catalog advertised "Comprehensive DG methods via dune-fem-dg"
    and "VEM (Virtual Element Method) support" as backend capabilities,
    and named DIRK23 / SDIRK22 / Heun / SSP-RK steppers with no API
    anywhere. Execution on 2026-08-03 refuted the premise: `import
    dune.femdg`, `import dune.fem.dg` and `import dune.vem` all raise
    ModuleNotFoundError. This keeps the capability claims from creeping
    back as bare assertions.
    """

    def _catalog_text(self) -> str:
        root = _REPO / "src" / "backends" / "dune"
        return "\n".join(p.read_text(encoding="utf-8")
                         for p in sorted(root.rglob("*.py")))

    def test_dune_fem_dg_is_named_only_as_absent(self) -> None:
        text = self._catalog_text()
        self.assertTrue(
            re.search(r"dune\.femdg.{0,400}ModuleNotFoundError", text,
                      re.S) or
            re.search(r"ModuleNotFoundError.{0,400}dune\.femdg", text,
                      re.S),
            "the catalog no longer records that dune.femdg is "
            "unimportable on this install")
        for m in re.finditer(r"pip install dune-fem-dg", text):
            self.fail("the catalog tells the reader to pip install "
                      "dune-fem-dg as if that made it part of this "
                      "backend; it is a separate package and was "
                      "measured absent")

    def test_vem_claim_is_negative(self) -> None:
        from backends.dune.generators import KNOWLEDGE
        vem = KNOWLEDGE["_general"]["vem"]
        self.assertIn("NOT AVAILABLE", vem)
        self.assertIn("ModuleNotFoundError", vem)


if __name__ == "__main__":                                  # pragma: no cover
    unittest.main()
