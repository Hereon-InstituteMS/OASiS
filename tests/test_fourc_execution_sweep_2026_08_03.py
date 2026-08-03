"""Regression: the 2026-08-03 4C execution sweep.

Every claim pinned here was produced by writing a minimal .4C.yaml,
running the deployed binary ``/home/alexander/4C/build/4C``
(4C 2026.2.0-dev, git 89519cf) and recording what actually happened.
The full probe log lives in the Tier-2 fixtures under
``scripts/tier2_fixtures/fourc/``:

  * ``result_description_gates_the_exit_code``
  * ``runtime_vtk_output_needs_both_sections``
  * ``thermo_prefixed_dirich_silently_ignored``
  * ``maxtime_truncates_numstep_silently``
  * ``restartevery_section_placement``
  * ``material_parameter_validation_bounds``
  * ``problem_size_is_optional_and_inert``
  * ``neumann_type_enum_vs_element_support``
  * ``structural_2d_solid_quad4_not_wall`` (rewritten era-agnostic)

This test is GEN-ONLY — it needs no 4C binary, so it runs in CI. It
guards the three things that a later edit could silently undo:

  1. the knowledge entries themselves are still present and still
     carry the executed-provenance marker, so nobody can quietly
     replace an executed claim with an inherited one;
  2. the Tier-2 fixture ``pitfall_index`` values still point at the
     pitfalls they were written for — index drift is invisible at
     review time and silently decouples a fixture from its claim
     (exactly the bug found in ``structural_dynamic_dynamictype_enum``
     on 2026-08-03, whose index 7 pointed past the end of a 7-entry
     list);
  3. the corrections this sweep made are not re-reverted: PROBLEM
     SIZE stays out of the mandatory-section list, and the
     ``result_description`` reference block keeps all 18 field-group
     names that ``valid_result_lines()`` actually registers.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

_FIXTURE_DIR = _REPO / "scripts" / "tier2_fixtures" / "fourc"

# Marker every entry added by the sweep carries. Chosen so that a
# re-worded claim keeps the marker only if it is still backed by a run.
_PROVENANCE = "Verified by execution 2026-08-03"

# (physics, pitfall_index, substring that must still be in that entry)
EXECUTED_ENTRIES = [
    ("input_format", 17, "RESULT DESCRIPTION"),
    ("input_format", 18, "line-buffered"),
    ("input_format", 19, "loose TOLERANCE"),
    ("input_format", 20, "expected 1 tests but performed 0"),
    ("input_format", 21, "PROBLEM SIZE is OPTIONAL"),
    ("input_format", 22, "not available."),
    ("input_format", 23, "has incorrect size"),
    ("input_format", 24, "not in range"),
    ("input_format", 25, "Unknown type of SurfaceNeumann condition"),
    ("input_format", 26, "INTERVAL_STEPS"),
    ("input_format", 27, "RESTARTEVERY"),
    ("input_format", 28, "--parameters"),
    ("structural_mechanics", 7, "VERSION-DEPENDENT"),
    ("structural_mechanics", 8, "EAS"),
    ("structural_mechanics", 9, "Expected parameter 'DENS'"),
    ("structural_mechanics", 10, "in_range[-1,0.5)"),
    ("structural_mechanics", 11, "Floating point exception (8)"),
    ("structural_mechanics", 12, "KINEM nonlinear"),
    ("thermal", 3, "SILENTLY DROPPED"),
    ("structural_dynamics", 7, "singular matrix"),
    ("structural_dynamics", 8, "Finalised step"),
    ("structural_dynamics", 9, "RESTARTEVERY"),
]

# The 18 groups registered by valid_result_lines() in
# src/global_legacy_module/4C_global_legacy_module.cpp. 4C echoes this
# exact list when an unknown group name is used, so it is directly
# checkable against the binary.
RESULT_GROUPS = {
    "STRUCTURE", "FLUID", "XFLUID", "ALE", "THERMAL", "LUBRICATION",
    "POROFLUIDMULTIPHASE", "SCATRA", "SSI", "SSTI", "STI",
    "RED_AIRWAY", "ARTNET", "FSI", "PARTICLE", "PARTICLEWALL",
    "RIGIDBODY", "CARDIOVASCULAR0D",
}


def _backend():
    from core.registry import load_all_backends, get_backend
    load_all_backends()
    return get_backend("fourc")


class TestExecutedEntriesPresent(unittest.TestCase):
    """Each swept claim is still there, still marked as executed."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = _backend()
        if cls.backend is None:
            raise unittest.SkipTest("fourc backend not registered")

    def test_entries_present_and_carry_execution_provenance(self):
        for physics, idx, needle in EXECUTED_ENTRIES:
            with self.subTest(physics=physics, index=idx):
                pitfalls = self.backend.get_knowledge(physics).get(
                    "pitfalls", [])
                self.assertGreater(
                    len(pitfalls), idx,
                    f"fourc::{physics} shrank to {len(pitfalls)} "
                    f"pitfalls; index {idx} no longer exists. If an "
                    f"entry was intentionally removed, remove it from "
                    f"EXECUTED_ENTRIES and from the Tier-2 fixture "
                    f"that references it in the same commit.")
                entry = pitfalls[idx]
                self.assertIn(
                    needle, entry,
                    f"fourc::{physics}#{idx} no longer contains "
                    f"{needle!r} — the pitfall list was probably "
                    f"reordered, which silently decouples the Tier-2 "
                    f"fixtures from their claims.")
                self.assertIn(
                    _PROVENANCE, entry,
                    f"fourc::{physics}#{idx} lost its "
                    f"{_PROVENANCE!r} marker. Executed claims must "
                    f"keep the note that says what was run; an "
                    f"unverified entry is worse than no entry.")

    def test_every_executed_entry_is_signal_parseable(self):
        """The sweep must not weaken the Signal: discipline."""
        sys.path.insert(0, str(_REPO / "scripts"))
        from verify_signal_clauses import _split_pitfall
        for physics, idx, _ in EXECUTED_ENTRIES:
            with self.subTest(physics=physics, index=idx):
                entry = self.backend.get_knowledge(physics)["pitfalls"][idx]
                cat, sig = _split_pitfall(entry)
                self.assertIsNotNone(
                    cat, f"fourc::{physics}#{idx} lost its [Category] "
                         f"prefix")
                self.assertTrue(
                    sig, f"fourc::{physics}#{idx} lost its parseable "
                         f"'Signal:' clause")


class TestFixtureIndicesStillPoint(unittest.TestCase):
    """Tier-2 fixture pitfall_index values must stay in range.

    A fixture whose index runs off the end of its pitfall list still
    'passes' the runner but verifies nothing, because the Signal
    harness looks the result up by ``backend::physics::index``. That
    is how ``structural_dynamic_dynamictype_enum`` sat at index 7 of a
    7-entry list until 2026-08-03.
    """

    def test_all_fourc_fixture_indices_are_in_range(self):
        backend = _backend()
        if backend is None:
            self.skipTest("fourc backend not registered")
        supported = {p.name for p in backend.supported_physics()}
        for fixture in sorted(_FIXTURE_DIR.glob("*/fixture.json")):
            meta = json.loads(fixture.read_text(encoding="utf-8"))
            physics = meta.get("physics", "")
            idx = int(meta.get("pitfall_index", -1))
            with self.subTest(fixture=fixture.parent.name):
                if physics not in supported:
                    # Legacy '_'-prefixed names never resolved to a
                    # physics row; they are recorded but unlinked.
                    self.assertTrue(
                        physics.startswith("_"),
                        f"{fixture.parent.name} targets physics "
                        f"{physics!r}, which fourc does not support "
                        f"and which is not a legacy '_' name.")
                    continue
                pitfalls = backend.get_knowledge(physics).get(
                    "pitfalls", [])
                self.assertLess(
                    idx, len(pitfalls),
                    f"{fixture.parent.name} points at "
                    f"fourc::{physics}#{idx} but that list only has "
                    f"{len(pitfalls)} entries — the fixture verifies "
                    f"nothing. Fix pitfall_index (and prefer appending "
                    f"new pitfalls at the END of a list so existing "
                    f"indices stay valid).")


class TestSweepCorrectionsHold(unittest.TestCase):
    """The two catalogue corrections must not be re-reverted."""

    @classmethod
    def setUpClass(cls) -> None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_fourc_knowledge", _REPO / "data" / "fourc_knowledge.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls.knowledge = mod.FOURC_KNOWLEDGE

    def test_problem_size_is_not_listed_as_mandatory(self):
        mandatory = self.knowledge["input_format"]["mandatory_sections"]
        for entry in mandatory:
            self.assertNotIn(
                "PROBLEM SIZE", entry,
                "PROBLEM SIZE is declared {.required = false} in "
                "4C_global_legacy_module_validparameters.cpp and a "
                "deck without it runs to completion (verified by "
                "execution 2026-08-03, fixture "
                "problem_size_is_optional_and_inert). It must not go "
                "back into mandatory_sections.")
        self.assertIn(
            "PROBLEM SIZE",
            self.knowledge["input_format"]["optional_sections"])

    def test_result_description_block_is_complete(self):
        rd = self.knowledge["result_description"]
        self.assertEqual(set(rd["field_groups"]), RESULT_GROUPS)
        # THERMAL, not THERMO — the single most likely wrong guess.
        self.assertIn("THERMAL", rd["field_groups"])
        self.assertNotIn("THERMO", rd["field_groups"])
        for q in ("dispx", "dispy", "dispz", "reactx", "stress",
                  "strain", "press"):
            self.assertIn(q, rd["structure_quantities"])
        for key in ("exit_semantics", "capture_note", "agent_recipe",
                    "required_keys", "yaml_example"):
            self.assertIn(key, rd)
        # No 'pitfalls' key here on purpose: the orphan audit in
        # tests/test_signal_verification.py counts catalogue rows that
        # carry pitfalls but are not registered physics names.
        self.assertNotIn("pitfalls", rd)


if __name__ == "__main__":
    unittest.main()
