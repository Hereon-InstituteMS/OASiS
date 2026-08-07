"""Regression: adversarial re-audit of the 2026-08-03 4C execution sweep.

Every correction pinned here was produced by RE-RUNNING the deployed
binary ``/home/alexander/4C/build/4C`` (4C 2026.2.0-dev, git 89519cf)
against the claim as written, and finding that the claim as written did
not survive.  Each test therefore guards a statement that was measured
to be FALSE once, so that it cannot quietly come back.

The five corrections, and the command that falsified the original text:

1. ``capture_note`` / input_format "block-buffered" pitfall claimed that
   redirecting 4C to a FILE preserves the RESULT DESCRIPTION failure
   diagnostic.  It does not — a regular file is fully buffered too::

       $ for i in 1 2 3; do 4C bad.4C.yaml out > f.log 2>&1; \\
             echo "rc=$? isWRONG=$(grep -c 'is WRONG' f.log)"; done
       rc=1 isWRONG=0
       rc=1 isWRONG=0
       rc=1 isWRONG=0
       $ stdbuf -oL -eL 4C bad.4C.yaml out > f.log 2>&1
       rc=1 isWRONG=1

2. The runtime-VTK pitfall claimed that the parent section plus
   ``IO/RUNTIME VTK OUTPUT/STRUCTURE`` with ``OUTPUT_STRUCTURE: true``
   yields 3 ``.vtu`` files.  With no field flag set it exits 1::

       OUTPUT_STRUCTURE only        -> rc=1, "No data was written or
                                       writer was already in final
                                       phase." (io_vtk_writer_base)
       OUTPUT_STRUCTURE+DISPLACEMENT-> rc=0, 3 .vtu / 3 .pvtu / 1 .pvd
       DISPLACEMENT only            -> rc=0, 0 files

   The branch's own Tier-2 fixture already used ``DISPLACEMENT: true``;
   only the prose omitted it.

3. The WALL ``EAS`` pitfall claimed EAS changes the answer *silently*.
   ``EAS full`` with ``KINEM linear`` is a hard error::

       ERROR: No EAS for geometrically linear WALL element   (rc=1)

4. ``result_description`` understated two key lists, both re-read from
   ``4C --parameters``: an ``ELEMENT`` selector is accepted by four
   groups (ARTNET, FLUID, POROFLUIDMULTIPHASE, RED_AIRWAY), and
   ``SPECIAL`` by nine, not three.

5. ``4C --parameters`` emits 2 926 432 bytes on stdout; the recorded
   2 926 462 was measured with ``2>&1`` and included 30 bytes of local
   X11 warning.

Plus one newly executed finding: ``NUE: -1.0`` passes the
``in_range[-1,0.5)`` validator (closed at the low end) and is then
killed by SIGFPE with no material diagnostic.

GEN-ONLY: no 4C binary needed, so this runs in CI.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _knowledge():
    """Load the catalogue dict directly — result_description is not a
    registered physics name, so the backend accessor cannot reach it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_fourc_knowledge_audit", ROOT / "data" / "fourc_knowledge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FOURC_KNOWLEDGE


class TestCaptureAdviceIsNotWrong(unittest.TestCase):
    """A file redirect does NOT preserve the result-test diagnostic."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.k = _knowledge()

    def test_capture_note_does_not_recommend_a_plain_file_redirect(self):
        note = self.k["result_description"]["capture_note"]
        self.assertIn("stdbuf", note)
        # The falsified phrasing: "(stdbuf -oL -eL) or redirect to a file".
        self.assertNotIn("or redirect to a file", note)
        # It must positively say that a file does not help.
        self.assertRegex(note.lower(), r"(does not rescue|does not help|not )")
        self.assertIn("buffered", note.lower())

    def test_buffering_pitfall_does_not_offer_a_file_as_an_alternative(self):
        pits = [p for p in self.k["input_format"]["pitfalls"]
                if "block-buffered" in p]
        self.assertEqual(len(pits), 1, "the buffering pitfall went missing")
        p = pits[0]
        self.assertNotIn("or to a file", p)
        self.assertIn("stdbuf -oL -eL", p)
        self.assertIn("fully buffered", p)


class TestRuntimeVtkNeedsAFieldFlag(unittest.TestCase):
    """OUTPUT_STRUCTURE alone crashes the VTK writer; it is not enough."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.k = _knowledge()

    def test_vtk_pitfall_names_a_field_flag(self):
        pits = [p for p in self.k["input_format"]["pitfalls"]
                if "RUNTIME VTK OUTPUT" in p and "INTERVAL_STEPS" in p]
        self.assertTrue(pits, "the runtime-VTK pitfall went missing")
        p = pits[0]
        self.assertIn("DISPLACEMENT", p)
        # and it must carry the observable signal for the loud branch
        self.assertIn("No data was written", p)
        self.assertIn("io_vtk_writer_base", p)

    def test_vtk_pitfall_no_longer_claims_output_structure_alone_works(self):
        pits = [p for p in self.k["input_format"]["pitfalls"]
                if "RUNTIME VTK OUTPUT" in p and "INTERVAL_STEPS" in p]
        p = pits[0]
        # The falsified sentence said either half alone produces ZERO
        # files and exit 0.  The sub-section-with-no-field case is exit 1.
        self.assertIn("exit 1", p)


class TestEasIsNotSilentUnderLinearKinematics(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.k = _knowledge()

    def test_eas_pitfall_records_the_hard_error(self):
        pits = [p for p in self.k["structural_mechanics"]["pitfalls"]
                if "On the WALL element EAS and" in p]
        self.assertTrue(pits, "the WALL EAS pitfall went missing")
        p = pits[0]
        self.assertIn("No EAS for geometrically linear WALL element", p)
        self.assertIn("KINEM nonlinear", p)
        # The falsified claim was that NEITHER keyword emits a warning.
        self.assertNotIn("neither emits a warning", p)

    def test_eas_numbers_are_same_node_same_kinematics(self):
        pits = [p for p in self.k["structural_mechanics"]["pitfalls"]
                if "On the WALL element EAS and" in p]
        self.assertTrue(pits, "the WALL EAS pitfall went missing")
        p = pits[0]
        # The corrected entry must state the probe node, otherwise the
        # percentages are not reproducible (the original mixed node 3 /
        # KINEM linear with node 2 / KINEM nonlinear).
        self.assertIn("node 2", p)
        self.assertIn("4.33567955849997223e-03", p)   # EAS none,  nonlinear
        self.assertIn("5.33749404753981662e-03", p)   # EAS full,  nonlinear


class TestResultDescriptionKeyListsAreComplete(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.rd = _knowledge()["result_description"]

    def test_element_selector_lists_all_four_groups(self):
        sel = self.rd["required_keys"]["NODE | LINE | SURFACE | VOLUME"]
        for g in ("ARTNET", "FLUID", "POROFLUIDMULTIPHASE", "RED_AIRWAY"):
            self.assertIn(g, sel)

    def test_special_lists_all_nine_groups(self):
        sp = self.rd["optional_keys"]["SPECIAL"]
        for g in ("CARDIOVASCULAR0D", "FSI", "PARTICLEWALL",
                  "POROFLUIDMULTIPHASE", "SCATRA", "SSI", "SSTI", "STI",
                  "STRUCTURE"):
            self.assertIn(g, sp)

    def test_op_lists_its_actual_enum_choices(self):
        op = self.rd["optional_keys"]["OP"]
        for c in ("sum", "max", "min", "unknown"):
            self.assertIn(c, op)


class TestExecutedConstantsAreTheReproducibleOnes(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.k = _knowledge()

    def test_result_description_uses_the_fixture_value(self):
        pits = [p for p in self.k["input_format"]["pitfalls"]
                if "RESULT DESCRIPTION is 4C's own numerical" in p]
        self.assertTrue(pits)
        p = pits[0]
        # The fixture's own executed value, re-measured 2026-08-03.
        self.assertIn("4.47909266337460053e-03", p)

    def test_parameters_byte_count_is_the_stdout_only_one(self):
        pits = [p for p in self.k["input_format"]["pitfalls"]
                if "--parameters" in p and "top-level keys" in p]
        self.assertTrue(pits)
        p = pits[0]
        self.assertIn("2 926 432", p)
        self.assertIn("478", p)   # the section count on this build


class TestNueLowerBoundIsFatal(unittest.TestCase):

    def test_nue_minus_one_sigfpe_is_documented(self):
        k = _knowledge()
        pits = [p for p in k["input_format"]["pitfalls"]
                if "in_range[-1,0.5)" in p and "SIGFPE" in p]
        self.assertTrue(
            pits,
            "NUE: -1.0 passes the validator and is then killed by SIGFPE "
            "(verified by execution 2026-08-03); the pitfall must stay.")
        p = pits[0]
        self.assertIn("signal 8", p)
        self.assertIn("136", p)


if __name__ == "__main__":
    unittest.main()
