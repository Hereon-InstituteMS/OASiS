"""Regression: the fourc rows that used to be "documented, not runnable".

HISTORY, AND WHY THIS FILE INVERTED
-----------------------------------
Written 2026-06-12, this file used to assert the opposite of what it asserts
now. A probe sweep had found ~21 fourc catalog rows whose generator templates
were placeholders full of literal `<...>` scalars and external mesh references
that aborted 4C in MatchTree. The response was to replace them with honest
reference stubs — YAML that says "Not a runnable input" and deliberately omits
MATERIALS so `validate_input` flags it and no probe can score it as a run — and
this file pinned that stubs stay stubs.

That was the right move for a template nobody had executed. It was never the
destination. A weak model handed a section list and no runnable skeleton cannot
produce a working 478-section 4C deck, and 4C was the only backend in the
project shipping stubs at all (deal.II 27/27 templated, scikit-fem 22/22,
FEBio 17/17, SPARTA 10/10).

So the rows below now carry decks that were RUN on the installed binary. The
test inverted with them: it pins that they never regress to stubs, and that the
two catalogs cannot disagree about which is which.

WHAT IS STILL A STUB, AND WHY THAT IS FINE
-------------------------------------------
`STILL_STUB_ROWS` is not an embarrassment list — it is the honest remainder,
and the test insists those rows keep announcing themselves as non-runnable
rather than quietly shipping something that aborts. A stub a user can read
beats a deck 4C rejects, because the deck looks like help.

This test is generation-only and needs no 4C binary. Execution is the separate
gate, `scripts/verify_fourc_decks.py --execute`, which runs every deck alone in
a fresh temp directory — the check that catches a deck depending on a sibling
file, which no amount of static inspection can.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

_PLACEHOLDER = re.compile(r"<[A-Za-z_][A-Za-z0-9_. ]*>")

# Rows that were stubs and are now executed decks. Kept as an explicit list
# rather than derived from the deck catalog: deriving it would make the test
# vacuously pass if the catalog were emptied.
PROMOTED_ROWS = [
    ("fsi", "fsi_2d"),
    ("fsi_xfem", "xfem_fsi_3d"),
    ("xfem_fluid", "xfem_3d"),
    ("fs3i", "fs3i_3d"),
    ("fpsi", "monolithic_3d"),
    ("ehl", "ehl_3d"),
    ("fbi", "penalty_3d"),
    ("pasi", "dem_impact_3d"),
    ("ssi", "monolithic_elch_3d"),
    ("ssti", "monolithic_3d"),
    ("sti", "monolithic_3d"),
    ("cardiac_monodomain", "monodomain_3d"),
    ("arterial_network", "single_artery_1d"),
    ("reduced_lung", "lung_1d"),
    ("multiscale", "fe2_3d"),
    ("beam_interaction", "beam_contact_3d"),
    ("beam_interaction", "beam_solid_meshtying_3d"),
    ("particle_pd", "plate_2d"),
    ("particle_pd", "impact_2d"),
    ("particle_sph", "hydrostatic_2d"),
    ("particle_sph", "dam_break_2d"),
    ("particle_dem", "settling_3d"),
    ("plasticity", "linear_2d"),
    ("plasticity", "nonlinear_3d"),
    ("fluid_turbulence", "les_channel_3d"),
    ("brownian_dynamics", "brownian_3d"),
]

# The honest remainder. Each must still announce itself as non-runnable.
STILL_STUB_ROWS = [
    ("porous_media", "terzaghi_2d"),
    ("porous_media", "consolidation_3d"),
]


class TestFourcPromotedRows(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("fourc")
        if cls.backend is None:
            raise unittest.SkipTest("fourc backend not registered")

    def _gen(self, physics: str, variant: str) -> str:
        return self.backend.generate_input(physics, variant, {})

    def test_promoted_rows_are_no_longer_stubs(self) -> None:
        """The exact phrases the old stub announced itself with must be gone.

        Checking for their ABSENCE rather than for some positive marker is
        deliberate: a half-finished promotion that leaves the stub text in a
        comment above a real deck would pass any positive check and would still
        tell the reader the deck cannot be run.
        """
        for physics, variant in PROMOTED_ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                content = self._gen(physics, variant)
                self.assertNotIn("Not a runnable", content)
                self.assertNotIn("reference stub", content)

    def test_promoted_rows_are_complete_decks(self) -> None:
        """Valid YAML with the two sections 4C cannot run without."""
        import yaml
        for physics, variant in PROMOTED_ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                data = yaml.safe_load(self._gen(physics, variant))
                self.assertIsInstance(data, dict)
                self.assertIn("PROBLEM TYPE", data)
                # The old stubs omitted MATERIALS on purpose so validate_input
                # would flag them. A promoted row must now pass that check.
                self.assertNotIn(
                    "Missing MATERIALS section",
                    self.backend.validate_input(self._gen(physics, variant)))

    def test_promoted_rows_carry_no_placeholders(self) -> None:
        for physics, variant in PROMOTED_ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                body = self._gen(physics, variant).split("TITLE:", 1)[-1]
                hits = _PLACEHOLDER.findall(body)
                self.assertFalse(
                    hits, f"{physics}/{variant} still has placeholders "
                          f"{hits[:5]}")

    def test_deck_catalog_and_promoted_list_agree(self) -> None:
        """Two lists of the same fact must not be allowed to drift apart."""
        from backends.fourc import decks
        self.assertEqual(
            sorted({(d.physics, d.variant) for d in decks.DECKS}),
            sorted(set(PROMOTED_ROWS)))

    def test_every_deck_is_advertised_by_the_catalog(self) -> None:
        """A deck no `template_variants` entry names is unreachable.

        This is the failure mode that hid `heat_transient_2d`: a working
        generator that no tool could select, because the catalog row did not
        list the variant.
        """
        advertised = {(p.name, v)
                      for p in self.backend.supported_physics()
                      for v in (p.template_variants or [])}
        from backends.fourc import decks
        for d in decks.DECKS:
            with self.subTest(row=f"{d.physics}/{d.variant}"):
                self.assertIn((d.physics, d.variant), advertised)

    def test_deck_files_exist_and_are_yaml(self) -> None:
        import yaml
        from backends.fourc import decks
        for d in decks.DECKS:
            with self.subTest(row=f"{d.physics}/{d.variant}"):
                self.assertTrue(d.path().is_file(), f"missing {d.filename}")
                self.assertIsInstance(yaml.safe_load(d.path().read_text()),
                                      dict)

    def test_deck_metadata_is_filled_in(self) -> None:
        """`evidence` is the field that distinguishes a deck that exits 0 from
        a deck that does something. An empty one is an unverified claim."""
        from backends.fourc import decks
        for d in decks.DECKS:
            with self.subTest(row=f"{d.physics}/{d.variant}"):
                self.assertGreaterEqual(len(d.summary), 40)
                self.assertGreaterEqual(len(d.evidence), 40)
                self.assertGreaterEqual(d.np, 1)
                self.assertTrue(d.upstream)


class TestFourcRemainingStubs(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("fourc")
        if cls.backend is None:
            raise unittest.SkipTest("fourc backend not registered")

    def test_remaining_stubs_announce_themselves(self) -> None:
        for physics, variant in STILL_STUB_ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                content = self.backend.generate_input(physics, variant, {})
                self.assertIn("Not a runnable", content)
                self.assertIn(
                    "Missing MATERIALS section",
                    self.backend.validate_input(content),
                    "a stub must stay flagged non-runnable, or a probe will "
                    "score it as a completed run")

    def test_stub_catalog_never_shadows_a_deck(self) -> None:
        from backends.fourc import decks
        for d in decks.DECKS:
            with self.subTest(row=f"{d.physics}/{d.variant}"):
                self.assertNotIn((d.physics, d.variant), set(STILL_STUB_ROWS))


if __name__ == "__main__":
    unittest.main()
