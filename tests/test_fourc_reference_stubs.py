"""Regression: fourc honest reference stubs for deep-multiphysics rows.

Created 2026-06-12 (task #29, cluster B). The probe sweep found ~21
fourc catalog rows whose generator templates are placeholders full of
literal <...> scalars + external mesh references (or, for a few, not
even YAML) that abort 4C in MatchTree. These rows model genuinely deep
multiphysics that a generic inline QUAD4/HEX8 mesh cannot carry:

  - two-mesh / multi-domain coupling (xfem fsi/fluid, fs3i, fpsi, ehl,
    fbi, pasi, ssi/ssti S2I electrodes, beam_interaction)
  - a second input file (multiscale fe2 micro RVE)
  - patient-derived 1-D topology (arterial_network, reduced_airways,
    reduced_lung)
  - an explicit particle cloud (particle_pd, particle_sph)
  - a wall-resolved periodic mesh (fluid_turbulence LES)
  - a build feature this 4C lacks (xfem cut needs Qhull)
  - stochastic statmech setup (brownian_dynamics)
  - single-mesh-but-special couplings (sti, cardiac_monodomain)

For these, generate_input now returns a documented reference stub
INSTEAD of the guaranteed-MPI_Abort placeholder. The stub is the
honest "documented, not runnable" state:
  * it is valid YAML and clearly marked "reference stub" /
    "Not a runnable input" so a user cannot mistake it for a job;
  * it carries PROBLEM TYPE but deliberately NO MATERIALS, so
    validate_input() flags it and the probe never reports it as a
    completed run — a stub is never counted as a pass.

This test is gen-only (no 4C binary). It pins that every stub row
stays an honest stub and never silently regresses to a placeholder
template carrying un-substituted <...> scalars.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

_PLACEHOLDER = re.compile(r"<[A-Za-z_][A-Za-z0-9_. ]*>")

STUB_ROWS = [
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
    # ("reduced_airways", "airways_1d"): promoted to a genuinely
    # runnable inline input (ported from the self-contained corpus case
    # red_airway_3airway_2acinus_awacinter.4C.yaml) — no longer a stub.
    ("reduced_lung", "lung_1d"),
    ("multiscale", "fe2_3d"),
    ("beam_interaction", "beam_contact_3d"),
    ("beam_interaction", "beam_solid_meshtying_3d"),
    ("particle_pd", "plate_2d"),
    ("particle_sph", "poiseuille_2d"),
    ("fluid_turbulence", "les_channel_3d"),
    ("brownian_dynamics", "brownian_3d"),
]

# XFEM rows must document the Qhull build limitation observed on this
# install (level-set cut aborts in Cut::TetMesh::call_q_hull).
XFEM_ROWS = [("fsi_xfem", "xfem_fsi_3d"), ("xfem_fluid", "xfem_3d")]


class TestFourcReferenceStubs(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("fourc")
        if cls.backend is None:
            raise unittest.SkipTest("fourc backend not registered")

    def _gen(self, physics: str, variant: str) -> str:
        return self.backend.generate_input(physics, variant, {})

    def test_stub_rows_are_valid_yaml(self) -> None:
        import yaml
        for physics, variant in STUB_ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                data = yaml.safe_load(self._gen(physics, variant))
                self.assertIsInstance(
                    data, dict,
                    f"{physics}/{variant} stub is not a YAML dict")
                self.assertIn("PROBLEM TYPE", data)

    def test_stub_rows_are_clearly_marked(self) -> None:
        """A stub is a SUCCESS state only if the user cannot run a
        comment by mistake — it must announce itself."""
        for physics, variant in STUB_ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                content = self._gen(physics, variant)
                self.assertIn("reference stub", content)
                self.assertIn("Not a runnable", content)

    def test_stub_rows_are_flagged_non_runnable(self) -> None:
        """The stub omits MATERIALS on purpose so validate_input()
        flags it: the probe skips the run stage and never reports a
        stub as a completed run. This is what keeps a stub honest."""
        for physics, variant in STUB_ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                errs = self.backend.validate_input(
                    self._gen(physics, variant))
                self.assertIn("Missing MATERIALS section", errs)

    def test_stub_yaml_body_has_no_placeholders(self) -> None:
        """The YAML body (after the comment block) must not carry
        un-substituted <...> scalars — that would be the placeholder
        template leaking through instead of the stub."""
        for physics, variant in STUB_ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                content = self._gen(physics, variant)
                body = content.split("TITLE:", 1)[-1]
                hits = _PLACEHOLDER.findall(body)
                self.assertFalse(
                    hits,
                    f"{physics}/{variant} YAML body has placeholders "
                    f"{hits[:5]} — placeholder template leaked.")

    def test_xfem_stubs_document_qhull_limitation(self) -> None:
        for physics, variant in XFEM_ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                content = self._gen(physics, variant)
                self.assertIn("Qhull", content)


if __name__ == "__main__":
    unittest.main()
