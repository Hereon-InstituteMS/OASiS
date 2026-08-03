"""Regression gate for the 2026-08-03 Kratos + SPARTA verification campaign.

Everything asserted here was established by EXECUTING the installed solvers
(Kratos Multiphysics 10.4.0 under /usr/bin/python3; SPARTA "24 Sep 2025",
spa_serial). The tier-2 fixtures under scripts/tier2_fixtures/{kratos,sparta}/
hold the executable evidence:

  kratos/cda_element_diffusion_vs_mass
  kratos/cda_conductivity_is_nodal
  kratos/constitutive_law_registry_names
  kratos/parameters_unknown_key_raises
  kratos/point_load_direction_process_rejects
  kratos/quad4_shear_locking_vs_quad8
  sparta/no_collide_is_free_molecular
  sparta/docpage_names_are_not_commands

This file is the CHEAP gate: it imports no solver, so it runs everywhere, and
it fails if someone reverts a corrected claim back to the falsified wording.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "data"))


def _kratos_knowledge() -> dict:
    from backends.kratos.generators import KNOWLEDGE
    return KNOWLEDGE


def _all_text(block: dict) -> str:
    import json
    return json.dumps(block, default=str)


class TestKratosFalsifiedClaimsStayFixed(unittest.TestCase):
    """Wordings that EXECUTION proved wrong must not come back."""

    # (physics, forbidden substring, what the execution showed instead)
    FORBIDDEN = [
        ("poisson",
         "produce solutions that differ by less than 1e-12 relative norm",
         "LaplacianElement2D3N and EulerianConvDiff2D3N differ by relative norm 1.0 "
         "in a stationary solve; EulerianConvDiff returns TEMPERATURE == 0"),
        ("poisson",
         "go on Properties object, NOT on nodes",
         "LaplacianElement2D3N reads CONDUCTIVITY nodally; the Properties value is "
         "ignored (LHS[0][0] = 1.0 with Properties 999, = 999.0 with nodal 999)"),
        ("structural_dynamics",
         "Wrong key is silently ignored and the scheme runs without damping",
         "Parameters::ValidateAndAssignDefaults raises on the unknown key and the "
         "analysis aborts before the first time step"),
        ("structural_dynamics",
         "KeyError 'echo_level' from AnalysisStage.RunSolutionLoop",
         "the real message is the C++ 'Error: Getting a value that does not exist. "
         "entry string : echo_level'"),
    ]

    # A corrected pitfall is allowed — and encouraged — to QUOTE the falsified
    # wording as provenance. What must never happen is the old wording standing
    # on its own as a live claim. So: wherever a falsified phrase appears, the
    # same pitfall string must also carry a supersession marker.
    SUPERSESSION_MARKERS = ("was WRONG", "had it exactly backwards", "was stale",
                            "prior catalog", "earlier catalog", "supersedes",
                            "Re-verified", "re-verified")

    def test_falsified_wordings_absent(self) -> None:
        knowledge = _kratos_knowledge()
        offenders = []
        for physics, forbidden, why in self.FORBIDDEN:
            for pitfall in knowledge[physics].get("pitfalls", []):
                if forbidden not in pitfall:
                    continue
                if not any(m in pitfall for m in self.SUPERSESSION_MARKERS):
                    offenders.append(
                        f"{physics}: {forbidden!r} appears without a supersession "
                        f"marker — execution showed: {why}")
            # the phrase must not live outside the pitfalls either
            rest = dict(knowledge[physics])
            rest.pop("pitfalls", None)
            if forbidden in _all_text(rest):
                offenders.append(
                    f"{physics}: {forbidden!r} present outside pitfalls — "
                    f"execution showed: {why}")
        self.assertEqual(offenders, [], "Falsified claim reintroduced:\n" + "\n".join(offenders))

    def test_corrections_present(self) -> None:
        knowledge = _kratos_knowledge()
        required = [
            ("poisson", "TEMPERATURE == 0.0 at EVERY node"),
            ("poisson", "nodal 999 + Properties 1 -> 999.0"),
            ("structural_dynamics", "NOT silently ignored"),
            ("structural_dynamics", "Getting a value that does not exist"),
            ("structural_dynamics", "66.6% of the Timoshenko value"),
            ("curved_mms", "REPRODUCED 2026-08-03"),
        ]
        missing = [f"{p}: {s!r}" for p, s in required
                   if s not in _all_text(knowledge[p])]
        self.assertEqual(missing, [], f"Verified correction missing: {missing}")


class TestKratosConstitutiveLawNames(unittest.TestCase):
    """Law names that resolve to nothing on the installed 10.4.0 must not be
    listed as usable law names. Checked against the registry AND the Python
    attributes by scripts/tier2_fixtures/kratos/constitutive_law_registry_names."""

    # Bare tokens that do not resolve, with the registered name that does.
    DEAD_NAMES = {
        "LinearElasticAxisymmetric2DLaw": "LinearElasticAxisym2DLaw",
        "HyperElasticIsotropicNeoHookean3DLaw": "HyperElasticNeoHookean3DLaw",
        "HyperElasticIsotropicNeoHookean2D/3DLaw": "HyperElasticNeoHookean3DLaw",
        "ThermalLinearElastic2DPlaneStrain": "ThermalLinearPlaneStrain",
    }
    # These appeared as standalone "laws" in KNOWLEDGE['constitutive_laws'].
    DEAD_FAMILIES = ("Yeoh", "Arruda-Boyce", "Blatz-Ko", "Mazars", "Perzyna",
                     "ModifiedCamClay", "CriticalStateLine", "RankineFragile",
                     "DruckerPragerViscoplastic")

    def test_dead_law_names_not_offered_as_usable(self) -> None:
        knowledge = _kratos_knowledge()
        offenders = []
        for physics in ("linear_elasticity", "mpm", "constitutive_laws", "dam"):
            laws = _all_text(knowledge[physics].get("constitutive_laws")
                             or knowledge[physics].get("laws") or {})
            for dead, live in self.DEAD_NAMES.items():
                if dead in laws and live not in laws:
                    offenders.append(f"{physics}: {dead!r} listed without {live!r}")
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_dead_families_removed_from_constitutive_laws(self) -> None:
        laws = _all_text(_kratos_knowledge()["constitutive_laws"].get("laws", {}))
        still = [f for f in self.DEAD_FAMILIES if f in laws]
        self.assertEqual(
            still, [],
            f"KNOWLEDGE['constitutive_laws']['laws'] still offers {still} as law "
            f"names; none of them resolves on Kratos 10.4.0 (neither as a registry "
            f"string nor as a Python attribute).")

    def test_corrected_law_names_are_offered(self) -> None:
        knowledge = _kratos_knowledge()
        elast = _all_text(knowledge["linear_elasticity"]["constitutive_laws"])
        for name in ("LinearElasticAxisym2DLaw", "KirchhoffSaintVenant3DLaw",
                     "ViscousGeneralizedMaxwell3D", "ViscousGeneralizedKelvin3D",
                     "SmallStrainIsotropicDamageFactory"):
            self.assertIn(name, elast)
        mpm = _all_text(knowledge["mpm"]["constitutive_laws"])
        for name in ("LinearElasticIsotropic3DLaw", "HenckyMCPlastic3DLaw",
                     "JohnsonCookThermalPlastic3DLaw"):
            self.assertIn(name, mpm)


class TestSpartaVerifiedKnowledge(unittest.TestCase):
    """SPARTA gained a verified command surface, setup sequence, output guide
    and pitfall set on 2026-08-03; the backend must keep serving them."""

    def _knowledge(self) -> dict:
        from core.registry import get_backend, load_all_backends
        load_all_backends()
        backend = get_backend("sparta")
        self.assertIsNotNone(backend, "sparta backend not registered")
        return backend.get_knowledge("rarefied_flow")

    def test_installed_build_block(self) -> None:
        build = self._knowledge()["installed_build"]
        self.assertEqual(build["version"], "24 Sep 2025")
        self.assertIn("spa_serial", build["binary"])
        self.assertEqual(build["compiled_styles"]["collide"], ["vss"])
        self.assertIn("diffuse", build["compiled_styles"]["surf_collide"])
        self.assertIn("kokkos_kk_styles", build["documented_but_absent_here"])

    def test_command_surface_separates_commands_from_doc_pages(self) -> None:
        surface = self._knowledge()["command_surface"]
        self.assertEqual(surface["n_true_commands"], 66)
        true_cmds = surface["true_commands"]
        self.assertEqual(len(true_cmds), 66)
        for real in ("create_box", "create_grid", "create_particles", "collide",
                     "species", "mixture", "surf_collide", "surf_modify", "run",
                     "stats", "stats_style", "dump", "fix", "compute"):
            self.assertIn(real, true_cmds)
        doc_only = surface["not_commands_doc_page_names_only"]
        self.assertEqual(len(doc_only), 55)
        for fake in ("compute_grid", "fix_ave_surf", "dump_image",
                     "surf_react_adsorb", "suffix"):
            self.assertIn(fake, doc_only)
            self.assertNotIn(fake, true_cmds)

    def test_setup_sequence_and_surface_errors(self) -> None:
        kn = self._knowledge()
        errs = kn["required_setup_sequence"]["hard_errors_verified"]
        self.assertIn("create_grid.cpp:44", errs["create_grid before create_box"])
        self.assertIn("random_mars.cpp:91", errs["no seed command"])
        silent = kn["required_setup_sequence"]["silently_accepted_do_not_rely_on_an_error"]
        self.assertIn("nrho=1.0", silent["no global nrho/fnum"])
        surf = kn["surfaces"]
        self.assertIn("surf.cpp:343", surf["hard_errors_verified"]
                      ["surfaces with no collision model"])
        self.assertIn("create_grid -> read_surf -> surf_collide -> surf_modify",
                      surf["mandatory_order"])

    def test_output_reading_block(self) -> None:
        out = self._knowledge()["reading_output"]
        self.assertIn("not a cumulative total", out["counters_are_per_step"])
        self.assertIn("mode vector", out["boundary_tally_idiom"])
        self.assertIn("compute reduce", out["surface_tally_idiom"])
        self.assertIn("NEVER makes that comparison", out["recommended_timestep_idiom"])

    def test_verified_pitfalls_are_served_for_every_physics(self) -> None:
        from core.registry import get_backend, load_all_backends
        load_all_backends()
        backend = get_backend("sparta")
        markers = ("Omitting the collide command is NEVER an error",
                   "SILENTLY overrides 'global nrho'",
                   "the z extent given to create_box is IGNORED",
                   "surf_collide specular transfers EXACTLY ZERO energy")
        for cap in backend.supported_physics():
            text = "\n".join(backend.get_knowledge(cap.name)["pitfalls"])
            for marker in markers:
                self.assertIn(marker, text,
                              f"sparta physics {cap.name!r} lost the verified pitfall "
                              f"{marker!r}")

    def test_validate_input_rejects_doc_page_command_forms(self) -> None:
        from core.registry import get_backend, load_all_backends
        load_all_backends()
        backend = get_backend("sparta")
        preamble = ("seed 1\ndimension 2\nboundary p p p\n"
                    "create_box 0 1 0 1 -0.5 0.5\ncreate_grid 2 2 1\n")
        self.assertEqual(backend.validate_input(preamble + "run 10\n"), [])
        for bad in ("compute_grid all all n", "fix_ave_surf all 1 1 1",
                    "dump_image all 100 img.ppm", "suffix kk"):
            errs = backend.validate_input(preamble + bad + "\nrun 10\n")
            self.assertTrue(errs, f"validate_input accepted the doc-page form {bad!r}")

    def test_verified_running_drops_the_nonexistent_surf_deck(self) -> None:
        import json
        kb = json.loads((_REPO / "src" / "backends" / "sparta"
                         / "sparta_knowledge.json").read_text())
        self.assertNotIn(
            "surf", kb["verified_running"],
            "examples/surf/ has in.surf.add / .move / .remove / .rotate / .slide "
            "but NO in.surf, so 'surf' could never have been executed as written.")
        self.assertIn("free", kb["verified_running"])


if __name__ == "__main__":
    unittest.main()
