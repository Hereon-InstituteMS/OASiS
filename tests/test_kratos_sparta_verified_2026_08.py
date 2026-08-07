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
            # NOT "TEMPERATURE == 0.0 at EVERY node": that 2026-08-03 wording was
            # itself falsified on 2026-08-03 (re-audit) (it recorded a DELTA_TIME = 0 artifact
            # of its own fixture). The element-level statement is what survives.
            ("poisson", "max T = 19.7392"),
            ("poisson", "setting the RHS to zero"),
            ("poisson", "nodal 999 + Properties 1 -> 999.0"),
            ("structural_dynamics", "NOT silently ignored"),
            ("structural_dynamics", "Getting a value that does not exist"),
            ("structural_dynamics", "66.6% of the Timoshenko value"),
            ("curved_mms", "REPRODUCED 2026-08-03"),
        ]
        missing = [f"{p}: {s!r}" for p, s in required
                   if s not in _all_text(knowledge[p])]
        self.assertEqual(missing, [], f"Verified correction missing: {missing}")

    def test_structural_hyperelastic_names_are_not_mpm_only(self) -> None:
        """HyperElasticNeoHookean* is registered by MPMApplication, not by
        StructuralMechanicsApplication. Offering it as a structural law makes
        Materials.json fail with 'Kratos components missing'. Proven 2026-08-03 (re-audit)
        with only SMA + CLA imported: HasConstitutiveLaw is False and
        KM.ReadMaterialsUtility raises."""
        elast = _all_text(_kratos_knowledge()["linear_elasticity"]["constitutive_laws"])
        self.assertIn("HyperElasticSimoTaylorNeoHookean3DLaw", elast)
        self.assertIn("MPMApplication", elast,
                      "the structural hyperelastic list must say that the "
                      "HyperElasticNeoHookean* family comes from MPMApplication")
        laws = _all_text(_kratos_knowledge()["constitutive_laws"]["laws"])
        self.assertNotIn("HyperElasticQuasiIncompressibleNeoHookean3DLaw", laws,
                         "not registered under ANY of the 26 importable "
                         "applications on 10.4.0 (checked 2026-08-03 (re-audit))")


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
    """SPARTA's knowledge was restructured from one flat JSON dump into
    per-physics modules under src/backends/sparta/generators/, the shape every
    FEM backend already uses. Two things must hold afterwards: the payload an
    agent receives stays small, structured and free of host paths, and the
    execution-verified claims keep being served."""

    def _backend(self):
        from core.registry import get_backend, load_all_backends
        load_all_backends()
        backend = get_backend("sparta")
        self.assertIsNotNone(backend, "sparta backend not registered")
        return backend

    def _knowledge(self) -> dict:
        return self._backend().get_knowledge("rarefied_flow")

    def _kb_file(self) -> dict:
        import json
        return json.loads((_REPO / "src" / "backends" / "sparta"
                           / "sparta_knowledge.json").read_text())

    # ── payload shape ────────────────────────────────────────────────────
    def test_knowledge_is_structured_not_a_manual_dump(self) -> None:
        """The old payload was ~43 KB per physics, 27 KB of it a verbatim copy
        of the SPARTA manual under 'relevant_commands'. That pushed the
        pitfalls past the client-side truncation point, so half the knowledge
        never reached the agent."""
        backend = self._backend()
        for cap in backend.supported_physics():
            kn = backend.get_knowledge(cap.name)
            self.assertNotIn("relevant_commands", kn,
                             f"{cap.name}: the verbatim manual dump is back")
            self.assertIn("key_commands", kn)
            self.assertIn("deck_skeleton", kn)
            self.assertIn("pitfalls", kn)
            self.assertLess(
                len(_all_text(kn)), 25_000,
                f"{cap.name}: knowledge payload grew back past 25 KB")

    def test_knowledge_carries_no_absolute_host_path(self) -> None:
        """'installed_build' used to hard-code four absolute paths from the
        machine the campaign ran on and served them for all ten physics."""
        import re
        host = re.compile(r"/home/[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+")
        backend = self._backend()
        for name in [c.name for c in backend.supported_physics()] + ["_general"]:
            hits = host.findall(_all_text(backend.get_knowledge(name)))
            self.assertEqual(hits, [], f"{name}: host paths leaked into knowledge")
        build = self._kb_file()["installed_build"]
        for key in ("distribution_root", "data_dir", "examples_dir"):
            self.assertNotIn(key, build,
                             f"sparta_knowledge.json['installed_build'] "
                             f"re-added the host path key {key!r}")

    def test_every_pitfall_carries_a_signal(self) -> None:
        backend = self._backend()
        missing = []
        for cap in backend.supported_physics():
            for i, pit in enumerate(backend.get_knowledge(cap.name)["pitfalls"]):
                if "Signal:" not in str(pit):
                    missing.append((cap.name, i))
        self.assertEqual(missing, [], "pitfalls without an observable Signal:")

    def test_every_physics_offers_an_executable_generator(self) -> None:
        from backends.sparta.generators import GENERATORS
        backend = self._backend()
        for cap in backend.supported_physics():
            self.assertTrue(cap.template_variants, f"{cap.name}: no variants")
            for v in cap.template_variants:
                self.assertIn(f"{cap.name}_{v}", GENERATORS)
                deck = backend.generate_input(cap.name, v, {})
                self.assertIn("run ", deck)
                self.assertIn("seed ", deck)
                self.assertIn("timestep ", deck,
                              f"{cap.name}/{v}: no timestep line — SPARTA's "
                              f"default dt is 1.0 s and the deck would hang")
                self.assertEqual(backend.validate_input(deck), [],
                                 f"{cap.name}/{v}: generated deck fails validation")

    # ── the verbatim command index moved, it did not disappear ───────────
    def test_command_surface_separates_commands_from_doc_pages(self) -> None:
        surface = self._kb_file()["command_surface"]
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

    def test_command_reference_is_reachable_on_demand(self) -> None:
        ref = self._backend().get_command_reference("compute_grid")
        self.assertNotIn("error", ref)
        self.assertEqual(ref["doc_page"], "compute_grid")
        self.assertEqual(ref["command"], "compute grid")
        self.assertIn("syntax", ref)

    def test_installed_build_block_kept_its_style_inventory(self) -> None:
        build = self._kb_file()["installed_build"]
        self.assertEqual(build["version"], "24 Sep 2025")
        self.assertEqual(build["compiled_styles"]["collide"], ["vss"])
        self.assertIn("diffuse", build["compiled_styles"]["surf_collide"])
        self.assertIn("kokkos_kk_styles", build["documented_but_absent_here"])
        facts = self._knowledge()["build"] if "build" in self._knowledge() else {}
        del facts  # served through knowledge(topic='overview'), see below
        general = self._backend().get_knowledge("_general")
        self.assertEqual(general["build"]["compiled_collide_styles"], ["vss"])
        self.assertIn("KOKKOS", general["build"]["accelerators"])

    # ── ordering / output knowledge survived the move ────────────────────
    def test_setup_ordering_errors_are_served(self) -> None:
        errs = self._knowledge()["hard_ordering_errors"]
        self.assertIn("create_grid.cpp:44", errs["create_grid before create_box"])
        self.assertIn("random_mars.cpp:91", errs["no seed command"])
        general = self._backend().get_knowledge("_general")
        self.assertIn("nrho = 1.0",
                      general["silently_accepted"]["no global nrho/fnum"])

    def test_output_reading_block(self) -> None:
        out = self._knowledge()["output_idioms"]
        self.assertIn("mode vector", out["boundary tally idiom"])
        self.assertIn("VECTOR", out["vector vs array rule"])
        general = self._backend().get_knowledge("_general")["reading_output"]
        self.assertIn("reduce sum", general["per-surf tally idiom"])

    # ── claims execution established, in their CORRECTED form ────────────
    def test_verified_pitfalls_are_served_for_every_physics(self) -> None:
        """These wordings replaced earlier ones that a critic pass falsified by
        execution — see FORBIDDEN_SPARTA below for what they replaced."""
        backend = self._backend()
        markers = (
            "silently overrides 'global nrho'",
            "the z extent handed to create_box is ignored",
            "no default timestep worth having",
            "does NOT convert the species, VSS, reaction or surface files",
        )
        for cap in backend.supported_physics():
            text = "\n".join(backend.get_knowledge(cap.name)["pitfalls"])
            for marker in markers:
                self.assertIn(marker, text,
                              f"sparta physics {cap.name!r} lost the verified "
                              f"pitfall {marker!r}")
        surf = "\n".join(backend.get_knowledge("surface_interaction")["pitfalls"])
        self.assertIn("reverses the normal velocity component", surf)

    def test_falsified_sparta_wordings_stay_dead(self) -> None:
        """Each string here was in the catalog and was disproved by running the
        installed binary. None may come back."""
        FORBIDDEN_SPARTA = [
            ("rarefied_flow", "Omitting the collide command is NEVER an error",
             "collide_modify, fix vibmode and react tce each abort with a hard "
             "error when no collide style is defined"),
            ("axisymmetric",
             "Use 'global ... axisymmetric yes' and radial particle weighting",
             "there is no such keyword: 'global axisymmetric yes' gives "
             "'Illegal global command (../update.cpp:1805)'; axisymmetry is the "
             "boundary letter 'a' on the lower y face"),
            ("chemistry", "Reaction file format (tce/qk) must match the 'react' "
             "style",
             "'react qk <a tce file>' is accepted without error or warning and "
             "fires reactions"),
            ("surface_interaction",
             "for conjugate heat transfer the wall T is updated each coupling "
             "window",
             "vague and untestable; replaced by the fix surf/temp ordering and "
             "bracket rules, each with its own error string"),
        ]
        backend = self._backend()
        for physics, forbidden, why in FORBIDDEN_SPARTA:
            text = _all_text(backend.get_knowledge(physics))
            self.assertNotIn(forbidden, text,
                             f"sparta {physics}: falsified wording came back "
                             f"({forbidden!r}) — execution showed: {why}")

    def test_validate_input_rejects_doc_page_command_forms(self) -> None:
        backend = self._backend()
        preamble = ("seed 1\ndimension 2\nboundary p p p\n"
                    "create_box 0 1 0 1 -0.5 0.5\ncreate_grid 2 2 1\n")
        self.assertEqual(backend.validate_input(preamble + "run 10\n"), [])
        for bad in ("compute_grid all all n", "fix_ave_surf all 1 1 1",
                    "dump_image all 100 img.ppm", "suffix kk"):
            errs = backend.validate_input(preamble + bad + "\nrun 10\n")
            self.assertTrue(errs, f"validate_input accepted the doc-page form {bad!r}")

    def test_validate_input_understands_line_continuations(self) -> None:
        """SPARTA continues a command when the line ends in '&'. Without
        joining, the continuation line's first word is read as a command and
        every multi-line 'fix adapt' deck is falsely rejected."""
        backend = self._backend()
        deck = ("seed 1\ndimension 2\nboundary p p p\n"
                "create_box 0 1 0 1 -0.5 0.5\ncreate_grid 2 2 1\n"
                "fix ad adapt 100 all refine coarsen value c_x[2] 2.0 4.5 &\n"
                "    combine min thresh less more cells 2 2 1\n"
                "run 10\n")
        self.assertEqual(backend.validate_input(deck), [])

    def test_verified_running_lists_only_decks_that_exist(self) -> None:
        """An entry belongs here only if examples/<name>/in.<name> exists.
        'surf' and 'emit' both failed that test and were removed; the same
        check must not silently regress."""
        kb = self._kb_file()
        for absent in ("surf", "emit", "adjust_temp"):
            self.assertNotIn(
                absent, kb["verified_running"],
                f"examples/{absent}/ ships in.{absent}.* variants but no file "
                f"called in.{absent}, so it could never have been executed "
                f"as written.")
        self.assertEqual(sorted(kb["verified_running"]),
                         ["ambi", "axi", "chem", "circle", "collide", "free"])


if __name__ == "__main__":
    unittest.main()
