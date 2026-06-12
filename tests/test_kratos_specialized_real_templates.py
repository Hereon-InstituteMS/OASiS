"""Regression: the Kratos specialized templates keep their 2026-06-12 tier.

Until 2026-06-12 all 16 specialized-application templates were
availability-probe stubs (import the app, print "available", write a
1-line summary).  The stub-replacement work split them into two tiers:

  REAL  — the application's wheel pip-installs cleanly into the 10.4.2
          stack and the template performs a verified solve (rc=0, .vtk
          output, physical quantity cross-checked against an analytic
          or control solution in the scratch run):
            poromechanics_2d, shallow_water_2d, dam_2d,
            constitutive_laws_2d, dem_structures_2d
            (+ dem_structures_coupling_2d alias), cable_net_2d,
            optimization_2d

  STUB  — the application is genuinely NOT pip-installable (no PyPI
          wheel / broken wheel / version-pinned wheel / no cp312
          wheel); the template stays an availability probe but states
          the exact reason and the install route:
            wind_engineering_2d, thermal_dem_2d, swimming_dem_2d,
            fem_to_dem_2d, chimera_2d, droplet_dynamics_2d,
            free_surface_2d, fluid_biomedical_2d, fluid_hydraulics_2d

This test pins both tiers statically (no Kratos import needed):

  * a REAL template must never silently degrade back into a stub
    (the exact failure mode the 2026-06-01 audit found: run reports
    "completed", rc=0, but nothing was solved);
  * a STUB template must keep telling the truth — it must carry the
    unavailability reason and install hint, and must never grow fake
    solve scaffolding while the app remains uninstallable;
  * the KNOWLEDGE pitfalls must agree with the tier (a replaced
    physics must not still claim its template is a stub — that is
    catalog drift in the opposite direction).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


# generator key -> (application import, physics knowledge key,
#                   summary keys the real solve must report)
REAL_TEMPLATES = {
    "poromechanics_2d": (
        "KratosMultiphysics.PoromechanicsApplication",
        "poromechanics",
        ("pore_pressure_base", "settlement_top"),
    ),
    "shallow_water_2d": (
        "KratosMultiphysics.ShallowWaterApplication",
        "shallow_water",
        ("max_free_surface", "probe_eta_change"),
    ),
    "dam_2d": (
        "KratosMultiphysics.DamApplication",
        "dam",
        ("crest_ux", "sum_reaction_x"),
    ),
    "constitutive_laws_2d": (
        "KratosMultiphysics.ConstitutiveLawsApplication",
        "constitutive_laws",
        ("max_equivalent_plastic_strain",),
    ),
    "dem_structures_2d": (
        "KratosMultiphysics.DemStructuresCouplingApplication",
        "dem_structures_coupling",
        ("contact_force_peak", "max_plate_deflection"),
    ),
    "dem_structures_coupling_2d": (
        "KratosMultiphysics.DemStructuresCouplingApplication",
        "dem_structures_coupling",
        ("contact_force_peak", "max_plate_deflection"),
    ),
    "cable_net_2d": (
        "KratosMultiphysics.CableNetApplication",
        "cable_net",
        ("cable_x_tension", "slider_x"),
    ),
    "optimization_2d": (
        "KratosMultiphysics.OptimizationApplication",
        "optimization",
        ("compliance_initial", "compliance_final"),
    ),
}

# generator key -> (application import, physics knowledge key)
STUB_TEMPLATES = {
    "wind_engineering_2d": (
        "KratosMultiphysics.WindEngineeringApplication", "wind_engineering"),
    "thermal_dem_2d": (
        "KratosMultiphysics.ThermalDEMApplication", "thermal_dem"),
    "swimming_dem_2d": (
        "KratosMultiphysics.SwimmingDEMApplication", "swimming_dem"),
    "fem_to_dem_2d": (
        "KratosMultiphysics.FemToDemApplication", "fem_to_dem"),
    "chimera_2d": (
        "KratosMultiphysics.ChimeraApplication", "chimera"),
    "droplet_dynamics_2d": (
        "KratosMultiphysics.DropletDynamicsApplication", "droplet_dynamics"),
    "free_surface_2d": (
        "KratosMultiphysics.FreeSurfaceApplication", "free_surface"),
    "fluid_biomedical_2d": (
        "KratosMultiphysics.FluidDynamicsBiomedicalApplication",
        "fluid_biomedical"),
    "fluid_hydraulics_2d": (
        "KratosMultiphysics.FluidDynamicsHydraulicsApplication",
        "fluid_hydraulics"),
}

STUB_MARKER = "availability probe"


def _generate(key: str) -> str:
    from backends.kratos.generators.specialized import GENERATORS
    return GENERATORS[key]({})


class TestRealTemplatesAreRealSolves(unittest.TestCase):
    """A replaced template must keep the shape of a real solve."""

    def test_real_templates_not_stubs(self) -> None:
        for key, (app_import, _, summary_keys) in REAL_TEMPLATES.items():
            with self.subTest(template=key):
                tmpl = _generate(key)
                compile(tmpl, key, "exec")
                self.assertNotIn(
                    STUB_MARKER, tmpl,
                    f"{key}: degraded back into an availability-probe stub")
                self.assertIn(
                    f"import {app_import}", tmpl,
                    f"{key}: no longer imports its application module — "
                    f"it cannot be exercising {app_import}")
                self.assertIn(
                    "results_summary.json", tmpl,
                    f"{key}: dropped the results_summary.json contract")
                self.assertIn(
                    "VtkOutput", tmpl,
                    f"{key}: dropped the VTK output — the sweep harness "
                    f"flags it as has_vtu=False (the old stub signature)")
                for sk in summary_keys:
                    self.assertIn(
                        sk, tmpl,
                        f"{key}: physical summary key '{sk}' vanished — "
                        f"the template no longer reports a checkable "
                        f"physical quantity")
                self.assertGreater(
                    len(tmpl.splitlines()), 80,
                    f"{key}: template shrank below any plausible real-solve "
                    f"size — availability stubs were ~20 lines")

    def test_real_physics_pitfalls_dropped_stub_claim(self) -> None:
        from backends.kratos.generators.specialized import KNOWLEDGE
        for key, (_, phys, _) in REAL_TEMPLATES.items():
            with self.subTest(template=key, physics=phys):
                pitfalls = KNOWLEDGE[phys].get("pitfalls", [])
                stale = [p for p in pitfalls
                         if "availability-probe STUB" in p]
                self.assertFalse(
                    stale,
                    f"{phys}: pitfall still claims the template is an "
                    f"availability-probe stub, but {key} is a real solve "
                    f"since 2026-06-12 — stale catalog entry")


class TestStubTemplatesStayHonest(unittest.TestCase):
    """A remaining stub must say it is one, and say exactly why."""

    def test_stub_templates_carry_reason_and_hint(self) -> None:
        for key, (app_import, _) in STUB_TEMPLATES.items():
            with self.subTest(template=key):
                tmpl = _generate(key)
                compile(tmpl, key, "exec")
                self.assertIn(
                    STUB_MARKER, tmpl,
                    f"{key}: lost its stub self-identification")
                self.assertIn(
                    f"import {app_import}", tmpl,
                    f"{key}: availability probe no longer probes "
                    f"{app_import}")
                # The honest-stub contract: emitted summary explains
                # itself with reason + install hint.
                for field in ("reason", "install_hint"):
                    self.assertIn(
                        f'"{field}"', tmpl,
                        f"{key}: stub summary dropped the '{field}' field "
                        f"— the run would report 'not installed' without "
                        f"saying why or how to fix it")
                self.assertLess(
                    len(tmpl.splitlines()), 60,
                    f"{key}: stub grew beyond probe size — if a real solve "
                    f"was added, move the key to REAL_TEMPLATES (and make "
                    f"sure the app actually pip-installs now)")

    def test_stub_physics_pitfalls_flag_the_stub(self) -> None:
        from backends.kratos.generators.specialized import KNOWLEDGE
        for key, (_, phys) in STUB_TEMPLATES.items():
            with self.subTest(template=key, physics=phys):
                pitfalls = KNOWLEDGE[phys].get("pitfalls", [])
                self.assertTrue(
                    any("STUB" in p for p in pitfalls),
                    f"{phys}: no pitfall warns that the {key} template is "
                    f"an availability-probe stub — the critic gate cannot "
                    f"surface the limitation")


if __name__ == "__main__":
    unittest.main()
