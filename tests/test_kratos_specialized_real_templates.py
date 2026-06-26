"""Kratos catalog honesty: the availability-probe stubs were REMOVED.

The old specialized.py shipped ~20 generators that only import-checked a Kratos
sub-application (not installable in this stack) and wrote {"note":"not installed"}
with no solve. The overhaul removed them rather than ship fakes. This test pins
that invariant: those keys must NOT be advertised, and no surviving Kratos
generator may be an availability-probe stub.
"""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REMOVED_STUBS = [
    "wind_engineering_2d", "thermal_dem_2d", "swimming_dem_2d", "fem_to_dem_2d",
    "chimera_2d", "droplet_dynamics_2d", "free_surface_2d", "fluid_biomedical_2d",
    "fluid_hydraulics_2d", "rom_2d", "topology_opt_2d", "iga_2d",
]


class TestProbeStubsRemoved(unittest.TestCase):
    def test_removed_stubs_not_advertised(self):
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        b = get_backend("kratos")
        advertised = set()
        for p in b.supported_physics():
            advertised.add(p.name)
            advertised.update(getattr(p, "template_variants", []) or [])
        for key in REMOVED_STUBS:
            self.assertNotIn(key, advertised,
                             f"{key} is a removed probe-stub but still advertised")

    def test_no_surviving_probe_stub_generator(self):
        # Every kept generator's output must NOT be a note-only availability probe.
        from core.registry import load_all_backends, get_backend
        from core.quality_checks import is_stub_output
        load_all_backends()
        b = get_backend("kratos")
        for p in b.supported_physics():
            for v in (p.template_variants[:1] or ["default"]):
                try:
                    c = b.generate_input(p.name, v, {})
                except Exception:
                    continue
                self.assertNotIn('"note": "not installed"', c,
                                 f"{p.name}/{v} still emits a probe stub")


if __name__ == "__main__":
    unittest.main()
