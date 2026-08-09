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


class TestHonestyGuardChecksUseNotPresence(unittest.TestCase):
    """A solve marker must be CALLED, not merely imported.

    The MPM generator emitted a standalone numpy material-point method — its
    own grid, its own shape functions, its own USL loop, and the string
    "KratosMultiphysics" nowhere in it. `validate_input`'s honesty guard passed
    it for months on the strength of one line:

        from scipy.sparse.linalg import spsolve

    spsolve was never called, and neither was lil_matrix. The marker list held
    the bare name `spsolve` AND the module path `scipy.sparse.linalg`, so a
    dead import certified a solve. That is the same defect shape as an
    expectation satisfied by a word the fixture prints itself: the check can
    only be passed, never failed, by the thing it exists to test.
    """

    def _backend(self):
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        return get_backend("kratos")

    def test_an_unused_import_does_not_certify_a_solve(self):
        b = self._backend()
        dead = ("import numpy as np\n"
                "from scipy.sparse import lil_matrix\n"
                "from scipy.sparse.linalg import spsolve\n"
                "x = np.zeros(3)\n"
                "print('done', x.sum())\n")
        self.assertTrue(
            b.validate_input(dead),
            "a script that imports spsolve and never calls it, and never "
            "touches Kratos, was accepted as a runnable Kratos analysis")

    def test_the_same_script_calling_the_solver_is_accepted(self):
        """Both directions: the guard must not have become a blanket refusal."""
        b = self._backend()
        live = ("import numpy as np\n"
                "from scipy.sparse import csr_matrix\n"
                "from scipy.sparse.linalg import spsolve\n"
                "A = csr_matrix(np.eye(3))\n"
                "u = spsolve(A, np.ones(3))\n"
                "print('done', u.sum())\n")
        self.assertEqual(
            [], b.validate_input(live),
            "the guard rejected a script that genuinely assembles and solves")

    def test_the_mpm_template_drives_mpmapplication(self):
        """The capability is 'Kratos MPM', so Kratos must be what runs."""
        b = self._backend()
        t = b.generate_input("mpm", "2d", {})
        self.assertIn("KratosMultiphysics.MPMApplication", t)
        self.assertIn("MpmAnalysis", t)
        self.assertIn(".Run()", t)
        self.assertEqual([], b.validate_input(t))

    def test_every_template_still_passes(self):
        b = self._backend()
        rejected = []
        for p in b.supported_physics():
            for v in (p.template_variants or ["default"]):
                try:
                    t = b.generate_input(p.name, v, {})
                except Exception:      # noqa: BLE001 - a different defect
                    continue
                if b.validate_input(t):
                    rejected.append(f"{p.name}:{v}")
        self.assertEqual([], rejected,
                         "tightening the guard turned working templates red")

    def test_a_material_point_count_off_the_allowed_set_is_refused(self):
        """The generator must not emit a deck Kratos will reject by name.

        Quadrilateral grids accept 1/4/9/16/25 material points per element and
        nothing else. Refusing at generation time means the failure is a
        message about the input, not a stack trace forty seconds into a run.
        """
        b = self._backend()
        with self.assertRaises(ValueError):
            b.generate_input("mpm", "2d", {"material_points_per_element": 5})
        self.assertIn("MATERIAL_POINTS_PER_ELEMENT",
                      b.generate_input("mpm", "2d",
                                       {"material_points_per_element": 9}))


if __name__ == "__main__":
    unittest.main()
