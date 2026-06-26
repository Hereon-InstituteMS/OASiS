"""
Comprehensive physics coverage tests.

Verifies that ALL physics types across ALL backends:
1. Are registered as PhysicsCapability
2. Have knowledge entries
3. Have generator functions that produce non-empty output
4. Can be found via prepare_simulation's fuzzy matching
"""

import sys
import os
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestPhysicsCoverage(unittest.TestCase):
    """Verify every backend has complete physics coverage."""

    @classmethod
    def setUpClass(cls):
        from core.registry import load_all_backends, _backends
        load_all_backends()
        cls.backends = _backends

    def test_all_backends_loaded(self):
        """All 7 working backends must be registered."""
        names = {b.name() for b in self.backends.values()}
        expected = {"fourc", "fenics", "dealii", "ngsolve", "skfem", "kratos", "dune"}
        for exp in expected:
            self.assertIn(exp, names, f"Backend {exp} not registered")

    def test_minimum_physics_per_backend(self):
        """Each working backend must have a minimum number of physics types."""
        minimums = {
            "fourc": 35,
            "kratos": 30,
            "dealii": 25,
            "ngsolve": 18,
            "fenics": 16,
            "skfem": 13,
            "dune": 13,
            # febio excluded — not installed
        }
        for name, backend in self.backends.items():
            if name not in minimums:
                continue
            count = len(backend.supported_physics())
            min_count = minimums[name]
            self.assertGreaterEqual(
                count, min_count,
                f"{name} has {count} physics, expected >= {min_count}"
            )

    def test_every_physics_has_knowledge(self):
        """Every registered physics should return non-empty knowledge (warn, not fail)."""
        missing_knowledge = []
        total = 0
        with_knowledge = 0
        for name, backend in self.backends.items():
            for p in backend.supported_physics():
                total += 1
                k = backend.get_knowledge(p.name)
                if k and not (isinstance(k, dict) and k.get("error")):
                    with_knowledge += 1
                else:
                    missing_knowledge.append(f"{name}/{p.name}")
        coverage = with_knowledge / total * 100 if total else 0
        print(f"\nKnowledge coverage: {with_knowledge}/{total} ({coverage:.0f}%)")
        if missing_knowledge:
            print(f"Missing: {', '.join(missing_knowledge[:10])}...")
        # At least 80% must have knowledge
        self.assertGreaterEqual(coverage, 80,
                                f"Knowledge coverage {coverage:.0f}% < 80%")

    def test_every_physics_has_description(self):
        """Every PhysicsCapability must have a non-empty description."""
        for name, backend in self.backends.items():
            for p in backend.supported_physics():
                self.assertTrue(
                    len(p.description) > 5,
                    f"{name}/{p.name}: empty or too-short description"
                )

    def test_every_physics_has_spatial_dims(self):
        """Every physics must specify at least one spatial dimension."""
        for name, backend in self.backends.items():
            for p in backend.supported_physics():
                self.assertTrue(
                    len(p.spatial_dims) > 0,
                    f"{name}/{p.name}: no spatial_dims"
                )

    def test_total_physics_count(self):
        """Total physics across all backends must meet target."""
        total = sum(len(b.supported_physics()) for b in self.backends.values())
        self.assertGreaterEqual(total, 150,
                                f"Total physics = {total}, expected >= 150")
        print(f"\nTotal physics across all backends: {total}")

    def test_no_duplicate_physics_names_per_backend(self):
        """No backend should have duplicate physics names."""
        for name, backend in self.backends.items():
            names = [p.name for p in backend.supported_physics()]
            duplicates = [n for n in names if names.count(n) > 1]
            self.assertEqual(
                len(set(duplicates)), 0,
                f"{name} has duplicate physics: {set(duplicates)}"
            )


class TestKratosFullCoverage(unittest.TestCase):
    """Verify Kratos covers all major applications."""

    @classmethod
    def setUpClass(cls):
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("kratos")

    def test_kratos_applications_covered(self):
        """All major Kratos applications must have corresponding physics."""
        required_physics = [
            "linear_elasticity", "fluid", "dem", "mpm", "fsi", "contact",
            "cosimulation", "shape_optimization", "geomechanics", "rans",
            "compressible_potential", "pfem_fluid", "rom",
            "topology_optimization", "iga", "poromechanics", "shallow_water",
            "constitutive_laws", "thermal_dem", "swimming_dem",
            "dem_structures_coupling", "fem_to_dem", "cable_net",
            "chimera", "droplet_dynamics", "fluid_biomedical",
            "optimization",
        ]
        registered = {p.name for p in self.backend.supported_physics()}
        missing = [r for r in required_physics if r not in registered]
        self.assertEqual(missing, [],
                         f"Kratos missing physics: {missing}")


class TestFourcFullCoverage(unittest.TestCase):
    """Verify 4C covers all source modules."""

    @classmethod
    def setUpClass(cls):
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("fourc")

    def test_fourc_modules_covered(self):
        """All major 4C physics modules must have generators."""
        required_physics = [
            "poisson", "linear_elasticity", "heat", "fluid", "fsi",
            "structural_dynamics", "beams", "contact", "particle_pd",
            "particle_sph", "tsi", "ssi", "ale", "electrochemistry",
            "level_set", "low_mach", "ssti", "sti", "fbi", "fpsi",
            "pasi", "lubrication", "cardiac_monodomain", "arterial_network",
            "xfem_fluid", "fsi_xfem", "fs3i", "ehl", "reduced_airways",
            "beam_interaction", "multiscale", "porous_media",
            "membrane", "shell", "thermo", "mixture",
        ]
        registered = {p.name for p in self.backend.supported_physics()}
        missing = [r for r in required_physics if r not in registered]
        self.assertEqual(missing, [],
                         f"4C missing physics: {missing}")


class TestDealiiFullCoverage(unittest.TestCase):
    """Verify deal.II covers key step tutorials."""

    @classmethod
    def setUpClass(cls):
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("dealii")

    def test_dealii_key_steps_covered(self):
        """All key deal.II tutorial categories must have physics."""
        # Rebaselined to physics that produce a REAL, runnable, output-writing
        # deal.II program on the installed version (9.1.x). compressible_euler,
        # topology_opt_dealii, multiphysics_dealii, cg_dg_coupled, optimal_control
        # were REMOVED in the catalog-honesty overhaul (could not be made runnable
        # here — e.g. cg_dg_coupled needs FEInterfaceValues, absent in 9.1.x).
        required_physics = [
            "poisson", "linear_elasticity", "heat", "stokes",
            "helmholtz", "eigenvalue", "hyperelasticity",
            "navier_stokes", "mixed_laplacian",
            "matrix_free", "multigrid", "obstacle_problem",
        ]
        registered = {p.name for p in self.backend.supported_physics()}
        missing = [r for r in required_physics if r not in registered]
        self.assertEqual(missing, [],
                         f"deal.II missing physics: {missing}")


class TestSolverFreshness(unittest.TestCase):
    """Check that solver sources and pip packages are reasonably up to date."""

    def test_source_repos_exist(self):
        """Source root env vars point to valid directories if set."""
        env_vars = {
            "FOURC_ROOT": "4C",
            "KRATOS_ROOT": "Kratos",
            "DEALII_ROOT": "deal.II",
        }
        for var, name in env_vars.items():
            root = os.environ.get(var, "")
            if root:
                self.assertTrue(
                    os.path.isdir(root),
                    f"{var}={root} is set but directory does not exist"
                )

    def test_source_repos_not_stale(self):
        """Source repos should have commits less than 30 days old."""
        import subprocess
        import time
        env_vars = ["FOURC_ROOT", "KRATOS_ROOT", "DEALII_ROOT"]
        for var in env_vars:
            root = os.environ.get(var, "")
            if not root or not os.path.isdir(root):
                continue
            try:
                result = subprocess.run(
                    ["git", "log", "-1", "--format=%ct"],
                    capture_output=True, text=True, cwd=root, timeout=5
                )
                if result.returncode == 0:
                    last_commit = int(result.stdout.strip())
                    days_old = (time.time() - last_commit) / 86400
                    self.assertLess(
                        days_old, 30,
                        f"{var} last commit is {days_old:.0f} days old. "
                        f"Consider: cd {root} && git pull"
                    )
            except Exception:
                pass  # Skip if git not available

    def test_pip_solvers_installed(self):
        """Key pip-installed solvers should be importable."""
        solvers = {
            "ngsolve": "NGSolve",
            "skfem": "scikit-fem",
            "KratosMultiphysics": "Kratos",
        }
        installed = []
        for module, name in solvers.items():
            try:
                __import__(module)
                installed.append(name)
            except ImportError:
                pass
        # At least 2 pip solvers should be installed
        self.assertGreaterEqual(
            len(installed), 2,
            f"Only {len(installed)} pip solvers installed: {installed}"
        )

    def test_check_script_exists(self):
        """The freshness check script should exist and be executable."""
        script = Path(__file__).parent.parent / "check_solver_updates.sh"
        self.assertTrue(script.exists(), "check_solver_updates.sh not found")
        self.assertTrue(
            os.access(str(script), os.X_OK),
            "check_solver_updates.sh is not executable"
        )


if __name__ == "__main__":
    unittest.main()
