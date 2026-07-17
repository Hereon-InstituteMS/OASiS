#!/usr/bin/env python3
"""
Regression tests for issue #40: "Python environments ignored".

check_availability() discovered the dune.fem interpreter in a conda env,
but run() unconditionally used the server's own venv Python (which
typically lacks dune.fem). Fix: a single resolver (_find_dune_python)
used by BOTH methods.

This suite verifies:
1. End-to-end: with a fake conda env, run() executes the SAME interpreter
   that check_availability() reported (the actual #40 regression).
2. DUNE_PYTHON / DUNE_CONDA_PREFIX explicit overrides win.
3. The resolver caches (probing spawns subprocesses; run() must not
   re-probe after check_availability()).
4. Negative path: no dune.fem anywhere -> NOT_INSTALLED with actionable
   hint, and run() fails cleanly with the same hint.
5. Generic invariant, ALL backends: check_availability() and run() must
   use the same executable-resolution helpers, so this bug class cannot
   silently reappear in any backend.
"""

import asyncio
import inspect
import os
import re
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backends.dune import backend as dune_mod
from backends.dune.backend import DuneBackend, _find_dune_python, _reset_dune_python_cache
from core.backend import BackendStatus


# ── helpers ──────────────────────────────────────────────────

FAKE_PYTHON_SCRIPT = """#!/bin/sh
# Fake dune-fem Python: answers `-c "import dune.fem; ..."` probes with OK
# and records its own path when executing a script (run() path).
if [ "$1" = "-c" ]; then
    case "$2" in
        *dune.fem*) echo OK; exit 0;;
        *) exit 1;;
    esac
fi
echo "$0" > "$(pwd)/interpreter_used.txt"
exit 0
"""

BROKEN_PYTHON_SCRIPT = """#!/bin/sh
# Fake Python WITHOUT dune.fem: every import probe fails.
exit 1
"""


def _write_fake_python(path: Path, script: str = FAKE_PYTHON_SCRIPT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


class _DuneEnvTestCase(unittest.TestCase):
    """Base: isolated cache + env for every test."""

    def setUp(self):
        _reset_dune_python_cache()
        self._env_patcher = mock.patch.dict(os.environ)
        self._env_patcher.start()
        os.environ.pop("DUNE_PYTHON", None)
        os.environ.pop("DUNE_CONDA_PREFIX", None)
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        _reset_dune_python_cache()
        self._env_patcher.stop()
        self._tmp.cleanup()


# ── 1. The actual #40 regression, end to end ─────────────────

class TestIssue40DuneCondaEnvRespected(_DuneEnvTestCase):
    """With dune.fem only in a conda env, run() must use THAT interpreter."""

    def _fake_home_with_dune_env(self, env_name="ofa-dune"):
        home = self.tmp / "home"
        return home, _write_fake_python(
            home / "miniconda3" / "envs" / env_name / "bin" / "python")

    def test_check_availability_finds_conda_env(self):
        home, fake_py = self._fake_home_with_dune_env()
        with mock.patch("pathlib.Path.home", return_value=home):
            status, msg = DuneBackend().check_availability()
        self.assertEqual(status, BackendStatus.AVAILABLE)
        self.assertIn(str(fake_py), msg)

    def test_run_uses_the_interpreter_check_availability_found(self):
        """The core #40 assertion: discovery and execution use ONE interpreter."""
        home, fake_py = self._fake_home_with_dune_env()
        work_dir = self.tmp / "job"
        backend = DuneBackend()
        with mock.patch("pathlib.Path.home", return_value=home):
            status, msg = backend.check_availability()
            self.assertEqual(status, BackendStatus.AVAILABLE)
            job = asyncio.run(backend.run("import dune.fem\n", work_dir))

        self.assertEqual(job.status, "completed", f"run() failed: {job.error}")
        marker = work_dir / "interpreter_used.txt"
        self.assertTrue(marker.is_file(),
                        "solve.py was not executed by the fake dune python — "
                        "run() picked a different interpreter than "
                        "check_availability() (issue #40 regression)")
        used = marker.read_text().strip()
        self.assertEqual(used, str(fake_py))
        # And it must be exactly the interpreter the availability message named.
        self.assertIn(used, msg)
        # The server's own venv python must NOT have been used.
        self.assertNotEqual(used, sys.executable)

    def test_dune_named_env_preferred_over_other_envs(self):
        """Priority 0 ('dune' in name) must beat priority 1 (any other env)."""
        home = self.tmp / "home"
        # 'aaa-shared' sorts before 'ofa-dune' — only priority can win here.
        _write_fake_python(home / "miniconda3" / "envs" / "aaa-shared" / "bin" / "python")
        dune_py = _write_fake_python(home / "miniconda3" / "envs" / "ofa-dune" / "bin" / "python")
        with mock.patch("pathlib.Path.home", return_value=home):
            self.assertEqual(_find_dune_python(), str(dune_py))

    def test_broken_dune_env_falls_through_to_working_env(self):
        """A dune-named env whose import fails must not shadow a working one."""
        home = self.tmp / "home"
        _write_fake_python(home / "miniconda3" / "envs" / "dune-broken" / "bin" / "python",
                           BROKEN_PYTHON_SCRIPT)
        good = _write_fake_python(home / "miniconda3" / "envs" / "shared-env" / "bin" / "python")
        with mock.patch("pathlib.Path.home", return_value=home):
            self.assertEqual(_find_dune_python(), str(good))


# ── 2. Explicit overrides ────────────────────────────────────

class TestDunePythonOverrides(_DuneEnvTestCase):

    def test_dune_python_env_var_wins_over_conda_env(self):
        home, _ = TestIssue40DuneCondaEnvRespected._fake_home_with_dune_env(self)
        override = _write_fake_python(self.tmp / "custom" / "bin" / "python")
        os.environ["DUNE_PYTHON"] = str(override)
        with mock.patch("pathlib.Path.home", return_value=home):
            self.assertEqual(_find_dune_python(), str(override))

    def test_dune_python_env_var_is_verified_not_trusted(self):
        """A DUNE_PYTHON that cannot import dune.fem must be rejected."""
        bad = _write_fake_python(self.tmp / "bad" / "bin" / "python", BROKEN_PYTHON_SCRIPT)
        good = _write_fake_python(
            self.tmp / "home" / "miniconda3" / "envs" / "ofa-dune" / "bin" / "python")
        os.environ["DUNE_PYTHON"] = str(bad)
        with mock.patch("pathlib.Path.home", return_value=self.tmp / "home"):
            self.assertEqual(_find_dune_python(), str(good))

    def test_dune_conda_prefix(self):
        env_root = self.tmp / "some-env"
        py = _write_fake_python(env_root / "bin" / "python")
        os.environ["DUNE_CONDA_PREFIX"] = str(env_root)
        with mock.patch("pathlib.Path.home", return_value=self.tmp / "empty-home"):
            self.assertEqual(_find_dune_python(), str(py))

    def test_run_respects_dune_python(self):
        override = _write_fake_python(self.tmp / "custom" / "bin" / "python")
        os.environ["DUNE_PYTHON"] = str(override)
        work_dir = self.tmp / "job"
        with mock.patch("pathlib.Path.home", return_value=self.tmp / "empty-home"):
            job = asyncio.run(DuneBackend().run("pass\n", work_dir))
        self.assertEqual(job.status, "completed", f"run() failed: {job.error}")
        self.assertEqual((work_dir / "interpreter_used.txt").read_text().strip(),
                         str(override))


class TestDuneSubprocessEnv(_DuneEnvTestCase):
    """run() must export the interpreter's prefix via CMAKE_PREFIX_PATH.

    dune-fem JIT-compiles C++ modules at run time; without the conda
    env's prefix on CMAKE_PREFIX_PATH the first JIT build fails with
    'Could not find ... dune-commonConfig.cmake' even though run()
    picked the right interpreter (issue #40 follow-up, verified live
    with dune-fem 2.12 on macOS).
    """

    def test_prefix_prepended(self):
        py = self.tmp / "some-env" / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("")
        os.environ.pop("CMAKE_PREFIX_PATH", None)
        env = dune_mod._dune_subprocess_env(str(py))
        expected = str((self.tmp / "some-env").resolve())
        self.assertEqual(env["CMAKE_PREFIX_PATH"], expected)

    def test_existing_prefix_path_preserved(self):
        py = self.tmp / "some-env" / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("")
        os.environ["CMAKE_PREFIX_PATH"] = "/opt/other"
        env = dune_mod._dune_subprocess_env(str(py))
        expected = str((self.tmp / "some-env").resolve())
        self.assertEqual(
            env["CMAKE_PREFIX_PATH"],
            f"{expected}{os.pathsep}/opt/other")

    def test_no_duplicate_when_already_present(self):
        py = self.tmp / "some-env" / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("")
        expected = str((self.tmp / "some-env").resolve())
        os.environ["CMAKE_PREFIX_PATH"] = expected
        env = dune_mod._dune_subprocess_env(str(py))
        self.assertEqual(env["CMAKE_PREFIX_PATH"], expected)

    # ── issue #44: force CMake's FindPython3 to the resolved interpreter ──
    # CMAKE_PREFIX_PATH alone let CMake grab Xcode's Python 3.9 on macOS.

    def test_conda_activation_vars_set_for_conda_env(self):
        env_root = self.tmp / "ofa-dune"
        (env_root / "conda-meta").mkdir(parents=True)
        py = env_root / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("")
        env = dune_mod._dune_subprocess_env(str(py))
        self.assertEqual(env["CONDA_PREFIX"], str(env_root.resolve()))
        self.assertEqual(env["CONDA_DEFAULT_ENV"], "ofa-dune")

    def test_conda_vars_not_set_for_non_conda_env(self):
        os.environ.pop("CONDA_PREFIX", None)
        py = self.tmp / "plain-venv" / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("")
        env = dune_mod._dune_subprocess_env(str(py))
        self.assertNotEqual(env.get("CONDA_PREFIX"),
                            str((self.tmp / "plain-venv").resolve()))

    def test_bin_dir_prepended_to_path(self):
        py = self.tmp / "some-env" / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("")
        env = dune_mod._dune_subprocess_env(str(py))
        self.assertEqual(env["PATH"].split(os.pathsep)[0],
                         str((self.tmp / "some-env" / "bin").resolve()))

    def test_findpython3_hints_set(self):
        py = self.tmp / "some-env" / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("")
        env = dune_mod._dune_subprocess_env(str(py))
        self.assertEqual(env["Python3_ROOT_DIR"],
                         str((self.tmp / "some-env").resolve()))
        self.assertEqual(env["Python3_EXECUTABLE"], str(py.resolve()))


# ── 3. Caching ───────────────────────────────────────────────

class TestDunePythonCache(_DuneEnvTestCase):

    def test_second_resolution_uses_cache_not_reprobe(self):
        home = self.tmp / "home"
        fake_py = _write_fake_python(home / "miniconda3" / "envs" / "ofa-dune" / "bin" / "python")
        with mock.patch("pathlib.Path.home", return_value=home):
            first = _find_dune_python()
        self.assertEqual(first, str(fake_py))
        # Remove the interpreter: a re-probe would now fail, the cache must hold.
        fake_py.unlink()
        with mock.patch("pathlib.Path.home", return_value=home):
            self.assertEqual(_find_dune_python(), str(fake_py))

    def test_reset_clears_cache(self):
        home = self.tmp / "home"
        fake_py = _write_fake_python(home / "miniconda3" / "envs" / "ofa-dune" / "bin" / "python")
        with mock.patch("pathlib.Path.home", return_value=home):
            self.assertEqual(_find_dune_python(), str(fake_py))
        fake_py.unlink()
        _reset_dune_python_cache()
        with mock.patch("pathlib.Path.home", return_value=home), \
             mock.patch.object(dune_mod, "get_python_executable", return_value=None):
            self.assertIsNone(_find_dune_python())


# ── 4. Negative path ─────────────────────────────────────────

class TestDuneNotInstalled(_DuneEnvTestCase):

    def test_not_installed_message_is_actionable(self):
        with mock.patch("pathlib.Path.home", return_value=self.tmp / "empty-home"), \
             mock.patch.object(dune_mod, "get_python_executable", return_value=None):
            status, msg = DuneBackend().check_availability()
        self.assertEqual(status, BackendStatus.NOT_INSTALLED)
        self.assertIn("conda create -n ofa-dune", msg)
        self.assertIn("DUNE_PYTHON", msg)

    def test_run_fails_cleanly_with_same_hint(self):
        with mock.patch("pathlib.Path.home", return_value=self.tmp / "empty-home"), \
             mock.patch.object(dune_mod, "get_python_executable", return_value=None):
            job = asyncio.run(DuneBackend().run("pass\n", self.tmp / "job"))
        self.assertEqual(job.status, "failed")
        self.assertIn("DUNE_PYTHON", job.error)


# ── 5. Generic invariant across ALL backends ─────────────────

class TestInterpreterResolutionConsistencyAllBackends(unittest.TestCase):
    """check_availability() and run() must resolve executables identically.

    Guards against the #40 bug class in every backend: extract the
    executable-resolution helpers referenced by each method's source and
    require the sets to match. A backend whose discovery uses helper A
    but whose execution uses helper B ships a discovery/execution split.
    """

    RESOLVER_RE = re.compile(
        r"\b(_find_\w+_python|_find_\w+_binary|get_python_executable)\b")

    # dealii resolves via CMake in both methods (check: `shutil.which("cmake")`
    # + test-compile; run: cmake configure + make). Its _find_dealii helper
    # locates headers for version reporting only, so the token-set invariant
    # does not apply. sparta/fourc additionally shell out to mpirun for np>1,
    # which is not an interpreter/binary *resolution* difference.
    EXEMPT = {"dealii"}

    @classmethod
    def setUpClass(cls):
        from core.registry import load_all_backends, _backends
        load_all_backends()
        cls.backends = dict(_backends)

    def test_every_backend_resolves_consistently(self):
        self.assertGreater(len(self.backends), 0, "no backends registered")
        for name, backend in self.backends.items():
            if name in self.EXEMPT:
                continue
            with self.subTest(backend=name):
                check_src = inspect.getsource(backend.check_availability)
                run_src = inspect.getsource(backend.run)
                check_resolvers = set(self.RESOLVER_RE.findall(check_src))
                run_resolvers = set(self.RESOLVER_RE.findall(run_src))
                self.assertTrue(
                    check_resolvers,
                    f"{name}: check_availability() uses no known resolver "
                    f"helper — extend RESOLVER_RE or EXEMPT with a comment")
                self.assertEqual(
                    check_resolvers, run_resolvers,
                    f"{name}: check_availability() resolves via "
                    f"{sorted(check_resolvers)} but run() via "
                    f"{sorted(run_resolvers)} — discovery and execution "
                    f"would use different interpreters/binaries (issue #40 "
                    f"bug class)")

    def test_dune_run_does_not_use_server_venv_resolver(self):
        """Directly pin the #40 fix: run() must not call get_python_executable()."""
        backend = self.backends.get("dune")
        self.assertIsNotNone(backend, "dune backend not registered")
        run_src = inspect.getsource(backend.run)
        self.assertNotIn("get_python_executable()", run_src)
        self.assertIn("_find_dune_python()", run_src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
