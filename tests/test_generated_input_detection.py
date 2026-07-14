"""Regression tests for run_with_generator's input-file detection.

Guards against issue #39: deal.II's generator writes ``main.cpp`` and
SPARTA's writes ``in.sparta``, neither of which was matched by the old
hardcoded glob ``["*.4C.yaml","*.yaml","input.*","solve.py","MainKratos.py"]``,
so run_with_generator failed with "Generator did not produce an input file"
even though the generator succeeded.

These tests need no installed backend — they exercise the pure detection
helper with a stub backend exposing only ``input_format``.
"""
import tempfile
from pathlib import Path

import pytest

from core.backend import find_generated_input, InputFormat


class _StubBackend:
    def __init__(self, fmt):
        self.input_format = fmt


class _MethodStubBackend:
    """Mimics the real backends, where ``input_format`` is a METHOD."""
    def __init__(self, fmt):
        self._fmt = fmt

    def input_format(self):
        return self._fmt


def test_input_format_as_method_takes_priority_over_legacy():
    """Real backends expose ``input_format`` as a method. The format-specific
    patterns must still win over the legacy glob: a decoy ``notes.yaml`` in a
    deal.II dir must not be picked over ``main.cpp`` (would happen if the helper
    read the bound method instead of calling it)."""
    with tempfile.TemporaryDirectory() as d:
        wd = Path(d)
        (wd / "generate_input.py").write_text("#")
        (wd / "notes.yaml").write_text("decoy: true")
        (wd / "main.cpp").write_text("int main(){return 0;}")
        found = find_generated_input(wd, _MethodStubBackend(InputFormat.CPP))
        assert found is not None and found.name == "main.cpp"


# The canonical file each backend's generator writes, by declared format.
FORMAT_FILES = [
    (InputFormat.CPP, "main.cpp", "int main(){return 0;}"),
    (InputFormat.SPARTA, "in.sparta", "run 0\n"),
    (InputFormat.PYTHON, "solve.py", "print(1)\n"),
    (InputFormat.YAML, "sim.4C.yaml", "PROBLEM TYPE: Fluid\n"),
    (InputFormat.XML, "input.feb", "<febio_spec/>\n"),
    (InputFormat.JSON, "MainKratos.py", "print(1)\n"),
]


@pytest.mark.parametrize("fmt,fname,content", FORMAT_FILES,
                         ids=[f.name for f, _, _ in FORMAT_FILES])
def test_detects_each_format(fmt, fname, content):
    """Every backend's generator output is detected, and the generator
    script itself is never mistaken for the input."""
    with tempfile.TemporaryDirectory() as d:
        wd = Path(d)
        (wd / "generate_input.py").write_text("# the generator, must be ignored")
        (wd / fname).write_text(content)
        found = find_generated_input(wd, _StubBackend(fmt))
        assert found is not None, f"{fmt.name}: nothing detected (issue #39 regression)"
        assert found.name == fname


def test_dealii_ignores_cmakelists_and_generator():
    """deal.II generator writes main.cpp + CMakeLists.txt; must pick main.cpp."""
    with tempfile.TemporaryDirectory() as d:
        wd = Path(d)
        (wd / "generate_input.py").write_text("#")
        (wd / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.13.4)\n")
        (wd / "main.cpp").write_text("int main(){return 0;}")
        found = find_generated_input(wd, _StubBackend(InputFormat.CPP))
        assert found is not None and found.name == "main.cpp"


def test_only_scripts_and_logs_returns_none():
    """A generator that produced no input file yields None (the honest
    'did not produce an input file' path)."""
    with tempfile.TemporaryDirectory() as d:
        wd = Path(d)
        (wd / "generate_input.py").write_text("#")
        (wd / "stderr.log").write_text("boom")
        (wd / "stdout.log").write_text("")
        assert find_generated_input(wd, _StubBackend(InputFormat.CPP)) is None


def test_fallback_picks_unexpected_filename():
    """Even a non-standard generator output is found via the last-resort
    fallback, so a valid run is never dropped on a naming quirk."""
    with tempfile.TemporaryDirectory() as d:
        wd = Path(d)
        (wd / "generate_input.py").write_text("#")
        (wd / "weird_name.inp").write_text("data")
        found = find_generated_input(wd, _StubBackend(InputFormat.SPARTA))
        assert found is not None and found.name == "weird_name.inp"


def test_works_without_backend():
    """Detection must still work if no backend is passed (legacy patterns)."""
    with tempfile.TemporaryDirectory() as d:
        wd = Path(d)
        (wd / "generate_input.py").write_text("#")
        (wd / "problem.4C.yaml").write_text("x: 1")
        found = find_generated_input(wd, None)
        assert found is not None and found.name == "problem.4C.yaml"
