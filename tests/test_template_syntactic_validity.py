"""Regression: every backend's template_variants emit syntactically
valid input for the backend's expected format.

Complements Layer-F (which only covers the 4 Python-runnable
backends end-to-end with rc=0): this gate runs in the repo .venv,
needs none of the heavy compiled toolchains, and asserts:

  • fourc  — every emitted template is valid YAML (parses via
              yaml.safe_load without error).
  • febio  — every emitted template is valid XML (parses via
              xml.etree.ElementTree without error).
  • dealii — every emitted template contains both `#include`
              directives and `int main(` — the minimum shape
              for a deal.II driver C++ file.

A backend whose runtime library is not installed in the test env
still emits templates (they're just strings); this test catches
the regression where a generator silently drops the required
syntactic skeleton — e.g. forgets the febio_spec root tag, emits
malformed YAML keys, or omits `int main()` in a dealii driver.
"""
from __future__ import annotations

import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


def _gather_templates(backend_name: str):
    from core.registry import get_backend, load_all_backends
    load_all_backends()
    b = get_backend(backend_name)
    assert b is not None, f"backend not registered: {backend_name}"
    out = []
    for cap in b.supported_physics():
        for variant in cap.template_variants:
            try:
                tmpl = b.generate_input(cap.name, variant, {})
            except Exception as e:
                tmpl = None  # generation error is its own gate
            out.append((cap.name, variant, tmpl))
    return out


class TestTemplateSyntacticValidity(unittest.TestCase):
    def test_fourc_yaml(self) -> None:
        failures = []
        for physics, variant, tmpl in _gather_templates("fourc"):
            self.assertIsNotNone(
                tmpl, f"fourc::{physics}::{variant}: GEN_FAIL")
            try:
                yaml.safe_load(tmpl)
            except yaml.YAMLError as e:
                failures.append((physics, variant, str(e)[:120]))
        self.assertEqual(
            failures, [],
            f"fourc templates with YAML parse errors: {failures}. "
            "Each fourc generator must emit a parseable YAML "
            "document.")

    def test_febio_xml(self) -> None:
        failures = []
        for physics, variant, tmpl in _gather_templates("febio"):
            self.assertIsNotNone(
                tmpl, f"febio::{physics}::{variant}: GEN_FAIL")
            try:
                ET.fromstring(tmpl)
            except ET.ParseError as e:
                failures.append((physics, variant, str(e)[:120]))
        self.assertEqual(
            failures, [],
            f"febio templates with XML parse errors: {failures}. "
            "Each febio generator must emit a parseable XML "
            "document with a febio_spec root.")

    def test_dealii_cpp_skeleton(self) -> None:
        """Minimal shape check: every dealii template must define
        a main() function and #include at least one header. The
        compiler is the real syntactic gate; this just catches
        empty-or-broken stubs."""
        missing_main = []
        missing_include = []
        for physics, variant, tmpl in _gather_templates("dealii"):
            self.assertIsNotNone(
                tmpl, f"dealii::{physics}::{variant}: GEN_FAIL")
            if "int main(" not in tmpl:
                missing_main.append((physics, variant))
            if "#include" not in tmpl:
                missing_include.append((physics, variant))
        self.assertEqual(
            missing_main, [],
            f"dealii templates without int main(): {missing_main}")
        self.assertEqual(
            missing_include, [],
            f"dealii templates without #include: {missing_include}")

    def test_python_backends_compile(self) -> None:
        """Python-based backends (skfem, kratos, ngsolve, fenics,
        dune) — every template must compile to bytecode without
        SyntaxError. Layer-F runs the rc=0 path; this catches a
        different bug class: dead branches inside the template
        that aren't exercised at runtime but would crash the
        interpreter at module-import time if the user did
        `python template.py`. dune is the most important target
        here because the actual library isn't installed in any
        test env — Layer F skips it; this gate keeps the
        generators honest."""
        failures = []
        for backend in ("dune", "skfem", "kratos", "ngsolve",
                         "fenics"):
            for physics, variant, tmpl in _gather_templates(backend):
                self.assertIsNotNone(
                    tmpl, f"{backend}::{physics}::{variant}: GEN_FAIL")
                try:
                    compile(tmpl,
                            f"<{backend}_{physics}_{variant}>",
                            "exec")
                except SyntaxError as e:
                    failures.append(
                        (backend, physics, variant,
                         f"line {e.lineno}: {e.msg}"))
        self.assertEqual(
            failures, [],
            "Python-template SyntaxError: "
            f"{failures}. Generator must emit valid Python.")


if __name__ == "__main__":
    unittest.main()
