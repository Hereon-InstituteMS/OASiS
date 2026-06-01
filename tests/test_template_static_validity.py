"""Regression: every catalog generator's `generate_input()` output
must be statically valid for its declared input format.

The four checks pin a baseline that prior audits established:

  - Python  (fenics, ngsolve, skfem, dune, febio[no], kratos):
       `compile(content, ..., 'exec')` must succeed.
  - YAML    (fourc):
       `yaml.safe_load_all(content)` must parse >=1 doc, unless
       the template is in test_yaml_template_parses.STUB_TEMPLATES.
  - XML     (febio):
       `xml.etree.ElementTree.fromstring(content)` must succeed.
  - C++     (deal.II):
       brace count balanced, `int main` present, no `{ident}` bare
       placeholders left over from format-string mishaps.

This is a static-only pass — it does NOT compile the C++ or
run the Python — but every class of bug fixed during the
2026-06-02 audit (ngsolve f-string brace escape, dune
helmholtz `{k_val}`, fourc particle_pd YAML colon-in-placeholder,
9 stub templates) was a static-detectable defect. Putting this
test in the suite means the next instance trips immediately.

When a backend adds a new physics/variant and this test
fails, the fix is the SAME spot the audit pointed at: the
generator function in src/backends/<be>/generators/<phys>.py.
"""
from __future__ import annotations

import re
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


class TestTemplateStaticValidity(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, all_backends
        load_all_backends()
        cls.backends = all_backends()
        if not cls.backends:
            raise unittest.SkipTest("no backends registered")
        from tests.test_yaml_template_parses import STUB_TEMPLATES  # type: ignore
        cls.stubs = STUB_TEMPLATES

    def _iter_templates(self, fmt_filter: str | None = None):
        for b in self.backends:
            fmt = b.input_format().value
            if fmt_filter is not None and fmt != fmt_filter:
                continue
            for p in b.supported_physics():
                for v in p.template_variants:
                    yield b, fmt, p.name, v

    def test_python_templates_compile(self) -> None:
        failures = []
        n = 0
        for b, fmt, ph, v in self._iter_templates("python"):
            n += 1
            try:
                content = b.generate_input(ph, v, {})
            except Exception as e:
                failures.append((b.name(), ph, v,
                                 f"generate_input: {type(e).__name__}: {e}"))
                continue
            try:
                compile(content, f"{b.name()}/{ph}/{v}", "exec")
            except SyntaxError as e:
                failures.append((b.name(), ph, v,
                                 f"SyntaxError line {e.lineno}: {e.msg}"))
        if failures:
            lines = "\n".join(
                f"  {be}/{ph}/{v}: {err}" for be, ph, v, err in failures)
            self.fail(
                f"{len(failures)} of {n} Python templates failed to "
                f"compile:\n{lines}")

    def test_yaml_templates_parse(self) -> None:
        failures = []
        n = 0
        for b, fmt, ph, v in self._iter_templates("yaml"):
            n += 1
            try:
                content = b.generate_input(ph, v, {})
            except Exception as e:
                failures.append((b.name(), ph, v,
                                 f"generate_input: {type(e).__name__}: {e}"))
                continue
            try:
                docs = list(yaml.safe_load_all(content))
            except yaml.YAMLError as e:
                msg = str(e).splitlines()[0][:160]
                failures.append((b.name(), ph, v, f"YAMLError: {msg}"))
                continue
            if not docs or all(d is None for d in docs):
                if (b.name(), ph, v) in self.stubs:
                    continue  # known stub
                failures.append(
                    (b.name(), ph, v,
                     "parsed as empty YAML (likely stub); add to "
                     "tests.test_yaml_template_parses.STUB_TEMPLATES "
                     "if intentional"))
        if failures:
            lines = "\n".join(
                f"  {be}/{ph}/{v}: {err}" for be, ph, v, err in failures)
            self.fail(
                f"{len(failures)} of {n} YAML templates broken:\n{lines}")

    def test_xml_templates_parse(self) -> None:
        failures = []
        n = 0
        for b, fmt, ph, v in self._iter_templates("xml"):
            n += 1
            try:
                content = b.generate_input(ph, v, {})
            except Exception as e:
                failures.append((b.name(), ph, v,
                                 f"generate_input: {type(e).__name__}: {e}"))
                continue
            try:
                ET.fromstring(content)
            except ET.ParseError as e:
                failures.append((b.name(), ph, v,
                                 f"ParseError: {str(e)[:160]}"))
        if failures:
            lines = "\n".join(
                f"  {be}/{ph}/{v}: {err}" for be, ph, v, err in failures)
            self.fail(
                f"{len(failures)} of {n} XML templates broken:\n{lines}")

    def test_cpp_templates_static_checks(self) -> None:
        """Brace balance, main() presence, and no leftover bare
        `{ident}` placeholders (f-string format-string mishaps).
        """
        # Bare {identifier} that occurs at statement start — i.e.,
        # NOT in legitimate C++ initializer-list / template-arg
        # contexts. (See conservative filter in audit prose.)
        placeholder_re = re.compile(r"\{[a-z_][a-z0-9_]{0,30}\}")
        failures = []
        n = 0
        for b, fmt, ph, v in self._iter_templates("cpp"):
            n += 1
            try:
                content = b.generate_input(ph, v, {})
            except Exception as e:
                failures.append((b.name(), ph, v,
                                 f"generate_input: {type(e).__name__}: {e}"))
                continue
            opens = content.count("{")
            closes = content.count("}")
            if opens != closes:
                failures.append(
                    (b.name(), ph, v,
                     f"brace imbalance: {opens} open vs {closes} close"))
                continue
            if "int main" not in content:
                failures.append(
                    (b.name(), ph, v, "no int main() found"))
                continue
            # Look for {ident} starting a statement / line (likely
            # f-string format leftover, not legitimate C++).
            suspects: list = []
            for m in placeholder_re.finditer(content):
                start = m.start()
                ctx_before = content[max(0, start - 30):start].rstrip()
                if ctx_before and ctx_before[-1] in "=,({<:":
                    continue  # initializer-list / template-arg context
                if ctx_before.endswith(";") or ctx_before.endswith("}"):
                    suspects.append((m.group(0),
                                     ctx_before[-20:] + m.group(0)))
            if suspects:
                failures.append(
                    (b.name(), ph, v,
                     f"{len(suspects)} suspicious placeholders: "
                     f"{suspects[:3]}"))
        if failures:
            lines = "\n".join(
                f"  {be}/{ph}/{v}: {err}" for be, ph, v, err in failures)
            self.fail(
                f"{len(failures)} of {n} C++ templates broken:\n{lines}")


if __name__ == "__main__":
    unittest.main()
