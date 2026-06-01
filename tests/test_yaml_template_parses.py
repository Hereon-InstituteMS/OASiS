"""Regression: every fourc YAML template that is NOT a known stub
must parse as valid YAML.

Caught 2026-06-02:
  Probing every (backend, physics, variant) tuple's
  generate_input() output through yaml.safe_load_all surfaced:

    1 catastrophic YAML syntax error:
      fourc/particle_pd/plate_2d -> "mapping values are not
      allowed here" caused by a colon-inside-a-`<placeholder>`
      block (`TIMESTEP: <dt from CFL: dt < 0.5 ...>`). YAML
      saw the inner `: ` as a nested-mapping start. Fixed by
      quoting the placeholder string.

    9 stub-comment templates (membrane / shell / thermo /
    mixture / constraint / brownian_dynamics /
    cardiovascular0d / fluid_turbulence) that return just a
    single `# Foo template — use ...` placeholder line. These
    parse as empty YAML documents and were silently surfaced
    by prepare_simulation as fully-formed templates. Fixed by
    adding a ⚠ STUB tag in the prepare_simulation surface.

This test guards both regressions:
  (1) every YAML template either parses to >=1 doc, OR carries
      the ⚠ STUB marker in the prepare_simulation surface.
  (2) no new YAML syntax errors slip in.

The stub list is hand-curated — when a stub gets a real
template, drop it from STUB_TEMPLATES. When a new physics ships
as a stub, add it here AND make sure prepare_simulation tags
it.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


# Templates that are intentionally one-line comment stubs at the
# time of writing. Each entry is (backend, physics, variant).
# Drop from here once the stub is replaced by a real template.
STUB_TEMPLATES = {
    ("fourc", "membrane", "membrane_2d"),
    ("fourc", "shell", "shell_3d"),
    ("fourc", "thermo", "thermo_2d"),
    ("fourc", "thermo", "thermo_3d"),
    ("fourc", "mixture", "mixture_3d"),
    ("fourc", "constraint", "constraint_3d"),
    ("fourc", "brownian_dynamics", "brownian_3d"),
    ("fourc", "cardiovascular0d", "windkessel_3d"),
    ("fourc", "fluid_turbulence", "les_channel_3d"),
}


class TestYamlTemplatesParse(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, all_backends
        load_all_backends()
        cls.backends = [b for b in all_backends()
                        if b.input_format().value == "yaml"]
        if not cls.backends:
            raise unittest.SkipTest("no yaml backends registered")

    def test_every_non_stub_yaml_template_parses(self) -> None:
        failures = []
        for b in self.backends:
            for p in b.supported_physics():
                for v in p.template_variants:
                    key = (b.name(), p.name, v)
                    try:
                        content = b.generate_input(p.name, v, {})
                    except Exception as e:
                        failures.append(
                            (key, f"generate_input raised: "
                                  f"{type(e).__name__}: {e}"))
                        continue
                    try:
                        docs = list(yaml.safe_load_all(content))
                    except yaml.YAMLError as e:
                        failures.append(
                            (key, f"YAMLError: "
                                  f"{str(e).splitlines()[0][:160]}"))
                        continue
                    # Empty / stub: must be in the known-stub set
                    # OR caller must add it.
                    if not docs or all(d is None for d in docs):
                        if key not in STUB_TEMPLATES:
                            failures.append((
                                key,
                                "parsed as empty YAML (likely a "
                                "one-line stub). If this is "
                                "intentional, add to "
                                "STUB_TEMPLATES; otherwise the "
                                "generator is broken."))
                        # else: known stub, skip without
                        # failure.
                        continue
        if failures:
            lines = "\n".join(
                f"  {be}/{ph}/{vr}: {err}" for (be, ph, vr), err in failures)
            self.fail(
                f"{len(failures)} YAML template issue(s):\n{lines}")

    def test_stub_templates_are_tagged_in_prepare_simulation(self) -> None:
        """Every entry in STUB_TEMPLATES must produce a
        '⚠ STUB' tag in the prepare_simulation surface so the
        LLM is not misled into thinking it's a working template.
        """
        try:
            from mcp.server.fastmcp import FastMCP
            from tools.consolidated import register_consolidated_tools
        except ImportError as exc:
            self.skipTest(f"FastMCP not installed: {exc}")

        mcp = FastMCP("t")
        register_consolidated_tools(mcp)
        tools = mcp._tool_manager._tools  # type: ignore[attr-defined]
        fn = tools["prepare_simulation"].fn

        missing = []
        for (backend_name, physics, _variant) in STUB_TEMPLATES:
            result = fn(solver=backend_name, physics=physics)
            if "⚠ STUB" not in result:
                # Extract the template section for diagnostics.
                idx = result.find("## Template")
                snippet = result[idx:idx + 200] if idx >= 0 else result[:200]
                missing.append((backend_name, physics, snippet))
        if missing:
            lines = "\n".join(
                f"  {be}/{ph}: {snip!r}" for be, ph, snip in missing)
            self.fail(
                f"{len(missing)} stub template(s) not tagged in "
                f"prepare_simulation output:\n{lines}\n\n"
                "The ⚠ STUB heuristic in _stub_template_tag "
                "must match these.")


if __name__ == "__main__":
    unittest.main()
