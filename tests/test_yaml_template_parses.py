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


# Templates that are intentionally reference stubs (no runnable
# inline template) at the time of writing. Each entry is
# (backend, physics, variant). Drop from here once the stub is
# replaced by a real template; add here when a new physics ships
# as a stub (and confirm prepare_simulation tags it ⚠ STUB).
#
# 2026-06-12 (task #29): membrane / shell / thermo_2d /
# thermo_3d / mixture / constraint / cardiovascular0d were
# REMOVED — they were one-line comment stubs but are now routed
# to self-contained inline-mesh inputs that run rc=0 on the 4C
# binary (see tests/test_fourc_inline_*). The deep-multiphysics
# rows below were ADDED — they return honest reference stubs via
# FourcBackend._reference_stub_template because they genuinely
# need a case-specific mesh (often two meshes), a second input
# file, patient-derived 1-D topology, an explicit particle cloud,
# a wall-resolved periodic mesh. See tests/test_fourc_reference_stubs.py,
# which now pins the opposite: those rows are executed decks.
STUB_TEMPLATES = {
    # 2026-08-07: every fourc row that used to sit here has been replaced by a
    # deck that was executed on the installed binary (see
    # src/backends/fourc/decks.py and tests/test_fourc_reference_stubs.py), so
    # nothing here is a fourc row any more. The set is empty: no backend in
    # this tree currently ships a stub template. Kept rather than deleted, so
    # the next one that does has an obvious place to be declared and gets
    # tagged instead of shipping silently.
    #
    # One claim removed with them: the note above used to say XFEM was blocked
    # because "xfem cut needs Qhull" and this build lacks it. It does not lack
    # it — the build has FOUR_C_WITH_QHULL:BOOL=ON, lib4C.so links
    # libqhull.so.6, and the XFEM decks run in both Tessellation and
    # DirectDivergence cut modes.
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

    def test_stub_templates_are_tagged(self) -> None:
        """Every entry in STUB_TEMPLATES must be recognised by the ⚠ STUB
        heuristic, so the LLM is not told a placeholder is a working template.

        Fixed 2026-08-07: this used to call prepare_simulation(solver, physics)
        with no variant and look for the tag anywhere in the reply. For a
        physics with several variants prepare_simulation picks ONE — for
        fourc/porous_media it picks the runnable single_phase_3d — so the test
        was inspecting a template other than the one it named, and passed or
        failed for reasons unrelated to the stub. It now feeds the heuristic
        the exact (physics, variant) content it is making a claim about.
        """
        from core.registry import get_backend
        from tools.consolidated import _stub_template_tag

        missing = []
        for (backend_name, physics, variant) in sorted(STUB_TEMPLATES):
            b = get_backend(backend_name)
            if b is None:
                continue
            content = b.generate_input(physics, variant, {})
            if "⚠ STUB" not in _stub_template_tag(content,
                                                  b.input_format().value):
                missing.append((backend_name, physics, variant,
                                content[:160]))
        if missing:
            lines = "\n".join(
                f"  {be}/{ph}/{vr}: {snip!r}" for be, ph, vr, snip in missing)
            self.fail(
                f"{len(missing)} stub template(s) not recognised by "
                f"_stub_template_tag:\n{lines}")

    def test_stub_variants_are_tagged_when_prepare_simulation_picks_them(
            self) -> None:
        """And when a stub variant IS the one prepare_simulation selects, the
        tag must reach the LLM-visible surface."""
        try:
            from mcp.server.fastmcp import FastMCP
            from tools.consolidated import register_consolidated_tools
        except ImportError as exc:
            self.skipTest(f"FastMCP not installed: {exc}")
        from core.registry import get_backend

        mcp = FastMCP("t")
        register_consolidated_tools(mcp)
        fn = mcp._tool_manager._tools["prepare_simulation"].fn  # type: ignore

        missing = []
        for (backend_name, physics, variant) in sorted(STUB_TEMPLATES):
            b = get_backend(backend_name)
            if b is None:
                continue
            # Only physics whose FIRST/selected variant is the stub can be
            # checked through this surface; the rest are covered above.
            row = next((p for p in b.supported_physics()
                        if p.name == physics), None)
            if row is None or not row.template_variants:
                continue
            if row.template_variants[0] != variant:
                continue
            if "⚠ STUB" not in fn(solver=backend_name, physics=physics):
                missing.append((backend_name, physics, variant))
        if missing:
            self.fail(f"stub not tagged in prepare_simulation: {missing}")


if __name__ == "__main__":
    unittest.main()
