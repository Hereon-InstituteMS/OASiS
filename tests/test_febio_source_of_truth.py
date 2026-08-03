"""Regression: FEBio catalog type strings vs the installed binary.

FEBio is an XML-input solver, so every "API name" the catalog can
get wrong is a string inside a .feb file: a module name, a
material type, a boundary-condition type, a plot variable. A
wrong string is either a parse error or — for <Module type=...> —
a segmentation fault with no diagnostic at all.

Unlike the Python backends, FEBio can be introspected directly:
febio4 ships an interactive console whose `list` command dumps
every registered factory as "<module>.<type string>
[SUPER_CLASS_ID]", and `list -m` dumps the module names. That
dump IS the source of truth for the installed BUILD — better than
grepping REGISTER_FECORE_CLASS out of a source tree, because a
source tree registers features that a given cmake configuration
never compiles (e.g. `pardiso` exists in the sources but not in a
USE_MKL=OFF build).

This test pins two things:

  (a) OFFLINE — the module list hard-coded in
      backends.febio.generators.deck_structure.FEBIO_MODULES and
      the per-module <analysis> enum table must stay in sync with
      what the catalog's own templates emit. Runs everywhere.

  (b) LIVE (skipped when no febio4 is installed) — that same
      hard-coded module list must equal what the binary reports.
      This is the check that catches version drift: the catalog
      carried `heat` and `biphasic-FSI` module names for months;
      FEBio 4.12 registers neither, and feeding either to
      <Module type=...> crashes the process.

Established 2026-08-03 against FEBio 4.12.0.86045466d.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


def _find_febio() -> Path | None:
    env = os.environ.get("FEBIO_BINARY")
    if env and Path(env).is_file():
        return Path(env)
    for c in (Path.home() / "FEBio" / "bin" / "febio4",
              Path.home() / "FEBioStudio" / "bin" / "febio4",
              Path("/opt/febio/bin/febio4"),
              Path("/usr/local/bin/febio4")):
        if c.is_file():
            return c
    w = shutil.which("febio4") or shutil.which("febio")
    return Path(w) if w else None


def _live_modules(binary: Path) -> set[str]:
    """Ask the binary for its registered module names."""
    p = subprocess.run([str(binary), "-nosplash"],
                       input="list -m\nquit\n",
                       capture_output=True, text=True, timeout=60)
    out = (p.stdout or "") + (p.stderr or "")
    # Lines look like "  1: solid" — the numbering restarts at 1
    # and the list is terminated by a blank line.
    names = set()
    for line in out.splitlines():
        m = re.match(r"^\s*(?:febio>)?\s*(\d+):\s*(\S.*?)\s*$", line)
        if m:
            names.add(m.group(2))
    return names


class TestFebioModuleCatalogOffline(unittest.TestCase):
    """No binary required."""

    def test_no_template_emits_an_unregistered_module(self) -> None:
        """GUARD: every shipped template must name a module FEBio
        actually registers.

        This assertion used to run the other way round. It pinned
        `offenders == {"heat_3d_bar": "heat",
        "biphasic_fsi_3d_block": "biphasic-FSI"}`, i.e. the suite
        REQUIRED those two templates to keep segfaulting the solver
        and went red if either was repaired. That is a test that
        certifies a defect. Both templates were repaired on
        2026-08-03 — heat_3d_bar now emits `thermo-fluid` (FEBio
        4.12 has no `heat` module and no solid-conduction solver;
        holding the fluid at rest reduces the thermo-fluid energy
        equation to Fourier conduction) — so the assertion is now
        the guard it should always have been: the offender set must
        be EMPTY.

        An unregistered <Module type> is not diagnosed. FEBio dies
        with SIGSEGV, no ERROR box, no .log file, and stdout frozen
        mid-line at `Reading file <name>.feb ...`. So a template
        that trips this test is not merely wrong, it is
        undebuggable from the outside."""
        from backends.febio.generators import (
            GENERATORS, FEBIO_MODULES)

        offenders: dict[str, str] = {}
        for key, gen in GENERATORS.items():
            root = ET.fromstring(gen({}))
            mod_el = root.find("Module")
            self.assertIsNotNone(
                mod_el, f"{key}: template has no <Module> tag; "
                        f"FEBio rejects the deck with 'First tag "
                        f"must be the Module section.'")
            mtype = mod_el.attrib.get("type")
            if mtype not in FEBIO_MODULES:
                offenders[key] = mtype
        self.assertEqual(
            offenders, {},
            "a shipped template names a module FEBio 4.12 does not "
            "register. Feeding one to <Module type=...> SEGFAULTS "
            "the solver with no diagnostic at all. Fix the template "
            f"to use one of {list(FEBIO_MODULES)}; do NOT re-pin the "
            "offender.")

    def test_analysis_enum_table_covers_every_module(self) -> None:
        from backends.febio.generators import (
            FEBIO_MODULES, FEBIO_ANALYSIS_VALUES)
        self.assertEqual(
            set(FEBIO_ANALYSIS_VALUES), set(FEBIO_MODULES),
            "every registered module installs its own <analysis> "
            "enum vocabulary (FE*Analysis setEnums), so the table "
            "must cover exactly the registered modules")
        for mod, vals in FEBIO_ANALYSIS_VALUES.items():
            self.assertTrue(
                vals, f"{mod}: empty <analysis> vocabulary")
            for v in vals:
                self.assertEqual(
                    v, v.upper(),
                    f"{mod}: analysis enum words are upper-case in "
                    f"FEBio's setEnums tables ({v!r})")

    def test_templates_declare_analysis_legal_for_their_module(
            self) -> None:
        """A word from another module's vocabulary is a hard parse
        error (`tag "analysis" (line N) : invalid value: X`)."""
        from backends.febio.generators import (
            GENERATORS, FEBIO_ANALYSIS_VALUES)
        bad: list[str] = []
        for key, gen in GENERATORS.items():
            root = ET.fromstring(gen({}))
            mod_el = root.find("Module")
            ctrl = root.find("Control")
            if mod_el is None or ctrl is None:
                continue
            mtype = mod_el.attrib.get("type")
            an = ctrl.find("analysis")
            if an is None or an.text is None:
                continue
            legal = FEBIO_ANALYSIS_VALUES.get(mtype)
            if legal is None:
                # phantom module — covered by the test above
                continue
            word = an.text.strip()
            if word.upper() not in legal and not word.isdigit():
                bad.append(f"{key}: <analysis>{word}</analysis> "
                           f"not legal in module {mtype!r} "
                           f"(legal: {legal})")
        self.assertEqual(bad, [], "; ".join(bad))


class TestFebioModuleCatalogLive(unittest.TestCase):
    """Requires the febio4 binary; skipped otherwise."""

    def setUp(self) -> None:
        self.binary = _find_febio()
        if self.binary is None:
            self.skipTest(
                "febio4 not installed; set FEBIO_BINARY or symlink "
                "~/FEBio/bin/febio4 to run the live source-of-truth "
                "check")

    def test_pinned_module_list_matches_the_binary(self) -> None:
        from backends.febio.generators import FEBIO_MODULES
        live = _live_modules(self.binary)
        self.assertTrue(
            live, "could not read a module list out of febio4 — the "
                  "`list -m` console command changed shape?")
        self.assertEqual(
            set(FEBIO_MODULES), live,
            "deck_structure.FEBIO_MODULES has drifted from the "
            "installed FEBio build. This list is what the catalog "
            "uses to decide whether a <Module type=...> value is "
            "safe; a value outside the live set SEGFAULTS FEBio "
            "with no diagnostic, so the pinned list must be "
            "updated (and any affected template knowledge with "
            f"it). live={sorted(live)}")

    def test_repaired_templates_actually_run(self) -> None:
        """GUARD, the live half of the inversion above.

        `heat_3d_bar` and `biphasic_fsi_3d_block` were the two
        templates the suite used to require to stay broken. Passing
        the XML check is not enough — a module name can be
        registered and the deck still solve nothing. So run them and
        judge on PROGRESS: the process must not be killed by a
        signal, a termination banner must appear, and at least one
        time step must be completed.

        A `N O R M A L   T E R M I N A T I O N` banner alone is
        explicitly NOT accepted as evidence: a deck with no
        <Control> and a deck using <linear_solver type="test"/> both
        produce clean-looking runs that solved nothing (see the
        tier-2 fixtures missing_control_silent_zero_result and
        null_test_linear_solver_reports_success)."""
        import tempfile
        from backends.febio.generators import GENERATORS

        for key in ("heat_3d_bar", "biphasic_fsi_3d_block"):
            with self.subTest(template=key):
                self.assertIn(key, GENERATORS)
                deck = GENERATORS[key]({})
                with tempfile.TemporaryDirectory() as td:
                    feb = Path(td) / f"{key}.feb"
                    feb.write_text(deck)
                    p = subprocess.run(
                        [str(self.binary), "-i", str(feb), "-nosplash"],
                        capture_output=True, text=True, timeout=600,
                        cwd=td)
                    blob = (p.stdout or "") + (p.stderr or "")
                    log = Path(td) / f"{key}.log"
                    if log.is_file():
                        blob += log.read_text(errors="replace")

                self.assertGreaterEqual(
                    p.returncode, 0,
                    f"{key}: febio4 was killed by signal "
                    f"{-p.returncode}. An unregistered <Module type> "
                    f"segfaults with no diagnostic; this template "
                    f"must never do that again.")
                self.assertTrue(
                    "N O R M A L   T E R M I N A T I O N" in blob
                    or "E R R O R   T E R M I N A T I O N" in blob,
                    f"{key}: no termination banner at all — the run "
                    f"died before FEBio could report anything.")
                m = re.search(
                    r"Number of time steps completed\s*\.*\s*:\s*(\d+)",
                    blob)
                self.assertIsNotNone(
                    m, f"{key}: FEBio never printed a completed-step "
                       f"count, so the solver did not start.")
                self.assertGreaterEqual(
                    int(m.group(1)), 1,
                    f"{key}: 0 time steps completed. The deck parsed "
                    f"but solved nothing — that is a broken template, "
                    f"whatever the banner says.")

    def test_backend_reports_available_and_versioned(self) -> None:
        """check_availability() must find the same binary this
        test found, so the catalog and the tests agree on which
        FEBio the knowledge describes."""
        from core.registry import get_backend, load_all_backends
        load_all_backends()
        be = get_backend("febio")
        self.assertIsNotNone(be)
        status, msg = be.check_availability()
        self.assertEqual(
            str(getattr(status, "name", status)), "AVAILABLE",
            f"febio backend reports {status}: {msg}")
        self.assertIn(str(self.binary), msg)


if __name__ == "__main__":
    unittest.main()
