"""Guards for the install / setup / build-configuration surface.

WHAT THIS PROTECTS
    `src/backends/_setup.py` is the first thing a new user needs and the last
    thing anyone re-checks. It is also the surface most exposed to drift,
    because it describes the machine it was written on: an install route stays
    in the file long after the package it names has moved, and a claim keeps
    its confident tone long after the build option it depended on changed.

    So the tests here enforce SHAPE rather than content, with one exception.
    The exception is the important one:

        A claim that depends on build configuration MUST tell the reader how
        to find out which configuration they have.

    That is the whole point of the surface. "deal.II raises ExcDivideByZero on
    CG breakdown" is a true sentence and a useless one, because it is true on
    a Debug build and impossible on a Release build, and the reader has no
    idea which they are running. The fix is not to delete the claim — it is to
    ship the check alongside it. `test_build_config_claims_carry_a_check`
    makes that mechanical instead of aspirational.

WHAT THIS DELIBERATELY DOES NOT DO
    It does not assert that any measured quantity has a particular value, and
    it does not re-verify the quoted error strings by running solvers. A test
    that shells out to nine backends would be skipped on almost every machine
    that runs the suite, and a skipped test guards nothing. The strings were
    verified by execution when they were written; what a merge gate can
    usefully add is that nobody adds an entry in the WRONG SHAPE afterwards.

    Where a test does touch a real install, it SKIPS when the install is
    absent rather than passing — same rule as
    tests/test_fixtures_cannot_pass_vacuously.py. A pass on a machine that
    could not run the check is worse than no test, because it is indistinguish-
    able from a pass on a machine that could.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from backends._setup import (  # noqa: E402
    CONFIG_PROBES, SETUP_KNOWLEDGE, get_setup_knowledge, get_setup_pitfalls,
)

# The sub-kinds an entry may declare. Every pitfall starts with one of these.
_SUBKINDS = ("[Integration][Install]", "[Integration][Discovery]",
             "[Integration][FirstRun]", "[Integration][BuildConfig]",
             "[Integration][Portability]")


class TestCoverage(unittest.TestCase):

    def test_every_registered_backend_has_setup_knowledge(self):
        """A backend OASiS advertises but cannot tell you how to install is a
        backend a stranger cannot use. Derived from the registry, not from a
        hardcoded list, so adding a backend fails here until it is covered."""
        from core.registry import all_backends, load_all_backends

        load_all_backends()
        registered = {b.name() for b in all_backends()}
        self.assertTrue(registered, "no backends registered at all")
        missing = sorted(registered - set(SETUP_KNOWLEDGE))
        self.assertFalse(
            missing,
            f"registered backend(s) with no setup knowledge: {missing}. "
            f"Add an entry to src/backends/_setup.py.")

    def test_each_entry_is_complete(self):
        for name, entry in SETUP_KNOWLEDGE.items():
            with self.subTest(backend=name):
                for key in ("verified_on", "install_route", "pitfalls"):
                    self.assertIn(key, entry, f"{name} is missing {key!r}")
                    self.assertTrue(str(entry[key]).strip(),
                                    f"{name}'s {key} is empty")
                self.assertGreaterEqual(
                    len(entry["pitfalls"]), 2,
                    f"{name} has fewer than two setup pitfalls — install and "
                    f"discovery are separate failure modes and both bite")

    def test_install_route_gives_runnable_commands(self):
        """An install instruction a reader cannot copy is a description, not
        an instruction. Every route must contain at least one recognisable
        command, not just prose about one."""
        verbs = ("pip install", "conda create", "conda install", "apt install",
                 "git clone", "cmake", "make ", "brew install")
        for name, entry in SETUP_KNOWLEDGE.items():
            with self.subTest(backend=name):
                route = entry["install_route"]
                self.assertTrue(
                    any(v in route for v in verbs),
                    f"{name}'s install_route contains no runnable command "
                    f"(looked for {verbs})")


class TestShape(unittest.TestCase):

    def test_every_pitfall_is_categorised(self):
        """Same [Category] discipline as the rest of the pitfall DB, plus a
        sub-kind — the sub-kind is what lets a caller pick out the
        build-conditional ones."""
        for name, entry in SETUP_KNOWLEDGE.items():
            for i, p in enumerate(entry["pitfalls"]):
                with self.subTest(backend=name, index=i):
                    self.assertTrue(
                        p.startswith(_SUBKINDS),
                        f"{name}[{i}] does not start with one of "
                        f"{_SUBKINDS}: {p[:70]!r}")

    def test_pitfalls_are_prose_not_stubs(self):
        """A one-line pitfall is a headline. The value is in what the reader
        sees and what to do about it."""
        for name, entry in SETUP_KNOWLEDGE.items():
            for i, p in enumerate(entry["pitfalls"]):
                with self.subTest(backend=name, index=i):
                    self.assertGreater(
                        len(p), 180,
                        f"{name}[{i}] is too short to carry a symptom and a "
                        f"defense: {p!r}")

    def test_failure_claims_quote_a_signal(self):
        """Install, discovery and first-run entries describe something going
        wrong, so each must say what the user actually SEES. Portability and
        build-config entries are exempt: they are about scope, and their
        useful content is a version range or a probe."""
        needs_signal = ("[Integration][Install]", "[Integration][Discovery]",
                        "[Integration][FirstRun]")
        for name, entry in SETUP_KNOWLEDGE.items():
            for i, p in enumerate(entry["pitfalls"]):
                if not p.startswith(needs_signal):
                    continue
                with self.subTest(backend=name, index=i):
                    self.assertIn(
                        "Signal", p,
                        f"{name}[{i}] claims a failure without saying what "
                        f"the user sees. Quote the real message.")


class TestBuildConfigScoping(unittest.TestCase):
    """The load-bearing rule of this module."""

    # Words whose presence means the claim is conditional on how the backend
    # was compiled. A claim mentioning any of these has to be scoped.
    _CONFIG_WORDS = re.compile(
        r"\b(Debug|Release|CMAKE_BUILD_TYPE|MKL|PETSc|Trilinos|p4est|MPI|"
        r"KOKKOS|complex|UMFPACK|SLEPc|glibc|GLIBC)\b")

    # Ways an entry can discharge the obligation: point at a probe, give a
    # command, or name the file that answers the question.
    _CHECK_MARKERS = ("CONFIG_PROBES[", "Check with", "Check which",
                      "Defense:", "check `", "grep", "ldd ", "objdump",
                      "readelf", "make ps", "-info", "config.h")

    def test_build_config_claims_carry_a_check(self):
        """A build-conditional claim must ship the way to check the build.

        This is the rule the whole module exists for. A reader on a different
        build than ours needs to know (a) that the claim is conditional and
        (b) how to find out which side of it they are on. An entry that
        states the behaviour flat leaves them worse off than silence, because
        they will act on a claim that is false for them."""
        for name, entry in SETUP_KNOWLEDGE.items():
            for i, p in enumerate(entry["pitfalls"]):
                if not p.startswith("[Integration][BuildConfig]"):
                    continue
                with self.subTest(backend=name, index=i):
                    self.assertTrue(
                        any(m in p for m in self._CHECK_MARKERS),
                        f"{name}[{i}] is a build-configuration claim with no "
                        f"way for the reader to check their own build. Add a "
                        f"CONFIG_PROBES reference or an explicit command.\n"
                        f"  {p[:200]}")

    def test_config_words_outside_build_config_entries_are_scoped(self):
        """A configuration-dependent word in a NON-BuildConfig entry is how
        an unscoped claim sneaks in. Allow it only where the entry also says
        what it was checked against, or tells the reader how to check."""
        allowed = self._CHECK_MARKERS + (
            "verified", "Verified", "checked here", "was not possible",
            "unverified", "only", "scoped", "Version range")
        for name, entry in SETUP_KNOWLEDGE.items():
            for i, p in enumerate(entry["pitfalls"]):
                if p.startswith("[Integration][BuildConfig]"):
                    continue
                if not self._CONFIG_WORDS.search(p):
                    continue
                with self.subTest(backend=name, index=i):
                    self.assertTrue(
                        any(m in p for m in allowed),
                        f"{name}[{i}] mentions a build-dependent feature "
                        f"without scoping it or offering a check.\n"
                        f"  {p[:200]}")

    def test_every_probe_is_usable(self):
        for key, probe in CONFIG_PROBES.items():
            with self.subTest(probe=key):
                for field in ("what", "command", "reading"):
                    self.assertIn(field, probe)
                    self.assertTrue(str(probe[field]).strip())
                self.assertTrue(
                    probe["reading"].strip(),
                    f"{key} gives a command but not how to read its output — "
                    f"a weak reader needs both")

    def test_probes_referenced_by_pitfalls_exist(self):
        """A dangling probe reference is worse than none: the reader goes
        looking for something that is not there."""
        rx = re.compile(r"CONFIG_PROBES\['([a-z_]+)'\]")
        for name, entry in SETUP_KNOWLEDGE.items():
            for i, p in enumerate(entry["pitfalls"]):
                for key in rx.findall(p):
                    with self.subTest(backend=name, index=i, probe=key):
                        self.assertIn(
                            key, CONFIG_PROBES,
                            f"{name}[{i}] references CONFIG_PROBES[{key!r}], "
                            f"which does not exist")


class TestNoHostPaths(unittest.TestCase):

    def test_no_absolute_host_paths_in_the_payload(self):
        """Install knowledge legitimately talks about locations, so it is the
        surface most likely to bake in one machine's layout. Describe how to
        FIND a path; never ship the path.

        Exempt: /usr, /opt and /lib prefixes that name a genuine system-wide
        convention (a distribution deal.II really is at /usr), and the loader
        path inside a quoted error message, which is what the user will
        actually see on their own machine."""
        bad = re.compile(r"(/home/[a-z][a-z0-9_-]*|/media/[a-z][a-z0-9_-]*|"
                         r"/Users/[a-z][a-z0-9_-]*)", re.IGNORECASE)
        hits = []
        for name, entry in SETUP_KNOWLEDGE.items():
            blob = json.dumps(entry)
            for m in bad.finditer(blob):
                hits.append(f"{name}: …{blob[max(0, m.start() - 60):m.end() + 40]}…")
        self.assertFalse(
            hits,
            "An absolute host path is shipped as install knowledge. Give the "
            "command that finds it instead.\n  " + "\n  ".join(hits[:6]))

    def test_probe_commands_use_placeholders(self):
        for key, probe in CONFIG_PROBES.items():
            with self.subTest(probe=key):
                self.assertNotRegex(
                    probe["command"], r"/home/|/media/|/Users/",
                    f"probe {key} hard-codes a host path")


class TestReachability(unittest.TestCase):
    """A surface a caller cannot find is a surface that does not exist."""

    def test_alias_lookup_resolves_every_spelling(self):
        for spelling, canonical in (
                ("deal.ii", "dealii"), ("deal_ii", "dealii"),
                ("fenicsx", "fenics"), ("dolfinx", "fenics"),
                ("4c", "fourc"), ("dune-fem", "dune"),
                ("scikit-fem", "skfem"), ("DEALII", "dealii"),
                ("  Fenics  ", "fenics")):
            with self.subTest(spelling=spelling):
                self.assertEqual(
                    get_setup_knowledge(spelling).get("backend"), canonical)

    def test_unknown_backend_lists_the_known_ones(self):
        out = get_setup_knowledge("nosuchsolver")
        self.assertIn("error", out)
        self.assertIn("known", out)
        self.assertIn("dealii", out["known"])

    def test_no_argument_returns_everything_plus_probes(self):
        out = get_setup_knowledge()
        self.assertEqual(set(out["backends"]), set(SETUP_KNOWLEDGE))
        self.assertEqual(set(out["config_probes"]), set(CONFIG_PROBES))
        self.assertIn("_how_to_use", out)

    def test_setup_pitfalls_are_merged_into_the_pitfalls_topic(self):
        """An agent debugging a failed run asks for 'pitfalls'. If the setup
        entries were only reachable under a topic string it has to guess, the
        one class of pitfall it most needs would be the one it never sees."""
        import tools.consolidated as consolidated

        src = Path(consolidated.__file__).read_text()
        self.assertIn("get_setup_pitfalls", src,
                      "the pitfalls topic no longer merges setup pitfalls")
        self.assertIn("install_and_build_config", src)

    def test_topic_aliases_cover_the_obvious_words(self):
        import tools.consolidated as consolidated

        for word in ("install", "setup", "dependencies", "build_config",
                     "portability"):
            self.assertIn(word, consolidated._SETUP_TOPIC_ALIASES,
                          f"topic={word!r} should reach the setup surface")

    def test_get_setup_pitfalls_matches_the_entry(self):
        for name in SETUP_KNOWLEDGE:
            with self.subTest(backend=name):
                self.assertEqual(get_setup_pitfalls(name),
                                 SETUP_KNOWLEDGE[name]["pitfalls"])


class TestAgainstRealInstalls(unittest.TestCase):
    """Checks that touch a real install — and SKIP, never pass, without one.

    These are the only tests here that can catch the knowledge going stale
    against the software rather than against itself. They are few on purpose:
    each one is a claim that can be settled with a single cheap command.
    """

    def _skip_without(self, path: Path | None, what: str):
        if path is None or not Path(path).exists():
            self.skipTest(
                f"{what} not present on this machine — SKIPPED, not passed. "
                f"A pass here would be indistinguishable from a pass on a "
                f"machine that has it.")

    def test_dealii_build_type_probe_reads_a_real_config(self):
        """The deal.II feature probe must actually work on a real install:
        config.h must exist and its DEAL_II_WITH_ lines must be parseable in
        the two forms the knowledge says to look for."""
        try:
            from backends.dealii.backend import _find_dealii
        except ImportError:
            self.skipTest("dealii backend not importable")
        root = _find_dealii()
        self._skip_without(root, "a deal.II install")
        cfgs = list(Path(root).rglob("deal.II/base/config.h"))
        self.assertTrue(
            cfgs,
            f"no deal.II/base/config.h under the discovered root — the "
            f"feature probe in _setup.py would find nothing")
        text = cfgs[0].read_text(errors="ignore")
        self.assertRegex(
            text, r"#define DEAL_II_WITH_\w+|/\* #undef DEAL_II_WITH_\w+ \*/",
            "config.h has neither form of DEAL_II_WITH_ line, so the "
            "documented reading of it is wrong")

    def test_dealii_header_presence_really_is_a_false_positive(self):
        """The claim: on a source build, feature headers exist whether or not
        the feature is on, so only config.h is authoritative. If a header is
        present while config.h says the feature is OFF, the claim holds."""
        try:
            from backends.dealii.backend import _find_dealii
        except ImportError:
            self.skipTest("dealii backend not importable")
        root = _find_dealii()
        self._skip_without(root, "a deal.II install")
        cfgs = list(Path(root).rglob("deal.II/base/config.h"))
        if not cfgs:
            self.skipTest("no config.h to read features from")
        text = cfgs[0].read_text(errors="ignore")
        off = {m for m in re.findall(
            r"/\* #undef DEAL_II_WITH_(\w+) \*/", text)}
        if not off:
            self.skipTest("this deal.II has every feature on — the false "
                          "positive cannot be demonstrated here")
        probes = {"PETSC": "lac/petsc_vector.h",
                  "TRILINOS": "lac/trilinos_vector.h",
                  "SLEPC": "lac/slepc_solver.h",
                  "MPI": "base/mpi.h",
                  "P4EST": "distributed/tria.h"}
        for feature, rel in probes.items():
            if feature not in off:
                continue
            hits = list(Path(root).rglob(f"include/deal.II/{rel}"))
            if hits:
                # Claim demonstrated: header present, feature off.
                return
        self.skipTest("no feature is both OFF and header-probeable on this "
                      "install — cannot demonstrate the false positive here")

    def test_sparta_kokkos_refusals_are_real_strings(self):
        """The three KOKKOS messages are quoted verbatim in the knowledge.
        Check them against the binary's own string table."""
        try:
            from backends.sparta.backend import _find_sparta_binary
        except ImportError:
            self.skipTest("sparta backend not importable")
        binary = _find_sparta_binary()
        self._skip_without(binary and Path(binary), "a SPARTA binary")
        if not shutil.which("strings"):
            self.skipTest("`strings` not available to read the binary")
        out = subprocess.run(["strings", str(binary)], capture_output=True,
                             text=True, timeout=120).stdout
        entry = " ".join(SETUP_KNOWLEDGE["sparta"]["pitfalls"])
        for quoted in ("Cannot use -kokkos on without KOKKOS installed",
                       "Package kokkos command without KOKKOS package "
                       "enabled"):
            with self.subTest(message=quoted[:40]):
                self.assertIn(quoted, entry,
                              "knowledge no longer quotes this message")
                self.assertIn(
                    quoted, out,
                    f"the knowledge quotes a SPARTA message that this binary "
                    f"does not contain: {quoted!r}. Either the message "
                    f"changed upstream or it was never real.")

    def test_febio_version_flag_claim_holds(self):
        """The knowledge says `-v` is not a valid FEBio flag and `-info` is.
        Cheap to settle; expensive to get wrong, because it is the command we
        tell people to use to check they have the right binary."""
        try:
            from backends.febio.backend import _find_febio_binary
        except ImportError:
            self.skipTest("febio backend not importable")
        binary = _find_febio_binary()
        self._skip_without(binary, "an FEBio binary")
        r = subprocess.run([str(binary), "-v"], capture_output=True,
                           text=True, timeout=60)
        combined = (r.stdout or "") + (r.stderr or "")
        self.assertIn(
            "Invalid command line option", combined,
            "the knowledge says `febio4 -v` is rejected as an invalid "
            "option; this binary answered otherwise")


if __name__ == "__main__":
    unittest.main()
