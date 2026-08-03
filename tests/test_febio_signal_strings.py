"""Regression: every FEBio `Signal:` string must be one FEBio can emit.

FEBio is a C++ binary, so every diagnostic it can ever print is a
string literal inside febio4 or one of its shared libraries. That makes
"can this message actually appear?" a decidable question, unlike the
Python backends where a message can be built at runtime.

It needed to be decidable, because it was not being decided. Twenty-five
`Signal:` strings in this catalog named messages the binary cannot
produce. Six of them were `NOX` / `DIVERGED` / `DIVERGED_LINE_SEARCH` /
`DIVERGED_FNORM_NAN` — Trilinos NOX status names — and one was
`KSPSolve`, a PETSc entry point. FEBio links neither library. Those
strings were imported from other FEM codes by a model writing plausible
text instead of reading output, and a consumer that greps for them waits
forever for a message that will never come.

Two layers, deliberately:

  * OFFLINE (runs everywhere) — a deny-list of foreign-toolkit tokens.
    No FEBio message contains them, so their appearance anywhere in the
    FEBio knowledge is proof of imported text. This is the layer that
    would have caught the original 25.

  * LIVE (needs febio4) — the general check. Every backticked literal
    inside a `Signal:` clause is split into static fragments, and each
    fragment must occur in the strings of febio4 plus every .so beside
    it. FEBio messages are printf templates (`invalid value for
    attribute "%s"`), so whole-string matching is impossible and
    fragment matching is the correct granularity.

CONVENTION THIS TEST ENFORCES, and which every FEBio pitfall must
follow: **inside a `Signal:` clause, backticks mean "text FEBio
printed".** Nothing else may be backticked there. Deck text the author
wrote, type names FEBio rejected, file names, shell commands — all of
those go in plain double quotes, in the WRONG:/RIGHT: part of the entry,
or after the Signal clause closes. Without that rule the check cannot
tell a real diagnostic from a string being quoted as an example of what
not to write, and a phantom hides behind the ambiguity. Text FEBio
echoes back out of the deck (through a %s, and therefore always inside
double quotes in the message) is exempt — it is a runtime value, not a
literal.
"""
from __future__ import annotations

import os
import re
import shutil
import string
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

_PRINTABLE = set(string.printable[:-5].encode())

# Everywhere a runtime value can sit inside a FEBio message. Used to cut
# a quoted Signal into the static runs that must exist in the binary.
_PLACEHOLDER = re.compile(r'"[^"]*"|<[^>]*>|\.\.\.|\b[A-Z]\b|%\w')


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


def _strings(path: Path, minlen: int = 6):
    """Pure-python `strings -n <minlen>`; no external tool needed."""
    out, cur = [], bytearray()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            for b in chunk:
                if b in _PRINTABLE:
                    cur.append(b)
                else:
                    if len(cur) >= minlen:
                        out.append(cur.decode("ascii", "replace"))
                    cur.clear()
    if len(cur) >= minlen:
        out.append(cur.decode("ascii", "replace"))
    return out


def _corpus(binary: Path) -> str:
    """febio4 plus every shared library that ships with it.

    FEBio's messages are spread across libfecore / libfebiomech /
    libfebiomix / libfebiofluid / ... — checking only the executable
    would report almost everything as a phantom.
    """
    files = [binary]
    real = binary.resolve()
    for d in {real.parent, real.parent.parent / "lib", binary.parent}:
        if d.is_dir():
            files += sorted(d.glob("*.so")) + sorted(d.glob("*.so.*"))
    seen, parts = set(), []
    for f in files:
        r = f.resolve()
        if r in seen or not r.is_file():
            continue
        seen.add(r)
        parts.extend(_strings(r))
    return "\n".join(parts)


def _febio_pitfalls():
    """Every pitfall string anywhere in the FEBio knowledge.

    Recurses: pitfall lists live both at the topic level and inside
    nested reference blocks (e.g. _general/adaptive_mesh_refinement),
    and a check that only looked one level down would leave the nested
    ones unguarded.
    """
    from backends.febio import generators as G

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "pitfalls" and isinstance(v, (list, tuple)):
                    for i, p in enumerate(v):
                        if isinstance(p, str):
                            yield path, i, p
                else:
                    yield from walk(v, f"{path}/{k}" if path else str(k))

    for topic, entry in sorted(G.KNOWLEDGE.items()):
        yield from walk(entry, topic)


def _signal_literals():
    """(topic, index, literal) for every backticked run of text that
    sits inside a `Signal:` clause."""
    for topic, i, p in _febio_pitfalls():
        for m in re.finditer(r"Signal:(.*?)(?=\(Executed|\(Live-verified"
                             r"|\(Verified|\(Source walk|$)", p, re.S):
            for lit in re.findall(r"`([^`]+)`", m.group(1)):
                yield topic, i, lit


# Tokens that belong to other finite-element toolkits. FEBio links none
# of them, so any occurrence is text imported from another code rather
# than read out of FEBio.
FOREIGN_TOKENS = (
    "NOX",                  # Trilinos nonlinear solver
    "DIVERGED_",            # Trilinos/PETSc status enums
    "CONVERGED_",
    "KSPSolve", "KSPSetUp", "SNESSolve",   # PETSc
    "PETSC", "PetscError",
    "Trilinos", "Epetra", "Tpetra", "Belos", "Amesos", "AztecOO",
    "dolfinx", "DOLFIN", "UFLException",   # FEniCS
    "deal.II ExcMessage", "ExcDimensionMismatch",   # deal.II
    "KratosMultiphysics",   # Kratos
    "ngsolve.", "NgException",              # NGSolve
    "write_vtu",            # not a FEBio concept: FEBio writes .xplt
)


class TestNoForeignSolverTokens(unittest.TestCase):
    """Offline. Runs on any host, with or without FEBio installed."""

    def test_no_foreign_toolkit_token_in_any_febio_pitfall(self) -> None:
        hits = []
        for topic, i, p in _febio_pitfalls():
            for tok in FOREIGN_TOKENS:
                if tok in p:
                    hits.append(f"{topic}#{i}: {tok!r}")
        self.assertEqual(
            hits, [],
            "FEBio knowledge names a symbol from a different FEM "
            "toolkit. FEBio links neither PETSc nor Trilinos, so a "
            "message containing these can never appear and a consumer "
            "grepping for it waits forever. Read the real diagnostic "
            "off a failing run instead of writing a plausible one.\n  "
            + "\n  ".join(hits))

    def test_every_pitfall_has_a_signal_clause(self) -> None:
        missing = [f"{t}#{i}" for t, i, p in _febio_pitfalls()
                   if "Signal:" not in p]
        self.assertEqual(
            missing, [],
            "every FEBio pitfall must state the observable symptom "
            f"under a 'Signal:' clause: {missing}")

    def test_no_falsified_annotation_left_inline(self) -> None:
        """A phantom must be DELETED, not annotated.

        An earlier audit corrected phantom Signals by leaving the
        phantom string in place and appending a bracketed
        `[FALSIFIED ...]` note. The consumer of this catalog is a small
        model that does not read the bracket; it reads the phantom and
        greps for it. Corrections must replace the text, not argue
        with it.
        """
        bad = []
        for topic, i, p in _febio_pitfalls():
            for marker in ("[FALSIFIED", "[CORRECTED"):
                if marker in p:
                    bad.append(f"{topic}#{i}: {marker}")
        self.assertEqual(
            bad, [],
            "an inline [FALSIFIED]/[CORRECTED] bracket is still "
            "sitting next to the text it falsifies. Delete the wrong "
            "string and state only what the binary really emits; the "
            "history belongs in the commit message.\n  "
            + "\n  ".join(bad))


class TestSignalStringsExistInBinary(unittest.TestCase):
    """Live. Needs febio4; skipped otherwise."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.binary = _find_febio()
        cls.blob = _corpus(cls.binary) if cls.binary else ""

    def setUp(self) -> None:
        if self.binary is None:
            self.skipTest(
                "febio4 not installed; set FEBIO_BINARY or symlink "
                "~/FEBio/bin/febio4 to run the phantom-Signal check")

    def test_corpus_is_plausible(self) -> None:
        """Guard the guard: if the corpus came back tiny we would
        report every Signal as a phantom."""
        self.assertGreater(
            len(self.blob), 200_000,
            "the FEBio string corpus is implausibly small — the "
            "shared libraries were probably not found, which would "
            "make every Signal look like a phantom")
        for known in ('invalid value for attribute "%s"',
                      "N O R M A L   T E R M I N A T I O N",
                      "failed to converge at time"):
            self.assertIn(
                known, self.blob,
                f"corpus is missing a message FEBio certainly emits "
                f"({known!r}) — the corpus build is broken, not the "
                f"catalog")

    def test_every_signal_fragment_occurs_in_the_binary(self) -> None:
        lower = self.blob.lower()
        phantoms = []
        for topic, i, lit in _signal_literals():
            s = lit.strip()
            # Shell commands and XML the USER writes are not FEBio
            # output; only solver messages are checked here.
            if re.match(r"^(printf|grep|strings|febio4|python|\$)", s):
                continue
            if s.startswith("<") or s.startswith("--"):
                continue
            # FEBio messages are printf templates, so only the STATIC
            # runs between substitutions survive into the binary.
            # Replace every place a runtime value can sit with a
            # separator, so the static runs on either side are checked
            # independently and never joined across a hole:
            #   "..."   a %s slot, always double-quoted in FEBio output
            #   <...>   our own placeholder notation
            #   ...     an elision
            #   N X D   a bare capital standing for a number
            #   %d %s   a template written out literally
            # Without the separator, `Component "" needs to have
            # property "solver" defined` would be searched for as one
            # run and reported as a phantom although both halves are
            # real. With it, the check needs no per-word escape hatch —
            # and that hatch was the hole: every long word of "element
            # inversion detected in porous domain" occurs somewhere in
            # the corpus, so an invented sentence passed.
            s = _PLACEHOLDER.sub("|", s)
            for frag in re.split(r"[^A-Za-z_ ]+", s):
                frag = frag.strip()
                if len(re.sub(r"[^A-Za-z]", "", frag)) < 6:
                    continue
                if frag.lower() in lower:
                    continue
                phantoms.append(f"{topic}#{i}: {frag!r} (in {lit!r})")
        self.assertEqual(
            phantoms, [],
            "a Signal: names text that occurs nowhere in febio4 or "
            "its shared libraries, so FEBio cannot print it and the "
            "Signal can never match. Capture the real message from a "
            "failing run.\n  " + "\n  ".join(phantoms))


if __name__ == "__main__":
    unittest.main()
