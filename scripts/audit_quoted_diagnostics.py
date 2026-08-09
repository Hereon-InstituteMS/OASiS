#!/usr/bin/env python3
"""Does the software actually print the error message we quote?

THE DEFECT THIS EXISTS FOR
--------------------------
Every verification agent that has executed knowledge rather than reading it has
come back with the same finding, and it is always the largest category:

    FEBio        25 invented error messages, 11 non-existent material names
    4C beams/contact/constraint   ~10 quoted diagnostics absent from the source
    NGSolve      "Newton did not converge after N iterations" — nowhere in
                 ngsolve/netgen; the real text is "Warning: Newton might not
                 converge! Error = "
    skfem        meshio "Cannot write complex array" — never emitted; the real
                 failure is KeyError: dtype('complex128')
    Kratos       FREESTREAM_VELOCITY / MACH_INFINITY — neither name exists
    scipy        "RuntimeError: matrix not positive definite" from cg — scipy
                 raises nothing at all; it returns info=1000 and a garbage
                 vector

Why this class is worse than a merely wrong entry: an agent uses the quoted
string to BUILD ITS GUARD. Told that cg raises, it writes an exception handler
and treats the absence of an exception as success — so the guard passes and the
garbage flows through. A fabricated diagnostic does not just fail to help; it
actively manufactures false confidence, which is the exact failure mode this
whole project exists to prevent.

The existing `audit_phantom_apis.py` checks API ATTRIBUTES via hasattr. It says
nothing about quoted message text, which is where the damage has actually been.

HOW THIS CHECKS
---------------
For each quoted fragment inside a `Signal:` clause, look for it in the software
that is supposed to emit it: the C++ source tree for compiled backends, the
installed package source for Python ones, and — importantly — their dependency
trees, because plenty of genuine messages come from Trilinos, PETSc, UMFPACK,
meshio or scipy rather than from the backend itself.

BUILT TO AVOID FALSE ACCUSATIONS, deliberately and at the cost of sensitivity.
Two earlier guards in this repo cried wolf: a delegation guard flagged `sorted`,
`dedent` and `format` as suspicious (~20 false accusations), and a disclosure
guard accused five honest 4C entries because it required a phrase verbatim that
the entries expressed slightly differently. A checker that is wrong often gets
ignored, and an ignored gate is worse than none — so this one:

  * searches dependency trees, not just the backend;
  * matches on the LONGEST STATIC FRAGMENT, since real messages interpolate at
    runtime ("DPoint 1 not in range [0:0[" is literal only in parts);
  * requires a fragment to be substantial before judging it at all;
  * reports UNKNOWN rather than ABSENT when it cannot see the source, because
    "I could not check" and "it is not there" are different claims and only one
    of them is an accusation.

Absence of a string from the source is strong evidence but not proof: messages
can be assembled from pieces, or come from a binary we cannot read. So the exit
status distinguishes "definitely absent" from "could not verify", and the report
says which is which.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Where each backend's implementation actually lives. A missing entry means the
# audit reports UNKNOWN for that backend rather than guessing.
SOURCE_HINTS: dict[str, list[str]] = {
    # 4C: the source tree is READ ONLY — we only ever grep it.
    #
    # `apps/` was missing and it is where the executable's own banners live.
    # 4C_global_full_main.cpp holds printf("processor %d finished normally\n"),
    # so the exit line every 4C fixture greps for was invisible to this audit
    # and two entries quoting it were reported as fabricated diagnostics. 16
    # files; the entry point of the shipped binary is not an optional part of
    # 4C's source. `unittests/` is deliberately NOT added: a string that exists
    # only in a test's expected output is not evidence the solver emits it.
    "fourc": ["/home/alexander/4C/src", "/home/alexander/4C/apps",
              "/home/alexander/4C/tests"],
    # FEBio is installed as a BINARY with no source tree — `/opt/febio` and
    # `/usr/local/febio` are both absent on this host, so the audit reported
    # UNKNOWN for every FEBio claim. The real install is below, and a binary is
    # a perfectly good corpus for checking whether a tag or a message exists:
    # FEBio's XML element names are compiled into it as literal strings. This
    # only works with `grep -a`; without it grep skips the file as binary and
    # answers "not found" for everything in it.
    # NOT `/home/alexander/FEBio` — that directory holds only `bin/febio4`,
    # which is a SYMLINK into the tree below, and `grep -r` does not follow
    # symlinks. Pointed there, the corpus was effectively empty: a positive
    # control for the literal string "febio" returned zero files, and the audit
    # duly reported 23 of 23 FEBio keys as unresolved. Against the real tree,
    # 13 of those 23 resolve immediately. A corpus that answers "no" to
    # everything is not evidence of fabrication, it is a broken instrument —
    # which is why every backend here needs a positive control before its
    # numbers are quoted.
    "febio": ["/home/alexander/Schreibtisch/febio-src",
              "/opt/febio", "/usr/local/febio"],
    # deal.II ships as C++ headers and sources, not a Python package, so the
    # module probe could never find it. 7125 headers and sources here.
    #
    # There are TWO deal.II installs on this host and they disagree: the system
    # /usr/include/deal.II is 9.1.1 with MPI/P4EST/PETSC/TRILINOS/SLEPC ON,
    # while the build the C++ fixtures actually compile against has all five
    # undefined. That difference matters for capability claims and a fixture was
    # found reading the wrong one. It does NOT matter here — this audit asks
    # only whether a symbol or message exists anywhere in deal.II — but the
    # source tree is listed first so answers come from the real thing.
    "dealii": ["/home/alexander/dealii", "/usr/include/deal.II"],
    # SPARTA is a C++ code with its own input-command corpus in doc/ and
    # examples/; both matter, since a command can be documented and exercised
    # without appearing as a literal in the source.
    "sparta": ["/home/alexander/Schreibtisch/sparta"],
    # Kratos: prefer the source-built 28-APPLICATION install over the repo
    # venv's wheel, which ships only core plus three applications.
    #
    # Against the 3-app wheel, 66 Kratos keys did not resolve and the audit
    # correctly refused to call them invented — a key from DEMApplication or
    # PoromechanicsApplication is invisible when those applications are not
    # installed, however real it is. Against this build (70 compiled libraries)
    # BIOT_COEFFICIENT, DEM_SURFACE_LOAD and COMPUTE_FEM_RESULTS_OPTION all
    # resolve immediately. They were real the whole time.
    #
    # The restraint paid twice over: PARTICLE_FRICTION, served as a required DEM
    # material key AND written into generated decks, is absent from the full
    # 28-app build too. Now that the corpus can answer, that absence means
    # something. The real keys are STATIC_FRICTION / DYNAMIC_FRICTION.
    "kratos": ["/mnt/kratos-tier2/kv/lib/python3.12/site-packages/"
               "KratosMultiphysics"],
}

# Python backends, PRIMARY MODULE FIRST. The first entry must be importable or
# the audit reports UNKNOWN — the secondary modules are not a substitute.
#
# Learned from FEniCSx: dolfinx is not in the repo venv (it lives in a conda
# env), but ufl, basix and petsc4py are. With a flat list, three of four
# resolving looked like enough source to judge, and every dolfinx message was
# reported absent — having been searched for in ufl. The primary module is the
# one that emits the messages; without it there is nothing to check against.
PY_MODULES: dict[str, list[str]] = {
    # slepc4py and ffcx are listed because FEniCSx entries quote their text:
    # the eigenvalue pitfalls quote SLEPc's EPS behaviour, and the form
    # compiler is where several "this form cannot be compiled" messages live.
    "fenics": ["dolfinx", "ufl", "basix", "petsc4py", "slepc4py", "ffcx"],
    "ngsolve": ["ngsolve", "netgen"],
    "skfem": ["skfem"],
    "kratos": ["KratosMultiphysics"],
    "dune": ["dune"],
    "sparta": [],
}

# Quoted text that is NOT an assertion about what the software prints. Authors
# legitimately quote code to run, keys to set, and commands to type; demanding
# that those appear in the source as literals produces accusations against
# perfectly good entries.
_NOT_A_DIAGNOSTIC = re.compile(
    r"^\s*(?:from|import)\s+\w"          # import statements
    r"|^\s*(?:pip|conda|apt|cmake|make|mpirun|python[0-9.]*)\s"  # commands
    r"|^\s*[\w.]+\s*=\s*[^=]"            # assignments
    # An API expression: dotted names with a call somewhere and no sentence
    # punctuation. `Basis.interpolate(u).grad` is a thing to WRITE, not a thing
    # the library PRINTS, and requiring it to appear verbatim in the source
    # accuses an entry that is simply showing correct usage.
    r"|^[\w.]+(?:\([^)]*\))?(?:\.[\w]+(?:\([^)]*\))?)+$"
    # A weak-form / expression fragment: arithmetic operators outside of any
    # sentence. `(sigma*n - sigma.Other()*n) * v * ds(skeleton=True)` is UFL an
    # author is showing, not text a library prints.
    r"|^[^A-Z]*[*+]\s*\w+.*\)\s*$"
    # A UFL FORM: anything ending in an integration measure. `inner(curl(s),
    # curl(s))*dx` and `(1.0 - theta) * ufl.dot(...) * ufl.dx` are weak forms
    # to WRITE. No library prints a measure at the end of a sentence, so this
    # cannot swallow a real diagnostic.
    r"|\*\s*(?:ufl\.)?d[xsSPvc](?:\([^)]*\))?\s*$"
    # A CALL EXPRESSION: starts with a (dotted) name immediately applied, and
    # contains no ": " — every Python and C++ diagnostic that mentions a call
    # carries the exception class or a colon before the explanation, so
    # requiring its absence keeps `create_matrix(): incompatible function
    # arguments` judgeable while excusing `action(a, u) - L`,
    # `ufl.as_vector((Az.dx(1), -Az.dx(0)))` and
    # `fem.Expression(expr, W.element.interpolation_points())`.
    r"|^(?![^:]*:\s)[\w.]+\(.*\)[\w.\s)+*/-]*$"
    # A CONFIGURATION key/value pair: `snes_monitor: None` is a PETSc option an
    # author is telling the reader to SET, not a line the solver prints.
    r"|^[\w.]+\s*:\s*(?:None|''|\"\"|True|False|-?[\d.]+)\s*$",
    re.IGNORECASE)

# Dependency trees whose messages legitimately surface through a backend.
SHARED_DEPS = ["scipy", "numpy", "meshio", "petsc4py", "mpi4py"]

_SIGNAL_RE = re.compile(r"Signal:\s*(.*)", re.IGNORECASE | re.DOTALL)

# Quote pairs must be found IN ORDER, consuming both delimiters, with the
# length filter applied AFTER pairing.
#
# The obvious regex — r"'([^']{12,200})'|..." — is wrong, and wrong in the
# direction that manufactures false accusations. On the real entry
#
#     Signal: writing 'SOLID QUAD4' for an ALE mesh raises
#             'expected ALE element type' from 4C_ale_factory.cpp
#
# `SOLID QUAD4` is 11 characters, one below the minimum, so the alternation
# skipped it and matched the NEXT available pair — which straddled from the
# closing quote of the first fragment to the opening quote of the second,
# yielding the prose "for an ALE mesh raises". That prose is of course not in
# the source, so the checker reported an ABSENT diagnostic while never looking
# at `expected ALE element type`, the string actually being asserted. Both
# halves are failures: a fabricated accusation and a missed check.
_QUOTE_CHARS = "'\"`"

# RETRACTIONS ARE NOT FABRICATIONS. When a verification pass falsifies a quoted
# diagnostic, the honest correction records what the entry USED to claim so the
# next reader knows it was checked and why it changed:
#
#     beams.py:294  "older quote 'beam element type not supported in Exodus'
#                    is in no 4C source file; the real message is ..."
#
# That string is absent from the source — deliberately, and the entry says so.
# Flagging it would accuse an author of fabricating precisely where they did the
# opposite, and a checker that punishes correct behaviour gets switched off. So
# a quoted fragment sitting within a retraction context is EXPECTED_ABSENT and
# reported separately: still visible, never counted as a defect.
_RETRACTION_CUES = re.compile(
    r"(?:older|earlier|previous|prior|former)\s+(?:quote|claim|wording|text|"
    r"message|version|signals?|catalog)"
    # "previously quoted", "used to quote", "this entry used to claim"
    r"|previously\s+(?:quoted|claimed|stated|said)"
    r"|used\s+to\s+(?:quote|claim|say|state|describe)"
    r"|(?:the\s+)?(?:claimed|alleged|purported|supposed)\b"
    r"|is\s+in\s+no\b|does\s+not\s+exist|do\s+not\s+exist|never\s+appears?"
    r"|is\s+not\s+emitted|are\s+not\s+emitted|nowhere\s+in\b"
    r"|no\s+such\s+(?:string|message|error)"
    r"|(?:was|is)\s+(?:fabricated|invented|falsified|retracted|wrong)"
    r"|not\s+found\s+anywhere|appears?\s+in\s+neither"
    # WIDENED after the FEniCSx pass. Sixteen of 57 flagged strings were
    # entries doing exactly the right thing — recording, in the entry, that a
    # message does NOT exist — and the cue list simply did not know the
    # phrasing. Punishing an honest retraction is the one failure this screen
    # must not have, because the response is to stop retracting.
    #
    #   "does NOT reproduce" / "do NOT reproduce"    x7
    #   "does not appear in current UFL"             x2
    #   "is emitted by nothing in dolfinx"           x3
    #   "There is no 'TypeError: ...'"               x1
    #   "is NOT observable"                          x1
    #   "understates it by thirty orders"            x1
    #   "are absent from dolfinx"                    x1
    r"|(?:does|do|did)\s+not\s+(?:reproduce|appear|hold|apply)"
    r"|(?:is|are|was|were)\s+(?:absent|missing)\s+from\b"
    r"|(?:is|are)\s+emitted\s+by\s+nothing"
    r"|there\s+(?:is|are)\s+no\b"
    r"|(?:is|are)\s+not\s+observable"
    r"|understates?\b|overstates?\b"
    r"|neither\s+reproduces?\b"
    # A name quoted INSIDE one of Python's own name-not-found messages is
    # being reported as missing, by the message itself. `AttributeError:
    # module 'dolfinx' has no attribute 'PETScKrylovSolver'` is an entry
    # stating that PETScKrylovSolver does not exist — demanding the corpus
    # contain it inverts the claim and accuses the entry of inventing the very
    # name it is warning about.
    r"|has\s+no\s+attribute|no\s+attribute\s+named"
    r"|cannot\s+import\s+name|no\s+module\s+named"
    # Attribution elsewhere: the entry says the text comes from the CALLER's
    # own script or from a hypothetical buggy wrapper, not from the backend.
    # Demanding it appear in the backend's source accuses the entry of
    # fabricating a message it never claimed the backend prints.
    r"|(?:a|the)\s+(?:script|wrapper|caller|driver|user\s+code)\b"
    r"[^;.]{0,60}?(?:prints?|reports?|says?|emits?)",
    re.IGNORECASE)

# A quoted phrase that the sentence goes on to call a CHECK is a description of
# a test the reader might write, not a message the software prints:
#     "...tidy enough to pass any 'is it finite / is it O(1)' check"
_ROLE_IS_NOT_A_MESSAGE = re.compile(
    r"^\s*(?:check|test|guard|assertion|sanity|heuristic)\b", re.IGNORECASE)

# A retraction governs only its OWN clause. Both bounds were learned from
# getting it wrong on the real corpus:
#
#   * A plain backward window over-suppressed. In "older quote 'X' is in no 4C
#     source file; the real message is 'Y'", the cue for X sat within 120 chars
#     of Y, so Y was excused too — and Y is exactly the string that MUST be
#     checked. Clause boundaries stop the bleed.
#   * A backward-only window under-detected. In "'X' does not exist in the
#     source" the cue follows the quote, so nothing before it looked like a
#     retraction and a documented absence was reported as a fabrication.
#
# So: look back to the nearest clause boundary, and forward a short distance.
#
# THE DASH IS NOT A CLAUSE BOUNDARY, and treating it as one cost two more false
# accusations on the FEniCSx corpus. A retraction is very often written as an
# em-dash PARENTHETICAL, with the retracted strings inside it:
#
#     Both strings this entry used to quote — 'A' and 'B' — are absent from
#     dolfinx
#
# The opening dash cut "used to quote" out of A's backward window and the
# closing dash cut "are absent from" out of B's forward window, so a sentence
# whose entire subject is a retraction registered as none. A dash pair sets
# off a phrase; it does not end the assertion, which is what this window is
# trying to bound. `;` and `.` do, and they are what the bleed test needs.
_CLAUSE_BOUNDARY = re.compile(r"[;.]")
_BACK_WINDOW = 120
_FORWARD_WINDOW = 60


def _is_retracted(text: str, start: int, end: int) -> bool:
    """Is this quote introduced or immediately marked as NOT emitted?"""
    back = text[max(0, start - _BACK_WINDOW):start]
    # Trim to the last clause boundary so a neighbouring clause's retraction
    # does not excuse this quote.
    bounds = list(_CLAUSE_BOUNDARY.finditer(back))
    if bounds:
        back = back[bounds[-1].end():]
    if _RETRACTION_CUES.search(back):
        return True
    fwd = text[end:end + _FORWARD_WINDOW]
    if _ROLE_IS_NOT_A_MESSAGE.search(fwd):
        return True
    m = _CLAUSE_BOUNDARY.search(fwd)
    if m:
        fwd = fwd[:m.start()]
    return bool(_RETRACTION_CUES.search(fwd))


def quoted_fragments(text: str) -> list[str]:
    """Quoted fragments, paired left to right so a pair cannot straddle.

    Scans for an opening delimiter, then its matching close, then continues
    AFTER that close — so consecutive quoted fragments are read as the author
    wrote them. Short fragments are dropped after pairing, never during it.
    """
    def _is_apostrophe(s: str, k: int) -> bool:
        """An apostrophe inside a word is contraction/possessive, not a quote.

        Found by an independent adjudicator: several of this screen's flags were
        prose. In "don't reach for a skfem.NewtonSolver — it doesn't converge",
        the apostrophes of `don't` and `doesn't` were read as a matching pair
        and the sentence between them extracted as a diagnostic — which of
        course is not in any source, so an author's plain English was reported
        as a fabricated error message.
        """
        if s[k] != "'":
            return False
        before = s[k - 1] if k > 0 else ""
        after = s[k + 1] if k + 1 < len(s) else ""
        return before.isalpha() and after.isalpha()

    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in _QUOTE_CHARS and not _is_apostrophe(text, i):
            j = i
            while True:
                j = text.find(ch, j + 1)
                if j == -1 or not _is_apostrophe(text, j):
                    break
            if j == -1:
                break  # unbalanced delimiter: stop rather than guess
            frag = text[i + 1:j].strip()
            if 12 <= len(frag) <= 200 and not _is_retracted(text, i, j + 1):
                out.append(frag)
            i = j + 1  # resume AFTER the closing delimiter
            continue
        i += 1
    return out


def cited_source_files(text: str) -> list[str]:
    """Source files an entry names as the origin of a diagnostic.

    A separate defect class, found while building this: `ale.py` attributes a
    message to `4C_ale_factory.cpp`, and no such file exists anywhere in the 4C
    tree. A citation to a file that does not exist cannot be checked by a
    reader and cannot be where the message comes from, so it is a fabrication
    of provenance even when the message itself is real.
    """
    return sorted(set(re.findall(r"\b(4C_[\w.]+\.(?:cpp|hpp|H|cc))\b", text)
                      + re.findall(r"\b([\w/]+\.(?:cpp|hpp|py))\b", text)))

# A fragment must contain this much unbroken literal text to be judged. Below
# it, absence means nothing — short fragments collide with ordinary prose.
MIN_STATIC_FRAGMENT = 16


def signal_of(entry: str) -> str:
    m = _SIGNAL_RE.search(entry)
    return m.group(1) if m else ""


# Text the RUNTIME wraps around a message, which is therefore never in the
# source that emits it. Stripping these is not leniency — leaving them on makes
# the check unable to confirm a correct quote, which is worse than useless: it
# reports the honest thing and the fabricated thing identically.
#
#   * The exception CLASS. A Python traceback reads "TypeError: <message>",
#     and the library only ever wrote <message> — the class name is prepended
#     by the interpreter as it formats the traceback. Quoting the line as the
#     user sees it is exactly what an entry SHOULD do, and before this strip
#     every such entry was unconfirmable. It cost seven false accusations in
#     one FEniCSx run, every one of them against text measured from a live run
#     hours earlier: "ValueError: Unexpected complex value in real expression."
#     is in ufl/algorithms/remove_complex_nodes.py, and "RuntimeError: Rank
#     mismatch between Constant and function space in DirichletBC" is in
#     dolfinx's cpp.abi3.so. Only the seven leading characters were missing.
#
#   * The MPI RANK prefix. PETSc stamps "[0] " on every line of a parallel
#     diagnostic, and "[0]PETSC ERROR: " on the stderr form. The literal in
#     libpetsc is the bare message. Worse than a plain miss: static_parts
#     splits on digits, so "[0] Zero pivot row 0 value 0." was searched for as
#     "] Zero pivot row" — a leading bracket that appears nowhere, turning a
#     stock PETSc message into four separate accusations.
_RUNTIME_WRAPPER = re.compile(
    r"^\s*(?:\[\d+\]\s*(?:PETSC\s+ERROR:\s*)?)?"     # MPI rank / PETSc stderr
    r"(?:(?:\w+\.)*\w*(?:Error|Exception|Warning):\s+)?")  # exception class


def strip_runtime_wrapper(fragment: str) -> str:
    """Drop the rank prefix and exception class the runtime adds."""
    return _RUNTIME_WRAPPER.sub("", fragment, count=1).strip()


def static_parts(fragment: str) -> list[str]:
    """Split a message on the bits a program fills in at runtime.

    "DPoint 1 not in range [0:0[" is literal only in parts — the numbers come
    from the run. Matching the whole string would report a genuine message as
    fabricated, which is the failure mode this checker must not have.
    """
    fragment = strip_runtime_wrapper(fragment)
    # Split on: numbers, quoted sub-values, paths, angle-bracket placeholders,
    # ellipses, and the format placeholders the codes use.
    parts = re.split(
        r"\d+|<[^>]*>|\{[^}]*\}|%[sdfg]|\.\.\.|'[^']*'|\"[^\"]*\"|/[\w./-]+",
        fragment)
    return [p.strip() for p in parts if len(p.strip()) >= MIN_STATIC_FRAGMENT]


# Backends do not share an interpreter. dolfinx lives only in the `fenics`
# conda env, deal.II only in `ofa-dealii`, and the suite runs from a venv that
# has neither. Probing with `sys.executable` alone therefore returned nothing
# for four of nine backends, and the audit reported UNKNOWN for all of them —
# an honest non-answer, but a non-answer covering half the corpus.
#
# So each candidate interpreter is tried in turn. The first that can locate the
# package wins; the search is read-only and nothing from these environments is
# imported into this process.
_CANDIDATE_PYTHONS = [
    sys.executable,
    "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python",
    str(Path.home() / "miniconda3" / "envs" / "fenics" / "bin" / "python"),
    str(Path.home() / "miniconda3" / "envs" / "fenicsc" / "bin" / "python"),
    str(Path.home() / "miniconda3" / "envs" / "ofa-dealii" / "bin" / "python"),
    str(Path.home() / "miniconda3" / "envs" / "dune-fem-env" / "bin" / "python"),
    "/usr/bin/python3",
]


def _py_package_paths(module_names: list[str],
                      prefer: str | None = None) -> list[Path]:
    """Locate each module, trying `prefer` first if one is given.

    `prefer` exists because a backend's corpus must be ONE installation.
    Without it the search order is global, so `sys.executable` — the repo venv
    — answers for every module it happens to contain, and the audit reads a
    different release of the library from the one the backend runs against.
    Measured, not hypothetical: dolfinx 0.10 resolved in the conda env while
    `ufl` resolved in the venv, which holds UFL 2024.2.0 against the 2025.2.1
    that dolfinx actually imports. A message reworded between those two
    releases would be reported ABSENT while the running software prints it.
    """
    paths: list[Path] = []
    order = ([prefer] if prefer else []) + [
        p for p in _CANDIDATE_PYTHONS if p != prefer]
    for name in module_names:
        for _py in order:
            if _py != sys.executable and not Path(_py).is_file():
                continue
            if _locate_with(_py, name, paths):
                break
    return paths


def _interpreter_for(module: str) -> str | None:
    """The first candidate interpreter that can locate this module."""
    for py in _CANDIDATE_PYTHONS:
        if py != sys.executable and not Path(py).is_file():
            continue
        if _locate_with(py, module, []):
            return py
    return None


def _locate_with(py: str, name: str, paths: list[Path]) -> bool:
    """Try one interpreter; append the package dir and return True on success."""
    for probe in (
        # First choice: import it and ask where it lives.
        f"import {name},os;print(os.path.dirname({name}.__file__))",
        # Fallback: a failed import is not an absent package. Kratos is
        # INSTALLED on this host and unimportable — `import KratosMultiphysics`
        # dies on "libc.so.6: version GLIBC_2.32 not found", which is itself one
        # of the warnings in this corpus. Requiring a working import hid the
        # entire Kratos source tree from every audit, and the fallback roots
        # (scipy, numpy) then made real Kratos variables look invented.
        #
        # For grepping a corpus we need the FILES, not a live module, and
        # find_spec locates them without executing any of the package.
        "import importlib.util as u;"
        f"s=u.find_spec({name!r});"
        "print(next(iter(getattr(s,'submodule_search_locations',[]) or []), '')"
        " if s else '')",
    ):
        try:
            proc = subprocess.run([py, "-c", probe], capture_output=True,
                                  text=True, timeout=60,
                                  stdin=subprocess.DEVNULL)
        except (subprocess.TimeoutExpired, OSError):
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            p = Path(proc.stdout.strip())
            if p.is_dir():
                paths.append(p)
                return True
    return False


def search_roots(backend: str) -> tuple[list[Path], list[Path], list[str]]:
    """Where to look, split into the backend's OWN source and shared deps.

    The split is not cosmetic — it decides whether a verdict may be issued at
    all. First version returned one flat list, and Kratos resolved to four
    roots of which ZERO were Kratos: `KratosMultiphysics` is not importable in
    the test venv, so only scipy, numpy and meshio were found. A non-empty root
    list meant the audit considered itself able to judge, and it then reported
    every Kratos diagnostic as absent — having searched scipy for them. Over
    200 false accusations, against exactly the backend whose knowledge had just
    been carefully verified by hand.

    So: shared dependency trees may only ever CONFIRM a message (they hold
    genuine ones, from PETSc, UMFPACK, meshio, scipy). They can never license
    an absence verdict. Without the backend's own source, the answer is
    UNKNOWN.
    """
    own: list[Path] = []
    missing: list[str] = []
    for hint in SOURCE_HINTS.get(backend, []):
        p = Path(hint)
        if p.is_dir():
            own.append(p)
        else:
            missing.append(hint)

    mods = PY_MODULES.get(backend)
    shared: list[Path] = []
    if mods:
        primary = mods[0]
        home = _interpreter_for(primary)
        found_primary = _py_package_paths([primary], prefer=home)
        if not found_primary:
            # Secondary modules cannot stand in for the one that emits the
            # messages. Report nothing rather than search the wrong package.
            missing.append(f"primary module {primary!r} not importable here")
            shared = _py_package_paths(SHARED_DEPS)
            return own, shared, missing
        own.extend(found_primary)
        # Secondaries come from the SAME installation as the primary.
        own.extend(_py_package_paths(mods[1:], prefer=home))
        # The vendored *.libs siblings hold the actual compiled library; the
        # package directory holds only a wrapper.
        own.extend(vendored_lib_dirs(own))
        # A conda package does NOT vendor: its library sits in the env's
        # lib/ directory, which is nobody's sibling. See linked_libraries().
        own.extend(linked_libraries(own))
        shared = _py_package_paths(SHARED_DEPS)
        shared.extend(vendored_lib_dirs(shared))
        shared.extend(linked_libraries(shared))
        shared.extend(cpython_runtime(own))
    return own, shared, missing


def cpython_runtime(package_dirs: list[Path]) -> list[Path]:
    """libpython, which writes a good share of every Python backend's errors.

    `TypeError`, `AttributeError`, `SystemError` and their text are produced by
    the INTERPRETER, not by the library the user was calling. An entry that
    quotes the traceback line it saw is quoting CPython, and searching only the
    backend for it is looking in the wrong building.

    Measured: `returned a result with an exception set` — the full quoted line
    being `SystemError: <cyfunction EPS.solve at 0x...> returned a result with
    an exception set`, reproduced live from a SLEPc solve — is in
    libpython3.12.so.1.0 and in nothing else on this machine. Without this root
    it was reported as a fabricated FEniCSx diagnostic, twice.

    Not reachable through DT_NEEDED: CPython extension modules on Linux do not
    link libpython, they resolve its symbols from the running interpreter. So
    it has to be found by prefix, and it goes in SHARED (it may confirm a
    message, never license an absence verdict).
    """
    out: list[Path] = []
    for d in package_dirs:
        for parent in d.parents:
            if parent.name == "site-packages":
                libdir = parent.parent.parent
                for cand in sorted(libdir.glob("libpython3*.so*")):
                    real = cand.resolve()
                    if real.is_file() and real not in out:
                        out.append(real)
                break
    return out


def longest_found_prefix(fragment: str, roots: list[Path],
                         floor: int = MIN_STATIC_FRAGMENT) -> str:
    """The longest leading sub-phrase of this fragment present in the source.

    WHY THIS IS NEEDED, and it is the single biggest limitation of the whole
    approach. Compiled backends assemble diagnostics at runtime, so the message
    a user sees never exists as one literal. Two strings that agents captured
    from REAL NGSolve runs are not greppable at all:

        "Trace of non-matrix called"                  0 hits
        "does not exist for H1HighOrderFESpace"       0 hits
                (from: Operator "biharmonic" does not exist for
                       H1HighOrderFESpace!)

    The second is plainly `"... does not exist for " + type_name + "!"` with the
    type name supplied by RTTI. Reporting either as fabricated would be a false
    accusation against a message the agent watched the software print.

    So when a full fragment is absent, look for the longest leading piece that
    IS present. A substantial hit means "assembled at runtime, consistent with
    the source" — evidence for the entry, not against it. Only a fragment with
    no substantial piece anywhere is reported as absent, and even that stays
    evidence rather than proof.
    """
    words = fragment.split()

    # Longest CONTIGUOUS WINDOW, not longest leading slice.
    #
    # This used to take words[:n] only, so it could see interpolation at the
    # END of a message ("... does not exist for " + type_name) and was blind to
    # interpolation at the FRONT. Every KRATOS_ERROR prepends "Error: " at
    # runtime, and many also interpolate a name at the tail, so the literal in
    # the binary is an INTERIOR run of the message a user sees:
    #
    #     Error: Getting a value that does not exist. entry string : strategy
    #     ^^^^^^^                                                    ^^^^^^^^
    #     added at runtime                                    interpolated
    #
    # Measured on the 28-application build: the full fragment and its leading
    # slices score 0, while the interior run scores 1. Twelve flagged Kratos
    # entries have this shape, several captured verbatim from real runs, so a
    # whole class of true messages was being reported absent.
    #
    # Windows are tried longest-first and the search returns on the first hit,
    # so the cost is a few greps in the common case rather than O(n^2).
    n = len(words)
    for length in range(n, 0, -1):
        if len(" ".join(words[:length])) < floor and length < n:
            # every window of this length is below the floor
            shortest = min(len(" ".join(words[i:i + length]))
                           for i in range(n - length + 1))
            if shortest < floor:
                continue
        for i in range(n - length + 1):
            cand = " ".join(words[i:i + length])
            if len(cand) < floor:
                continue
            # A window that trims BOTH ends has lost its anchors, so it must
            # still be MOST of the message. Runtime insertions are short — a
            # "Error: " prefix, one interpolated name — so a literal that
            # accounts for only half the words is not that message, it is a
            # coincidence of common English.
            #
            # Measured, before this rule: the invented
            #   "PARTICLE_FRICTION must be positive for every DEM material"
            # was excused by the 20-character run "must be positive for",
            # which occurs somewhere in any large C++ corpus. Requiring 60% of
            # the words rejects it (4 of 8) while keeping the real messages:
            #   "Error: Getting a value ... entry string : strategy"  10/12
            #   "Error: STEP not found in process info of Main"        6/9
            if i > 0:
                # A PURE SUFFIX IS NOT THIS FUNCTION'S JOB — see
                # longest_found_suffix, which answers the same question under a
                # stricter, separately measured rule (skip at most ONE token,
                # because a message's tail is often generic English and an
                # unconstrained suffix search waved through 3 of 10 deliberate
                # fabrications). Two mechanisms for one question is the trap:
                # this one runs FIRST, so without this line the looser rule
                # decides every case and the stricter one never executes.
                # Measured on the merged tree before this line existed:
                # "cannot assemble: form has not been compiled" was excused by
                # "has not been compiled" and "solver exploded during the
                # quadrature loop" by "the quadrature loop" — both inventions,
                # both reported as ASSEMBLED rather than ABSENT.
                if i + length == n:
                    continue
                # A window that trims BOTH ends has lost its anchors, so it must
                # still be MOST of the message.
                if length < 0.6 * n:
                    continue
            if grep_literal(cand, roots):
                return cand
    return ""


def longest_found_suffix(fragment: str, roots: list[Path],
                         floor: int = MIN_STATIC_FRAGMENT) -> str:
    """The same idea from the other end, because interpolation goes there too.

    `longest_found_prefix` assumes the runtime fills in the TAIL. Half the time
    it fills in the HEAD, and then a prefix search fails on its very first
    word. nanobind writes

        "<function name>(): incompatible function arguments"

    so the literal in dolfinx's cpp.abi3.so is `incompatible function
    arguments` and the quoted line starts with `create_matrix():`, which is in
    no binary because nanobind builds it from the bound signature. Searching
    only forwards reported both nanobind messages as fabricated — two more
    accusations against text captured from a live run.

    THE HEAD MUST LOOK LIKE AN INTERPOLATED VALUE, and without that condition
    this probe is actively harmful. A message's TAIL is often generic English,
    so an unconstrained suffix search excuses inventions wholesale. Measured on
    a deliberately fabricated set, it waved through three of ten:

        "ADIOS2 VTX only supports Lagrange elements"
                              <= "supports Lagrange elements"
        "cannot assemble: form has not been compiled"
                              <= "has not been compiled"
        "solver exploded during the quadrature loop"
                              <= "the quadrature loop"

    None of those three is emitted by anything; the tail merely reads like
    something that would be. A screen that excuses fabrications is worse than
    one that flags real text, because nobody investigates a pass.

    So the part being skipped has to be a single token — the shape of a name
    the runtime substitutes (`create_matrix():`, `__init__():`) — and not a
    run of prose. That keeps both nanobind messages and rejects all three
    inventions above.
    """
    words = fragment.split()
    for n in range(len(words)):
        if n > 1:
            break  # skipping more than one token is prose, not interpolation
        cand = " ".join(words[n:])
        if len(cand) < floor:
            break
        if grep_literal(cand, roots):
            return cand
    return ""


# printf conversions, commonest first — CPython and the C++ codes use these.
_PRINTF_CONVERSIONS = ("%s", "%U", "%i", "%d", "%zu", "%ld", "%.200s")
_MIN_TEMPLATE = 14


def assembled_from_template(fragment: str, roots: list[Path]) -> str:
    """Does the corpus hold a FORMAT STRING that produces this text?

    The last honest gap. Some messages are interpolated in the MIDDLE, so
    neither a prefix nor a suffix search can reach them, and no literal search
    ever will. CPython's missing-argument error is the type case:

        libpython3.12.so.1.0:  %U() missing %i required %s argument%s: %U
        what the user sees:    TypeError: NonlinearProblem.__init__() missing
                               1 required keyword-only argument:
                               'petsc_options_prefix'

    Every word of that is CPython's, the entry quoted it correctly, and it was
    reported ABSENT four times because the count and the word `keyword-only`
    sit between the anchors. Reporting a template hit as ASSEMBLED — never as
    PRESENT, never as ABSENT — is the accurate answer: the corpus demonstrably
    contains the machinery that prints this, and no stronger claim is
    available from a literal search.

    Probes three-word windows with the middle token replaced by a conversion,
    which is cheap (a few dozen greps) and only runs on the accusation path.
    On the case above it finds `missing %i required` and `required %s
    argument`, both in libpython.
    """
    # Runs on the WHOLE fragment, not on the pieces static_parts produced —
    # those are split AT the interpolated values, so the anchors this needs to
    # bridge have already been separated into different pieces. Passing parts
    # was measured and found the template for none of the four CPython cases.
    words = strip_runtime_wrapper(fragment).split()
    for i in range(len(words) - 2):
        left = words[i].lstrip("'\"([")
        right = words[i + 2].rstrip(":;,.'\")]")
        if not left or not right:
            continue
        for conv in _PRINTF_CONVERSIONS:
            cand = f"{left} {conv} {right}"
            if len(cand) < _MIN_TEMPLATE:
                continue
            if grep_literal(cand, roots):
                return cand
    return ""


def grep_literal(needle: str, roots: list[Path]) -> bool:
    """Is this exact text anywhere under these roots, text OR binary?

    Uses -F (fixed string) so the metacharacters that fill real diagnostics —
    brackets, parentheses, dots — are taken literally, and -a so COMPILED
    libraries are searched as text.

    THE -a IS NOT OPTIONAL, and leaving it out invalidated a whole backend's
    result. grep skips binary content by default, silently: on ngslib.so it
    reported 0 matches for `FESpace` and `ngcore`, strings that must be in any
    NGSolve build. Every C++-level NGSolve diagnostic was therefore recorded as
    fabricated, which produced a measured "89% fabricated" that was itself
    fabricated. Two strings agents had captured from real runs —
    `Trace of non-matrix called` and `does not exist for H1HighOrderFESpace` —
    were among the accused, and both are real:

        libngfem.so   "Trace of non-matrix called"   1
        libngcomp.so  "does not exist for "          2

    See vendored_lib_dirs() for the other half of that bug.
    """
    if not roots:
        return False
    try:
        proc = subprocess.run(
            ["grep", "-rlaFq", "--", needle, *[str(r) for r in roots]],
            capture_output=True, text=True, timeout=300,
            stdin=subprocess.DEVNULL)
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def vendored_lib_dirs(package_dirs: list[Path]) -> list[Path]:
    """The `*.libs` siblings where wheels vendor their real shared libraries.

    A manylinux wheel ships a thin pybind11 wrapper in the package directory
    and the actual compiled library next door. For NGSolve, `ngsolve/ngslib.so`
    is 194 KB of wrapper while the code lives in
    `netgen_mesher.libs/libngfem.so`, `libngcomp.so`, `libngsolve.so`.

    Searching only the package directory therefore searches the wrapper and
    misses everything the backend actually says. Combined with grep's silent
    binary skip, that is what produced an entirely false NGSolve verdict.
    """
    out: list[Path] = []
    seen: set[Path] = set()
    for d in package_dirs:
        parent = d.parent
        if not parent.is_dir():
            continue
        try:
            for sib in parent.iterdir():
                if (sib.is_dir() and sib.name.endswith(".libs")
                        and sib not in seen):
                    seen.add(sib)
                    out.append(sib)
        except PermissionError:
            continue
    return out


def linked_libraries(package_dirs: list[Path]) -> list[Path]:
    """The compiled libraries a package LINKS AGAINST, wherever they live.

    THE CONDA-SHAPED HALF OF THE `vendored_lib_dirs` BUG, and it invalidated
    the entire first FEniCSx verdict. `vendored_lib_dirs` handles the wheel
    layout, where the real library sits in a `*.libs` SIBLING of the package
    directory. Conda does not vendor at all: the package directory holds only
    the pybind11/nanobind wrapper

        envs/fenics/lib/python3.12/site-packages/dolfinx/cpp.abi3.so

    while the code that emits the messages is two levels up and off to the
    side, in the ENVIRONMENT's library directory:

        envs/fenics/lib/libdolfinx.so.0.10.0     2.2 MB
        envs/fenics/lib/libpetsc.so.3.24.5        29 MB

    Neither is a sibling of anything the old search could reach, so the audit
    searched the wrapper and answered for the library. The positive control is
    unambiguous — "Newton solver did not converge" is a stock dolfinx message:

        package dir      miss
        libdolfinx.so    HIT

    That single gap produced 10 false accusations in one FEniCSx run. All ten
    are stock PETSc or dolfinx text: `Zero pivot in LU factorization`,
    `SNES Function norm`, `MUMPS error in numerical factorization: INFOG(`,
    `SNESSolve has not converged`, `Degree of output Function must be same as
    mesh degree`. Every one of them was in the software the whole time, in a
    file the audit could not see. This is the FEBio symlink lesson in a
    different costume, and it is why a positive control has to run before any
    absence is quoted.

    Resolution is by DT_NEEDED rather than by globbing the env's lib/, which
    is 3.6 GB and 2557 shared objects here — grepping that per fragment is not
    a search, it is a stall. The needed-list is what the backend is actually
    built on, so the corpus stays both correct and small (~80 MB for FEniCSx),
    and the transitive closure picks up the second rank — MUMPS reached
    through PETSc — which is where several genuine messages live.

    Reads ELF headers with `readelf -d`; nothing here is loaded or executed.
    """
    prefixes: list[Path] = []
    queue: list[Path] = []
    for d in package_dirs:
        # .../<prefix>/lib/pythonX.Y/site-packages/<pkg> -> <prefix>/lib
        for parent in d.parents:
            if parent.name == "site-packages" and parent.parent.parent.is_dir():
                libdir = parent.parent.parent
                if libdir.name == "lib" and libdir not in prefixes:
                    prefixes.append(libdir)
                break
        if d.is_dir():
            queue.extend(sorted(d.rglob("*.so*"))[:40])
        elif d.is_file():
            queue.append(d)
    if not prefixes:
        return []

    out: list[Path] = []
    seen: set[str] = set()
    scanned: set[Path] = set()
    while queue:
        obj = queue.pop()
        if obj in scanned or not obj.is_file():
            continue
        scanned.add(obj)
        for name in _dt_needed(obj):
            if name in seen:
                continue
            seen.add(name)
            for libdir in prefixes:
                cand = libdir / name
                if cand.is_file():
                    real = cand.resolve()
                    if real not in out:
                        out.append(real)
                        queue.append(real)
                    break
    return out


def _dt_needed(obj: Path) -> list[str]:
    """DT_NEEDED entries of an ELF object, or [] if it is not one."""
    try:
        proc = subprocess.run(["readelf", "-d", str(obj)],
                              capture_output=True, text=True, timeout=60,
                              stdin=subprocess.DEVNULL,
                              env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"})
    except (subprocess.TimeoutExpired, OSError):
        return []
    if proc.returncode != 0:
        return []
    return re.findall(r"\(NEEDED\)[^\[]*\[([^\]]+)\]", proc.stdout)


def _json_entries(be_dir: Path) -> list[tuple[Path, str]]:
    """Warnings held in JSON rather than in Python string literals.

    Not every backend keeps its knowledge in code. SPARTA's lives entirely in
    `sparta_knowledge.json`, so the AST walk below found ZERO entries for it and
    the audit cheerfully reported "0 keys checked, 0 unresolved" — a clean bill
    of health for a backend it had not looked at. A gate that reports OK on an
    empty reading is worse than one that reports UNKNOWN, because nobody
    investigates a pass.
    """
    out: list[tuple[Path, str]] = []

    def walk(node, path: Path) -> None:
        if isinstance(node, str):
            if "Signal:" in node:
                out.append((path, node))
        elif isinstance(node, dict):
            for v in node.values():
                walk(v, path)
        elif isinstance(node, list):
            for v in node:
                walk(v, path)

    for jf in sorted(be_dir.rglob("*.json")):
        try:
            walk(json.loads(jf.read_text(errors="ignore")), jf)
        except (json.JSONDecodeError, OSError):
            continue
    return out


def collect_entries(backend: str) -> list[tuple[Path, str]]:
    be_dir = REPO / "src" / "backends" / backend
    out = []
    if not be_dir.is_dir():
        return out
    out.extend(_json_entries(be_dir))
    for py in sorted(be_dir.rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(errors="ignore"))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and "Signal:" in node.value):
                out.append((py, node.value))
    return out


def audit_backend(backend: str) -> dict:
    own, shared, missing = search_roots(backend)
    entries = collect_entries(backend)
    all_roots = own + shared

    absent, present, unjudgeable = [], 0, 0
    assembled: list[dict] = []
    for path, entry in entries:
        sig = signal_of(entry)
        if not sig:
            continue
        for frag in quoted_fragments(sig):
            if _NOT_A_DIAGNOSTIC.search(frag):
                continue  # a command or code snippet, not an asserted message
            parts = static_parts(frag)
            if not parts:
                unjudgeable += 1
                continue
            if not own:
                # No source for THIS backend: a miss would prove nothing, so
                # do not produce one. See search_roots for what this cost.
                unjudgeable += 1
                continue
            # Present if ANY substantial static part is found in the backend
            # OR in a shared dependency — genuine messages routinely come from
            # PETSc, UMFPACK, meshio or scipy rather than from the backend.
            # Requiring all parts would flag messages assembled from several
            # literals, which is common.
            if any(grep_literal(p, all_roots) for p in parts):
                present += 1
                continue
            # Not present whole. Before accusing, look for the piece that IS
            # there — compiled backends and the interpreter assemble messages
            # at runtime, and a substantial hit means the entry matches the
            # source's format string rather than inventing it. Three probes,
            # because interpolation happens at all three places: the tail
            # (prefix search), the head (suffix search) and the middle
            # (printf template). Each was added after it caught a real message
            # the previous ones reported as fabricated.
            prefix = ""
            for probe in (longest_found_prefix, longest_found_suffix):
                for p in parts:
                    prefix = probe(p, all_roots)
                    if prefix:
                        break
                if prefix:
                    break
            if not prefix:
                prefix = assembled_from_template(frag, all_roots)
            if prefix:
                assembled.append({
                    "file": str(path.relative_to(REPO)),
                    "fragment": frag[:160],
                    "matched_prefix": prefix,
                })
                continue
            absent.append({
                "file": str(path.relative_to(REPO)),
                "fragment": frag[:160],
                "searched_for": parts[:3],
            })
    return {
        "backend": backend,
        "roots_searched": [str(r) for r in own],
        "shared_roots": [str(r) for r in shared],
        "roots_missing": missing,
        "entries": len(entries),
        "fragments_present": present,
        "fragments_absent": absent,
        "fragments_assembled": assembled,
        "fragments_unjudgeable": unjudgeable,
        "verdict": ("UNKNOWN — the backend's own source is not available here"
                    if not own
                    else ("CLEAN" if not absent else "ABSENT_STRINGS_FOUND")),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("backends", nargs="*",
                    help="backends to audit (default: all with a known source)")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    args = ap.parse_args()

    names = args.backends or sorted(
        set(SOURCE_HINTS) | {k for k, v in PY_MODULES.items() if v})
    results = [audit_backend(b) for b in names]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            print(f"\n=== {r['backend']} — {r['verdict']} ===")
            print(f"  entries {r['entries']}, quoted fragments present "
                  f"{r['fragments_present']}, absent {len(r['fragments_absent'])}, "
                  f"unjudgeable {r['fragments_unjudgeable']}")
            if r["roots_missing"]:
                print(f"  could NOT search: {', '.join(r['roots_missing'])}")
            for a in r["fragments_absent"][:20]:
                print(f"    ABSENT  {a['file']}")
                print(f"            {a['fragment']}")
            # ASSEMBLED is NOT a pass. It says a format string consistent with
            # the quote exists, which is the strongest thing a literal search
            # can say about an interpolated message — and it is also where a
            # fabrication shaped like a real one would hide. Measured on a
            # deliberately invented set, one of twelve landed here. Printing
            # the bucket keeps that one visible instead of silently clean.
            for a in r["fragments_assembled"][:20]:
                print(f"    assembled (unresolved) {a['file']}")
                print(f"            {a['fragment']}")
                print(f"            template in corpus: {a['matched_prefix']}")
        total = sum(len(r["fragments_absent"]) for r in results)
        unknown = [r["backend"] for r in results if r["verdict"].startswith("UNKNOWN")]
        print(f"\n{total} quoted diagnostics not found in the software that is "
              f"supposed to emit them.")
        if unknown:
            print(f"NOT CHECKED (no source available here): {', '.join(unknown)} "
                  f"— that is 'could not verify', NOT 'verified clean'.")

    # Exit 2 only for definite absences. An unverifiable backend must not turn
    # the gate red, or the gate gets disabled on the machines that lack a
    # source tree — and a disabled gate protects nothing.
    return 2 if any(r["fragments_absent"] for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
