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
    "fourc": ["/home/alexander/4C/src", "/home/alexander/4C/tests"],
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
    "fenics": ["dolfinx", "ufl", "basix", "petsc4py"],
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
    r"|^[^A-Z]*[*+]\s*\w+.*\)\s*$",
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
    r"message|version)"
    r"|(?:the\s+)?(?:claimed|alleged|purported|supposed)\b"
    r"|is\s+in\s+no\b|does\s+not\s+exist|do\s+not\s+exist|never\s+appears?"
    r"|is\s+not\s+emitted|are\s+not\s+emitted|nowhere\s+in\b"
    r"|no\s+such\s+(?:string|message|error)"
    r"|(?:was|is)\s+(?:fabricated|invented|falsified|retracted|wrong)"
    r"|not\s+found\s+anywhere|appears?\s+in\s+neither",
    re.IGNORECASE)

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
_CLAUSE_BOUNDARY = re.compile(r"[;.]|\s--\s|\s—\s")
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


def static_parts(fragment: str) -> list[str]:
    """Split a message on the bits a program fills in at runtime.

    "DPoint 1 not in range [0:0[" is literal only in parts — the numbers come
    from the run. Matching the whole string would report a genuine message as
    fabricated, which is the failure mode this checker must not have.
    """
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


def _py_package_paths(module_names: list[str]) -> list[Path]:
    paths = []
    for name in module_names:
        for _py in _CANDIDATE_PYTHONS:
            if _py != sys.executable and not Path(_py).is_file():
                continue
            if _locate_with(_py, name, paths):
                break
    return paths


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
        found_primary = _py_package_paths([primary])
        if not found_primary:
            # Secondary modules cannot stand in for the one that emits the
            # messages. Report nothing rather than search the wrong package.
            missing.append(f"primary module {primary!r} not importable here")
            shared = _py_package_paths(SHARED_DEPS)
            return own, shared, missing
        own.extend(found_primary)
        own.extend(_py_package_paths(mods[1:]))
        # The vendored *.libs siblings hold the actual compiled library; the
        # package directory holds only a wrapper.
        own.extend(vendored_lib_dirs(own))
        shared = _py_package_paths(SHARED_DEPS)
        shared.extend(vendored_lib_dirs(shared))
    return own, shared, missing


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
    for n in range(len(words), 0, -1):
        cand = " ".join(words[:n])
        if len(cand) < floor:
            break
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
            # Not present whole. Before accusing, look for the longest leading
            # piece — compiled backends assemble messages at runtime, and a
            # substantial hit means the entry matches the source's format
            # string rather than inventing it.
            prefix = ""
            for p in parts:
                prefix = longest_found_prefix(p, all_roots)
                if prefix:
                    break
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
