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
    "febio": ["/opt/febio", "/usr/local/febio"],
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
    r"|^[\w.]+(?:\([^)]*\))?(?:\.[\w]+(?:\([^)]*\))?)+$",
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
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in _QUOTE_CHARS:
            j = text.find(ch, i + 1)
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


def _py_package_paths(module_names: list[str]) -> list[Path]:
    paths = []
    for name in module_names:
        try:
            proc = subprocess.run(
                [sys.executable, "-c",
                 f"import {name},os;print(os.path.dirname({name}.__file__))"],
                capture_output=True, text=True, timeout=60,
                stdin=subprocess.DEVNULL)
            if proc.returncode == 0 and proc.stdout.strip():
                p = Path(proc.stdout.strip())
                if p.is_dir():
                    paths.append(p)
        except (subprocess.TimeoutExpired, OSError):
            continue
    return paths


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
        shared = _py_package_paths(SHARED_DEPS)
    return own, shared, missing


def grep_literal(needle: str, roots: list[Path]) -> bool:
    """Is this exact text anywhere under these roots?

    Uses grep -F (fixed string) so regex metacharacters in a diagnostic — and
    they are everywhere: brackets, parentheses, dots — are taken literally.
    """
    if not roots:
        return False
    try:
        proc = subprocess.run(
            ["grep", "-rlFq", "--", needle, *[str(r) for r in roots]],
            capture_output=True, text=True, timeout=300,
            stdin=subprocess.DEVNULL)
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def collect_entries(backend: str) -> list[tuple[Path, str]]:
    be_dir = REPO / "src" / "backends" / backend
    out = []
    if not be_dir.is_dir():
        return out
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
            else:
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
