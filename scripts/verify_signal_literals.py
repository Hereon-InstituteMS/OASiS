#!/usr/bin/env python3
"""Can the software actually EMIT the error messages our Signals quote?

WHY THIS EXISTS
---------------
Every pitfall in OASiS ships a ``Signal:`` clause — the observable symptom an
agent is told to look for. `verify_signal_clauses.py` checks two things about
those clauses: that they name an entity the catalogs know (tier 0) and that they
use symptom vocabulary (tier 1). A hand-written fixture (tier 2) covers a small
minority.

An adversarial audit of the FEBio knowledge showed what those tiers miss. It
found 25 quoted Signal strings that appear nowhere in the FEBio binary or any of
its libraries — among them ``DIVERGED_FNORM_NAN`` and ``KSPSolve: zero-pivot``,
which are Trilinos and PETSc names for a code that links neither. They passed
tier 0 and tier 1 comfortably: they name plausible entities and they read
exactly like observable symptoms. They were simply invented.

That is the worst failure mode in the whole knowledge base. A pitfall whose
signal cannot occur is not merely useless — it tells an agent that the absence
of a message it could never have seen means the problem is not present.

WHAT THIS CHECKS
----------------
For each backend, the literal strings quoted inside Signal clauses are matched
against a corpus of everything that backend could print: the string table of its
compiled binaries and shared libraries, or the source of its installed Python
package. A literal that appears nowhere in that corpus cannot be emitted, and is
reported as UNMATCHABLE.

WHAT IT DOES NOT PROVE — WITH A MEASURED EXAMPLE
------------------------------------------------
A match is not proof the message appears in the situation the pitfall describes
— only that the software contains it. That still needs a tier-2 fixture. The
value here is the opposite direction: a NON-match is near-proof of fabrication,
it is cheap, and it covers the entire surface rather than a sample.

That caveat is not hypothetical, and an audit produced the instance. A FEBio
pitfall claimed the `LU` linear solver reports

    Linear solver failed to find solution. Aborting run.

The string IS in the binary — so it passes this checker, and it passed the
tier-0 allowlist too. But it is not what an LU failure prints. The real message
is `Fatal error in factorization of stiffness matrix. Aborting run.`, confirmed
by running it. A REAL string attributed to the WRONG TRIGGER survives every
presence check there is.

So the tiers stack as: presence (this script) rules out invention; ATTRIBUTION
needs the wrong variant triggered and the message observed, which only a tier-2
fixture does. Presence is necessary and nowhere near sufficient, and anyone
reading a green run from this script should read that sentence twice. Reporting
"0 unmatchable signals" as though it meant "signals verified" is the mistake this
paragraph exists to prevent.

STATUS: A LEAD GENERATOR, NOT YET A GATE
----------------------------------------
Read the output as "look at these", not "these are fabricated". Measured
against a careful manual audit of FEBio, which found 25 fabricated messages
across 81 pitfalls, this script extracts only a handful of literals per backend
— it sees the identifiers a Signal quotes, and most Signals describe their
symptom in prose instead. It is nowhere near the coverage of a human pass.

Its false positives are known and are of three kinds:

  * PYTHON BUILTINS — `ImportError` is not in scikit-fem's source because it is
    in the interpreter.
  * TUTORIAL SYMBOLS — deal.II's `assemble_system` and `output_results` are
    defined in the step-N example programs, not in libdeal_II, so a Signal that
    names them is correct and this reports it anyway.
  * INCOMPLETE CORPORA — a name implemented in a backend's compiled layer is
    absent from its Python source. Pass every relevant .so, or expect noise.

What it is genuinely good at is the case that motivated it: DIAGNOSTICS
BORROWED FROM ANOTHER LIBRARY. `KSPSolve: DIVERGED_BREAKDOWN` and
`DIVERGED_INDEFINITE_PC` are in the NGSolve knowledge; NGSolve's library
contains no PETSc KSP strings at all. `DIVERGED_FNORM_NAN` and `NOX` were in
the FEBio knowledge, which links neither PETSc nor Trilinos. Those are not
typos — they are error messages from a different code, written into a pitfall
as though observed, and an agent told to watch for them will wait forever.

Making this a standing gate needs prose-symptom extraction and a per-backend
allowlist for the three false-positive classes above. Until then it earns its
keep by pointing a human at the right lines.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BACKENDS = REPO / "src" / "backends"

# WHAT IS CHECKED, AND WHY IT IS NARROW
#
# A first version of this extractor pulled anything between quotes out of a
# Signal clause. It reported 95-100% of literals as unmatchable on every
# backend, which is not a finding, it is a broken tool: what it had actually
# collected was parenthesised PROSE ("(cylinder in cross-flow) shows no shedding
# when dt"), which of course appears in no binary. A check that condemns
# everything is worth exactly as much as one that condemns nothing.
#
# So this now checks one precise, falsifiable thing: the IDENTIFIERS a Signal
# names — API symbols and error-code tokens, no spaces — must exist in the
# software. That is the class the audits actually caught: DIVERGED_FNORM_NAN and
# KSPSolve in FEBio (Trilinos and PETSc names, in a code that links neither),
# MortarFacetBasis in scikit-fem, GridTools::get_boundary_ids in deal.II. Whole
# quoted sentences are deliberately NOT checked: runtime formatting means they
# legitimately never appear verbatim, and demanding they do produces a wall of
# false accusations that gets the tool switched off.
_SIGNAL = re.compile(r"Signal:\s*(.{0,400}?)(?:\.\s|\"\s*,|\Z)", re.S)

# An identifier: no spaces, and shaped like code rather than English.
_IDENT = re.compile(r"[`\"']([A-Za-z_][\w:.<>]{5,80})[`\"']")

# It must look like code: snake_case, CamelCase, SCREAMING_CASE, or a
# qualified name. A quoted ordinary word is prose.
_CODE_SHAPED = re.compile(r"_|::|\.|[a-z][A-Z]|^[A-Z]{4,}$")

# Words that are code-shaped by accident but are ordinary English in context.
_STOPWORDS = {"e.g.", "i.e.", "etc.", "vs.", "cf."}

MIN_FRAGMENT = 12          # shortest static fragment worth searching for


def _corpus_from_binaries(paths: list[Path]) -> str:
    """Everything a compiled artefact could print, via its string table."""
    chunks = []
    for p in paths:
        if not p.exists():
            continue
        try:
            out = subprocess.run(["strings", "-n", "6", str(p)],
                                 capture_output=True, text=True, timeout=600)
            chunks.append(out.stdout)
        except (OSError, subprocess.SubprocessError):
            continue
    return "\n".join(chunks)


def _corpus_from_python(roots: list[Path]) -> str:
    chunks = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            try:
                chunks.append(p.read_text(errors="ignore"))
            except OSError:
                continue
    return "\n".join(chunks)


def _longest_static_fragment(literal: str) -> str:
    """The longest run without a runtime-substituted part.

    "Section 'X' is not a valid section name." exists in the binary only as the
    pieces either side of the substitution, so searching for the whole sentence
    finds nothing even though the message is real.
    """
    parts = re.split(r"%[sdfgx]|\{[^}]*\}|'[^']*'|\"[^\"]*\"|<[^>]*>|\b\d+\b",
                     literal)
    return max((p.strip() for p in parts), key=len, default="")


def signal_literals(text: str) -> list[str]:
    """Identifiers a Signal clause names. See the note above on why only these."""
    out = []
    for sig in _SIGNAL.finditer(text):
        for q in _IDENT.finditer(sig.group(1)):
            lit = q.group(1).strip().rstrip(".")
            if lit.lower() in _STOPWORDS or not _CODE_SHAPED.search(lit):
                continue
            out.append(lit)
    return out


def _match_target(lit: str) -> str:
    """What to search the corpus for.

    A qualified name is searched by its final component: `GridTools::get_boundary_ids`
    exists in a library as the symbol `get_boundary_ids`, and namespaces are
    mangled rather than spelled out. Searching the qualified form would miss
    every genuine symbol.
    """
    for sep in ("::", "."):
        if sep in lit:
            tail = lit.rsplit(sep, 1)[-1]
            if len(tail) >= 5:
                return tail
    return lit


def check_backend(name: str, corpus: str) -> dict:
    root = BACKENDS / name
    if not root.exists():
        return {"backend": name, "error": "no such backend directory"}
    literals, per_file = [], {}
    for p in root.rglob("*.py"):
        found = signal_literals(p.read_text(errors="ignore"))
        if found:
            per_file[str(p.relative_to(REPO))] = found
            literals.extend(found)

    unmatched = []
    for lit in sorted(set(literals)):
        if lit in corpus or _match_target(lit) in corpus:
            continue
        unmatched.append(lit)

    total = len(set(literals))
    return {
        "backend": name,
        "literals_checked": total,
        "unmatchable": len(unmatched),
        "percent_unmatchable": round(100.0 * len(unmatched) / total, 1) if total else 0.0,
        "corpus_size": len(corpus),
        "examples": unmatched[:25],
        "files": {k: len(v) for k, v in per_file.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", action="append", default=None)
    ap.add_argument("--binary", action="append", default=[],
                    help="backend=path  (repeatable) — compiled artefact")
    ap.add_argument("--pysrc", action="append", default=[],
                    help="backend=path  (repeatable) — installed package root")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    sources: dict[str, dict[str, list[Path]]] = {}
    for spec, kind in [(args.binary, "bin"), (args.pysrc, "py")]:
        for item in spec:
            if "=" not in item:
                print(f"bad --{kind}: {item}", file=sys.stderr)
                return 2
            be, path = item.split("=", 1)
            sources.setdefault(be, {}).setdefault(kind, []).append(Path(path))

    names = args.backend or sorted(sources)
    if not names:
        print("give at least one --binary or --pysrc", file=sys.stderr)
        return 2

    report = []
    for name in names:
        src = sources.get(name, {})
        corpus = (_corpus_from_binaries(src.get("bin", []))
                  + "\n" + _corpus_from_python(src.get("py", [])))
        if not corpus.strip():
            print(f"[{name}] SKIPPED — no corpus; a check with nothing to match "
                  f"against would report everything as fabricated")
            continue
        r = check_backend(name, corpus)
        report.append(r)
        if "error" in r:
            print(f"[{name}] {r['error']}")
            continue
        print(f"[{name}] {r['unmatchable']}/{r['literals_checked']} quoted Signal "
              f"literals ({r['percent_unmatchable']}%) cannot be emitted "
              f"(corpus {r['corpus_size'] // 1000}k chars)")
        for ex in r["examples"][:8]:
            print(f"      UNMATCHABLE: {ex}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
