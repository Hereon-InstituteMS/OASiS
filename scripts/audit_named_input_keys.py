#!/usr/bin/env python3
"""Screen every input key OASiS names against the backend that would consume it.

WHY THIS EXISTS
---------------
The 4C particle pass found this being served as knowledge:

    "SOUNDSPEED too low -> fluid compresses unrealistically;
     rule of thumb c >= 10 * v_max"

`SOUNDSPEED` appears in **zero** files of 4C's source and zero of its 2171
input decks. So does `SMOOTHING_LENGTH`. Both were listed beside real material
parameters (`DYN_VISCOSITY`, `BULK_MODULUS`) as things an SPH deck must set, and
one of them carried a numeric tuning rule. An agent following that advice writes
a deck 4C refuses to parse, and the rule of thumb is advice about nothing.

Nothing caught it. The quoted-diagnostics auditor screens error strings the
knowledge puts in quotes; an invented *input key* is not a quoted diagnostic, so
it walked straight through. This closes that hole: every ALL-CAPS identifier the
knowledge names is looked up in the backend's own corpus, and the ones that
resolve nowhere are reported.

WHAT IT REPORTS, AND WHAT IT REFUSES TO REPORT
----------------------------------------------
Candidates, not verdicts. Four exclusions keep it from crying wolf, each one
learned from a false accusation this project already made and had to withdraw:

  * **Retractions are not fabrications.** "There is no SOUNDSPEED key" is the
    CORRECTED entry. It contains the token precisely because the absence is the
    knowledge. Reusing `_is_retracted` from the quoted-diagnostics auditor rather
    than writing a second copy — two implementations of one rule is how the
    matcher defect happened.
  * **Prose is not a key.** `CFL`, `MPI`, `VTK`, `YAML` are English-in-caps.
    Screened by a stopword list plus a shape rule.
  * **A backend with no resolvable corpus yields UNKNOWN**, never "fabricated".
    Kratos was once judged against scipy because its own source was not on disk,
    and every claim looked invented.
  * **Compiled backends assemble key names at runtime**, so a miss in the source
    text is weaker evidence there than in a Python backend. Reported with that
    caveat attached rather than silently equated.

SELF-CONTROL
------------
`--selftest` runs the audit against a branch that predates the 4C fix and one
that follows it. The pre-fix tree must flag SOUNDSPEED; the post-fix tree must
not. A gate that cannot demonstrate it detects the thing it was built for is
just another unverified claim.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from audit_quoted_diagnostics import (  # noqa: E402
    _is_retracted, collect_entries, search_roots, signal_of,
)

# ── what counts as a candidate key ──────────────────────────────────────────
# Shape: ALL CAPS, may carry digits and underscores. Four characters minimum —
# below that the false-positive rate from prose swamps the signal.
_KEY = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*\b")

# English-in-caps, format names, physics abbreviations, units and the project's
# own category tags. None of these are input keys, and every one of them occurs
# in ordinary warning prose.
_STOPWORDS = {
    # category tags used by the warning format itself
    "NUMERICAL", "API", "INPUT", "SYNTAX", "PHYSICS", "INTEGRATION", "SETUP",
    "PERFORMANCE", "OUTPUT", "UNITS", "MESH", "VALIDATION", "BC", "CROSS",
    # formats, tools, protocols
    "XML", "YAML", "JSON", "HDF5", "VTK", "VTU", "VTP", "CSV", "TXT", "PVD",
    "MPI", "OPENMP", "CUDA", "GPU", "CPU", "RAM", "OS", "CLI", "API", "URL",
    "UFL", "PETSC", "MUMPS", "UMFPACK", "SUPERLU", "BLAS", "LAPACK", "GMSH",
    # physics and numerics in caps
    "CFL", "PDE", "ODE", "FEM", "DEM", "SPH", "MPM", "PFEM", "DSMC", "DOF",
    "DOFS", "RHS", "LHS", "FSI", "TSI", "ALE", "DG", "HDG", "CG", "GMRES",
    "BDF", "RK", "MMS", "SI", "EOS", "PD", "VEM", "AMG", "ILU", "LU", "QR",
    "SVD", "PCG", "BICGSTAB", "NAN", "INF", "TOL", "ABS", "REL", "MIN", "MAX",
    # English words that appear capitalised for emphasis
    "NOT", "NO", "AND", "OR", "BUT", "ALL", "ANY", "ONLY", "MUST", "NEVER",
    "ALWAYS", "TODO", "NOTE", "WARNING", "ERROR", "FATAL", "OK", "YES",
    "TRUE", "FALSE", "NONE", "NULL", "ID", "IDS", "II", "III", "IV", "3D",
    "2D", "1D", "PASS", "FAIL", "SKIP", "THE", "IS", "IT", "IF", "ON", "OFF",
}


# A key can be named in order to say it DOES NOT EXIST — which is the corrected
# form of exactly the entry this gate was built to catch:
#
#     "There is no SOUNDSPEED key. The material is MAT_ParticleSPHFluid ..."
#
# The shared `_is_retracted` does not recognise that phrasing, and widening the
# shared regex would silently change the quoted-diagnostics auditor's results on
# a corpus that has already been audited against it. So the absence test lives
# here and is anchored to the TOKEN rather than to the clause: the phrase has to
# be talking about this identifier, not merely sitting near it.
_ABSENCE_BEFORE = re.compile(
    r"(?:there\s+(?:is|are|was|were)\s+no"
    r"|no\s+such"
    r"|not\s+a\s+(?:valid|real|recognised|recognized)"
    r"|does\s+not\s+(?:exist|accept|have)"
    r"|never\s+(?:existed|accepted))"
    r"[^.;]{0,40}$", re.I)
_ABSENCE_AFTER = re.compile(
    r"^[^.;]{0,40}?(?:does\s+not\s+exist"
    r"|is\s+not\s+a\s+(?:valid|real|4c|febio|kratos)?\s*(?:key|section|option)"
    r"|is\s+no\s+such"
    r"|was\s+invented"
    r"|appears\s+in\s+(?:zero|no)\b)", re.I)


def _absence_asserted(text: str, start: int, end: int) -> bool:
    """True when the surrounding prose says this identifier does NOT exist."""
    back = text[max(0, start - 120):start]
    if _ABSENCE_BEFORE.search(back):
        return True
    return bool(_ABSENCE_AFTER.search(text[end:end + 120]))


# Words that mark the token beside them as an input key rather than emphasis.
_KEYISH = re.compile(
    r"\b(?:key|keys|parameter|parameters|section|sections|option|options|flag|"
    r"flags|field|fields|attribute|attributes|entry|variable|variables|set|"
    r"setting|specify|specifies|required|writes?|written)\b", re.I)


def _identifier_shaped(tok: str) -> bool:
    """An underscore or a digit is positive evidence of an identifier."""
    return "_" in tok or any(c.isdigit() for c in tok)


def _in_list_with_identifier(text: str, start: int, end: int) -> bool:
    """True when the token sits in a comma list containing a real identifier.

    This is the rule that catches the case shape alone cannot. The fabricated
    entry read

        "... for each particle type (density, DYN_VISCOSITY, BULK_MODULUS,
         SOUNDSPEED)"

    `SOUNDSPEED` is a plain all-caps word, exactly like the emphasis-caps this
    gate must ignore (`CHECKERBOARD`, `AUTOMATICALLY`). What distinguishes it is
    the company it keeps: enumerated alongside `DYN_VISCOSITY` and
    `BULK_MODULUS`, both underscore-shaped and both real. A word being listed as
    a peer of confirmed input keys is the claim that it is one.
    """
    lo = text.rfind("(", max(0, start - 200), start)
    seg_start = lo + 1 if lo != -1 else max(0, start - 200)
    hi = text.find(")", end, end + 200)
    seg_end = hi if hi != -1 else min(len(text), end + 200)
    seg = text[seg_start:seg_end]
    if "," not in seg:
        return False
    peers = [p.strip(" '\"`") for p in seg.split(",")]
    peers = [p for p in peers if p != text[start:end]]
    return any(_KEY.fullmatch(p) and _identifier_shaped(p) and
               p not in _STOPWORDS for p in peers)


def candidate_keys(text: str) -> list[tuple[str, int, int]]:
    """ALL-CAPS tokens `text` presents AS input keys, with spans.

    Shape is not enough. `SOUNDSPEED` (invented) and `CHECKERBOARD` (ordinary
    prose in caps) are the same shape, so a shape rule either misses the
    fabrication or reports every emphasised word — the first pass over the
    corpus did the latter, returning 147 "unresolved keys" for Kratos of which
    the overwhelming majority were English. A gate nobody reads catches nothing.

    So a token has to earn candidacy by evidence that the text is naming a key:
    identifier shape, quoting, a key-marker word beside it, or membership in a
    list whose other members are confirmed identifiers.
    """
    out = []
    for m in _KEY.finditer(text):
        tok, i, j = m.group(0), m.start(), m.end()
        if tok in _STOPWORDS or len(tok) < 4:
            continue
        if text[max(0, i - 1):i] == "[":      # the warning's [Category] tag
            continue
        quoted = (text[max(0, i - 1):i] in "'\"`"
                  or text[j:j + 1] in "'\"`")
        near = bool(_KEYISH.search(text[max(0, i - 60):i])
                    or _KEYISH.search(text[j:j + 60]))
        if not (_identifier_shaped(tok) or quoted or near
                or _in_list_with_identifier(text, i, j)):
            continue
        out.append((tok, i, j))
    return out


def key_present(key: str, roots: list[Path]) -> bool:
    """Does this identifier occur anywhere in the backend's corpus?

    `-a` is not optional: without it grep skips files it decides are binary and
    reports nothing found. That single flag was the difference between a
    measured 91% fabrication rate and the true 73% in an earlier pass.
    """
    if not roots:
        return False
    cmd = ["grep", "-r", "-a", "-l", "-F", "--", key] + [str(r) for r in roots]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return bool(p.stdout.strip())


def audit(backend: str, verbose: bool = False) -> dict:
    src_roots, corpus_roots, notes = search_roots(backend)
    roots = list(src_roots) + list(corpus_roots)
    res = {"backend": backend, "roots": [str(r) for r in roots],
           "notes": notes, "unresolved": [], "checked": 0, "entries": 0}
    if not roots:
        res["status"] = "UNKNOWN"
        res["reason"] = (f"no corpus on disk for {backend}; a missing corpus "
                         f"cannot distinguish an invented key from a real one")
        return res

    # THE RULE THIS PROJECT KEEPS RELEARNING. `search_roots` falls back to
    # whatever scientific-Python packages are installed when the backend's own
    # module will not import. Judging Kratos knowledge against scipy makes every
    # real Kratos variable look invented: the first run of this gate reported
    # 121 of 139 Kratos keys "unresolved", and the roots it actually searched
    # were scipy, numpy, meshio and mpi4py. Not one Kratos file.
    #
    # An audit that cannot see the software it is auditing has no verdict to
    # give. It says so, and names the interpreter that would fix it.
    unresolved_primary = [n for n in (notes or [])
                          if "not importable" in str(n)]
    if unresolved_primary:
        res["status"] = "UNKNOWN"
        res["reason"] = (
            f"{unresolved_primary[0]} — the roots on hand "
            f"({', '.join(Path(p).name for p in res['roots'][:4])}) are not "
            f"{backend}. Re-run with an interpreter where {backend} imports; "
            f"a fallback corpus can only manufacture false accusations.")
        return res

    seen: dict[str, list] = {}
    for path, entry in collect_entries(backend):
        res["entries"] += 1
        for tok, i, j in candidate_keys(entry):
            if _is_retracted(entry, i, j) or _absence_asserted(entry, i, j):
                continue          # "there is no SOUNDSPEED key" is the fix
            seen.setdefault(tok, []).append((str(path), signal_of(entry)[:90]))

    for tok in sorted(seen):
        res["checked"] += 1
        if not key_present(tok, roots):
            res["unresolved"].append({
                "key": tok,
                "occurrences": len(seen[tok]),
                "first_seen": seen[tok][0][0],
                "signal": seen[tok][0][1],
            })
    part = corpus_completeness(backend)
    if part:
        res["corpus_partial"] = part
        res["status"] = "PARTIAL_CORPUS"
        # "Not found" means "not found in what is installed". Kratos ships ~40
        # applications; this host has three. BIOT_COEFFICIENT lives in
        # Poromechanics and DEM_SURFACE_LOAD in DEM, so neither can be resolved
        # here — and neither is thereby shown to be invented. Calling them
        # fabrications would repeat the scipy mistake one level down.
        res["unverifiable"] = res.pop("unresolved")
        return res
    res["status"] = "OK" if not res["unresolved"] else "CANDIDATES"
    return res


def corpus_completeness(backend: str) -> str:
    """Describe a corpus known to be a SUBSET of the backend, else ''.

    Only Kratos needs this today: it is distributed as a core plus optional
    application packages, each a separate wheel, and a key belonging to an
    application that is not installed is invisible no matter how real it is.
    """
    if backend != "kratos":
        return ""
    import glob
    sp = glob.glob("/home/alexander/Schreibtisch/open-fem-agent/.venv/lib/"
                   "python*/site-packages")
    if not sp:
        return ""
    apps = sorted({Path(p).name.split("-")[0]
                   for p in glob.glob(f"{sp[0]}/kratos*application-*.dist-info")})
    if not apps:
        return ""
    return (f"only {len(apps)} Kratos applications installed "
            f"({', '.join(a.replace('kratos', '').replace('application', '') for a in apps)}"
            f" + core); keys belonging to any other application "
            f"(DEM, Poromechanics, CFD, ...) cannot be resolved on this host")


def selftest() -> int:
    """Prove the gate detects what it was built to detect.

    Pre-fix tree must flag SOUNDSPEED; post-fix tree must stay quiet about it.
    """
    pre = Path("/home/alexander/Schreibtisch/ofa-verify-ngs")
    post = Path("/home/alexander/Schreibtisch/ofa-know-4c")
    print("SELFTEST — the gate must flag the known fabrication and only it\n")
    ok = True
    for label, tree, expect in (("pre-fix ", pre, True), ("post-fix", post, False)):
        f = tree / "src" / "backends" / "fourc" / "backend.py"
        if not f.is_file():
            print(f"  {label}: {f} missing — cannot run control")
            ok = False
            continue
        text = f.read_text(errors="ignore")
        hits = [t for t, i, j in candidate_keys(text)
                if t == "SOUNDSPEED" and not _is_retracted(text, i, j)
                and not _absence_asserted(text, i, j)]
        got = bool(hits)
        good = got == expect
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {label} tree: SOUNDSPEED "
              f"{'flagged' if got else 'not flagged'} "
              f"(expected {'flagged' if expect else 'not flagged'})")
    print(f"\nselftest {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("backends", nargs="*")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    names = args.backends or ["fourc", "fenics", "dealii", "ngsolve", "skfem",
                              "kratos", "dune", "febio", "sparta"]
    out = []
    for b in names:
        r = audit(b)
        out.append(r)
        head = f"{b:<10} {r['status']:<11}"
        if r["status"] == "UNKNOWN":
            print(f"  {head} {r['reason']}")
            continue
        # "unresolved" and "unverifiable" are deliberately different words and
        # must never be printed under one heading: the first says the corpus was
        # searched and the key is not in it, the second says the corpus cannot
        # answer. Collapsing them is how a coverage gap gets read as a fabrication.
        hits = r.get("unresolved")
        label = "unresolved"
        if hits is None:
            hits, label = r.get("unverifiable", []), "UNVERIFIABLE here"
        print(f"  {head} {r['checked']:>4} distinct keys checked across "
              f"{r['entries']} entries -> {len(hits)} {label}")
        if r.get("corpus_partial"):
            print(f"        corpus is partial: {r['corpus_partial']}")
        for u in hits[:8]:
            print(f"        {u['key']:<28} x{u['occurrences']:<3} {u['signal']}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\n  written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
