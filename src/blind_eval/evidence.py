"""Did the named codes actually run, and did they actually couple?

The pre-existing gate (``grade_blind.code_ran``) is a case-insensitive
**substring** scan for the code's name over every ``.log/.out/.txt/.json`` under
the run directory, excluding four filenames.  The task statement names both
codes.  So an agent that writes any note mentioning "deal.II" satisfies the
deal.II evidence requirement without deal.II existing on the machine.

Combine that with a coupled instance whose two subdomains carry the same field
and the same material, and the rational strategy for either arm is: solve the
whole problem monolithically in one code, write a file mentioning the second,
submit.  The result converges at order two, passes every integrity gate, and the
cell has measured single-code PDE solving — which the single-code cells already
measure.  A coupling-uplift number computed from that would be believed and
empty.

What this module requires instead
---------------------------------
1. **Structured solver output, per code.**  Not the code's name: a line only
   that solver emits, carrying a NUMBER it computed.  ``dolfinx`` printing a DOF
   count, deal.II's "Number of active cells", NGSolve's ``ndof``, Kratos's
   solver banner with a time.  A name is prose; a number is a run.

2. **The two codes' evidence must come from different files.**  One monolithic
   solve writes one solver's output.  This does not stop a determined forger,
   and is not claimed to.

3. **A partitioned-iteration residual history, as a required artefact.**  A
   coupled run that cannot produce one did not couple.  It must have at least
   three iterations, be positive, decrease by at least a factor of ten from
   first to last, end at or below the interface tolerance the task prescribed,
   and agree with the iteration count the agent reports in ``RESULT.txt``.
   A monolithic solve has no such history at all.

Honesty about what this is
--------------------------
This raises the cost of faking from "write one text file" to "implement a
plausible partitioned iteration and log it".  It is not proof of execution and
is not described as such.  The verdicts are three-valued — ``PROVEN``,
``NOT_PROVEN``, ``CONTRADICTED`` — because a gate that can only say yes or no
forces an unproven run to be scored as either fabricated or fine, and it is
neither.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

# Files that are the runner's or the agent's own prose, never solver output.
NOT_EVIDENCE = {"trajectory.txt", "trajectory_live.txt", "task.txt",
                "RESULT.txt", "prompt.txt", "notes.txt", "notes.md",
                "README.md", "ledger.json", "plan.md"}

# Structured signatures: each must capture a number the solver computed.
# Deliberately several alternatives per code — solvers differ across versions,
# and a gate that only knows one phrasing fails honest runs.
PER_CODE_SIGNATURES = {
    "fenics": [
        r"num_dofs[^0-9]{0,12}(\d{2,})",
        r"number of (?:dofs|degrees of freedom)[^0-9]{0,12}(\d{2,})",
        r"dolfinx[^\n]{0,80}?(\d{2,})\s*(?:cells|dofs|vertices)",
        r"(?:num_cells|topology\.index_map\(\s*\d+\s*\)\.size_local)"
        r"[^0-9]{0,12}(\d{2,})",
        r"petsc[^\n]{0,60}converged[^\n]{0,40}?(\d+)\s*iterations",
        r"ksp[_ ]converged[^\n]{0,40}?(\d+)",
    ],
    "ngsolve": [
        r"ndof\s*[:=]\s*(\d{2,})",
        r"assemble[^\n]{0,40}?(\d{2,})\s*(?:elements|dofs)",
        r"ne\s*=\s*(\d{2,})[^\n]{0,40}nv\s*=\s*(\d{2,})",
        r"netgen[^\n]{0,60}?(\d{2,})\s*(?:elements|points)",
        r"call\s+solver[^\n]{0,40}?(\d+)",
    ],
    "dealii": [
        r"number of active cells\s*[:=]\s*([\d,]{2,})",
        r"number of degrees of freedom\s*[:=]\s*([\d,]{2,})",
        r"(\d+)\s+cg iterations needed",
        r"solver[^\n]{0,30}converged[^\n]{0,30}?(\d+)\s*iterations",
    ],
    "kratos": [
        r"kratos multiphysics[^\n]{0,80}?(\d+\.\d+)",
        r"(?:solving|solve) time\s*[:=]?\s*(\d+\.\d+)",
        r"::\s*solve\s*[^\n]{0,40}?(\d+)",
        r"number of (?:nodes|elements)\s*[:=]\s*(\d{2,})",
    ],
    "skfem": [
        r"(?:ndofs|n_dofs|N\s*=)\s*[:=]?\s*(\d{2,})",
        r"skfem[^\n]{0,60}?(\d{2,})\s*(?:elements|dofs)",
    ],
    "dune": [
        r"(?:dune|gridview)[^\n]{0,60}?(\d{2,})\s*(?:elements|entities|dofs)",
        r"number of (?:elements|dofs)\s*[:=]\s*(\d{2,})",
    ],
    "4C": [
        r"number of (?:nodes|elements)\s*[:=]\s*(\d{2,})",
        r"4c[^\n]{0,80}?(\d+\.\d+)\s*s",
    ],
    "febio": [
        r"normal termination[^\n]{0,80}",
        r"total elapsed time\s*[:=]\s*([\d:\.]+)",
    ],
}

READABLE_SUFFIXES = (".log", ".out", ".txt", ".json", ".csv", ".err", ".dat")
MAX_FILE_BYTES = 8_000_000


@dataclass
class EvidenceItem:
    code: str
    verdict: str              # PROVEN | NOT_PROVEN
    files: list = field(default_factory=list)
    matches: list = field(default_factory=list)
    detail: str = ""


@dataclass
class EvidenceReport:
    per_code: list = field(default_factory=list)
    coupling: dict = field(default_factory=dict)
    verdict: str = "NOT_PROVEN"
    notes: list = field(default_factory=list)

    @property
    def proven(self) -> bool:
        return self.verdict == "PROVEN"


def _candidate_files(work: Path):
    for f in sorted(work.rglob("*")):
        if not f.is_file() or f.name in NOT_EVIDENCE:
            continue
        if f.suffix.lower() not in READABLE_SUFFIXES:
            continue
        try:
            if f.stat().st_size > MAX_FILE_BYTES:
                continue
            yield f, f.read_text(errors="ignore")
        except OSError:
            continue


def code_evidence(work: Path, code: str) -> EvidenceItem:
    """Structured, number-bearing solver output for one code."""
    pats = PER_CODE_SIGNATURES.get(code)
    if not pats:
        return EvidenceItem(code, "NOT_PROVEN",
                            detail=f"no structured signature is known for "
                                   f"{code!r}; a bare name is not evidence")
    files, matches = [], []
    for f, text in _candidate_files(work):
        low = text.lower()
        for p in pats:
            m = re.search(p, low)
            if m:
                files.append(str(f.relative_to(work)))
                matches.append(f"{p} -> {m.group(0)[:80]!r}")
                break
    if files:
        return EvidenceItem(code, "PROVEN", files, matches,
                            f"{len(files)} file(s) carry structured "
                            f"{code} solver output")
    return EvidenceItem(code, "NOT_PROVEN", detail=(
        f"no file under work/ carries structured {code} solver output. "
        f"The code's NAME appearing in prose is not evidence: the task "
        f"statement names it, so the agent's notes contain it as a matter of "
        f"course."))


# ──────────────────────────────────────────────────────────────────────
RESIDUAL_FILE = re.compile(r"^(?:coupling_)?residual_level(\d+)\.csv$")


def read_residual_history(work: Path) -> dict:
    """Per-level partitioned-iteration residual histories."""
    out = {}
    for f in sorted(work.rglob("*.csv")):
        m = RESIDUAL_FILE.match(f.name)
        if not m:
            continue
        rows = []
        try:
            with open(f, newline="", errors="ignore") as fh:
                for r in csv.reader(fh):
                    if not r or not any(c.strip() for c in r):
                        continue
                    try:
                        rows.append((int(float(r[0])), float(r[1])))
                    except (ValueError, IndexError):
                        continue          # header
        except OSError:
            continue
        if rows:
            out[int(m.group(1))] = rows
    return out


def coupling_evidence(work: Path, iface_tol: float = 1e-6,
                      min_iterations: int = 3,
                      min_decrease: float = 10.0,
                      claimed_iterations: int | None = None) -> dict:
    """A partitioned iteration leaves a residual history.  Require it.

    This is the artefact a monolithic solve cannot produce at all, which is why
    it carries most of the weight now that the per-code check is understood to
    be forgeable.
    """
    hist = read_residual_history(work)
    if not hist:
        return {"verdict": "NOT_PROVEN", "levels": 0, "detail":
                "no partitioned-iteration residual history "
                "(residual_level<k>.csv). A coupled run that cannot produce "
                "one did not couple; a monolithic solve has none."}
    problems, per_level = [], {}
    for lvl, rows in sorted(hist.items()):
        vals = [v for _, v in rows]
        info = {"iterations": len(vals), "first": vals[0], "last": vals[-1]}
        if len(vals) < min_iterations:
            problems.append(f"level {lvl}: only {len(vals)} iteration(s)")
        if any((not math.isfinite(v)) or v <= 0 for v in vals):
            problems.append(f"level {lvl}: non-positive or non-finite residual")
        elif vals[0] / max(vals[-1], 1e-300) < min_decrease:
            problems.append(f"level {lvl}: residual fell only "
                            f"{vals[0] / max(vals[-1], 1e-300):.3g}x — an "
                            f"iteration that never moved did not couple")
        if vals and vals[-1] > iface_tol * 1.5:
            problems.append(f"level {lvl}: final residual {vals[-1]:.3g} "
                            f"exceeds the prescribed {iface_tol:g}")
        if len(set(vals)) == 1:
            problems.append(f"level {lvl}: residual is constant")
        per_level[lvl] = info
    if claimed_iterations is not None and hist:
        finest = max(hist)
        got = len(hist[finest])
        if got != claimed_iterations:
            problems.append(
                f"RESULT.txt claims COUPLING_ITERATIONS={claimed_iterations} "
                f"but the finest-level history has {got} rows")
    return {"verdict": "PROVEN" if not problems else "CONTRADICTED",
            "levels": len(hist), "per_level": per_level,
            "detail": "; ".join(problems) or
                      f"{len(hist)} level(s) with a well-formed, converging "
                      f"partitioned-iteration residual history"}


def assess(work: Path, codes: list, coupled: bool, iface_tol: float = 1e-6,
           claimed_iterations: int | None = None) -> EvidenceReport:
    """The whole evidence question for one run."""
    rep = EvidenceReport()
    for c in codes:
        rep.per_code.append(code_evidence(work, c))
    unproven = [e.code for e in rep.per_code if e.verdict != "PROVEN"]

    if coupled and len(codes) > 1:
        seen = {e.code: set(e.files) for e in rep.per_code}
        shared = set.intersection(*seen.values()) if all(seen.values()) else set()
        if shared and all(len(v) == 1 for v in seen.values()):
            rep.notes.append(
                f"both codes' only evidence is the same file "
                f"{sorted(shared)}: one solve writes one solver's output, so "
                f"this does not distinguish two runs from one")
        rep.coupling = coupling_evidence(
            work, iface_tol=iface_tol, claimed_iterations=claimed_iterations)

    if unproven:
        rep.verdict = "NOT_PROVEN"
        rep.notes.append(f"no structured solver output for: {', '.join(unproven)}")
    elif coupled and rep.coupling.get("verdict") != "PROVEN":
        rep.verdict = "NOT_PROVEN"
        rep.notes.append(f"coupling evidence: {rep.coupling.get('detail')}")
    else:
        rep.verdict = "PROVEN"
    return rep
