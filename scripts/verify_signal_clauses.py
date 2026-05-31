#!/usr/bin/env python3
"""
Signal-clause verification harness.

The Open-FEM-Agent paper (§3.2 / Table 1) sells the pitfall DB as
the distinctive value-add. Each pitfall entry in
``backends.<backend>.generators.*.KNOWLEDGE['pitfalls']`` ships a
``Signal:`` clause stating the observable symptom — the string the
post-execution critic is supposed to match against actual error /
result text.

The senior-AI-scientist critic (2026-05-31) flagged the
unfalsified state of these signals as the second-largest risk:
"every encoded pitfall is a claim the project cannot defend."
This harness operationalises the verification in three tiers:

  * **Tier 0** — structural. The Signal text references at least
    one entity (class name, function name, error class) that is
    real and known to the canonical catalogs. Cheap; catches
    typos like "FE_Simplex" (missing the trailing P).
  * **Tier 1** — semantic. The Signal text uses observable-symptom
    vocabulary: "report", "error", "diverges", "converges to",
    "exits", "raises", "warns", "stalls", "oscillates", a
    quoted-error pattern, or a numerical observation. Catches
    vague non-actionable signals.
  * **Tier 2** — operational. An intentional-failure regression
    fixture compiles + runs and the Signal text appears in the
    captured stderr / stdout. This is the strongest tier but
    each fixture is hand-written and per-backend, so the work is
    multi-week. The harness here records which Signal entries
    have Tier-2 verification and which are still on the
    Tier-0+1 floor.

Usage:
    python scripts/verify_signal_clauses.py
    python scripts/verify_signal_clauses.py --backend dealii
    python scripts/verify_signal_clauses.py --tier 2  # only run operational

Output: ``scripts/scan_results/signal_verification.json`` with
per-pitfall verification status.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "scripts" / "scan_results" / "signal_verification.json"


# Observable-symptom vocabulary — Tier-1 semantic check looks for
# at least one of these in the Signal: clause.
OBSERVABLE_VOCAB = (
    "report", "error", "exit", "raise", "warn", "stall",
    "converge", "diverge", "oscillat", "break", "crash",
    "abort", "missing", "undefined", "differs", "drop", "drift",
    "appears", "shows", "matches", "wrong", "zero", "nan",
    "checkerboard", "pattern", "amplitude", "value",
    "grows", "shrinks", "exceeds", "below", "above",
    "larger", "smaller", "slower", "faster",
    "wall-time", "wall time", "iteration", "iterations",
    "reaches", "returns",
)


@dataclass
class SignalVerification:
    backend: str
    physics: str
    pitfall_index: int                     # 0-based index into pitfalls list
    pitfall_category: str                  # [Syntax]/[Physics]/...
    signal_text: str                       # the Signal: clause itself
    tier0_passed: bool = False
    tier0_entities_matched: list = field(default_factory=list)
    tier1_passed: bool = False
    tier1_vocab_hits: list = field(default_factory=list)
    tier2_passed: bool = False
    tier2_status: str = "not_attempted"    # not_attempted / harness_pending / passed / failed
    notes: list = field(default_factory=list)


_PITFALL_PREFIX_RE = re.compile(
    r"^\s*\[(Syntax|Physics|Numerical|API|Integration)\]")
_SIGNAL_RE = re.compile(r"\bSignal:\s*(.+?)$", re.IGNORECASE | re.DOTALL)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def _split_pitfall(text: str) -> tuple[str | None, str | None]:
    """Return (category, signal_text) — both None if not parseable."""
    m_cat = _PITFALL_PREFIX_RE.match(text)
    cat = m_cat.group(1) if m_cat else None
    m_sig = _SIGNAL_RE.search(text)
    sig = m_sig.group(1).strip() if m_sig else None
    return cat, sig


def _load_canonical_entities(backend: str) -> set[str]:
    """Set of names a Signal can plausibly reference: catalog
    element + mesh-generator names, plus a small set of well-known
    error classes / external symbols (SolverCG breakdown, EPS_,
    MPI_ERR_*, ...).
    """
    out: set[str] = set()
    if backend == "dealii":
        try:
            mod = importlib.import_module(
                "backends.dealii.element_catalog")
            out.update(getattr(mod, "ELEMENT_NAMES", set()))
            out.update(getattr(mod, "MESH_GENERATOR_NAMES", set()))
        except ImportError:
            pass
    # Common cross-backend error/observable names we treat as
    # canonical for the Tier-0 entity check. This set is the
    # "namespace of grepable strings" — anything a
    # post-execution critic can search for in actual deal.II
    # output, plus the well-known method/algorithm names that
    # identify which mathematical machinery the Signal is talking
    # about.
    out.update({
        # ── Krylov solvers ─────────────────────────────────────
        "SolverCG", "SolverGMRES", "SolverFGMRES", "SolverMinRes",
        "SolverBiCGStab", "SolverDirect", "SolverFIRE",
        "SolverQMRS", "SolverRichardson", "SolverControl",
        # ── External-package solvers / linear-algebra layers ─
        "SUNDIALS", "KINSOL", "ARKode", "SLEPc", "PETSc",
        "Trilinos", "TrilinosWrappers", "MUMPS", "UMFPACK",
        "PETScWrappers", "EPS",
        # ── MPI / runtime errors ───────────────────────────────
        "MPI_Init", "MPI_Comm_size", "MPI_ERR_COMM",
        "MPI_InitFinalize", "Utilities",
        # ── Mesh / DoF infrastructure ──────────────────────────
        "GridGenerator", "GridIn", "GridOut", "GridTools",
        "Triangulation", "DoFHandler", "DoFRenumbering",
        "AffineConstraints", "ConstraintMatrix", "MappingQ",
        "MappingQEulerian", "SolutionTransfer", "MatrixFree",
        "FEValuesExtractors", "FEValues", "FEValuesBase",
        "FEInterfaceValues", "FEEvaluation",
        "DataOut", "DataOutInterface", "ParaView",
        "KellyErrorEstimator", "Quadrature",
        "VectorTools", "TimerOutput", "VectorOperation",
        "SparseMatrix", "BlockSparseMatrix",
        "SparsityPattern", "BlockSparsityPattern",
        "Turek", "Ghia",  # ASCII shortforms of Schäfer-Turek / Ghia
        "Stokes", "Navier", "Maxwell", "Helmholtz",
        "Laplace", "Poisson",  # method names users would grep for
        # ── Common exception classes ──────────────────────────
        "ExcMessage", "ExcDimensionMismatch", "ExcIndexRange",
        "ExcNotImplemented", "ExcInternalError",
        "ExcInitializeNotInitialized",
        # ── Preconditioners / smoothers ────────────────────────
        "AMG", "BoomerAMG", "SSOR", "SOR", "ILU", "ILUT",
        "PreconditionAMG", "PreconditionSSOR",
        "PreconditionJacobi", "PreconditionChebyshev",
        "PreconditionILU", "PreconditionBlock",
        "MGSmootherRelaxation",
        # ── Numerical methods / discretisation names ──────────
        "Newton", "Picard", "Oseen", "Jacobi", "Crank-Nicolson",
        "Newmark", "BDF2", "Theta", "RungeKutta", "Heun",
        # ── Stable element pairs / stabilisations ─────────────
        "Taylor-Hood", "MINI", "Vanka", "SUPG", "GLS", "VMS",
        "PML", "Sommerfeld", "Nitsche",
        # ── Catalog-internal mathematical observables ─────────
        "Hankel", "Bessel", "Schäfer-Turek", "Schaefer-Turek",
        "Ghia", "Kirsch", "Euler-Bernoulli", "Boussinesq",
        "LBB",  # Ladyzhenskaya-Babuška-Brezzi
        "Lagrange",
        # ── deal.II output-stream markers the critic can grep ─
        "breakdown", "convergence", "checkerboard", "Mesh",
    })
    return out


def _tier0_check(signal: str, entities: set[str],
                 result: SignalVerification) -> None:
    """Tier 0: Signal references at least one canonical entity."""
    matched: list[str] = []
    for tok in _IDENT_RE.findall(signal):
        if len(tok) < 3:
            continue
        if tok in entities:
            matched.append(tok)
    result.tier0_entities_matched = sorted(set(matched))
    result.tier0_passed = len(matched) >= 1


def _tier1_check(signal: str, result: SignalVerification) -> None:
    """Tier 1: Signal uses observable-symptom vocabulary."""
    low = signal.lower()
    hits = sorted({w for w in OBSERVABLE_VOCAB if w in low})
    # Plus quoted error fragments (anything inside backticks or
    # double quotes within the signal counts as a concrete error
    # observable).
    if re.search(r"`[^`]+`", signal) or re.search(r"'[^']+'", signal):
        hits.append("quoted-error-fragment")
    # Numerical-comparison phrases ("by 10+%", "differs by", "off by")
    if re.search(r"\bdiffers?\b|\boff by\b|\bby \d", low):
        hits.append("numerical-comparison")
    result.tier1_vocab_hits = sorted(set(hits))
    result.tier1_passed = len(hits) >= 1


def _harvest_pitfalls(backend: str) -> list[tuple[str, int, str]]:
    """Walk a backend's KNOWLEDGE dicts and return
    (physics, index, pitfall_text) triples.

    Goes through the backend API so we exercise the same path the
    agent uses — same rationale as the diff tool's loader.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from core.registry import load_all_backends, get_backend
    try:
        load_all_backends()
    except Exception:
        pass
    b = get_backend(backend)
    if b is None:
        return []
    out: list[tuple[str, int, str]] = []
    try:
        physics_iter = list(b.supported_physics())
    except Exception:
        return out
    for p in physics_iter:
        try:
            knowledge = b.get_knowledge(p.name)
        except Exception:
            continue
        if not isinstance(knowledge, dict):
            continue
        pitfalls = knowledge.get("pitfalls", [])
        if not isinstance(pitfalls, list):
            continue
        for i, entry in enumerate(pitfalls):
            if isinstance(entry, str):
                out.append((p.name, i, entry))
    return out


def verify_backend(backend: str) -> list[SignalVerification]:
    entities = _load_canonical_entities(backend)
    results: list[SignalVerification] = []
    for physics, idx, text in _harvest_pitfalls(backend):
        cat, sig = _split_pitfall(text)
        result = SignalVerification(
            backend=backend, physics=physics, pitfall_index=idx,
            pitfall_category=cat or "(no-prefix)",
            signal_text=sig or "",
        )
        if not cat:
            result.notes.append(
                "pitfall lacks [Category] prefix (PR #26 Table-1 "
                "convention) — Tier 0/1 not applicable")
            results.append(result)
            continue
        if not sig:
            result.notes.append(
                "pitfall lacks `Signal:` clause — cannot match in "
                "post-execution critic; the entry is descriptive, "
                "not detection-actionable")
            results.append(result)
            continue
        _tier0_check(sig, entities, result)
        _tier1_check(sig, result)
        # Tier 2 is fixture-based; record harness-pending for now.
        result.tier2_status = "harness_pending"
        results.append(result)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="dealii",
                    help="restrict to one backend; default: dealii")
    args = ap.parse_args()

    results = verify_backend(args.backend)

    totals = {
        "n_pitfalls": len(results),
        "with_category_prefix": sum(
            1 for r in results
            if r.pitfall_category != "(no-prefix)"),
        "with_signal_clause": sum(
            1 for r in results if r.signal_text),
        "tier0_passed": sum(1 for r in results if r.tier0_passed),
        "tier1_passed": sum(1 for r in results if r.tier1_passed),
        "tier0_and_1_passed": sum(
            1 for r in results
            if r.tier0_passed and r.tier1_passed),
        "tier2_passed": sum(1 for r in results if r.tier2_passed),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "backend": args.backend,
        "totals": totals,
        "results": [asdict(r) for r in results],
    }, indent=2))

    print(f"\n{args.backend} signal verification:")
    for k, v in totals.items():
        if k == "n_pitfalls":
            print(f"  {k:30s} {v:>4d}")
        elif k.startswith("tier"):
            pct = (100.0 * v / totals["n_pitfalls"]
                   if totals["n_pitfalls"] else 0)
            print(f"  {k:30s} {v:>4d} / {totals['n_pitfalls']} "
                  f"({pct:.0f}%)")
        else:
            print(f"  {k:30s} {v:>4d}")
    print(f"\nFull report: {OUTPUT.relative_to(REPO_ROOT)}")

    # Per-result diagnostics for the entries that didn't pass
    # Tier 0 or 1 — the Signal text needs a small fix.
    bad = [r for r in results
           if r.signal_text
           and not (r.tier0_passed and r.tier1_passed)]
    if bad:
        print(f"\n{len(bad)} Signal clauses to review:")
        for r in bad[:10]:
            print(f"  {r.physics}#{r.pitfall_index} "
                  f"[{r.pitfall_category}] tier0={r.tier0_passed} "
                  f"tier1={r.tier1_passed}")
            print(f"    signal: {r.signal_text[:120]!r}")
        if len(bad) > 10:
            print(f"  ... +{len(bad) - 10} more")


if __name__ == "__main__":
    main()
