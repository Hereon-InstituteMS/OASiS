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
    # Tier-0 split per critic 2026-05-31 round 2:
    #   tier0_code_symbol_matched  — Signal references ≥1 real code
    #     symbol (deal.II class, exception, library identifier).
    #     This is the HONEST Tier-0 gate.
    #   tier0_domain_names_matched — Signal references ≥1 textbook
    #     concept (Stokes, Newton, Turek). Decorative; NOT
    #     sufficient on its own for Tier 0.
    #   tier0_passed                — true iff tier0_code_symbol_matched
    tier0_passed: bool = False
    tier0_code_symbol_matched: list = field(default_factory=list)
    tier0_domain_names_matched: list = field(default_factory=list)
    tier0_entities_matched: list = field(default_factory=list)  # back-compat union
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


def _load_entity_split(backend: str) -> tuple[set[str], set[str]]:
    """Return (code_symbols, domain_names).

    Senior-AI-scientist critic 2026-05-31 round 2: the prior
    single-set design let domain words ("Stokes", "Poisson",
    "Newton") count as Tier-0 hits, so every elasticity pitfall
    scored on "Poisson ratio" and every flow pitfall on "Stokes".
    That made Tier-0 a near-tautology and inflated the headline
    pass rate.

    Splitting them:
      * **code_symbols** — names a programmer would `grep -r` for
        in deal.II source: classes (FE_Q, SolverCG, GridIn,
        DataOut, VectorTools), exceptions (ExcMessage,
        ExcDimensionMismatch), function names (compress,
        condense), and library-internal identifiers (KINSOL,
        SLEPc, EPS_TARGET_REAL). A Tier-0 pass requires at least
        one match here.
      * **domain_names** — words a textbook would use: physics
        names (Stokes, Maxwell, Laplace), method names (Newton,
        Newmark, BDF2, Crank-Nicolson), benchmark people
        (Turek, Ghia, Kirsch). Decorative; NEVER sufficient on
        their own for Tier 0.
    """
    code_symbols: set[str] = set()
    domain_names: set[str] = set()

    if backend == "dealii":
        try:
            mod = importlib.import_module(
                "backends.dealii.element_catalog")
            code_symbols.update(getattr(mod, "ELEMENT_NAMES", set()))
            code_symbols.update(getattr(mod, "MESH_GENERATOR_NAMES", set()))
        except ImportError:
            pass

    code_symbols.update({
        # ── Krylov solvers ─────────────────────────────────────
        "SolverCG", "SolverGMRES", "SolverFGMRES", "SolverMinRes",
        "SolverBiCGStab", "SolverDirect", "SolverFIRE",
        "SolverQMRS", "SolverRichardson", "SolverControl",
        # ── External-package solvers / linear-algebra layers ─
        "SUNDIALS", "KINSOL", "ARKode", "SLEPc", "PETSc",
        "Trilinos", "TrilinosWrappers", "MUMPS", "UMFPACK",
        "PETScWrappers", "EPS", "EPS_TARGET_REAL",
        # ── MPI / runtime errors ───────────────────────────────
        "MPI_Init", "MPI_Comm_size", "MPI_ERR_COMM",
        "MPI_InitFinalize",
        # ── Mesh / DoF infrastructure ──────────────────────────
        "GridGenerator", "GridIn", "GridOut", "GridTools",
        "Triangulation", "DoFHandler", "DoFRenumbering",
        "AffineConstraints", "ConstraintMatrix", "MappingQ",
        "MappingQEulerian", "SolutionTransfer", "MatrixFree",
        "FEValuesExtractors", "FEValues", "FEValuesBase",
        "FEInterfaceValues", "FEEvaluation", "FECollection",
        "FESeries", "QCollection", "QGauss", "MeshWorker",
        "DataOut", "DataOutInterface", "DataOutFaces",
        "KellyErrorEstimator", "Quadrature",
        "VectorTools", "TimerOutput", "VectorOperation",
        "SparseMatrix", "BlockSparseMatrix", "BlockVector",
        "SparsityPattern", "BlockSparsityPattern",
        "SparsityTools", "IndexSet", "DoFTools",
        "VectorizedArray",
        # ── Common exception classes ──────────────────────────
        "ExcMessage", "ExcDimensionMismatch", "ExcIndexRange",
        "ExcNotImplemented", "ExcInternalError",
        "ExcInitializeNotInitialized", "ExcSolverFail",
        "ExcInvalidIterator",
        # ── Preconditioners / smoothers ────────────────────────
        "PreconditionAMG", "PreconditionSSOR",
        "PreconditionJacobi", "PreconditionChebyshev",
        "PreconditionILU", "PreconditionBlock",
        "PreconditionBlockSSOR", "PreconditionIdentity",
        "MGSmootherRelaxation",
        # ── External library output markers a critic can grep ─
        "BoomerAMG",  # HYPRE class name, real C++ symbol
        "p4est",
        # ── Differentiation / AD API ──────────────────────────
        "Differentiation",
        # ── Output ─────────────────────────────────────────────
        "write_vtu", "write_vtu_with_pvtu_record",
        "write_pvd_record",
    })

    domain_names.update({
        # ── Numerical method names (textbook concepts, not
        #    classes — e.g. "Newton" is a method, but in code
        #    you see SolverControl + iterate loops, not a
        #    "Newton" class).
        "Newton", "Picard", "Oseen", "Jacobi",
        "Crank-Nicolson", "Newmark", "BDF2", "Theta",
        "RungeKutta", "Heun", "Euler",
        # ── Stable pairs / stabilisations / concepts ──────────
        "Taylor-Hood", "MINI", "Vanka", "SUPG", "GLS", "VMS",
        "PML", "Sommerfeld", "Nitsche", "AMG", "SSOR", "SOR",
        "ILU", "ILUT",  # algorithm names sit between code/domain
        # ── Benchmark people / domain-specific math objects ──
        "Hankel", "Bessel", "Turek", "Schäfer-Turek",
        "Schaefer-Turek", "Ghia", "Kirsch", "Euler-Bernoulli",
        "Boussinesq", "Hermite",
        # ── Physics / PDE / math concepts ─────────────────────
        "Stokes", "Navier", "Maxwell", "Helmholtz",
        "Laplace", "Poisson", "Lagrange", "LBB",
        # ── Vocabulary in pitfall prose that's NOT a code symbol
        "breakdown", "convergence", "checkerboard", "Mesh",
        "ParaView",  # tool name, not a class
    })

    return code_symbols, domain_names


def _load_canonical_entities(backend: str) -> set[str]:
    """Back-compat alias — union of code-symbols and domain-names."""
    c, d = _load_entity_split(backend)
    return c | d


def _tier0_check(signal: str, code_symbols: set[str],
                 domain_names: set[str],
                 result: SignalVerification) -> None:
    """Tier 0: Signal references ≥1 CODE SYMBOL.

    Per critic 2026-05-31 round 2: domain names (Stokes, Poisson,
    Newton) are NOT sufficient on their own. The honest Tier-0
    gate is "Signal references at least one real deal.II code
    symbol that a post-execution critic could grep for in actual
    output". Domain names are recorded separately as soft
    indicators but do not flip tier0_passed.
    """
    code_matched: list[str] = []
    domain_matched: list[str] = []
    for tok in _IDENT_RE.findall(signal):
        if len(tok) < 3:
            continue
        if tok in code_symbols:
            code_matched.append(tok)
        elif tok in domain_names:
            domain_matched.append(tok)
    result.tier0_code_symbol_matched = sorted(set(code_matched))
    result.tier0_domain_names_matched = sorted(set(domain_matched))
    result.tier0_entities_matched = sorted(
        set(code_matched) | set(domain_matched))
    result.tier0_passed = len(code_matched) >= 1


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
    code_symbols, domain_names = _load_entity_split(backend)
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
        _tier0_check(sig, code_symbols, domain_names, result)
        _tier1_check(sig, result)
        # Tier 2 — load operational results from the fixture
        # runner output (if present).
        result.tier2_status = _tier2_lookup(backend, physics, idx)
        result.tier2_passed = (result.tier2_status == "passed")
        results.append(result)
    return results


def _tier2_lookup(backend: str, physics: str, idx: int) -> str:
    """Look up Tier-2 fixture result for one pitfall.

    Reads ``scripts/scan_results/tier2_results.json`` if present.
    Returns one of: ``passed`` / ``failed`` / ``harness_pending`` /
    ``not_attempted`` (when there is no fixture for this pitfall).
    """
    path = OUTPUT.parent / "tier2_results.json"
    if not path.is_file():
        return "harness_pending"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return "harness_pending"
    key = f"{backend}::{physics}::{idx}"
    entry = data.get(key)
    if not isinstance(entry, dict):
        return "not_attempted"
    return str(entry.get("status", "harness_pending"))


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
        "tier0_code_symbol_only": sum(
            1 for r in results
            if r.tier0_code_symbol_matched
            and not r.tier0_domain_names_matched),
        "tier0_with_domain_decoration": sum(
            1 for r in results
            if r.tier0_code_symbol_matched
            and r.tier0_domain_names_matched),
        "tier0_domain_names_only": sum(
            1 for r in results
            if r.tier0_domain_names_matched
            and not r.tier0_code_symbol_matched),
        "tier1_passed": sum(1 for r in results if r.tier1_passed),
        "tier0_and_1_passed": sum(
            1 for r in results
            if r.tier0_passed and r.tier1_passed),
        "tier2_passed": sum(1 for r in results if r.tier2_passed),
        "tier2_attempted": sum(
            1 for r in results
            if r.tier2_status in ("passed", "failed")),
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
