#!/usr/bin/env python3
"""
Layer A→B — catalog-vs-scan diff.

Reads the per-backend capability scans under ``scripts/scan_results/``
(produced by ``scan_backend_capabilities.py``) and compares them
against the MCP's own catalog (the ``KNOWLEDGE`` dicts under
``src/backends/<backend>/generators/``).

Per backend, emits a JSON gap report with three buckets:

* **drift** — catalog mentions an entity that does NOT appear in the
  source scan. Either upstream removed/renamed the entity, the
  catalog was wrong from day one, or the scan couldn't enumerate
  the category (no-information case — flagged but not counted as
  drift).
* **coverage_gap** — source scan lists an entity that does NOT appear
  anywhere in the catalog. This is the opposite direction: a real
  upstream capability the MCP doesn't surface to the agent. Each
  entry here is a candidate catalog entry / template / pitfall.
* **shared** — entities present in both, for sanity. Big number
  here means the catalog is reasonably aligned with upstream for
  that bucket.

The matching is intentionally coarse on this first pass: we tokenize
the catalog strings on whitespace and parentheses and compare against
scan names case-insensitively after stripping a small set of common
suffixes (``Element``, ``Law``, ``Process``). Per-backend refinements
can land later as evidence accumulates from reviewing the report.

Usage:
    python scripts/diff_catalog_vs_scan.py [--backend NAME]

Outputs land under ``scripts/scan_results/diffs/<backend>.json`` plus
a top-level ``gap_summary.json``.

This script INTENTIONALLY does not fail when gaps are present — gaps
are the deliverable, not a regression. The companion test
``tests/test_catalog_vs_scan.py`` asserts only on the report's
structural validity.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_RESULTS = REPO_ROOT / "scripts" / "scan_results"
DIFFS_DIR = SCAN_RESULTS / "diffs"

# All seven backends that have a generators/ module under src/backends/.
BACKENDS = ("kratos", "fourc", "skfem", "fenics", "ngsolve", "dealii", "dune")


# ── Capability bucket → scan-field mapping ──────────────────────────────
# Maps the catalog category (the key inside each physics entry's
# KNOWLEDGE dict) to the scan field it should be compared against.
# Same category can map differently per backend — e.g. Kratos
# constitutive_laws maps to scan.constitutive_laws, but NGSolve has
# no equivalent so it stays empty.

CATEGORY_TO_SCAN_FIELD = {
    "elements": ("elements", "element_families"),
    "constitutive_laws": ("constitutive_laws",),
    "boundary_conditions": ("conditions",),
    "mesh_generators": ("mesh_generators",),
    "variables": ("variables",),
    "element_families": ("element_families",),
    # NGSolve / skfem-style alias.
    "spaces": ("element_families",),
}


@dataclass
class BackendDiff:
    backend: str
    catalog_loaded: bool = False
    scan_loaded: bool = False
    drift: dict[str, list[str]] = field(default_factory=dict)
    # Coverage gaps split into two tiers (Open-FEM-Agent §3.2 calls
    # this category "missing capability"):
    #   * truly_missing  — scan name appears NOWHERE in the catalog
    #     across any string in any physics entry. Highest-priority
    #     encoding work: the agent has no way to know this exists.
    #   * uncategorised  — scan name is referenced somewhere in the
    #     catalog (often in a pitfall string or a description) but
    #     not under the matching categorical key. The agent CAN see
    #     it but only in free-form prose, so retrieval is weaker.
    #     Lower priority — fix is to lift the mention into a
    #     structured key.
    truly_missing: dict[str, list[str]] = field(default_factory=dict)
    uncategorised: dict[str, list[str]] = field(default_factory=dict)
    shared_counts: dict[str, int] = field(default_factory=dict)
    no_info_buckets: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ── tokenizer ───────────────────────────────────────────────────────────


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")
# Suffixes we strip when comparing catalog tokens to scan tokens so
# "SmallDisplacementElement" matches "SmallDisplacement" in scans that
# elided the trailing "Element" (or vice versa). Order matters —
# longer suffixes first.
_NORM_SUFFIXES = ("Element", "Law", "Process", "FE", "FESpace")


def _tokens_from_string(s: str) -> set[str]:
    """Extract candidate entity names from a catalog string.

    A typical catalog string is
        ``"SmallDisplacementElement2D3N/4N/6N (linear, small strain)"``
    or
        ``"FE_Q (continuous Lagrange)"``.
    The tokenizer pulls the identifier-shaped substrings, which
    overshoots ("D3N", "linear", "small", "strain") but that's fine
    — the intersection with the scan's name set filters out the
    overshoot. False positives on intersection are not possible;
    overshoots just inflate the *un-matched* token set, which we
    don't report.
    """
    return {m.group(0) for m in _TOKEN_RE.finditer(s)}


def _normalise(name: str) -> str:
    n = name
    for suf in _NORM_SUFFIXES:
        if n.endswith(suf) and len(n) > len(suf):
            n = n[: -len(suf)]
            break
    return n.lower()


def _normalise_set(names: list[str] | set[str]) -> set[str]:
    return {_normalise(n) for n in names if isinstance(n, str)}


# ── catalog walk ────────────────────────────────────────────────────────


def _walk_catalog_strings(node, out_by_category: dict[str, list[str]],
                          global_pool: list[str],
                          current_category: str = "") -> None:
    """Recursively collect string entries from a KNOWLEDGE dict.

    Populates two things in parallel:
      * ``out_by_category[cat]`` — strings whose closest enclosing
        dict key matches CATEGORY_TO_SCAN_FIELD; used for drift +
        categorised-coverage analysis.
      * ``global_pool`` — every string we visit, regardless of
        enclosing key; used to detect "the catalog mentions this
        somewhere but not under the right category" (uncategorised
        gap).
    """
    if isinstance(node, dict):
        for k, v in node.items():
            cat = k if k in CATEGORY_TO_SCAN_FIELD else current_category
            _walk_catalog_strings(v, out_by_category, global_pool, cat)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _walk_catalog_strings(item, out_by_category, global_pool,
                                  current_category)
    elif isinstance(node, str):
        global_pool.append(node)
        if current_category:
            out_by_category.setdefault(current_category, []).append(node)


def _load_catalog(backend: str) -> tuple[dict[str, list[str]], list[str]]:
    """Harvest catalog strings the AGENT sees, via the MCP backend API.

    Earlier this function reached straight into each backend's
    ``generators.KNOWLEDGE`` attribute — but the seven backends use
    three different layouts (top-level dict / get_knowledge() /
    none-at-package-level), so the comparison was apples-to-oranges
    and gave deal.II / fenics / fourc empty catalogs (and therefore
    fake "coverage_gap" entries). Going through ``Backend.get_knowledge
    (physics)`` measures what the LLM actually retrieves and is
    naturally backend-agnostic.

    Returns ``{category: [string, ...]}`` aggregated across every
    physics the backend says it supports. An import failure (a
    backend whose runtime is broken — e.g. DUNE today) is caught
    by the caller, which records it as a note.
    """
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    from core.registry import get_backend, load_all_backends
    # Ensure the registry has loaded everything; idempotent.
    try:
        load_all_backends()
    except Exception:
        pass
    b = get_backend(backend)
    if b is None:
        raise RuntimeError(f"backend {backend!r} not registered")
    out: dict[str, list[str]] = {}
    global_pool: list[str] = []
    try:
        physics_iter = list(b.supported_physics())
    except Exception as e:
        raise RuntimeError(f"supported_physics() raised: {e}") from e
    for p in physics_iter:
        try:
            knowledge = b.get_knowledge(p.name)
        except Exception:
            continue
        if isinstance(knowledge, dict) and knowledge:
            _walk_catalog_strings(knowledge, out, global_pool)
    return out, global_pool


# ── diff ────────────────────────────────────────────────────────────────


def diff_one(backend: str) -> BackendDiff:
    d = BackendDiff(backend=backend)

    # Load scan
    scan_path = SCAN_RESULTS / f"{backend}.json"
    if not scan_path.is_file():
        d.notes.append(f"no scan at {scan_path.relative_to(REPO_ROOT)}")
        return d
    scan = json.loads(scan_path.read_text())
    d.scan_loaded = True

    # Load catalog
    try:
        catalog_by_category, catalog_global_pool = _load_catalog(backend)
        d.catalog_loaded = True
    except Exception as e:
        d.notes.append(
            f"catalog import failed: {type(e).__name__}: {str(e)[:200]}")
        return d

    # Build the global-pool token set once — used to split
    # coverage_gap into truly_missing vs uncategorised.
    global_token_set: set[str] = set()
    for s in catalog_global_pool:
        for tok in _tokens_from_string(s):
            if len(tok) >= 3:
                global_token_set.add(_normalise(tok))

    for catalog_category, scan_fields in CATEGORY_TO_SCAN_FIELD.items():
        catalog_strings = catalog_by_category.get(catalog_category, [])
        # Aggregate scan names from every matching scan field.
        scan_names: list[str] = []
        for fld in scan_fields:
            v = scan.get(fld)
            if isinstance(v, list):
                scan_names.extend(v)
            elif isinstance(v, dict):
                for sub in v.values():
                    if isinstance(sub, list):
                        scan_names.extend(sub)
        # Also pull from `other` — backends like deal.II park
        # solver / preconditioner / cell_types here.
        other = scan.get("other", {})
        if isinstance(other, dict) and catalog_category in other:
            v = other[catalog_category]
            if isinstance(v, list):
                scan_names.extend(v)

        if not catalog_strings and not scan_names:
            # Nothing to compare — quietly skip.
            continue
        if not scan_names:
            # Scanner couldn't enumerate this category. Don't claim
            # drift; the scan has no information.
            d.no_info_buckets.append(catalog_category)
            continue

        scan_set = _normalise_set(scan_names)

        # Catalog tokens — extract identifier substrings from every
        # catalog string, then keep those that look like scan names
        # (intersection with scan_set on normalised form).
        catalog_token_set: set[str] = set()
        catalog_orig_by_norm: dict[str, str] = {}
        for s in catalog_strings:
            for tok in _tokens_from_string(s):
                if len(tok) < 3:
                    continue
                norm = _normalise(tok)
                catalog_token_set.add(norm)
                catalog_orig_by_norm.setdefault(norm, tok)

        shared = catalog_token_set & scan_set
        coverage_gap_norms = scan_set - catalog_token_set
        # Drift: catalog tokens that look like they SHOULD be in the
        # scan but aren't. The challenge is filtering doc-prose
        # tokens ("Hood", "Lagrange", "Taylor", "MUST", "True") that
        # also happen to match the identifier shape. A token is
        # considered "name-shaped" only if it satisfies one of the
        # below — each rule reflects an actual upstream naming
        # convention rather than a guess:
        #   * contains a digit (Kratos's "2D3N", FE_P1NC, ...);
        #   * contains an internal underscore (FE_Q, fe_dgp, ...);
        #   * starts with a known canonical prefix (FE, Element,
        #     Mesh, Solver, Precondition, KM, dune, netgen);
        #   * is mixed-case with a lowercase character somewhere
        #     after the first character (camelCase or PascalCase
        #     proper, not single-word title case like "Hood").
        # Words that fail all four are treated as prose, not names.
        _CANONICAL_PREFIXES = (
            "FE", "Element", "Mesh", "Solver", "Precondition",
            "KM", "dune", "netgen", "DISPLACEMENT", "PRESSURE",
            "VELOCITY", "TEMPERATURE", "Kratos",
        )

        def _name_shaped(tok: str) -> bool:
            if any(c.isdigit() for c in tok):
                return True
            if "_" in tok[1:-1]:  # internal underscore
                return True
            if any(tok.startswith(p) for p in _CANONICAL_PREFIXES):
                return True
            # Mixed-case: has uppercase AND lowercase, AND at least
            # one case transition in the interior. Single title-
            # cased English words ("Hood", "Lagrange") have
            # uppercase only at position 0, so they fail.
            upper = [i for i, c in enumerate(tok) if c.isupper()]
            lower_after_upper = any(
                i > 0 and tok[i].islower()
                and any(j < i and tok[j].isupper() for j in range(i))
                for i in range(1, len(tok)))
            has_upper_after_lower = any(
                i > 0 and tok[i].isupper()
                and any(j < i and tok[j].islower() for j in range(i))
                for i in range(1, len(tok)))
            if upper and len(upper) >= 2 and (lower_after_upper or has_upper_after_lower):
                return True
            return False

        drift_norms = set()
        for orig in catalog_strings:
            for tok in _tokens_from_string(orig):
                if len(tok) < 4:
                    continue
                if not _name_shaped(tok):
                    continue
                norm = _normalise(tok)
                if norm not in scan_set and norm in catalog_token_set:
                    drift_norms.add(norm)

        # Build the original-name reports (use the first catalog
        # spelling we saw for drift; use the scan spelling for the
        # gap reports so the entries are actionable).
        scan_orig_by_norm = {_normalise(n): n for n in scan_names}
        if drift_norms:
            d.drift[catalog_category] = sorted(
                catalog_orig_by_norm[n] for n in drift_norms)
        if coverage_gap_norms:
            # Split coverage gaps: truly_missing (not anywhere in
            # the catalog) vs uncategorised (mentioned somewhere
            # but not under the matching category).
            truly = sorted(
                scan_orig_by_norm[n] for n in coverage_gap_norms
                if n not in global_token_set)
            uncat = sorted(
                scan_orig_by_norm[n] for n in coverage_gap_norms
                if n in global_token_set)
            if truly:
                d.truly_missing[catalog_category] = truly
            if uncat:
                d.uncategorised[catalog_category] = uncat
        d.shared_counts[catalog_category] = len(shared)

    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="",
                    help="restrict to one backend; default does all 7")
    args = ap.parse_args()

    targets = [args.backend] if args.backend else BACKENDS
    DIFFS_DIR.mkdir(parents=True, exist_ok=True)

    overall_summary: dict[str, dict] = {}
    for backend in targets:
        d = diff_one(backend)
        path = DIFFS_DIR / f"{backend}.json"
        path.write_text(json.dumps(asdict(d), indent=2))
        n_drift = sum(len(v) for v in d.drift.values())
        n_truly = sum(len(v) for v in d.truly_missing.values())
        n_uncat = sum(len(v) for v in d.uncategorised.values())
        print(f"  {backend:8s}  drift={n_drift:>4d}  "
              f"truly_missing={n_truly:>4d}  "
              f"uncategorised={n_uncat:>4d}  "
              f"shared_buckets={len(d.shared_counts):>2d}  "
              f"no_info={len(d.no_info_buckets):>2d}  "
              f"({'catalog ok' if d.catalog_loaded else 'CATALOG IMPORT FAILED'})")
        overall_summary[backend] = {
            "drift": n_drift,
            "truly_missing": n_truly,
            "uncategorised": n_uncat,
            "shared_buckets": d.shared_counts,
            "no_info_buckets": d.no_info_buckets,
            "catalog_loaded": d.catalog_loaded,
            "scan_loaded": d.scan_loaded,
            "notes": d.notes,
        }

    (DIFFS_DIR / "gap_summary.json").write_text(
        json.dumps(overall_summary, indent=2))
    print(f"\nresults under {DIFFS_DIR}/")


if __name__ == "__main__":
    main()
