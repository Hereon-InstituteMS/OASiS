#!/usr/bin/env python3
"""Batch-ingest saved session journals into community-knowledge candidates.

Reads one or more journals saved by the MCP server (``data/sessions/
session_*.json``), runs the post-session analyzer over each, dedupes
across the batch, and either prints the resulting candidates or saves
them to ``data/community_knowledge/pending/`` for later inclusion in
the catalog (the same path the ``session_insights`` MCP tool writes
to during a live session).

Usage:
    python scripts/ingest_session.py <path>             # show candidates only
    python scripts/ingest_session.py <path> --approve   # also save to pending/
    python scripts/ingest_session.py <path> --solver fourc   # filter by solver

<path> may be a single ``session_*.json`` file or a directory; the
directory case scans for ``session_*.json`` recursively-via-glob.

This is the offline equivalent of calling ``session_insights('ingest',
path=...)`` from inside Claude Code -- useful for community
contributors who have a folder of journals from past usage and want
to triage them in one go.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.session_analyzer import (  # noqa: E402
    CandidateKnowledge,
    analyze_journal_file,
    filter_against_existing,
    format_candidates,
)


_RETRY_SUFFIX_RE = re.compile(r"\s*\(retry \d+\)\s*$", re.IGNORECASE)


def _normalise_title(title: str) -> str:
    """Collapse a candidate title into a dedup-friendly form.

    Strips leading/trailing whitespace, lowercases, removes a trailing
    ``(retry N)`` suffix and collapses internal whitespace.  Matches the
    spirit of the analyzer's intra-file fuzzy dedup (SequenceMatcher
    >0.8) without pulling SequenceMatcher into the cross-file path.
    """
    cleaned = _RETRY_SUFFIX_RE.sub("", title)
    return " ".join(cleaned.lower().split())


def _gather(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("session_*.json"))
    if path.is_file():
        return [path]
    return []


def _save_batch(candidates: list, out_dir: Path) -> Path:
    """Write the deduped batch to ``community_knowledge/pending/``.

    The filename includes a fresh uuid so multiple batch ingestions
    accumulate rather than overwrite each other.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_id = uuid.uuid4().hex[:12]
    out = out_dir / f"session_batch_{batch_id}.json"
    out.write_text(
        json.dumps([c.to_dict() for c in candidates], indent=2, default=str)
    )
    return out


def main() -> int:
    description = (__doc__ or "").splitlines()[0] if __doc__ else ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "path", type=Path,
        help="File or directory of session_*.json journals to ingest.",
    )
    parser.add_argument(
        "--approve", action="store_true",
        help="Save the deduped candidate batch to "
             "data/community_knowledge/pending/ "
             "(otherwise this is a dry run).",
    )
    parser.add_argument(
        "--solver", default="",
        help="Only keep candidates for this solver (e.g. 'fourc').",
    )
    args = parser.parse_args()

    sources = _gather(args.path)
    if not sources:
        print(f"No session_*.json found at {args.path}", file=sys.stderr)
        return 1

    all_candidates = []
    errors = []
    for s in sources:
        try:
            all_candidates.extend(analyze_journal_file(s))
        except Exception as e:  # noqa: BLE001
            errors.append((s, e))

    # De-duplicate on a normalised (category, solver, title) key.  The
    # in-file analyzer already runs fuzzy dedup (SequenceMatcher >0.8),
    # but cross-file candidates with cosmetic differences (trailing
    # whitespace, capitalisation, retry-count suffix) need to be
    # collapsed too -- otherwise the contributor sees N near-identical
    # entries for the same root cause.
    best: dict[tuple[str, str, str], CandidateKnowledge] = {}
    for c in all_candidates:
        key = (
            c.category.strip().lower(),
            (c.solver or "").strip().lower(),
            _normalise_title(c.title),
        )
        if key not in best or c.confidence > best[key].confidence:
            best[key] = c
    deduped = list(best.values())

    if args.solver:
        deduped = [c for c in deduped if c.solver == args.solver]

    # Cross-check against the catalog's existing pitfalls so we don't
    # propose entries the upstream knowledge base already covers.
    existing_pitfalls = _existing_pitfalls(args.solver)
    novel = filter_against_existing(deduped, existing_pitfalls)

    print(
        f"Ingested {len(sources)} journal file(s); "
        f"{len(all_candidates)} raw -> {len(deduped)} deduped -> "
        f"{len(novel)} novel after catalog filter."
    )
    if errors:
        print("Errors:")
        for s, e in errors:
            print(f"  {s.name}: {e}")

    if not novel:
        return 0

    print()
    print(format_candidates(novel))

    if args.approve:
        out_dir = REPO_ROOT / "data" / "community_knowledge" / "pending"
        out = _save_batch(novel, out_dir)
        print(f"\nSaved {len(novel)} candidate(s) to: {out.relative_to(REPO_ROOT)}")
        print("Open a PR adding this file to share your findings upstream.")
    else:
        print("\n(Dry run.  Re-run with --approve to save to pending/.)")

    return 0


def _existing_pitfalls(solver: str) -> list[str]:
    """Lightweight version of consolidated._collect_existing_pitfalls()
    that avoids importing the MCP server module (which spins up its
    FastMCP context).  Walks the registered backends directly.
    """
    from core.registry import load_all_backends, _backends
    pitfalls: list[str] = []
    try:
        load_all_backends()
    except Exception:
        return pitfalls
    for b in _backends.values():
        if solver and b.name() != solver:
            continue
        try:
            for p in b.supported_physics():
                k = b.get_knowledge(p.name)
                if k and isinstance(k, dict) and "pitfalls" in k:
                    for pit in k["pitfalls"]:
                        if isinstance(pit, str):
                            pitfalls.append(pit)
                        elif isinstance(pit, dict) and "text" in pit:
                            pitfalls.append(pit["text"])
        except Exception:
            continue
    return pitfalls


if __name__ == "__main__":
    raise SystemExit(main())
