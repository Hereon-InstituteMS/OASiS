"""Guard: OASiS's knowledge must describe the CODE, never the ANSWER.

An audit found evaluation-specific content reachable through the very tool the
server instructions tell every agent to call first. Confirmed by execution:

  * campaign identifiers inside agent-readable pitfalls — "the E5 prototype
    run (2026-08-01)", "(T14 campaign fix.)";
  * MEASURED convergence orders shipped as knowledge — "L2 EOCs
    1.984/1.996/1.999", "observed orders 1.93 / 1.99 / 1.99", "Richardson
    orders 2.018/2.004";
  * pre-solved coupled cases — an exact solution and a tuned relaxation answer
    for the same geometry the evaluation uses.

Knowledge like that makes the evaluation measure itself: the agent can read the
answer to a convergence study out of the tool being assessed.

THE RULE THIS TEST ENFORCES
    Knowledge may state how the code behaves, how it fails, what a keyword
    means in the installed version, which element locks, which default changed.
    It may NOT state the answer to a problem: a measured convergence order we
    produced, an exact solution, a tuned parameter for a specific setup, or any
    reference to our own evaluation campaign.

Published literature benchmarks WITH a citation are expertise, not answers, and
are deliberately allowed.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Where knowledge an agent can reach actually lives.
KNOWLEDGE_ROOTS = [REPO / "src" / "backends", REPO / "src" / "tools",
                   REPO / "data"]

# Identifiers of our own evaluation. None of these may appear in knowledge.
CAMPAIGN_TOKENS = [
    r"\bE[1-5]\s+prototype\b", r"\bT1[0-9]\s+campaign\b", r"\bT[0-9]+\s+campaign\b",
    r"\bcampaign\s+fix\b", r"\bevaluation\s+campaign\b",
    r"\bB2/E[1-5]\b", r"\bheld-?out\s+(?:cell|instance|evaluation)\b",
]

# "we measured this order on this install" — the answer to a convergence study.
MEASURED_ORDER_PATTERNS = [
    r"observed\s+orders?\s*[:=]?\s*\d+\.\d+",
    r"\bEOCs?\b[^.\n]{0,40}\d\.\d{2,}",
    r"Richardson\s+orders?\s+\d+\.\d+",
    r"verified\s+live[^.\n]{0,60}\d\.\d{2,}",
]

ALLOWED_SUFFIXES = {".py", ".json", ".md", ".txt", ".yaml", ".yml"}
# Tests and fixtures legitimately discuss our campaigns; knowledge does not.
SKIP_PARTS = {"tests", "scripts", "benchmarks", "__pycache__", ".git",
              "data/sessions", "postmortems"}


def _knowledge_files():
    for root in KNOWLEDGE_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix not in ALLOWED_SUFFIXES:
                continue
            rel = p.relative_to(REPO).as_posix()
            if any(part in rel.split("/") for part in SKIP_PARTS):
                continue
            if "data/sessions" in rel or "postmortems" in rel:
                continue
            yield p


def _hits(pattern: str):
    rx = re.compile(pattern, re.IGNORECASE)
    found = []
    for p in _knowledge_files():
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for m in rx.finditer(text):
            line = text[:m.start()].count("\n") + 1
            snippet = text[max(0, m.start() - 60):m.end() + 60].replace("\n", " ")
            found.append(f"{p.relative_to(REPO)}:{line}: …{snippet.strip()}…")
    return found


@pytest.mark.parametrize("pattern", CAMPAIGN_TOKENS)
def test_no_campaign_identifier_in_knowledge(pattern):
    """Knowledge must not reference our own evaluation."""
    hits = _hits(pattern)
    assert not hits, (
        "Evaluation-campaign reference found in agent-reachable knowledge.\n"
        "Knowledge describes the code, never our tests.\n  " + "\n  ".join(hits[:8]))


@pytest.mark.parametrize("pattern", MEASURED_ORDER_PATTERNS)
def test_no_measured_convergence_orders_in_knowledge(pattern):
    """The worst contamination: the answer to a convergence study, shipped.

    A general statement of a method's THEORETICAL order is fine — that is
    textbook code knowledge. Our measurements are not.
    """
    hits = _hits(pattern)
    assert not hits, (
        "Measured convergence order found in agent-reachable knowledge.\n"
        "An agent can read the answer to a convergence study out of the tool "
        "being evaluated.\n  " + "\n  ".join(hits[:8]))


# Prose ABOUT exact solutions is legitimate and must not be flagged: the
# mesh-independence tool's docstring correctly says it is "for problems WITHOUT
# an exact solution", and a pitfall may warn that no closed form exists. Only a
# formula being SUPPLIED is contamination, so require a right-hand side that
# actually defines one (an equals/colon followed by an expression in x, t or a
# number) and exclude negating context.
_NEGATING = re.compile(
    r"(?:without|no|lacks?|absent|not\s+have|unavailable|if\s+an?)\s+"
    r"(?:an?\s+)?(?:known\s+|analytical\s+|closed[- ]form\s+)?exact\s+solution",
    re.IGNORECASE)


def test_no_exact_solution_shipped_for_a_coupled_case():
    """Pre-solved cases were reachable via knowledge(topic='coupling'):
    'Exact solution: T(x) = 100*(1-x)' with a per-solver error table."""
    rx = re.compile(r"[Ee]xact\s+solution\s*[:=]\s*([A-Za-z_]\w*\s*\([^)]*\)\s*=|"
                    r"[-+]?\d|[A-Za-z_]\w*\s*[-+*/^])")
    hits = []
    for f in _knowledge_files():
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        for m in rx.finditer(text):
            ctx = text[max(0, m.start() - 120):m.end() + 40]
            if _NEGATING.search(ctx):
                continue                      # "for problems WITHOUT an exact solution"
            line = text[:m.start()].count("\n") + 1
            hits.append(f"{f.relative_to(REPO)}:{line}: …{ctx.replace(chr(10),' ').strip()[:150]}…")
    assert not hits, (
        "An exact solution is shipped in knowledge; the agent can read the "
        "answer instead of computing it.\n  " + "\n  ".join(hits[:8]))
