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
    r"\bB2/E[1-5]\b", r"\bheld-?out\s+(?:cell|instance)\b",
]
# NOTE: "held-out evaluation" is deliberately NOT a token here. Unlike a
# campaign identifier, the phrase legitimately appears in OASiS's own design
# prose (e.g. the verification gate explaining why an ablation flag exists).
# Flagging it would pressure a maintainer into deleting correct documentation,
# which is a worse outcome than the residual risk: an entry that named a
# specific held-out CELL or INSTANCE is still caught above, and the measured
# numbers such an entry would carry are caught by MEASURED_ORDER_PATTERNS.

# "we measured this order on this install" — the answer to a convergence study.
#
# The first four patterns were written from the contamination an audit had
# already found, and matched only that phrasing. An adversarial re-audit then
# found the same class of content passing straight through, in different words:
# `prepare_simulation('ngsolve','poisson')` was returning
# "1.669e-01 / 3.923e-02 / 8.013e-03 / 1.763e-03 (rates 2.09, 2.29, 2.18)" —
# a complete measured convergence table, handed to the agent by the tool being
# evaluated. It said "rates" rather than "observed orders" and that was the
# whole of its disguise.
#
# So these match the SHAPE of a measurement rather than one way of introducing
# it: a list of rates, a run of slash-separated error values, and a dated
# provenance stamp for one of our own runs.
MEASURED_ORDER_PATTERNS = [
    r"observed\s+orders?\s*[:=]?\s*\d+\.\d+",
    r"\bEOCs?\b[^.\n]{0,40}\d\.\d{2,}",
    r"Richardson\s+orders?\s+\d+\.\d+",
    r"verified\s+live[^.\n]{0,60}\d\.\d{2,}",
    # "(rates 2.09, 2.29, 2.18)" — a measured sequence, whatever it is called.
    r"\brates?\s+\d+\.\d{2,}\s*,\s*\d+\.\d{2,}",
    # "1.669e-01 / 3.923e-02 / 8.013e-03" — a convergence table. Three or more
    # slash-separated values in a row is our error series, not a citation.
    r"\d\.\d{2,}e[-+]\d+\s*/\s*\d\.\d{2,}e[-+]\d+\s*/\s*\d\.\d{2,}e[-+]\d+",
    # "gave orders 2.083 / 1.997 / 1.668" — the same measurement, slash-
    # separated instead of comma-separated, and introduced by "gave" rather
    # than "observed". Found still shipping after the first two patterns were
    # added, which is why these match the shape rather than the wording.
    r"\borders?\s+\d+\.\d{2,}\s*/\s*\d+\.\d{2,}",
]

# A PARAMETER TUNED FOR ONE SPECIFIC PROBLEM.
#
# This is the shape the other patterns all missed, and it is the one that gives
# most away: not a measured result but a settled ANSWER — "for the unit square
# split at x=0.5, use theta=0.42". An agent reading that skips the work the
# evaluation is asking it to do, and it compromises every future draw from the
# distribution, not just the instance it names.
#
# The discriminator is SPECIFIC GEOMETRY plus a CONCRETE VALUE. General guidance
# must survive untouched, because it is the knowledge we are trying to build:
#   * a formula is general        — "theta ~ 1/(1+rho)"
#   * a problem CLASS is general — "theta = 0.5 for problems with source terms"
#   * a named domain with numbers, next to a value, is not.
_TUNED_FOR_A_GEOMETRY = re.compile(
    r"(?:for|on)\s+(?:the\s+)?"
    r"(?:unit\s+(?:square|cube|interval)|\(0\s*,\s*[\d/.]+\)|"
    r"\d+\s*[x×]\s*\d+\s+(?:mesh|grid|domain)|"
    r"(?:square|cube|rectangle|box|domain)\s+split\s+at)"
    # The bridge must allow '.', because the geometry itself carries decimals —
    # "split at x=0.5". A first version excluded '.' to stop the match running
    # across a sentence boundary, and that made it miss the primary case while
    # catching the two easier ones. Bounded length does the same job.
    r"[^\n]{0,80}?"
    r"(?:use|set|take|choose|with)\s+\w{1,24}\s*(?:=|\bof\b|\bto\b)\s*"
    r"[-+]?\d*\.?\d+",
    re.IGNORECASE)

# TWO PATTERNS THAT WERE TRIED AND DELIBERATELY NOT KEPT
#
# A gate has to be precise or it gets switched off, and a gate that fires on
# correct content teaches people to ignore it. Both of these were measured
# against the purged branch before being rejected:
#
#   r"(?:verified empirically|live-verified|measured)\s+\d{4}-\d{2}-\d{2}"
#     — 98 hits on the ALREADY-PURGED branch. A date stamp records when we
#       checked something; it is untidy provenance, but an agent cannot read an
#       answer out of it. Making the merge gate fail in 98 places for that would
#       force a mass edit that buys no safety.
#
#   r"/home/[a-z][a-z0-9_-]*/"
#     — 16,435 hits, most of them in `src/backends/_installed_api.py`, a
#       generated manifest of where things are installed ON THIS MACHINE. That
#       file is supposed to contain local paths; that is its job. Six genuinely
#       misplaced paths in deal.II prose are a real finding, but they are a
#       code-review item, not something this pattern can isolate.
#
# What is kept above catches measured ANSWERS, and is clean on the purged
# branch — so a failure means new contamination, every time.

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


def test_no_absolute_host_paths_in_served_knowledge_payloads():
    """A `*_knowledge.json` is handed to the agent wholesale. Absolute paths
    from the machine it was written on are wrong everywhere else.

    An audit found SPARTA's `installed_build` block shipping four absolute host
    paths through the live `knowledge` tool, for all ten physics — the only
    backend doing it.

    Deliberately narrow. A broader sweep over `src/backends` was measured and
    rejected: it fires 16,435 times, overwhelmingly inside the generated install
    manifest whose whole job is to record local paths, and on ordinary comments
    that mention a path while explaining something. Search-path fallbacks in a
    backend's own `backend.py` are likewise legitimate — the code has to look
    somewhere. What is not legitimate is a path baked into the knowledge PAYLOAD
    an agent is served as fact.
    """
    rx = re.compile(r"[\"'](/home/|/media/)[a-z][a-z0-9_/-]*")
    hits = []
    for root in KNOWLEDGE_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*_knowledge.json"):
            text = p.read_text(errors="ignore")
            for m in rx.finditer(text):
                line = text[:m.start()].count("\n") + 1
                hits.append(f"{p.relative_to(REPO)}:{line}: "
                            f"{text[m.start():m.start() + 90]}")
    assert not hits, (
        "An absolute host path is served to agents as knowledge. Describe how "
        "to locate the install; do not hard-code one machine's layout.\n  "
        + "\n  ".join(hits[:8]))


def test_no_parameter_tuned_for_one_specific_problem():
    """The shape every other pattern missed, and the one that gives most away.

    Not a measured result but a settled ANSWER: "for the unit square split at
    x=0.5, use theta=0.42". An agent reading that skips the work the evaluation
    is asking it to do — and unlike a leaked instance, it compromises every
    future draw from the problem distribution, because the tuning is general.

    Calibrated in both directions, since general guidance is the knowledge we
    are trying to build and a guard that eats it would be worse than none:
    formulas ("theta ~ 1/(1+rho)"), problem CLASSES ("theta = 0.5 for problems
    with source terms"), qualitative rules ("put Dirichlet on the softer
    subdomain") and element facts ("P1 locks at nu = 0.4999") all pass.
    """
    hits = _hits(_TUNED_FOR_A_GEOMETRY.pattern)
    assert not hits, (
        "a parameter tuned for one specific geometry is shipped in "
        "agent-reachable knowledge. State the rule or the formula, not the "
        "answer for one domain.\n  " + "\n  ".join(hits[:8]))


def test_the_tuned_parameter_guard_discriminates():
    """Proof the predicate works, because an untested guard is decoration."""
    for planted in ("for the unit square split at x=0.5 use theta=0.42",
                    "on the unit cube with 16 elements per side, set "
                    "relaxation = 0.3",
                    "for the 32x32 mesh choose dt of 0.001"):
        assert _TUNED_FOR_A_GEOMETRY.search(planted), planted
    for legitimate in ("theta ~ 1/(1+rho) with rho the conductance ratio",
                       "theta = 0.5 is required for problems WITH source terms",
                       "put Dirichlet on the softer subdomain",
                       "for nearly incompressible materials use a mixed "
                       "formulation",
                       "P1 elements lock at nu = 0.4999; use P2",
                       "the unit square is the usual test domain"):
        assert not _TUNED_FOR_A_GEOMETRY.search(legitimate), legitimate


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
