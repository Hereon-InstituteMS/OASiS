"""The leakage gate: does this prompt disclose the solution?

Run by the auditor (which may read keys), never by the agent.  It answers one
question about a blind task text:

    can a reader obtain the solution field in closed form from this text?

Five independent rules, each with its own severity.  The rules are deliberately
**symbolic and numeric**, not lexical pattern-matching over mathematical
function names — that distinction is the whole design.

The trap this design exists to avoid
------------------------------------
An earlier attempt flagged ``sin(pi ...)`` wherever it appeared, and so fired
on the SOURCE TERM.  The source term is derived from the hidden solution and
will contain trigonometric and exponential functions in essentially every
manufactured problem; the agent is *supposed* to have it.  Flagging that makes
the harness unusable, and worse, it trains the operator to ignore the gate.

So this gate never asks "does a sine appear".  It asks:

* ``L1`` does the text use language that names or requests an exact solution?
* ``L2`` does any parsable expression in the text equal, or scale to, the
  hidden solution?  (Symbolic equality — catches a leak written in any
  algebraically equivalent form, which a substring test cannot.)
* ``L3`` does any number in the text match a true probe value?
* ``L4`` does the published source term contain ``c * u_exact`` as a surviving
  sub-sum, so the solution can be read off it?  (This is a *design* verdict on
  the manufactured field, not a drafting mistake in the prose.)
* ``L5`` does the text ask the agent to compute an error against a reference it
  is not given — which both leaks the existence of one and makes the agent's
  own number ungradeable?

``L4`` is the rule that catches the two real leaks in the pre-existing campaign
(``B1``, ``D3``); see :func:`blind_eval.derive.embedded_multiple`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import sympy as sp

from .derive import embedded_multiple, is_proportional

SYMS = {c: sp.Symbol(c, real=True) for c in ("x", "y", "z", "t")}

# ── L1 ────────────────────────────────────────────────────────────────
# Phrases that name or request the solution.  Kept narrow on purpose: each
# entry must be a phrase that has no legitimate place in a blind task text.
FORBIDDEN_PHRASES = (
    "exact solution", "the exact solution", "u_exact", "uexact", "u exact",
    "manufactured solution", "manufactured displacement", "manufactured field",
    "manufactured exact", "method of manufactured", "mms",
    "analytical solution", "analytic solution", "closed form solution",
    "closed-form solution", "reference solution", "true solution",
    "ground truth", "known solution", "the solution is", "solution is given by",
    "exact eigenvalue", "exact value", "the exact values",
)

# ── L5 ────────────────────────────────────────────────────────────────
# Asking the agent to grade itself against a reference implies a reference.
SELF_GRADING_PATTERNS = (
    r"error\s+(against|versus|vs\.?|relative to|w\.?r\.?t\.?)\s+(the\s+)?"
    r"(exact|analytic|analytical|true|reference|manufactured)",
    r"(l2|l\^2|h1|h\^1|rms)\s*(-|\s)?(norm\s+)?error\s+against",
    r"compare\s+(your\s+)?(solution|result|values)\s+(against|to|with)\s+"
    r"(the\s+)?(exact|analytic|analytical|true|reference)",
    r"relative\s+l2\s+error\s+of\b.*\bvs\b",
    r"convergence\s+(rate|order)s?\s+(against|versus)\s+(the\s+)?exact",
)

# A candidate mathematical expression: a run of expression characters that
# mentions a coordinate and contains an operator.  Deliberately generous —
# false candidates cost only a failed sympify.
_CAND = re.compile(r"[0-9A-Za-z_.\*/+\-\^()\s]{6,400}")
_HAS_COORD = re.compile(r"\b[xyzt]\b")
_HAS_OP = re.compile(r"[\*/^]|\*\*")
_NUMBER = re.compile(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+[eE][-+]?\d+")


@dataclass
class Finding:
    rule: str
    severity: str          # CRITICAL | HIGH | MEDIUM | INFO
    detail: str


@dataclass
class GateReport:
    task_id: str
    findings: list = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not any(f.severity in ("CRITICAL", "HIGH") for f in self.findings)

    def add(self, rule, severity, detail):
        self.findings.append(Finding(rule, severity, detail))

    def summary(self) -> str:
        if not self.findings:
            return f"{self.task_id}: CLEAN"
        worst = min(self.findings,
                    key=lambda f: ("CRITICAL", "HIGH", "MEDIUM", "INFO").index(f.severity))
        return (f"{self.task_id}: {'CLEAN' if self.clean else 'LEAK'} "
                f"({len(self.findings)} finding(s), worst={worst.severity})")


# ──────────────────────────────────────────────────────────────────────
def _sympify(text: str):
    try:
        e = sp.sympify(text, locals=SYMS, evaluate=True)
    except Exception:
        return None
    if not isinstance(e, sp.Basic) or not e.free_symbols & set(SYMS.values()):
        return None
    return e


def _candidate_expressions(text: str, max_candidates: int = 400):
    """Pull plausible mathematical expressions out of prose."""
    seen, out = set(), []
    for m in _CAND.finditer(text):
        s = m.group(0).strip(" \t\n.,;:")
        if len(s) < 6 or not _HAS_COORD.search(s) or not _HAS_OP.search(s):
            continue
        # also try progressively trimmed left/right ends: prose usually wraps
        # the expression in words that break sympify.
        for cand in _trims(s):
            if cand in seen:
                continue
            seen.add(cand)
            e = _sympify(cand)
            if e is not None:
                out.append((cand, e))
                break
        if len(out) >= max_candidates:
            break
    return out


def _trims(s: str):
    """Yield the string and a few end-trimmed variants, longest first."""
    yield s
    toks = s.split()
    for lo in range(0, min(4, len(toks))):
        for hi in range(len(toks), max(len(toks) - 4, lo), -1):
            t = " ".join(toks[lo:hi])
            if len(t) >= 6:
                yield t


def _exact_expressions(key: dict):
    """Every sealed solution component in the key, as sympy expressions."""
    ex = key.get("exact_solution")
    if ex is None:
        return []
    groups = ex.values() if isinstance(ex, dict) else [ex]
    out = []
    for g in groups:
        for comp in (g if isinstance(g, list) else [g]):
            try:
                out.append(sp.sympify(comp, locals=SYMS))
            except Exception:
                pass
    return out


def _source_expressions(key: dict):
    src = key.get("source_term")
    if src is None:
        return []
    groups = src.values() if isinstance(src, dict) else [src]
    out = []
    for g in groups:
        for comp in (g if isinstance(g, list) else [g]):
            try:
                out.append((str(comp), sp.sympify(comp, locals=SYMS)))
            except Exception:
                pass
    return out


def scan(task_text: str, key: dict, task_id: str = "?",
         true_probe_values: list | None = None,
         numeric_rtol: float = 1e-6) -> GateReport:
    """Audit one blind task text against its sealed key."""
    rep = GateReport(task_id=task_id)
    low = task_text.lower()
    exacts = _exact_expressions(key)
    sources = _source_expressions(key)

    # ── L1 language ───────────────────────────────────────────────────
    for phrase in FORBIDDEN_PHRASES:
        if phrase in low:
            rep.add("L1_PHRASE", "HIGH",
                    f"task text contains {phrase!r}, which names or requests a "
                    f"solution the agent must not have")

    # ── L5 self-grading request ───────────────────────────────────────
    for pat in SELF_GRADING_PATTERNS:
        m = re.search(pat, low)
        if m:
            rep.add("L5_SELF_GRADING", "HIGH",
                    f"task asks the agent to grade itself against a reference: "
                    f"{m.group(0)!r}")

    # ── L2 symbolic disclosure ────────────────────────────────────────
    # Excise the legitimate source term first.  Without this the gate would
    # rediscover the source term inside the prompt and call it a leak — the
    # exact false alarm this design is built to avoid.
    stripped = task_text
    for raw, _ in sources:
        stripped = stripped.replace(raw, " <SOURCE_TERM> ")
    # also excise whitespace-normalised variants (printers differ)
    for raw, _ in sources:
        flat = re.sub(r"\s+", "", raw)
        stripped = re.sub(re.escape(flat), " <SOURCE_TERM> ",
                          re.sub(r"\s+", "", stripped))

    if exacts:
        for cand_text, cand in _candidate_expressions(stripped):
            for u in exacts:
                try:
                    if sp.simplify(cand - u) == 0:
                        rep.add("L2_SYMBOLIC", "CRITICAL",
                                f"an expression in the task is symbolically "
                                f"identical to the sealed solution: {cand_text!r}")
                        break
                    prop, c = is_proportional(cand, u)
                    if prop:
                        rep.add("L2_SYMBOLIC", "CRITICAL",
                                f"an expression in the task equals ({c}) x the "
                                f"sealed solution: {cand_text!r}")
                        break
                except Exception:
                    continue

    # ── L3 numeric disclosure ─────────────────────────────────────────
    if true_probe_values:
        nums = []
        for m in _NUMBER.finditer(task_text):
            try:
                nums.append(float(m.group(0)))
            except ValueError:
                pass
        for v in true_probe_values:
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v == 0:
                continue
            for n in nums:
                if abs(n - v) <= numeric_rtol * max(abs(v), 1e-30):
                    rep.add("L3_NUMERIC", "CRITICAL",
                            f"the task prints {n!r}, which matches a true probe "
                            f"value of the sealed solution")
                    break

    # ── L4 source-term embedding (design verdict) ─────────────────────
    for u in exacts:
        for raw, f in sources:
            if raw not in task_text and re.sub(r"\s+", "", raw) not in re.sub(r"\s+", "", task_text):
                continue                      # this source component is not published here
            prop, c = is_proportional(f, u)
            if prop:
                rep.add("L4_SOURCE_INVERTIBLE", "CRITICAL",
                        f"the published source term equals ({c}) x the sealed "
                        f"solution: one division recovers it (eigenfunction "
                        f"manufactured field)")
                continue
            # Severity turns on how much work recovery takes.  A sub-sum of the
            # terms AS PRINTED is a naked-eye leak; an embedding that only
            # appears after expand() needs deliberate symbolic algebra.
            comb, c_print = printed_subsum(f, u)
            if comb is not None:
                rep.add("L4_PRINTED_SUBSUM", "HIGH",
                        f"{len(comb)} of the {len(sp.Add.make_args(f))} source "
                        f"terms AS PRINTED sum to ({c_print}) x the sealed "
                        f"solution — recoverable by inspection: "
                        + " + ".join(str(t) for t in comb)[:200])
                continue
            coeff, frac = embedded_multiple(f, u)
            if coeff is not None and frac >= 0.999:
                rep.add("L4_SOURCE_EMBEDS", "MEDIUM",
                        f"after expansion, every term of the sealed solution "
                        f"survives in the published source term scaled by "
                        f"({coeff}). Not visible in the printed form, but one "
                        f"expand() and collect() recovers it.")
            elif coeff is not None and frac > 0:
                rep.add("L4_SOURCE_PARTIAL", "INFO",
                        f"{frac:.0%} of the sealed solution's terms appear in the "
                        f"published source term scaled by ({coeff})")
    return rep


def printed_subsum(f, u, max_scan: int = 60):
    """Subset of ``f``'s terms **as printed** summing to ``c*u``.

    Recovery difficulty is what severity should track, and that depends on the
    form the agent actually reads.  ``B1`` publishes ``12*pi**2*x*y*(x-1)*(y-1)*
    cos(2*pi*x)`` as one printed term — the solution is right there.  ``D3``
    needs two or three terms collected.  Both are inspection-level.  An
    embedding that only surfaces after ``expand()`` is a lesser exposure and is
    graded lower.

    Enumerating subsets is exponential and was far too slow to run routinely, so
    the search is driven by monomials instead: a subset summing to ``c*u`` must
    reproduce every monomial of ``u``, so pick one monomial of ``u``, and only
    the printed terms containing it can be in the subset.  That fixes ``c``, and
    the candidate subset is then read off directly.  Linear in the number of
    terms, and exact whenever each monomial of ``u`` is contributed by a single
    printed term — which is the case for every real instance encountered.

    Returns ``(terms, coefficient)`` or ``(None, None)``.
    """
    terms = list(sp.Add.make_args(f))
    if not terms or len(terms) > max_scan:
        return None, None

    # cheap exact hit first: a single printed term proportional to u (B1)
    for t in terms:
        try:
            ratio = sp.simplify(sp.cancel(sp.together(t / u)))
        except Exception:
            continue
        if not ratio.free_symbols and ratio != 0:
            return [t], ratio

    U = sp.expand(u)
    u_terms = sp.Add.make_args(U)
    if len(u_terms) < 2:
        return None, None
    expanded = [sp.expand(t) for t in terms]

    # pick the monomial of u that the fewest printed terms could supply
    def parts(e):
        return {sp.Mul(*[a for a in sp.Mul.make_args(m) if a.free_symbols]):
                sp.Mul(*[a for a in sp.Mul.make_args(m) if not a.free_symbols])
                for m in sp.Add.make_args(e)}

    u_parts = parts(U)
    term_parts = [parts(e) for e in expanded]
    best_key, best_hits = None, None
    for mono in u_parts:
        hits = [i for i, tp in enumerate(term_parts) if mono in tp]
        if hits and (best_hits is None or len(hits) < len(best_hits)):
            best_key, best_hits = mono, hits
    if best_key is None:
        return None, None

    for i in best_hits:
        c = sp.simplify(term_parts[i][best_key] / u_parts[best_key])
        if c.free_symbols or c == 0:
            continue
        target = parts(sp.expand(c * U))
        subset = [j for j, tp in enumerate(term_parts)
                  if any(m in target for m in tp)]
        if not subset:
            continue
        if sp.simplify(sp.expand(sum(terms[j] for j in subset) - c * u)) == 0:
            return [terms[j] for j in subset], c
    return None, None
