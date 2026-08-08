"""Observed convergence order from mesh halving alone — no exact solution.

The owner's requirement: *"convergence to be tested by heuristics and mesh
halving"*.  The pre-existing grader computes the order from the **true** error
(sealed solution evaluated on the probe grid), so it cannot produce a number at
all until the keys are unsealed.  This module closes that gap: it derives the
observed order from the agent's own reported values across a refinement
sequence, and never touches a key.

Why this is not a weaker substitute
-----------------------------------
For a discretisation converging at order ``p`` with exact halving,

    u_h = u + C h^p + o(h^p)

so successive differences satisfy ``(u_1 - u_2) / (u_2 - u_3) -> 2^p``.  That
identity involves only computed solutions.  The estimator is therefore
available for *any* problem, including the ones with no closed form — which is
the situation every real engineering problem is in, and the reason a
reference-free convergence check is the honest one.

It is also an **independent cross-check on the key**.  If the self-convergence
order and the true-error order disagree, either the manufactured solution in
the key is wrong or the run is not what it claims.  Running both and comparing
catches a class of error that running either alone cannot.

Heuristics reported alongside the order
---------------------------------------
``bounded``            every reported value is finite and within a sane range
``monotone_differences``  successive differences shrink (the sequence settles)
``order_near_theory``  the observed order is close to the theoretical rate
``probe_consistency``  per-probe orders agree with each other
``gci``                Roache's Grid Convergence Index — an error bar on the
                       finest level, again with no exact solution
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SelfConvergence:
    order: float | None = None            # primary, norm-based
    order_per_step: list = field(default_factory=list)
    per_probe_orders: list = field(default_factory=list)
    diffs: list = field(default_factory=list)
    richardson_estimate: list = field(default_factory=list)
    gci: float | None = None
    heuristics: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.order is not None and all(
            v for k, v in self.heuristics.items() if k != "order_near_theory")


def _rms(vals) -> float:
    return math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else 0.0


def self_convergence(levels: list[list[float]],
                     theoretical_order: float | None = None,
                     order_tol: float = 0.5,
                     refinement_ratio: float = 2.0,
                     magnitude_limit: float = 1e12,
                     safety_factor: float = 1.25) -> SelfConvergence:
    """Observed order from the agent's values alone.

    ``levels`` is coarse-to-fine; every entry is the vector of values at the
    same fixed probe points.  Requires at least three levels — two levels give
    a difference but no ratio, hence no order.
    """
    out = SelfConvergence()
    n = len(levels)
    if n < 3:
        out.notes.append(f"need >= 3 mesh levels for a ratio; got {n}")
        return out
    width = {len(v) for v in levels}
    if len(width) != 1:
        out.notes.append(f"levels report different probe counts: {sorted(width)}")
        return out

    flat = [v for lv in levels for v in lv]
    bounded = all(math.isfinite(v) and abs(v) < magnitude_limit for v in flat)
    out.heuristics["bounded"] = bounded
    if not bounded:
        out.notes.append("non-finite or implausibly large values reported")
        return out

    # ── successive differences, in the RMS norm over the probe grid ──
    diffs = []
    for k in range(n - 1):
        diffs.append(_rms([a - b for a, b in zip(levels[k], levels[k + 1])]))
    out.diffs = diffs
    out.heuristics["monotone_differences"] = all(
        diffs[i] > diffs[i + 1] for i in range(len(diffs) - 1))

    logr = math.log(refinement_ratio)
    steps = []
    for i in range(len(diffs) - 1):
        if diffs[i + 1] <= 0 or diffs[i] <= 0:
            out.notes.append(f"zero difference between levels {i + 1}/{i + 2}: "
                             f"the solution stopped changing (exactly-representable "
                             f"solution, or values were copied between levels)")
            continue
        steps.append(math.log(diffs[i] / diffs[i + 1]) / logr)
    out.order_per_step = steps
    if not steps:
        return out
    out.order = sum(steps) / len(steps)

    # ── per-probe orders, as a consistency diagnostic ────────────────
    per = []
    for j in range(len(levels[0])):
        d1 = levels[-3][j] - levels[-2][j]
        d2 = levels[-2][j] - levels[-1][j]
        if d1 != 0 and d2 != 0 and (d1 / d2) > 0:
            per.append(math.log(abs(d1 / d2)) / logr)
    out.per_probe_orders = per
    if per:
        med = sorted(per)[len(per) // 2]
        spread = sorted(per)[int(0.9 * (len(per) - 1))] - sorted(per)[int(0.1 * (len(per) - 1))]
        out.heuristics["probe_consistency"] = spread <= 1.0
        out.notes.append(f"per-probe order median={med:.3f}, p10-p90 spread={spread:.3f}")
    else:
        out.heuristics["probe_consistency"] = True
        out.notes.append("per-probe orders undefined (differences vanish pointwise)")

    # ── Richardson extrapolation + GCI on the finest level ───────────
    p = out.order
    if p > 0:
        denom = refinement_ratio ** p - 1.0
        ext = [levels[-1][j] + (levels[-1][j] - levels[-2][j]) / denom
               for j in range(len(levels[-1]))]
        out.richardson_estimate = ext
        rel = [abs(e - a) / max(abs(e), 1e-30)
               for e, a in zip(ext, levels[-1])]
        out.gci = safety_factor * _rms(rel)

    if theoretical_order is not None:
        out.heuristics["order_near_theory"] = abs(p - theoretical_order) <= order_tol

    return out


def cross_check(self_order: float | None, true_order: float | None,
                tol: float = 0.5) -> dict:
    """Compare the key-free order against the key-based one.

    Disagreement means the sealed solution, the problem statement, or the run
    is wrong.  Running both is how that gets noticed instead of shipped.
    """
    if self_order is None or true_order is None:
        return {"agree": None, "note": "one of the two orders is unavailable"}
    d = abs(self_order - true_order)
    return {"agree": d <= tol, "delta": d,
            "note": ("self-convergence and true-error orders agree"
                     if d <= tol else
                     "MISMATCH: the key-free and key-based orders disagree; "
                     "suspect the sealed solution or the submission")}
