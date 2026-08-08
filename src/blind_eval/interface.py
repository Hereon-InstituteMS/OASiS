"""Interface grading quantities — the ones that move when the coupling breaks.

Why the field alone is not enough
---------------------------------
The primary, key-free grade is the observed order from mesh halving.  It is
reference-free, which is its virtue, and it is blind to the single most
important coupled failure mode, which is its limit: **a partitioned scheme that
iterates the wrong transmission condition converges cleanly to the wrong fixed
point, at order two.**  Measured on a real Dirichlet-Neumann solve
(``scripts/mutate_coupling.py``):

    variant                       order_u (mesh halving)   true order   jump_q
    correct                                1.838              1.884     1.6e-14
    wrong transmitted quantity             1.865              0.112     3.0
    reversed interface mapping             1.847              0.405     0.75
    receiver's coefficient 25% wrong       1.847              0.002     1.6e-14

Every mutation self-converges at order ~1.85.  The key-free order cannot
separate any of them from the correct run.  The two-sided flux jump separates
two of the three with the keys still sealed, and the true error separates all
three once the key is opened.  Hence: order from halving stays the primary
grade, the interface quantities are added, and neither is described as
sufficient alone.

What is graded, and against what
--------------------------------
``jump_u``    ``|u_A - u_B|`` at fixed interface probes. Reference-free.
``jump_q``    ``|q_A + q_B| / |q_A|`` with OUTWARD normals, so continuity means
              the sum vanishes. Reference-free, and its correct limit is known
              to be zero without any solution — which is exactly what the
              mesh-halving order cannot give.
``q_profile`` the whole flux profile, not one scalar. A scalar summary of the
              interface is invariant under an interface-mapping reversal — the
              mean interface value and the net flux were measured UNCHANGED to
              six digits under ``MUT_MAP`` — while the profile moves. A single
              number cannot grade a coupling.
``q_true``    flux error against the sealed exact interface flux (phase 2).

Fabrication
-----------
The flux is agent-reported, so an agent could report ``q_B = -q_A`` by
construction.  :func:`recover_flux_from_field` therefore recomputes the
interface flux from the agent's own submitted FIELD values by one-sided
quadratic extrapolation on the probe columns, and :func:`flux_consistency`
compares the two.  Faking the flux then requires faking a consistent field.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path


def _rms(v) -> float:
    v = [float(a) for a in v]
    return math.sqrt(sum(a * a for a in v) / len(v)) if v else 0.0


def read_interface_csv(path: Path, ncoord: int, nval: int, nflux: int):
    """``coords..., values..., fluxes...`` — any bad row invalidates the file."""
    pts, vals, flux = [], [], []
    with open(path, newline="", errors="ignore") as fh:
        rows = [r for r in csv.reader(fh) if r and any(c.strip() for c in r)]
    if not rows:
        return None, "empty file"
    start = 0
    try:
        float(rows[0][0])
    except (ValueError, IndexError):
        start = 1
    need = ncoord + nval + nflux
    for r in rows[start:]:
        if len(r) < need:
            return None, f"row has {len(r)} columns, need {need}"
        try:
            nums = [float(c) for c in r[:need]]
        except ValueError:
            return None, "unparsable numeric value"
        if not all(math.isfinite(v) for v in nums):
            return None, "non-finite value"
        pts.append(tuple(nums[:ncoord]))
        vals.append(tuple(nums[ncoord:ncoord + nval]))
        flux.append(tuple(nums[ncoord + nval:]))
    return (pts, vals, flux), "ok"


@dataclass
class InterfaceReport:
    levels: list = field(default_factory=list)
    jump_u_rel: list = field(default_factory=list)
    jump_q_rel: list = field(default_factory=list)
    q_profile_order: float | None = None
    consistency: list = field(default_factory=list)
    verdict: str = "NOT_ASSESSED"
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.verdict}: jump_q_rel={self.jump_q_rel}, "
                f"jump_u_rel={self.jump_u_rel}")


def two_sided_jumps(side_a, side_b, coord_tol: float = 1e-6):
    """``jump_u`` and ``jump_q`` at matched interface probe points.

    The two subdomains report at the SAME physical points with OPPOSITE outward
    normals, so a correct coupling has ``u_A - u_B = 0`` and ``q_A + q_B = 0``.
    Points are matched by coordinate, not by row order, so an agent that writes
    one side reversed is detected rather than silently graded against itself.
    """
    (pa, va, qa), (pb, vb, qb) = side_a, side_b
    if len(pa) != len(pb):
        return None, f"sides report {len(pa)} and {len(pb)} interface points"
    idx = {tuple(round(c, 9) for c in p): i for i, p in enumerate(pb)}
    du, dq, scale_u, scale_q = [], [], [], []
    for i, p in enumerate(pa):
        key = tuple(round(c, 9) for c in p)
        j = idx.get(key)
        if j is None:
            best, bd = None, coord_tol
            for k, jj in idx.items():
                d = max(abs(a - b) for a, b in zip(k, p))
                if d < bd:
                    best, bd = jj, d
            j = best
        if j is None:
            return None, f"subdomain B reports no point at {p}"
        du += [a - b for a, b in zip(va[i], vb[j])]
        dq += [a + b for a, b in zip(qa[i], qb[j])]
        scale_u += list(va[i])
        scale_q += list(qa[i])
    return {"jump_u": _rms(du), "jump_q": _rms(dq),
            "jump_u_rel": _rms(du) / max(_rms(scale_u), 1e-30),
            "jump_q_rel": _rms(dq) / max(_rms(scale_q), 1e-30)}, "ok"


def recover_flux_from_field(field_pts, field_vals, iface_pts, k_normal,
                            normal_axis: int, iface_coord: float,
                            outward_sign: float):
    """Independent flux estimate from the agent's own FIELD submission.

    One-sided quadratic extrapolation of the normal derivative onto the
    interface, using the three probe columns nearest to it.  Accurate to
    ``O(h_probe^2)`` — far short of grading an order, and ample to catch an
    ``O(1)`` fabricated flux, which is all it is for.

    ``outward_sign`` is ``+1`` when the subdomain's outward normal points along
    ``+e_axis`` and ``-1`` otherwise.
    """
    rows: dict = {}
    for p, v in zip(field_pts, field_vals):
        key = tuple(round(c, 9) for i, c in enumerate(p) if i != normal_axis)
        rows.setdefault(key, []).append((p[normal_axis], v))
    out = []
    for p in iface_pts:
        key = tuple(round(c, 9) for i, c in enumerate(p) if i != normal_axis)
        col = sorted(rows.get(key, []), key=lambda a: abs(a[0] - iface_coord))
        if len(col) < 3:
            out.append(None)
            continue
        (x0, v0), (x1, v1), (x2, v2) = col[0], col[1], col[2]
        ncomp = len(v0)
        est = []
        for c in range(ncomp):
            y0, y1, y2 = v0[c], v1[c], v2[c]
            # derivative at iface_coord of the quadratic through the 3 points
            d01 = (y1 - y0) / (x1 - x0)
            d12 = (y2 - y1) / (x2 - x1)
            d012 = (d12 - d01) / (x2 - x0)
            dv = d01 + d012 * (2 * iface_coord - x0 - x1)
            est.append(-k_normal * dv * outward_sign)
        out.append(tuple(est))
    return out


def flux_consistency(reported, recovered, rtol: float = 0.25):
    """Does the reported interface flux match one recomputed from the field?"""
    pairs = [(r, g) for r, g in zip(reported, recovered) if g is not None]
    if not pairs:
        return {"verdict": "NOT_ASSESSED",
                "detail": "no probe column had three points to extrapolate from"}
    num = _rms([a - b for r, g in pairs for a, b in zip(r, g)])
    den = max(_rms([a for r, _ in pairs for a in r]), 1e-30)
    rel = num / den
    return {"verdict": "CONSISTENT" if rel <= rtol else "INCONSISTENT",
            "relative_difference": rel, "tolerance": rtol,
            "detail": ("the reported interface flux does not follow from the "
                       "submitted field: one of the two was not computed from "
                       "the other" if rel > rtol else
                       "reported flux agrees with the field-derived estimate")}


def order_from_halving(seq) -> float | None:
    s = [float(v) for v in seq]
    if len(s) < 2 or any(v <= 0 for v in s):
        return None
    return sum(math.log2(s[i] / s[i + 1]) for i in range(len(s) - 1)) / (len(s) - 1)


def assess(per_level: list, jump_tol: float = 5e-3) -> InterfaceReport:
    """``per_level`` is coarse-to-fine, each entry the output of two_sided_jumps."""
    rep = InterfaceReport()
    rep.levels = list(range(1, len(per_level) + 1))
    rep.jump_u_rel = [d["jump_u_rel"] for d in per_level]
    rep.jump_q_rel = [d["jump_q_rel"] for d in per_level]
    if not per_level:
        rep.verdict = "NOT_ASSESSED"
        return rep
    worst_q = max(rep.jump_q_rel)
    worst_u = max(rep.jump_u_rel)
    if worst_q > jump_tol or worst_u > jump_tol:
        rep.verdict = "INTERFACE_NOT_SATISFIED"
        rep.notes.append(
            f"the two subdomains disagree at the interface: max relative "
            f"field jump {worst_u:.3e}, max relative FLUX jump {worst_q:.3e}, "
            f"tolerance {jump_tol:g}. A flux jump that does not fall under "
            f"refinement means the scheme converged to a fixed point of the "
            f"wrong transmission condition — which the observed order cannot "
            f"see, because convergence to a wrong answer is still convergence.")
    else:
        rep.verdict = "INTERFACE_SATISFIED"
    return rep
