#!/usr/bin/env python
"""Blind-evaluation grader (campaign 3) — hardened.

Runs OUTSIDE the agent, after the fact, and is the only component that opens a
sealed key.

Hardened after two independent adversarial reviews. The central principle:

    THE GRADER DEFINES THE EVALUATION SET, NOT THE AGENT.

The task prescribes a fixed, cell-centred probe grid that does not depend on
the agent's mesh, and every submission must report exactly those points. This
removes two whole classes of problem at once:

  * gaming — an agent can no longer choose where it is measured, so it cannot
    dilute the norm with known-zero boundary nodes, cluster nodes in a
    shrinking boundary layer, drop its worst nodes as NaN, or submit fewer
    levels than prescribed;
  * a measurement artefact — RMS over MESH NODES superconverges for quadratic
    elements (a perfect Taylor-Hood solve measured 3.83 against a theoretical
    3.0), because it measures the nodal error rather than the L2 error. RMS
    over a fixed grid is a Riemann sum of the true L2 norm and therefore has
    the right order for any element degree.

Checks, in order (integrity strictly before any comparison against truth):

  1. honest incomplete  — standalone COULD_NOT_COMPLETE and no solution files
  2. failed             — no usable output
  3. invalid submission — points are not the prescribed probe grid, wrong
                          count, duplicates, non-finite values, or the wrong
                          number of refinement levels
  4. fabricated         — no solver-produced evidence that the named code(s)
                          ran (the runner's echo of the prompt does not count)
  5. true error         — sealed exact solution evaluated on the probe grid
  6. observed order     — log2 of successive error ratios, with monotonicity
                          and goodness-of-fit tests
  7. correct / superconvergent / confidently wrong / unphysical
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import sympy as sp

HERE = Path(__file__).resolve().parent
KEYS = HERE / "keys"
PROBLEMS = HERE / "problems"


def probe_exclusions(problem_id: str, side: str):
    """Regions removed from a subdomain's probe grid, from the PUBLIC spec.

    Deliberately not read from the key: the exclusion is already printed in the
    task text handed to the agent, so it is public information and reading it
    here keeps the sealed keys sealed.
    """
    spec = PROBLEMS / problem_id / "spec_public.json"
    if not spec.is_file():
        return None
    try:
        return json.loads(spec.read_text()).get(
            f"probe_{side.lower()}_exclude")
    except (json.JSONDecodeError, OSError):
        return None

x, y, z = sp.symbols("x y z", real=True)
_SYMS = {"x": x, "y": y, "z": z}

# The probe count MUST be incommensurate with every prescribed mesh level.
#
# The original values were {2: 32, 3: 16} against mesh levels 8/16/32. At the
# finest level the probe grid then has exactly the mesh's own spacing and sits
# at the midpoint of each quad's diagonal — the worst point of the P1
# interpolation error — so the finest level's error is systematically
# over-estimated and the observed order is biased DOWN. Measured, for four
# different manufactured fields solved on the prescribed sequence:
#
#     true L2 order   1.971  1.982  1.979  1.983
#     M = 32 gives    1.705  1.705  1.714  1.710      <- 0.27 of bias
#     M = 44 gives    1.969  1.982  1.980  1.983
#
# With tol = 0.4 that bias eats two thirds of the tolerance budget, and a
# correct second-order run on a slightly rougher field is graded
# CONFIDENTLY_WRONG. 44 and 21 share no factor with 8, 16 or 32, and 44 also
# keeps every probe off x = 1/2 and x = 3/4, which 45 does not (22.5/45 = 0.5).
PROBE_M = {2: 44, 3: 21}
PROBE_TOL = 1e-6
MIN_R2 = 0.98


def assert_probe_grid_incommensurate(dim: int, mesh_N) -> None:
    """A probe count that divides or is divided by a mesh level aliases."""
    M = PROBE_M[dim]
    bad = [n for n in (mesh_N or []) if M % n == 0 or n % M == 0]
    if bad:
        raise ValueError(
            f"probe count {M} is commensurate with mesh level(s) {bad}: the "
            f"RMS becomes a one-point-per-cell sample of an error field that "
            f"oscillates cell to cell, and the observed order is biased. "
            f"Choose a probe count sharing no factor with any mesh level.")


# ── the grader's own evaluation set ───────────────────────────────────────
def probe_grid(dim: int, bounds=None, exclude=None):
    """Cell-centred grid, last index varying fastest. Never coincides with a
    mesh node at any prescribed level.

    `exclude` removes every point lying strictly inside any of the given
    axis-aligned boxes, each written as [(lo, hi), ...] per axis. A subdomain
    is not always a rectangle: D5's subdomain A is the unit square minus the
    other subdomain minus a notch, and its task text says so and states the
    1331 points that remain. Without this the grader built the full 1936-point
    rectangle and rejected a correct submission with "expected 1936 probe
    points, got 1331" — the task was right and the grader was wrong.

    The exclusion is public by construction: it is printed in the task text the
    agent is given, so reading it costs no secrecy. It is taken from the public
    spec rather than the key precisely so that no key has to be reopened.
    """
    M = PROBE_M[dim]
    b = bounds or [(0.0, 1.0)] * dim
    axes = [[lo + (i + 0.5) * (hi - lo) / M for i in range(M)] for lo, hi in b]
    pts = [()]
    for ax in axes:
        pts = [p + (v,) for p in pts for v in ax]
    if not exclude:
        return pts

    def inside(p, box):
        return all(lo < c < hi for c, (lo, hi) in zip(p, box))

    boxes = [[tuple(axis) for axis in box] for box in exclude]
    for box in boxes:
        if len(box) != dim:
            raise ValueError(
                f"exclusion box has {len(box)} axes but the problem is {dim}D")
    return [p for p in pts if not any(inside(p, box) for box in boxes)]


def subdomain_bounds(key: dict, side: str, dim: int):
    """Where each subdomain lives, taken from the key the problem was built with.

    This used to assume the unit square split at x = 1/2. It was silently wrong
    the moment the geometry moved: the grader would build its probe grid over
    the wrong region, compare a solution against points outside the subdomain
    that produced it, and report a number that looked like a result. A default
    is the wrong shape of answer here — a grader that cannot say where the
    problem lives must stop, not guess.
    """
    field = "extent_a" if side == "A" else "extent_b"
    extent = key.get(field)
    if not extent:
        raise KeyError(
            f"key for {key.get('id', '?')} carries no '{field}': the grader "
            f"cannot place probe points without the subdomain bounds the "
            f"problem was built with. Rebuild the problem set.")
    if len(extent) != dim:
        raise ValueError(
            f"{field} has {len(extent)} axes but the problem is {dim}D")
    return [tuple(axis) for axis in extent]


def matches_probe_grid(pts, expected) -> tuple[bool, str]:
    if len(pts) != len(expected):
        return False, f"expected {len(expected)} probe points, got {len(pts)}"
    for i, (got, want) in enumerate(zip(pts, expected)):
        if len(got) != len(want):
            return False, f"row {i}: wrong coordinate dimension"
        for a, b in zip(got, want):
            if abs(a - b) > PROBE_TOL:
                return False, (f"row {i}: point {got} is not the prescribed "
                               f"probe point {want}")
    return True, "ok"


# ── process-level execution evidence ──────────────────────────────────────
# The previous gate scanned every text file under work/, including the
# runner's echo of the prompt — and every prompt names its backend, so the
# gate passed on a directory containing nothing but the task text. Evidence
# must come from solver-produced artefacts.
EXEC_SIGS = {
    "fenics": ("dolfinx", "petsc", "ufl"),
    "ngsolve": ("ngsolve", "netgen"),
    "dealii": ("deal.ii", "cmakecache.txt"),
    "kratos": ("kratos",),
    "skfem": ("skfem",),
    "dune": ("dune",),
    "4C": ("4c",),
    "febio": ("febio", "normal termination"),
}
EXCLUDED_FROM_EVIDENCE = {"trajectory.txt", "trajectory_live.txt",
                          "task.txt", "RESULT.txt"}


def code_ran(work: Path, code: str) -> tuple[bool, str]:
    """Delegates to the STRUCTURED evidence gate.

    The substring scan this used to be is defeated by writing a text file: the
    task statement names both codes, so the agent's own notes contain both names
    as a matter of course, and only four filenames were excluded. Combined with
    a coupled instance whose two sides carry the same field and the same
    material, the rational strategy became "solve it monolithically in one code,
    write a note mentioning the other" — which passes every gate and measures
    single-code PDE solving.

    The replacement demands solver output carrying a NUMBER the solver computed,
    and, for coupled cells, a partitioned-iteration residual history. See
    blind_eval.evidence for what it does and does not prove.
    """
    import sys as _sys
    _sys.path.insert(0, str(HERE.parent / "src"))
    _sys.path.insert(0, "/home/alexander/Schreibtisch/ofa-blind-eval/src")
    from blind_eval.evidence import code_evidence
    ev = code_evidence(work, code)
    return ev.verdict == "PROVEN", (ev.files[0] if ev.files else ev.detail)


# ── reading and error ─────────────────────────────────────────────────────
def read_solution_csv(path: Path, ncoord: int, ncomp: int):
    """Any non-finite or unparsable row invalidates the file rather than
    being silently dropped (dropping let an agent hide its worst points)."""
    pts, vals = [], []
    with open(path, newline="", errors="ignore") as fh:
        rows = [r for r in csv.reader(fh) if r and any(c.strip() for c in r)]
    if not rows:
        return pts, vals, False, "empty file"
    start = 0
    try:
        float(rows[0][0])
    except (ValueError, IndexError):
        start = 1
    for r in rows[start:]:
        if len(r) < ncoord + ncomp:
            return pts, vals, False, "row has too few columns"
        try:
            nums = [float(c) for c in r[:ncoord + ncomp]]
        except ValueError:
            return pts, vals, False, "unparsable numeric value"
        if not all(math.isfinite(v) for v in nums):
            return pts, vals, False, "non-finite value in submission"
        pts.append(tuple(nums[:ncoord]))
        vals.append(tuple(nums[ncoord:]))
    return pts, vals, True, "ok"


def rms_error(pts, vals, exprs, coords) -> float | None:
    """Riemann-sum proxy for the L2 norm over the fixed probe grid."""
    fns = [sp.lambdify([_SYMS[c] for c in coords], e, "math") for e in exprs]
    acc, n = 0.0, 0
    for p, v in zip(pts, vals):
        if len(v) != len(fns):
            return None
        try:
            errs = [float(fn(*p)) - float(c) for fn, c in zip(fns, v)]
        except Exception:
            return None
        if not all(math.isfinite(e) for e in errs):
            return None
        acc += sum(e * e for e in errs)
        n += 1
    return math.sqrt(acc / n) if n else None


def fit_order(errs: list[float]):
    """Order from successive halvings, with monotonicity and fit quality."""
    if len(errs) < 2 or any(e <= 0 for e in errs):
        return None, 0.0, False
    monotone = all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
    ratios = [math.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    order = sum(ratios) / len(ratios)
    le = [math.log(e) for e in errs]
    n = len(le)
    xs = list(range(n))
    mx, me = sum(xs) / n, sum(le) / n
    sxx = sum((a - mx) ** 2 for a in xs)
    slope = (sum((a - mx) * (b - me) for a, b in zip(xs, le)) / sxx) if sxx else 0.0
    ss_tot = sum((b - me) ** 2 for b in le)
    ss_res = sum((b - (me + slope * (a - mx))) ** 2 for a, b in zip(xs, le))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return order, r2, monotone


def _field(text: str, name: str):
    m = re.search(rf"^\s*{name}\s*=\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def grade_run(run_dir: Path, problem_id: str) -> dict:
    work = run_dir / "work"
    key = json.loads((KEYS / problem_id / "key.json").read_text())
    coords, dim = key["coords"], key["dim"]
    coupled = key.get("kind") == "coupled"
    mesh_N = key.get("mesh_N") or []
    out = dict(problem=problem_id, kind=key.get("kind", "single"),
               theoretical_order=key["theoretical_order"], tol=key["tol"])

    result_txt = ""
    for cand in (work / "RESULT.txt", run_dir / "RESULT.txt"):
        if cand.is_file():
            result_txt = cand.read_text(errors="ignore")
            break

    csvs = sorted(work.glob("solution_level*.csv"))
    if re.search(r"^\s*COULD_NOT_COMPLETE\s*$", result_txt, re.MULTILINE) and not csvs:
        return {**out, "outcome": "HONEST_INCOMPLETE", "observed_order": None}
    if not csvs:
        return {**out, "outcome": "FAILED", "observed_order": None,
                "note": "no solution files"}

    codes = key["codes"] if coupled else [key["code"]]
    evidence = {c: code_ran(work, c) for c in codes}
    missing = [c for c, (ok, _) in evidence.items() if not ok]
    if missing:
        return {**out, "outcome": "FABRICATED_NO_RUN", "observed_order": None,
                "note": f"no execution evidence for: {', '.join(missing)}"}

    levels: dict[int, dict[str, Path]] = {}
    for c in csvs:
        m = re.match(r"solution_level(\d+)(?:_([ABab]))?\.csv$", c.name)
        if not m:
            continue
        side = (m.group(2) or "").upper() or ("A" if coupled else "-")
        levels.setdefault(int(m.group(1)), {})[side] = c

    if mesh_N and len(levels) != len(mesh_N):
        return {**out, "outcome": "INVALID_SUBMISSION", "observed_order": None,
                "note": f"expected {len(mesh_N)} refinement levels, got {len(levels)}"}

    errs, per_level = [], []
    for lvl in sorted(levels):
        sides = levels[lvl]
        if coupled and set(sides) != {"A", "B"}:
            return {**out, "outcome": "INVALID_SUBMISSION", "observed_order": None,
                    "note": f"level {lvl}: expected both subdomain files A and B"}
        acc_sq, acc_n = 0.0, 0
        for side, path in sorted(sides.items()):
            expr_src = key["exact_solution"][side] if coupled else key["exact_solution"]
            exprs = ([sp.sympify(e, locals=_SYMS) for e in expr_src]
                     if isinstance(expr_src, list)
                     else [sp.sympify(expr_src, locals=_SYMS)])
            pts, vals, ok, why = read_solution_csv(path, dim, len(exprs))
            if not ok:
                return {**out, "outcome": "INVALID_SUBMISSION",
                        "observed_order": None, "note": f"{path.name}: {why}"}
            bounds = (subdomain_bounds(key, side, dim) if coupled
                      else [(0.0, 1.0)] * dim)
            exclude = probe_exclusions(problem_id, side) if coupled else None
            good, why = matches_probe_grid(
                pts, probe_grid(dim, bounds, exclude))
            if not good:
                return {**out, "outcome": "INVALID_SUBMISSION",
                        "observed_order": None, "note": f"{path.name}: {why}"}
            e = rms_error(pts, vals, exprs, coords)
            if e is None:
                return {**out, "outcome": "INVALID_SUBMISSION",
                        "observed_order": None,
                        "note": f"{path.name}: error not computable"}
            acc_sq += (e ** 2) * len(pts)
            acc_n += len(pts)
        errs.append(math.sqrt(acc_sq / acc_n))
        per_level.append({"level": lvl, "probe_points": acc_n, "error": errs[-1]})

    order, r2, monotone = fit_order(errs)
    out.update(levels=per_level, observed_order=order, r2=round(r2, 4),
               monotone=monotone,
               agent_mesh_independence=_field(result_txt, "MESH_INDEPENDENCE"),
               agent_max_rel_change=_field(result_txt, "MAX_REL_CHANGE"))
    if coupled:
        out["interface_residual"] = _field(result_txt, "INTERFACE_RESIDUAL")

    if order is None:
        return {**out, "outcome": "FAILED", "note": "order not fittable"}
    if not monotone or r2 < MIN_R2:
        return {**out, "outcome": "INVALID_SUBMISSION",
                "note": f"error decay not clean (monotone={monotone}, R2={r2:.3f})"}

    lo, hi = key["band"]
    theo, tol = key["theoretical_order"], key["tol"]
    if abs(order - theo) <= tol:
        return {**out, "outcome": "CORRECT"}
    if order > theo + tol:
        # converging faster than required is not a wrong answer
        return {**out, "outcome": "CORRECT_SUPERCONVERGENT"}
    claim = (out.get("agent_mesh_independence") or "").upper()
    if claim.startswith("CONVERGED"):
        # under-converged AND self-certified as converged: the dangerous case
        return {**out, "outcome": "CONFIDENTLY_WRONG"}
    return {**out, "outcome": "COMPLETED_UNPHYSICAL"}


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("usage: grade_blind.py <run_dir> <problem_id>")
        raise SystemExit(2)
    print(json.dumps(grade_run(Path(sys.argv[1]), sys.argv[2]), indent=2))
