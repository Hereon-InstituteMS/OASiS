#!/usr/bin/env python3
"""Two-phase blind grader: mesh-halving order first, sealed key second.

Wraps the campaign's existing ``grade_blind.py`` rather than replacing it — the
submission-integrity work in there (grader-owned probe grid, execution
evidence, non-finite rejection) is good and is reused as imported functions.

What this adds is the part the owner asked for and the existing grader does not
have:

    **PHASE 1 — no key.**  The observed convergence order is computed from the
    agent's own reported values across the refinement sequence, by Richardson
    mesh halving.  Needs no exact solution, so it runs immediately after a
    campaign, with the keys still sealed and the passphrase still not typed.

    **PHASE 2 — key.**  Optionally decrypts the sealed key in memory and
    computes the true error and the key-based order, exactly as before.

Running phase 1 alone is a real capability, not a fallback: it is the only
convergence evidence available for a problem with no closed form, which is the
situation every real engineering problem is in.  Running both and comparing is
strictly better than either — a disagreement means the sealed solution, the
problem statement, or the run is wrong, and that is a class of error neither
phase catches alone.

Usage:
    blind_grade.py <run_dir> <problem_id>              # phase 1 only
    blind_grade.py <run_dir> <problem_id> --with-key   # prompts for passphrase
"""
from __future__ import annotations

import argparse
import getpass
import json
import math
import re
import sys
from pathlib import Path

CAMPAIGN = Path("/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(CAMPAIGN))

from blind_eval import keyvault, selfconv                       # noqa: E402
from blind_eval import evidence as EV                           # noqa: E402
from blind_eval import interface as IF                          # noqa: E402
import grade_blind as GB                                        # noqa: E402

# The three grades of evidence are NOT interchangeable and must never be pooled
# into one aggregate number.
EVIDENCE_GRADES = {
    1: "exact manufactured solution — proves the answer is CORRECT",
    2: "monolithic reference — proves the SPLIT did not change the answer; a "
       "discretisation bug shared by both solves makes them agree while both "
       "are wrong",
    3: "mesh-halving order only — proves the scheme CONVERGES, not that it "
       "converges to the right thing. Never sufficient alone for a coupled cell.",
}


def _interface_phase(work: Path, spec: dict) -> dict:
    """Reference-free interface quantities: the two-sided jumps.

    This is the part of phase 1 that can fail a coupling the observed order
    cannot. The correct limit of the flux jump is known to be zero without any
    reference solution, which is precisely what mesh halving does not give.
    """
    dim = spec.get("dim", 2)
    ncomp = len(spec.get("components", ["u"]))
    per_level, notes = [], []
    lvls: dict = {}
    for c in sorted(work.glob("interface_level*.csv")):
        m = re.match(r"interface_level(\d+)_([ABab])\.csv$", c.name)
        if m:
            lvls.setdefault(int(m.group(1)), {})[m.group(2).upper()] = c
    if not lvls:
        return {"verdict": "NOT_SUBMITTED",
                "note": "no interface_level<k>_<side>.csv files. The interface "
                        "flux is the only reference-free quantity that can "
                        "detect convergence to the wrong transmission "
                        "condition; without it the cell is graded on order "
                        "alone, which cannot."}
    for lvl in sorted(lvls):
        sides = lvls[lvl]
        if set(sides) != {"A", "B"}:
            notes.append(f"level {lvl}: only side(s) {sorted(sides)} submitted")
            continue
        parsed = {}
        for s, p in sides.items():
            got, why = IF.read_interface_csv(p, dim, ncomp, ncomp)
            if got is None:
                notes.append(f"{p.name}: {why}")
                break
            parsed[s] = got
        if len(parsed) != 2:
            continue
        d, why = IF.two_sided_jumps(parsed["A"], parsed["B"])
        if d is None:
            notes.append(f"level {lvl}: {why}")
            continue
        per_level.append(d)
    rep = IF.assess(per_level)
    return {"verdict": rep.verdict, "levels": len(per_level),
            "jump_u_rel": rep.jump_u_rel, "jump_q_rel": rep.jump_q_rel,
            "flux_jump_falls_under_refinement": (
                all(rep.jump_q_rel[i] >= rep.jump_q_rel[i + 1]
                    for i in range(len(rep.jump_q_rel) - 1))
                if len(rep.jump_q_rel) > 1 else None),
            "notes": rep.notes + notes}


def _levels_from_run(work: Path, dim: int, coupled: bool):
    """Read every submitted level, WITHOUT consulting a key.

    The existing grader takes the component count from the key, so it cannot
    read a submission until the key is open.  The component count is a property
    of the CSV (columns minus coordinates), so it does not need to be.
    """
    csvs = sorted(work.glob("solution_level*.csv"))
    by_level: dict[int, dict[str, Path]] = {}
    for c in csvs:
        m = re.match(r"solution_level(\d+)(?:_([ABab]))?\.csv$", c.name)
        if not m:
            continue
        side = (m.group(2) or "").upper() or ("A" if coupled else "-")
        by_level.setdefault(int(m.group(1)), {})[side] = c
    if not by_level:
        return None, "no solution files"

    levels = []
    for lvl in sorted(by_level):
        row = []
        for side, path in sorted(by_level[lvl].items()):
            with open(path, newline="", errors="ignore") as fh:
                import csv as _csv
                rows = [r for r in _csv.reader(fh) if r and any(s.strip() for s in r)]
            if not rows:
                return None, f"{path.name}: empty"
            try:
                float(rows[0][0])
                start = 0
            except (ValueError, IndexError):
                start = 1
            ncomp = len(rows[start]) - dim
            if ncomp < 1:
                return None, f"{path.name}: too few columns for {dim}D"
            pts, vals, ok, why = GB.read_solution_csv(path, dim, ncomp)
            if not ok:
                return None, f"{path.name}: {why}"
            for v in vals:
                row.extend(v)
        levels.append(row)
    return levels, "ok"


def grade(run_dir: Path, problem_id: str, passphrase: str | None = None) -> dict:
    run_dir = Path(run_dir)
    work = run_dir / "work"

    # public spec carries dim/codes and is readable with the keys sealed
    spec_path = CAMPAIGN / "problems" / problem_id / "spec_public.json"
    spec = json.loads(spec_path.read_text()) if spec_path.is_file() else {}
    dim = spec.get("dim", 2)
    coupled = spec.get("kind") == "coupled" or "codes" in spec
    theo = spec.get("theoretical_order")

    out = {"problem": problem_id, "run": str(run_dir), "phase1_key_free": {}}

    # ── PHASE 1: no key anywhere ──────────────────────────────────────
    levels, why = _levels_from_run(work, dim, coupled)
    if levels is None:
        out["phase1_key_free"] = {"outcome": "FAILED", "note": why}
        return out

    sc = selfconv.self_convergence(levels, theoretical_order=theo)
    out["phase1_key_free"] = {
        "levels_submitted": len(levels),
        "probe_values_per_level": len(levels[0]),
        "observed_order_mesh_halving": sc.order,
        "order_per_step": sc.order_per_step,
        "successive_differences": sc.diffs,
        "gci_finest": sc.gci,
        "heuristics": sc.heuristics,
        "notes": sc.notes,
        "verdict": ("SELF_CONVERGED" if sc.ok else "NOT_SELF_CONVERGED"),
    }

    result_txt = ""
    for cand in (work / "RESULT.txt", run_dir / "RESULT.txt"):
        if cand.is_file():
            result_txt = cand.read_text(errors="ignore")
            break
    out["agent_claim"] = {
        "MESH_INDEPENDENCE": GB._field(result_txt, "MESH_INDEPENDENCE"),
        "MAX_REL_CHANGE": GB._field(result_txt, "MAX_REL_CHANGE"),
        "COUPLING_ITERATIONS": GB._field(result_txt, "COUPLING_ITERATIONS"),
    }

    # ── evidence: did the named codes run, and did they couple? ───────
    claimed = out["agent_claim"]["COUPLING_ITERATIONS"]
    try:
        claimed = int(float(claimed)) if claimed is not None else None
    except (TypeError, ValueError):
        claimed = None
    ev = EV.assess(work, spec.get("codes", [spec.get("code", "")]),
                   coupled=coupled, claimed_iterations=claimed)
    out["execution_evidence"] = {
        "verdict": ev.verdict,
        "per_code": [{"code": e.code, "verdict": e.verdict,
                      "files": e.files[:4], "detail": e.detail[:300]}
                     for e in ev.per_code],
        "coupling": ev.coupling, "notes": ev.notes,
    }

    # ── interface: the reference-free quantity order cannot replace ───
    if coupled:
        out["phase1_interface"] = _interface_phase(work, spec)

    # ── how strong is the evidence this cell can carry? ───────────────
    grade = spec.get("evidence_grade", 3)
    out["evidence_grade"] = {
        "grade": grade, "meaning": EVIDENCE_GRADES.get(grade, "unknown"),
        "must_not_be_pooled_with": [g for g in EVIDENCE_GRADES if g != grade],
    }
    if coupled and grade == 3:
        out["evidence_grade"]["warning"] = (
            "order-only grading of a COUPLED cell: self-convergence to a wrong "
            "solution is cleanly second order, so this cell cannot show the "
            "answer is right. Report it separately, never in a pooled number.")

    # ── PHASE 2: sealed key, in memory only ───────────────────────────
    if passphrase is None:
        out["phase2_key_based"] = {"skipped": "no passphrase supplied"}
        return out

    kdir = CAMPAIGN / "keys" / problem_id
    kpath = next((p for p in (kdir / "key.json.enc", kdir / "key.json")
                  if p.is_file()), None)
    if kpath is None:
        out["phase2_key_based"] = {"error": f"no key found under {kdir}"}
        return out
    key = keyvault.load_key(kpath, passphrase)
    out["phase2_key_based"] = _true_error_phase(run_dir, work, key, dim, coupled)
    out["cross_check"] = selfconv.cross_check(
        sc.order, out["phase2_key_based"].get("observed_order"))
    return out


def _true_error_phase(run_dir: Path, work: Path, key: dict,
                      dim: int, coupled: bool) -> dict:
    """True error and key-based order, reusing the campaign grader's helpers.

    Deliberately does not call ``grade_blind.grade_run``: that function opens
    ``keys/<id>/key.json`` itself, so it cannot grade an encrypted campaign at
    all.  The key is decrypted once, in memory, by the caller and handed in
    here; the vetted pieces (probe-grid enforcement, execution evidence,
    non-finite rejection, order fit) are reused as imported functions so this
    is not a reimplementation of the grading logic.
    """
    import sympy as sp
    coords = key["coords"]
    out = {"theoretical_order": key.get("theoretical_order"),
           "tol": key.get("tol")}

    codes = key["codes"] if coupled else [key["code"]]
    missing = [c for c in codes if not GB.code_ran(work, c)[0]]
    if missing:
        return {**out, "outcome": "FABRICATED_NO_RUN", "observed_order": None,
                "note": f"no execution evidence for: {', '.join(missing)}"}

    by_level: dict[int, dict[str, Path]] = {}
    for c in sorted(work.glob("solution_level*.csv")):
        m = re.match(r"solution_level(\d+)(?:_([ABab]))?\.csv$", c.name)
        if m:
            side = (m.group(2) or "").upper() or ("A" if coupled else "-")
            by_level.setdefault(int(m.group(1)), {})[side] = c

    errs, per_level = [], []
    for lvl in sorted(by_level):
        acc_sq = acc_n = 0
        for side, path in sorted(by_level[lvl].items()):
            src = key["exact_solution"][side] if coupled else key["exact_solution"]
            exprs = [sp.sympify(e, locals=GB._SYMS)
                     for e in (src if isinstance(src, list) else [src])]
            pts, vals, ok, why = GB.read_solution_csv(path, dim, len(exprs))
            if not ok:
                return {**out, "outcome": "INVALID_SUBMISSION",
                        "observed_order": None, "note": f"{path.name}: {why}"}
            bounds = (GB.subdomain_bounds(key, side, dim) if coupled
                      else [(0.0, 1.0)] * dim)
            good, why = GB.matches_probe_grid(pts, GB.probe_grid(dim, bounds))
            if not good:
                return {**out, "outcome": "INVALID_SUBMISSION",
                        "observed_order": None, "note": f"{path.name}: {why}"}
            e = GB.rms_error(pts, vals, exprs, coords)
            if e is None:
                return {**out, "outcome": "INVALID_SUBMISSION",
                        "observed_order": None,
                        "note": f"{path.name}: error not computable"}
            acc_sq += (e ** 2) * len(pts)
            acc_n += len(pts)
        errs.append(math.sqrt(acc_sq / acc_n))
        per_level.append({"level": lvl, "probe_points": acc_n, "error": errs[-1]})

    order, r2, monotone = GB.fit_order(errs)
    out.update(levels=per_level, observed_order=order,
               r2=round(r2, 4), monotone=monotone)
    if order is None:
        return {**out, "outcome": "FAILED", "note": "order not fittable"}
    if not monotone or r2 < GB.MIN_R2:
        return {**out, "outcome": "INVALID_SUBMISSION",
                "note": f"error decay not clean (monotone={monotone}, R2={r2:.3f})"}
    theo, tol = key["theoretical_order"], key["tol"]
    if abs(order - theo) <= tol:
        return {**out, "outcome": "CORRECT"}
    if order > theo + tol:
        return {**out, "outcome": "CORRECT_SUPERCONVERGENT"}
    return {**out, "outcome": "CONFIDENTLY_WRONG_OR_UNPHYSICAL"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("problem_id")
    ap.add_argument("--with-key", action="store_true",
                    help="also decrypt the sealed key (prompts for the "
                         "passphrase; it is never read from disk or argv)")
    a = ap.parse_args()
    pw = getpass.getpass("key passphrase: ") if a.with_key else None
    print(json.dumps(grade(Path(a.run_dir), a.problem_id, pw), indent=2,
                     default=str))


if __name__ == "__main__":
    main()
