#!/usr/bin/env python3
"""Rebuild the campaign-3 instances whose source term discloses the solution.

Three of the eleven pre-registered blind problems fail the leak gate:

    B1  one PRINTED source term is 12*pi**2 * u_exact  (naked-eye)
    D3  two/three printed terms sum to pi**2 * u_A and 4*pi**2 * u_B (naked-eye)
    B2  4*pi**2 * u_exact survives expansion (needs one expand(), not naked-eye)

``DESIGN.md`` admits B1 and D3 ("two instances leak by naked inspection") and
does not know about B2.  All three share one root cause:

    **the hidden field is an eigenfunction of the operator in at least one
    direction**, so applying the operator returns a multiple of the field
    itself, which then sits in the published source term.

``B1``/``B2`` use ``cos(2 pi x)`` / ``sin(pi x) sin(pi y)``; ``D3`` uses a
``sin(pi y)`` transverse profile, and ``-k d2/dy2`` of it is ``k pi^2`` times
itself.  The fix is therefore not cosmetic: replace the eigenfunction factors
with profiles that vanish on the boundary (so the boundary data still discloses
nothing) but are **not** eigenfunctions, so the operator maps them somewhere
else and the source term stops carrying a copy of the answer.

Physics, backend, element, mesh sequence, theoretical order and geometry are
all preserved, so the problem set keeps its intended shape.  Only the hidden
field changes.  Nothing is invalidated by this: the campaign has produced zero
runs.

Run:
    .venv/bin/python scripts/blind_rebuild_leaky.py [--apply]

Without ``--apply`` it verifies and reports, writing nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sympy as sp

CAMPAIGN = Path("/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind")
REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))
sys.path.insert(0, str(CAMPAIGN))

from blind_eval.leakgate import scan                      # noqa: E402
from blind_eval.derive import is_proportional             # noqa: E402

import build_problems as BP                               # noqa: E402
import build_coupled as BC                                # noqa: E402

x, y = BP.x, BP.y


# ── replacement hidden fields ─────────────────────────────────────────
# Each vanishes on the whole boundary (so "u = 0 on the boundary" stays the
# honest, zero-disclosure boundary datum) and none is an eigenfunction of its
# operator.  The exponential modulation is what breaks the eigenfunction
# relation while keeping the field smooth and O(1).
def field_B1():
    """Anisotropic diffusion, K = [[3,-1],[-1,2]] — was xy(1-x)(1-y)cos(2 pi x)."""
    return 4 * x * (1 - x) * y * (1 - y) * sp.exp(x + 2 * y)


def field_B2():
    """Variable-coefficient diffusion — was sin(pi x) sin(pi y)(1 + xy/5).

    The coefficient a(x,y) = 2 + exp(-x) sin(pi y) is UNCHANGED, so the
    published source term still contains sin(pi y).  That is deliberate: it
    keeps a live negative control in the problem set for the gate's central
    distinction — trigonometry in the source term is not a leak.
    """
    return 4 * x * (1 - x) * y * (1 - y) * sp.exp(2 * x - y)


def shape_D3():
    """Transverse profile for the two-material problem — was sin(pi y).

    ``-k d2/dy2 sin(pi y) = k pi^2 sin(pi y)`` is precisely the eigenfunction
    relation that put ``pi^2 * u`` into D3's source term.  ``y(1-y)exp(y)``
    still vanishes at y = 0 and y = 1 and is not an eigenfunction.
    """
    return y * (1 - y) * sp.exp(y)


def build_B1():
    spec, _, _, _, _, coords, K = BP.problem_B1()
    u = field_B1()
    f = BP.diffusion_source(u, coords, K)
    ok, residual = BP.verify_diffusion(u, f, coords, K)
    return spec, u, f, ok, residual, BP.build_scalar_task(spec, f)


def build_B2():
    spec, _, _, _, _, coords, K = BP.problem_B2()
    u = field_B2()
    f = BP.diffusion_source(u, coords, K)
    ok, residual = BP.verify_diffusion(u, f, coords, K)
    return spec, u, f, ok, residual, BP.build_scalar_task(spec, f)


def build_D3():
    """Re-solve the interface coefficients against the new transverse profile.

    The transmission conditions are solved symbolically, exactly as the original
    builder does, because hand-carried constants are how a construction quietly
    stops satisfying its own interface conditions.
    """
    coords, xi, LX = (x, y), BC.XI, BC.LX
    kA, kB = sp.Integer(1), sp.Integer(4)
    shape = shape_D3()
    a = x + x ** 2
    c0, c1, c2 = sp.symbols("c0 c1 c2")
    b = c0 + c1 * (x - xi) + c2 * (x - xi) ** 2
    sol = sp.solve([
        sp.Eq(b.subs(x, xi), a.subs(x, xi)),                       # continuity
        sp.Eq(kB * sp.diff(b, x).subs(x, xi),
              kA * sp.diff(a, x).subs(x, xi)),                     # flux match
        sp.Eq(b.subs(x, LX), 0),                                   # far wall
    ], [c0, c1, c2], dict=True)
    assert len(sol) == 1, f"interface conditions did not determine B: {sol}"
    uA, uB = a * shape, b.subs(sol[0]) * shape
    KA, KB = kA * sp.eye(2), kB * sp.eye(2)
    fA = sp.simplify(BP.diffusion_source(uA, coords, KA))
    fB = sp.simplify(BP.diffusion_source(uB, coords, KB))

    checks = {
        "residual_A": sp.simplify(
            -BP.div_vec(KA * BP.grad(uA, coords), coords) - fA) == 0,
        "residual_B": sp.simplify(
            -BP.div_vec(KB * BP.grad(uB, coords), coords) - fB) == 0,
        "continuity": sp.simplify(uA.subs(x, xi) - uB.subs(x, xi)) == 0,
        "flux_continuity": sp.simplify(
            (KA * BP.grad(uA, coords))[0].subs(x, xi)
            - (KB * BP.grad(uB, coords))[0].subs(x, xi)) == 0,
        "outer_bc_x0": sp.simplify(uA.subs(x, 0)) == 0,
        "outer_bc_xL": sp.simplify(uB.subs(x, LX)) == 0,
        "outer_bc_y0": sp.simplify(uA.subs(y, 0)) == 0 and sp.simplify(uB.subs(y, 0)) == 0,
        "outer_bc_y1": sp.simplify(uA.subs(y, 1)) == 0 and sp.simplify(uB.subs(y, 1)) == 0,
    }
    return uA, uB, fA, fB, checks


def gate(pid, task, exact, source):
    key = {"exact_solution": exact, "source_term": source}
    return scan(task, key, pid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the rebuilt problems and keys")
    args = ap.parse_args()

    results, failures = [], []

    for pid, builder in (("B1", build_B1), ("B2", build_B2)):
        spec, u, f, ok, residual, task = builder()
        homog = BP.is_homogeneous_on_boundary(u)
        rep = gate(pid, task, sp.sstr(u), sp.sstr(f))
        prop, _ = is_proportional(f, u)
        good = ok and homog and rep.clean and not prop
        results.append((pid, ok, homog, rep, task, spec, u, f))
        print(f"[{pid}] residual_zero={ok} homogeneous_bc={homog} "
              f"gate={'CLEAN' if rep.clean else 'LEAK'} "
              f"findings={[fi.rule for fi in rep.findings]}")
        if not good:
            failures.append(pid)

    uA, uB, fA, fB, checks = build_D3()
    d3_task = (CAMPAIGN / "problems" / "D3" / "task.txt").read_text()
    old = json.loads((CAMPAIGN / "keys" / "D3" / "key.json").read_text())
    new_task = d3_task
    for side, old_f, new_f in (("A", old["source_term"]["A"], sp.sstr(fA)),
                               ("B", old["source_term"]["B"], sp.sstr(fB))):
        assert old_f in new_task, f"D3 task text does not contain source {side}"
        new_task = new_task.replace(old_f, new_f)
    rep3 = gate("D3", new_task, {"A": sp.sstr(uA), "B": sp.sstr(uB)},
                {"A": sp.sstr(fA), "B": sp.sstr(fB)})
    all_checks = all(checks.values())
    print(f"[D3] {'  '.join(f'{k}={v}' for k, v in checks.items())}")
    print(f"[D3] gate={'CLEAN' if rep3.clean else 'LEAK'} "
          f"findings={[fi.rule for fi in rep3.findings]}")
    if not (all_checks and rep3.clean):
        failures.append("D3")

    if failures:
        print(f"\nREFUSING: {failures} did not pass. Nothing written.")
        return 1
    print("\nAll three rebuilt instances verify and pass the leak gate.")

    if not args.apply:
        print("(dry run — pass --apply to write)")
        return 0

    for pid, ok, homog, rep, task, spec, u, f in results:
        (CAMPAIGN / "problems" / pid / "task.txt").write_text(task, encoding="utf-8")
        kp = CAMPAIGN / "keys" / pid / "key.json"
        key = json.loads(kp.read_text())
        key.update(exact_solution=sp.sstr(u), source_term=sp.sstr(f),
                   residual="0", source_term_verified=True,
                   homogeneous_boundary=homog,
                   rebuilt_reason="source term embedded a multiple of the "
                                  "hidden solution; eigenfunction factor replaced")
        kp.write_text(json.dumps(key, indent=2), encoding="utf-8")
        print(f"wrote {pid}")

    (CAMPAIGN / "problems" / "D3" / "task.txt").write_text(new_task, encoding="utf-8")
    k3 = json.loads((CAMPAIGN / "keys" / "D3" / "key.json").read_text())
    k3.update(exact_solution={"A": sp.sstr(uA), "B": sp.sstr(uB)},
              source_term={"A": sp.sstr(fA), "B": sp.sstr(fB)},
              verification_detail={k: bool(v) for k, v in checks.items()},
              rebuilt_reason="sin(pi y) transverse profile is an eigenfunction "
                             "of -k d2/dy2, putting k*pi^2*u into the source term")
    (CAMPAIGN / "keys" / "D3" / "key.json").write_text(
        json.dumps(k3, indent=2), encoding="utf-8")
    print("wrote D3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
