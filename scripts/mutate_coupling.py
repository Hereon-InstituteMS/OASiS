#!/usr/bin/env python3
"""Which coupled grading quantity actually detects a broken coupling?

The campaign grades coupled cells on the convergence order of the field and, in
the paper's band-graded cells, on the interface temperature.  Both were asserted
to be adequate.  This script tries to fool them.

Method: build a coupled problem, solve it with a real partitioned
Dirichlet-Neumann iteration, then break exactly one thing in the physics and
re-solve.  For every candidate grading quantity, measure how far the mutated run
moves.  A quantity that does not move is a quantity that cannot grade.

Candidates evaluated (not assumed):
    T_iface_scalar    the mean interface value            -- what the paper bands
    T_iface_profile   the whole interface value profile
    u_order_halving   order of the field from mesh halving -- the primary grade
    u_true_error      RMS error against the sealed field   -- phase 2
    q_iface_scalar    the net interface flux
    q_iface_profile   the interface flux profile
    q_jump_two_sided  |q_A + q_B| / |q_A|, reference-free

Mutations applied:
    MUT_KRATIO   export the raw normal derivative, import it after multiplying
                 by the receiver's own conductivity.  EXACTLY the identity when
                 the two subdomains share a material -- which is why the
                 pre-existing D1/D2/D4 cannot see it.
    MUT_SIGN     the transferred flux arrives with the wrong sign
    MUT_MAP      the interface mapping is reversed
    MUT_KB       the receiver solves with a 25% wrong conductivity
    MUT_SCALE    both conductivities scaled together (conductance ratio
                 preserved) -- the mutation that leaves interface temperature
                 exactly invariant in the Dirichlet-driven problem the paper's
                 band-graded coupled cells use

Run:
    .venv/bin/python scripts/mutate_coupling.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from blind_eval import coupled as C            # noqa: E402
from blind_eval import femdd as F              # noqa: E402

x, y = C.X, C.Y
LX, XI = sp.Rational(3, 2), sp.Rational(3, 5)
LEVELS = (8, 16, 32)
PROBE_M = 45            # incommensurate with every mesh level -- see report


# ──────────────────────────────────────────────────────────────────────
def instance(kA, kB, tag):
    """A coupled instance: two materials, one hidden field, exact by design.

    ``kA == kB`` reproduces the shape of the pre-existing D1/D2/D4, where the
    same field and the same material sit on both sides.
    """
    pm = C.ProductMaterial([XI], [kA, kB], [], [1])
    H, Zt = pm.eta_at(LX), pm.zeta_at(1)

    def U(s, t):
        return s * (H - s) * t * (Zt - t) * (sp.Rational(2, 3)
                                             + sp.Rational(1, 5) * s
                                             + sp.Rational(3, 7) * t)

    cells = pm.field(U)
    uA, uB = cells[(0, 0)], cells[(1, 0)]
    KA = sp.nsimplify(kA) * sp.eye(2)
    KB = sp.nsimplify(kB) * sp.eye(2)
    fA = C.poisson_source(uA, (x, y), KA)
    fB = C.poisson_source(uB, (x, y), KB)
    rep = C.check_scalar_transmission(uA, uB, KA, KB, (x, y), x, XI)
    return dict(tag=tag, kA=float(kA), kB=float(kB), uA=uA, uB=uB,
                fA=fA, fB=fB, KA=KA, KB=KB, transmission=rep)


def _lam(e):
    fn = sp.lambdify((x, y), e, "numpy")

    def g(X, Yv):
        return np.asarray(fn(X, Yv), dtype=float) * np.ones_like(np.asarray(X, float))
    return g


def solve(inst, n, **mut):
    """One partitioned Dirichlet-Neumann solve at mesh level ``n``."""
    xi, lx = float(XI), float(LX)
    nA = max(2, int(round(xi * n)))
    nB = max(2, int(round((lx - xi) * n)))
    gA = F.Grid(0.0, xi, 0.0, 1.0, nA, n)
    gB = F.Grid(xi, lx, 0.0, 1.0, nB, n)
    A = F.Side(gA, np.eye(2) * inst["kA"], _lam(inst["fA"]), "x1", ("x0", "y0", "y1"))
    kB_used = mut.pop("kB_used", inst["kB"])
    B = F.Side(gB, np.eye(2) * kB_used, _lam(inst["fB"]), "x0", ("x1", "y0", "y1"))
    res = F.dn_couple(A, B, theta=0.5, max_iter=400, tol=1e-12, **mut)
    return gA, gB, res


def quantities(inst, per_level):
    """Every candidate grading quantity, from the same set of solves."""
    kA, kB = inst["kA"], inst["kB"]
    ufA, ufB = _lam(inst["uA"]), _lam(inst["uB"])
    bA = ((0.0, float(XI)), (0.0, 1.0))
    bB = ((float(XI), float(LX)), (0.0, 1.0))
    PA = F.probe_grid_2d(*bA, PROBE_M)
    PB = F.probe_grid_2d(*bB, PROBE_M)
    exA, exB = ufA(PA[:, 0], PA[:, 1]), ufB(PB[:, 0], PB[:, 1])

    vals, errs, tface, qface, qjump = [], [], [], [], []
    for gA, gB, r in per_level:
        vA = F.evaluate(gA, r.uA, PA)
        vB = F.evaluate(gB, r.uB, PB)
        vals.append(np.concatenate([vA, vB]))
        errs.append(F.rms(np.concatenate([vA - exA, vB - exB])))
        tface.append(r.trace_A.copy())
        # nodal flux functionals are integrals of q against the hat functions;
        # dividing by the nodal "length" makes them a comparable density
        w = np.full(len(r.iface_y), 1.0 / max(len(r.iface_y) - 1, 1))
        w[0] = w[-1] = 0.5 / max(len(r.iface_y) - 1, 1)
        qface.append(r.flux_A / w)
        denom = max(np.abs(r.flux_A).sum(), 1e-30)
        qjump.append(float(np.abs(r.flux_A + r.flux_B).sum() / denom))

    exact_t = _lam(inst["uA"].subs(x, XI))(per_level[-1][2].iface_y * 0 + 0,
                                           per_level[-1][2].iface_y)
    return dict(
        T_iface_scalar=float(np.mean(tface[-1])),
        T_iface_profile=tface[-1],
        T_iface_exact=float(np.mean(exact_t)),
        u_order_halving=F.order_from_halving(
            [F.rms(vals[i] - vals[i + 1]) for i in range(len(vals) - 1)] or [1, 1])
        if len(vals) >= 3 else None,
        u_selfconv_diffs=[F.rms(vals[i] - vals[i + 1]) for i in range(len(vals) - 1)],
        u_true_error=errs,
        u_true_order=F.order_from_halving(errs),
        q_iface_scalar=float(np.sum(per_level[-1][2].flux_A)),
        q_iface_profile=qface[-1],
        q_jump_two_sided=qjump[-1],
        q_jump_all_levels=qjump,
        iterations=[r.iterations for _, _, r in per_level],
        converged=[bool(r.converged) for _, _, r in per_level],
    )


def rel(a, b) -> float:
    a, b = np.atleast_1d(np.asarray(a, float)), np.atleast_1d(np.asarray(b, float))
    d = np.linalg.norm(a - b)
    s = max(np.linalg.norm(b), 1e-30)
    return float(d / s)


MUTATIONS = {
    "correct": lambda inst: {},
    "MUT_KRATIO": lambda inst: dict(export_scale_A=1.0 / inst["kA"],
                                    import_scale_B=inst["kB"]),
    "MUT_SIGN": lambda inst: dict(import_scale_B=-1.0),
    "MUT_MAP": lambda inst: dict(flip_map=True),
    "MUT_KB": lambda inst: dict(kB_used=1.25 * inst["kB"]),
}


def run_instance(inst) -> dict:
    out = {"tag": inst["tag"], "kA": inst["kA"], "kB": inst["kB"],
           "transmission": inst["transmission"].summary(),
           "transmission_vacuous": inst["transmission"].vacuous,
           "variants": {}}
    base = None
    for name, mk in MUTATIONS.items():
        per = []
        ok = True
        for n in LEVELS:
            try:
                per.append(solve(inst, n, **dict(mk(inst))))
            except Exception as exc:                      # a diverged mutation
                ok = False
                out["variants"][name] = {"error": f"{type(exc).__name__}: {exc}"}
                break
        if not ok:
            continue
        q = quantities(inst, per)
        if name == "correct":
            base = q
        q["moved_vs_correct"] = {
            k: rel(q[k], base[k]) for k in
            ("T_iface_scalar", "T_iface_profile", "q_iface_scalar",
             "q_iface_profile")
        } if base is not None else {}
        out["variants"][name] = q
    return out


def _fmt(v):
    if v is None:
        return "   --  "
    return f"{v:8.4f}"


def report(res: dict):
    print(f"\n=== {res['tag']}   k_A={res['kA']:g}  k_B={res['kB']:g} ===")
    print(f"    transmission check: {res['transmission']}")
    hdr = (f"{'variant':<12} {'T_iface':>10} {'moved_T%':>9} {'q_net':>11} "
           f"{'moved_q%':>9} {'jump_q':>9} {'order_u':>8} {'true_ord':>9} "
           f"{'iters':>6}")
    print(hdr)
    print("-" * len(hdr))
    for name, q in res["variants"].items():
        if "error" in q:
            print(f"{name:<12} DIVERGED / {q['error'][:60]}")
            continue
        mv = q.get("moved_vs_correct", {})
        print(f"{name:<12} {q['T_iface_scalar']:10.6f} "
              f"{100 * mv.get('T_iface_scalar', 0):9.3f} "
              f"{q['q_iface_scalar']:11.6f} "
              f"{100 * mv.get('q_iface_scalar', 0):9.3f} "
              f"{q['q_jump_two_sided']:9.2e} "
              f"{_fmt(q['u_order_halving'])} {_fmt(q['u_true_order'])} "
              f"{max(q['iterations']):6d}")


def band_demonstration():
    """The conductance-ratio-preserving mutation, in the setting the paper bands.

    The band-graded coupled cells use a Dirichlet-driven split conduction
    problem with no source term.  There the interface temperature depends on the
    two conductivities ONLY through their ratio, so scaling both leaves it
    exactly invariant while the flux scales with them.  A +-20% band on
    interface temperature therefore cannot separate a correct coupling from one
    with 100% wrong material data.
    """
    print("\n=== the paper's band-graded quantity, attacked exactly ===")
    xl, xi, xr, tl, tr = 0.0, 0.6, 1.1, 320.0, 300.0

    def closed_form(kl, kr):
        cl, cr = kl / (xi - xl), kr / (xr - xi)
        t = (cl * tl + cr * tr) / (cl + cr)
        return t, cl * (tl - t)

    rows = []
    t0, q0 = closed_form(0.8, 1.5)
    for s in (1.0, 2.0, 5.0, 0.25):
        t, q = closed_form(0.8 * s, 1.5 * s)
        rows.append((s, t, 100 * abs(t - t0) / abs(t0),
                     q, 100 * abs(q - q0) / abs(q0)))
    print(f"{'scale':>6} {'T_iface':>12} {'T moved %':>10} "
          f"{'q_iface':>12} {'q moved %':>10}  {'+-20% band on T':>16}")
    for s, t, dt, q, dq in rows:
        verdict = "PASSES" if dt <= 20 else "fails"
        print(f"{s:6.2f} {t:12.6f} {dt:10.4f} {q:12.6f} {dq:10.4f}  {verdict:>16}")
    print("  -> every ratio-preserving conductivity error passes a band on T,")
    print("     and every one of them is caught by the flux.")
    return rows


def main():
    print(__doc__.split("Run:")[0])
    results = []
    for kA, kB, tag in ((1, 1, "OLD SHAPE  same material both sides (D1/D2/D4)"),
                        (1, 4, "NEW SHAPE  two-material interface (D3-style)")):
        inst = instance(sp.Integer(kA), sp.Integer(kB), tag)
        r = run_instance(inst)
        report(r)
        results.append(r)

    band = band_demonstration()

    out = REPO / "data" / "coupled_grading_sensitivity.json"
    ser = []
    for r in results:
        rr = {"tag": r["tag"], "kA": r["kA"], "kB": r["kB"],
              "transmission": r["transmission"],
              "transmission_vacuous": r["transmission_vacuous"], "variants": {}}
        for n, q in r["variants"].items():
            if "error" in q:
                rr["variants"][n] = q
                continue
            rr["variants"][n] = {
                k: (v if not isinstance(v, np.ndarray) else None)
                for k, v in q.items() if k != "T_iface_profile"
                and k != "q_iface_profile"}
            rr["variants"][n]["moved_vs_correct"] = q.get("moved_vs_correct", {})
        ser.append(rr)
    out.write_text(json.dumps({"instances": ser,
                               "band_demonstration": band}, indent=2,
                              default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
