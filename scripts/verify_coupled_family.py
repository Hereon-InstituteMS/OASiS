#!/usr/bin/env python3
"""Solve every redesigned coupled instance and check that grading works on it.

Building a manufactured solution and verifying its algebra says the PROBLEM is
right.  It does not say the problem is GRADEABLE.  Two things still have to be
measured on each instance before it can be put in front of a paid run:

* that a correct partitioned solve really does produce the theoretical order on
  the prescribed mesh sequence and the prescribed probe grid, so a correct agent
  is not graded as wrong;
* that **phase 1 and phase 2 agree** — the key-free order from mesh halving and
  the key-based order from the true error.  A disagreement means the sealed
  solution, the problem statement, or the solve is wrong, and it is a class of
  error neither phase catches alone.

Instances are solved with the harness's own P1 solver
(``blind_eval.femdd``), on the meshes the task prescribes, with the interfaces
on mesh lines exactly as the task promises.  What cannot be solved with it is
reported as such rather than quietly omitted.

Run:
    .venv/bin/python scripts/verify_coupled_family.py [--only D1 D3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "campaign3_blind"))

from blind_eval import coupled as C            # noqa: E402
from blind_eval import femdd as F              # noqa: E402
import build_coupled_v2 as B                   # noqa: E402

x, y = B.x, B.y
# A DEMONSTRATION seed, deliberately not the campaign's.
#
# The campaign's hidden fields are drawn from a CSPRNG at build time and the
# draw seed is written only into the sealed key. Hard-coding that seed in a
# version-controlled file would let anyone with the repository re-derive every
# hidden field by re-running the builder -- which defeats the whole point of a
# builder that holds no answers. This file rebuilds instances from the same
# FAMILY to measure that the family is gradeable; it never touches the campaign
# instances. tests/test_blind_campaign_integrity.py fails if a tracked file
# ever contains a live draw seed.
SEED = 11111111
LEVELS = (8, 16, 32)


def _f(e, dim=2):
    v = (B.x, B.y, B.z)[:dim]
    fn = sp.lambdify(v, e, "numpy")

    def g(*a):
        return np.asarray(fn(*a), dtype=float) * np.ones_like(np.asarray(a[0], float))
    return g


def _probes(extent, M):
    return F.probe_grid_2d(tuple(extent[0]), tuple(extent[1]), M)


# ── the straight-interface scalar instances: full partitioned solve ───
def verify_dn(spec, fields, sources, mats, reaction_b=0.0):
    KA = np.array(sp.Matrix(mats[0]).tolist(), dtype=float)
    KB = np.array(sp.Matrix(mats[1]).tolist(), dtype=float)
    xi, lx = float(B.XI), float(B.LX)
    M = spec["probe_M"]
    PA, PB = _probes(spec["extent_a"], M), _probes(spec["extent_b"], M)
    exA = _f(fields["A"])(PA[:, 0], PA[:, 1])
    exB = _f(fields["B"])(PB[:, 0], PB[:, 1])

    vals, errs, jumps, iters = [], [], [], []
    for n in LEVELS:
        gA = F.Grid(0.0, xi, 0.0, 1.0, max(2, int(round(xi * n))), n)
        gB = F.Grid(xi, lx, 0.0, 1.0, max(2, int(round((lx - xi) * n))), n)
        A = F.Side(gA, KA, _f(sources["A"]), "x1", ("x0", "y0", "y1"))
        Bs = F.Side(gB, KB, _f(sources["B"]), "x0", ("x1", "y0", "y1"),
                    reaction=reaction_b)
        # relaxation from the conductance ratio, as the served knowledge says
        rho = (KA[0, 0] / xi) / (KB[0, 0] / (lx - xi))
        r = F.dn_couple(A, Bs, theta=1.0 / (1.0 + rho), max_iter=600, tol=1e-12)
        vA = F.evaluate(gA, r.uA, PA)
        vB = F.evaluate(gB, r.uB, PB)
        vals.append(np.concatenate([vA, vB]))
        errs.append(F.rms(np.concatenate([vA - exA, vB - exB])))
        denom = max(np.abs(r.flux_A).sum(), 1e-30)
        jumps.append(float(np.abs(r.flux_A + r.flux_B).sum() / denom))
        iters.append(r.iterations)
    diffs = [F.rms(vals[i] - vals[i + 1]) for i in range(len(vals) - 1)]
    return {"phase1_order_mesh_halving": F.order_from_halving(diffs),
            "phase2_order_true_error": F.order_from_halving(errs),
            "true_errors": errs, "flux_jump": jumps,
            "coupling_iterations": iters, "solver": "partitioned DN"}


# ── monolithic solves for the instances the DN driver cannot serve ────
def verify_monolithic_scalar(spec, fields, sources, k_of_cell, holes,
                             box, probe_sets):
    """One solve over the whole (possibly notched) domain with piecewise k."""
    def K(xc, yc):
        out = np.zeros((len(xc), 2, 2))
        kk = np.array([k_of_cell(a, b) for a, b in zip(xc, yc)], dtype=float)
        out[:, 0, 0] = out[:, 1, 1] = kk
        return out

    def src(xc, yc):
        out = np.zeros(len(xc))
        for (bx, by), e in sources.items():
            m = ((xc >= bx[0]) & (xc <= bx[1]) & (yc >= by[0]) & (yc <= by[1]))
            if m.any():
                out[m] = _f(e)(xc[m], yc[m])
        return out

    errs, vals = [], []
    for n in LEVELS:
        g = F.Grid(box[0][0], box[0][1], box[1][0], box[1][1],
                   int(round((box[0][1] - box[0][0]) * n)),
                   int(round((box[1][1] - box[1][0]) * n)), holes=tuple(holes))
        A, b = F.assemble(g, K, src)
        bd = g.boundary_nodes()
        u = F.solve_with_dirichlet(A, b, bd, np.zeros(len(bd)), dead=g.dead)
        row_v, row_e = [], []
        for pts, ex in probe_sets:
            got = F.evaluate(g, u, pts)
            row_v.append(got)
            row_e.append(got - ex)
        vals.append(np.concatenate(row_v))
        errs.append(F.rms(np.concatenate(row_e)))
    diffs = [F.rms(vals[i] - vals[i + 1]) for i in range(len(vals) - 1)]
    return {"phase1_order_mesh_halving": F.order_from_halving(diffs),
            "phase2_order_true_error": F.order_from_halving(errs),
            "true_errors": errs, "solver": "monolithic reference"}


def verify_d4(spec, fields, sources, mats):
    """Vector elasticity, monolithic, with the shear modulus jumping at x = XI."""
    (lam, muA), (_, muB) = mats
    xi, lx = float(B.XI), float(B.LX)
    lam_f, muA_f, muB_f = float(lam), float(muA), float(muB)
    M = spec["probe_M"]
    PA, PB = _probes(spec["extent_a"], M), _probes(spec["extent_b"], M)
    ex = {}
    for side, P in (("A", PA), ("B", PB)):
        ex[side] = np.column_stack([_f(c)(P[:, 0], P[:, 1])
                                    for c in fields[side]])

    fA = [_f(c) for c in sources["A"]]
    fB = [_f(c) for c in sources["B"]]

    def load(xc, yc):
        left = xc < xi
        fx = np.where(left, fA[0](xc, yc), fB[0](xc, yc))
        fy = np.where(left, fA[1](xc, yc), fB[1](xc, yc))
        return fx, fy

    errs, vals = [], []
    for n in LEVELS:
        g = F.Grid(0.0, lx, 0.0, 1.0, int(round(lx * n)), n)
        cen = g.pts[g.tris].mean(axis=1)  # monolithic: no relaxation involved
        mu = np.where(cen[:, 0] < xi, muA_f, muB_f)
        A, b = F.assemble_elasticity(g, lam_f, mu, load)
        bn = g.boundary_nodes()
        dof = np.concatenate([2 * bn, 2 * bn + 1])
        u = F.solve_with_dirichlet(A, b, dof, np.zeros(len(dof)))
        gotA = F.evaluate_vector(g, u, PA)
        gotB = F.evaluate_vector(g, u, PB)
        vals.append(np.concatenate([gotA.ravel(), gotB.ravel()]))
        errs.append(F.rms(np.concatenate([(gotA - ex["A"]).ravel(),
                                          (gotB - ex["B"]).ravel()])))
    diffs = [F.rms(vals[i] - vals[i + 1]) for i in range(len(vals) - 1)]
    return {"phase1_order_mesh_halving": F.order_from_halving(diffs),
            "phase2_order_true_error": F.order_from_halving(errs),
            "true_errors": errs, "solver": "monolithic reference (vector)"}


NOT_SOLVED_HERE = {
    "D2": "3D; the verification solver in blind_eval.femdd is 2D. Verified "
          "symbolically to exact zero and numerically by finite differences; "
          "its 2D analogue D3 is solved and graded here and shares the "
          "construction exactly.",
    "D8": "transient; the verification solver is steady. Verified symbolically "
          "and by finite differences including the time derivative; its steady "
          "analogue D3 is solved and graded here.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--from-keys", action="store_true",
                    help="verify the SHIPPED instances by loading their sealed "
                         "keys, rather than rebuilding from the family")
    a = ap.parse_args()
    if a.from_keys:
        return main_from_keys(Path("/home/alexander/Schreibtisch/"
                                   "qwen_uplift_test/campaign3_blind/keys"))

    out = {}
    for fn in B.BUILDERS:
        pid = fn.__name__.split("_")[-1]
        if a.only and pid not in a.only:
            continue
        d = B.Draw(SEED)
        spec, fields, sources, coords, mats = fn(d)
        if pid in NOT_SOLVED_HERE:
            out[pid] = {"solver": "not solved here",
                        "note": NOT_SOLVED_HERE[pid]}
        elif pid == "D4":
            out[pid] = verify_d4(spec, fields, sources, mats)
        elif pid == "D5":
            kv = {(0, 0): 1.0, (1, 0): 2.5, (1, 1): 5.0, (0, 1): 2.0}

            def k_of(xc, yc):
                return kv[(1 if xc > 0.5 else 0, 1 if yc > 0.5 else 0)]

            src = {}
            for (i, j), e in sources.items():
                bx = (0.5, 1.0) if i else (0.0, 0.5)
                by = (0.5, 1.0) if j else (0.0, 0.5)
                src[(bx, by)] = e
            # The SHIPPED probe rule, not a convenient substitute: subdomain A
            # is the 44x44 grid over the whole square minus the points inside B
            # and inside the notch, and B is its own 44x44 grid. Verifying on a
            # different probe set would measure a different thing from the one
            # the campaign grades.
            M = spec["probe_M"]
            PA = F.probe_grid_2d((0.0, 1.0), (0.0, 1.0), M)
            drop = (((PA[:, 0] > 0.5) & (PA[:, 1] < 0.5))
                    | ((PA[:, 0] > 0.75) & (PA[:, 1] > 0.75)))
            PA = PA[~drop]
            PB = F.probe_grid_2d((0.5, 1.0), (0.0, 0.5), M)

            def exact_at(P):
                out = np.zeros(len(P))
                for (i, j), e in fields.items():
                    m = (((P[:, 0] > 0.5) == bool(i))
                         & ((P[:, 1] > 0.5) == bool(j)))
                    if m.any():
                        out[m] = _f(e)(P[m, 0], P[m, 1])
                return out

            probe_sets = [(PA, exact_at(PA)), (PB, exact_at(PB))]
            out[pid] = verify_monolithic_scalar(
                spec, fields, src, k_of, [(0.75, 1.0, 0.75, 1.0)],
                [(0.0, 1.0), (0.0, 1.0)], probe_sets)
        else:
            out[pid] = verify_dn(spec, fields, sources, mats,
                                 reaction_b=12.0 if pid == "D7" else 0.0)
        out[pid]["evidence_grade"] = spec["evidence_grade"]
        out[pid]["contrast"] = spec["material_contrast"]

    hdr = (f"{'id':<4}{'grade':>6}{'contrast':>18}{'phase1':>9}{'phase2':>9}"
           f"{'agree':>7}{'flux jump':>11}  solver")
    print(hdr)
    print("-" * len(hdr))
    bad = []
    for pid, r in out.items():
        p1 = r.get("phase1_order_mesh_halving")
        p2 = r.get("phase2_order_true_error")
        agree = (abs(p1 - p2) <= 0.5) if (p1 and p2) else None
        fj = max(r["flux_jump"]) if r.get("flux_jump") else None
        print(f"{pid:<4}{r['evidence_grade']:>6}{str(r.get('contrast','')):>18}"
              f"{('%.3f' % p1) if p1 else '   --':>9}"
              f"{('%.3f' % p2) if p2 else '   --':>9}"
              f"{str(agree):>7}"
              f"{('%.2e' % fj) if fj is not None else '     --':>11}  "
              f"{r['solver']}")
        if r.get("note"):
            print(f"      {r['note']}")
        if p2 is not None and abs(p2 - 2.0) > 0.4:
            bad.append(f"{pid}: true-error order {p2:.3f} is outside the "
                       f"instance's own tolerance")
        if agree is False:
            bad.append(f"{pid}: phase 1 and phase 2 disagree ({p1:.3f} vs {p2:.3f})")
    dest = REPO / "data" / "coupled_family_verification.json"
    dest.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {dest}")
    if bad:
        print("\nPROBLEMS:")
        for b in bad:
            print("  -", b)
        return 1
    print("\nEvery solved instance grades at its theoretical order, and the "
          "key-free and key-based orders agree.")
    return 0




# ──────────────────────────────────────────────────────────────────────
# Verifying the instances that will ACTUALLY be graded
# ──────────────────────────────────────────────────────────────────────
# Rebuilding from a demonstration seed verifies the FAMILY. That is most of what
# matters — gradeability is a property of the construction, the geometry and the
# probe grid — but it is not the shipped instance. This mode loads the sealed
# key, solves the instance that will actually be graded, and prints only orders.
# The materials below are PUBLIC: every one of them is stated in the task text.
PUBLIC_MATERIALS = {
    "D1": ("tensor", [[1, sp.Rational(1, 2)], [sp.Rational(1, 2), 2]],
           [[3, sp.Rational(1, 2)], [sp.Rational(1, 2), 5]], 0.0),
    "D3": ("scalar", 1, 4, 0.0),
    "D6": ("scalar", 1, 1000, 0.0),
    "D7": ("scalar", 1, 3, 12.0),
}


def verify_from_key(pid, keys_dir: Path):
    """Solve the SHIPPED instance, from its sealed key. Prints orders only."""
    kp = keys_dir / pid / "key.json"
    if not kp.is_file():
        return {"solver": "key not readable (sealed or encrypted)",
                "note": "unseal to verify the shipped instance"}
    key = json.loads(kp.read_text())
    spec_path = (REPO / "campaign3_blind" / "problems" / pid /
                 "spec_public.json")
    spec = json.loads(spec_path.read_text())
    kind, ka, kb, react = PUBLIC_MATERIALS[pid]
    if kind == "tensor":
        KA, KB = sp.Matrix(ka), sp.Matrix(kb)
    else:
        KA, KB = sp.Integer(ka) * sp.eye(2), sp.Integer(kb) * sp.eye(2)
    fields = {s: sp.sympify(key["exact_solution"][s]) for s in ("A", "B")}
    sources = {s: sp.sympify(key["source_term"][s]) for s in ("A", "B")}
    out = verify_dn(spec, fields, sources, (KA, KB), reaction_b=react)
    out["evidence_grade"] = spec["evidence_grade"]
    out["contrast"] = spec["material_contrast"]
    out["source"] = "shipped instance, from its sealed key"
    return out


def main_from_keys(keys_dir: Path):
    out = {}
    for pid in sorted(PUBLIC_MATERIALS):
        out[pid] = verify_from_key(pid, keys_dir)
    bad = []
    print(f"{'id':<4}{'phase1':>9}{'phase2':>9}{'agree':>7}{'flux jump':>11}  source")
    for pid, r in out.items():
        p1 = r.get("phase1_order_mesh_halving")
        p2 = r.get("phase2_order_true_error")
        if p1 is None:
            print(f"{pid:<4}{'--':>9}{'--':>9}{'--':>7}{'--':>11}  {r['solver']}")
            continue
        agree = abs(p1 - p2) <= 0.5
        fj = max(r["flux_jump"])
        print(f"{pid:<4}{p1:9.3f}{p2:9.3f}{str(agree):>7}{fj:11.2e}  {r['source']}")
        if abs(p2 - 2.0) > 0.4 or not agree:
            bad.append(pid)
    dest = REPO / "data" / "coupled_shipped_instance_verification.json"
    dest.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {dest}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
