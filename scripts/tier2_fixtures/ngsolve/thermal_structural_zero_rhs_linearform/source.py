"""Tier-2: LinearForm(0*v*dx) collapses to a form with no TestFunction.

Claim: ngsolve thermal_structural#4 — a symbolic-zero RHS, LinearForm(0*v*dx),
collapses before construction, leaving a form with no TestFunction, and
Assemble() then raises NgException 'Linearform must have TestFunction'. Use the
no-integrand constructor LinearForm(V); f.Assemble() to build an empty RHS.

Wrong variant: LinearForm(ZERO_COEFFICIENT * vT * dx).Assemble() with
ZERO_COEFFICIENT = 0, i.e. the shipped pure-conduction RHS written the obvious
way.

Observed on NGSolve 6.2.2604 (2026-08-03), unit_square maxh=0.3,
H1(order=2, dirichlet='left|right'):
  * NgException, str() exactly 'Linearform must have TestFunction';
  * a nonzero constant (1e-30 * v * dx) is NOT affected -- the collapse is
    symbolic, triggered by the literal 0, not by smallness;
  * LinearForm(V); f.Assemble() succeeds and yields an all-zero vector, which is
    then usable in the template's heat solve: the lifted Dirichlet solution
    still reaches T_max = 100.

Mutation control: T2_MUTATE=1 sets ZERO_COEFFICIENT to 1.0 instead of the
literal 0, so the integrand no longer collapses symbolically and the form
keeps its TestFunction.  'zero_integrand_raises=True', 'exc_type=NgException'
and 'Linearform must have TestFunction' then disappear.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import ngsolve as ngs
from netgen.geom2d import unit_square

# WRONG: the literal 0 that makes the integrand collapse symbolically
MUTATE = os.environ.get("T2_MUTATE") == "1"

# Mutation: T2_MUTATE=1 makes the coefficient nonzero, so the integrand no
# longer collapses symbolically and the LinearForm keeps its TestFunction.
ZERO_COEFFICIENT = 1.0 if MUTATE else 0

T_HOT, T_COLD = 100.0, 0.0


def main() -> int:
    ok = True
    print(f"ngsolve_version={ngs.__version__}")
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.3))
    V_T = ngs.H1(mesh, order=2, dirichlet="left|right")
    uT, vT = V_T.TnT()
    print(f"thermal_space_type={V_T.type} ndof={V_T.ndof}")

    # --- WRONG variant: symbolic-zero integrand ---------------------------
    msg, typ = "", ""
    try:
        f_bad = ngs.LinearForm(ZERO_COEFFICIENT * vT * ngs.dx)
        f_bad.Assemble()
    except Exception as exc:                        # noqa: BLE001
        msg, typ = str(exc), type(exc).__name__
    print(f"zero_integrand_raises={bool(msg)} exc_type={typ} msg={msg!r}")
    if "Linearform must have TestFunction" not in msg:
        print(f"FAIL: LinearForm(0*v*dx).Assemble() did not raise the "
              f"documented NgException; got {msg!r}", file=sys.stderr)
        ok = False

    # the collapse is symbolic, not a smallness threshold
    tiny_msg = ""
    try:
        f_tiny = ngs.LinearForm(1e-30 * vT * ngs.dx)
        f_tiny.Assemble()
    except Exception as exc:                        # noqa: BLE001
        tiny_msg = str(exc)
    print(f"tiny_but_nonzero_coefficient_assembles={not tiny_msg}")
    if tiny_msg:
        print(f"FAIL: a nonzero 1e-30 coefficient also failed ({tiny_msg!r}), "
              f"so the collapse is not the symbolic zero", file=sys.stderr)
        ok = False

    # --- RIGHT variant: no-integrand constructor --------------------------
    f_ok = ngs.LinearForm(V_T)
    f_ok.Assemble()
    nrm = float(ngs.Norm(f_ok.vec))
    print(f"no_integrand_ctor_assembles=True rhs_norm={nrm:.3e}")
    print(f"no_integrand_ctor_rhs_is_zero={nrm == 0.0}")
    if nrm != 0.0:
        print(f"FAIL: LinearForm(V).Assemble() produced a nonzero RHS "
              f"({nrm:.3e})", file=sys.stderr)
        ok = False

    # and it carries the template's heat solve
    a = ngs.BilinearForm(ngs.grad(uT) * ngs.grad(vT) * ngs.dx).Assemble()
    gfT = ngs.GridFunction(V_T)
    gfT.Set(ngs.IfPos(0.5 - ngs.x, T_HOT, T_COLD),
            definedon=mesh.Boundaries("left|right"))
    f_ok.vec.data -= a.mat * gfT.vec
    gfT.vec.data += a.mat.Inverse(V_T.FreeDofs()) * f_ok.vec
    t_max = float(max(gfT.vec))
    print(f"heat_solve_t_max={t_max:.6g}")
    print(f"empty_rhs_still_solves_conduction={abs(t_max - T_HOT) < 1e-9}")
    if abs(t_max - T_HOT) >= 1e-9:
        print(f"FAIL: the empty-RHS heat solve gave T_max={t_max:.6g}",
              file=sys.stderr)
        ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
