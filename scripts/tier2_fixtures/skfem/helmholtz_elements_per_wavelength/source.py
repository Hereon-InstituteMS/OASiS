"""Tier-2: fewer than 5 elements per wavelength gives a visibly wrong phase velocity.

Claim: skfem helmholtz#1 — use at least 10 elements per wavelength
(lambda = 2*pi/k); below 5 elements per wavelength the computed phase velocity is
visibly wrong (a propagating wave reaches the boundary at the wrong time by
10-30%); the dispersion error is O(k h)^2 for P1 and becomes catastrophic for
h*k > 1.

Wrong variant: the same outgoing-wave problem is discretised at 3 elements per
wavelength (k*h = 2.09, i.e. h*k > 1) and the discrete wavenumber k_h is
measured two independent ways — from the assembled interior stencil (a root of
``row(A) @ exp(-i*kappa*x) = 0``, free of any boundary influence) and from the
end-to-end complex solve of the driven strip.  Both are compared against the
analytic P1 dispersion relation cos(k_h h) = (1-(kh)^2/3)/(1+(kh)^2/6).

Scaled-down configuration: a one-element-thick MeshQuad strip, k=8, 4
wavelengths long (26 DOFs at the coarse resolution, 642 at the finest).  The
strip carries the 1D dispersion of a 2D mesh's propagation direction exactly —
for a y-constant field the Q1 tensor operator reduces to h_y*(K_x - k^2 M_x) —
at a fraction of a full 2D mesh's cost.

Observed on skfem 12.0.1 (2026-08-06): 3 elements/wavelength -> k_h/k = 0.879,
i.e. the wave arrives 12% early; 20 elements/wavelength -> 0.41% error; the
error decays at rate ~2 in h; and above k*h = 2*sqrt(3) ~ 3.46 the discrete
operator admits no propagating mode at all.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.optimize import brentq
from skfem import (Basis, BilinearForm, ElementQuad1, FacetBasis, MeshQuad,
                   asm, condense, solve)

K_WAVE = 8.0
N_WAVE = 4
NPPW_COARSE = 3.0     # the mistake: below the rule of thumb of 10 (and of 5)
NPPW_FINE = 20.0
NPPW_RATE = (5.0, 10.0, 20.0, 40.0)
NPPW_NO_MODE = 1.5    # k*h = 4.19 > 2*sqrt(3): no propagating discrete mode


def strip(nppw: float):
    lam = 2.0 * np.pi / K_WAVE
    L = N_WAVE * lam
    nx = int(round(nppw * N_WAVE))
    h = L / nx
    m = (MeshQuad.init_tensor(np.linspace(0.0, L, nx + 1), np.array([0.0, h]))
         .with_boundaries({"left": lambda x: x[0] < 1e-9 * L,
                           "right": lambda x: x[0] > L - 1e-9 * L}))
    e = ElementQuad1()
    b = Basis(m, e)

    @BilinearForm(dtype=complex)
    def helmholtz(u, v, w):
        return (u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1]
                - K_WAVE ** 2 * u * v)

    @BilinearForm(dtype=complex)
    def abc(u, v, w):
        return 1j * K_WAVE * u * v

    A_int = asm(helmholtz, b)
    A = A_int + asm(abc, FacetBasis(m, e, facets="right"))
    return m, b, A_int.tocsr(), A, L, h


def stencil_wavenumber(m, A_int, L, h):
    """Root of the assembled interior stencil for a plane-wave ansatz."""
    xs = m.p[0]
    bottom = np.where(m.p[1] < 1e-14)[0]
    j = int(bottom[np.argmin(np.abs(xs[bottom] - 0.5 * L))])
    row = A_int[j]

    def resid(kappa: float) -> float:
        w = np.exp(-1j * kappa * xs)
        r = np.asarray(row @ w).ravel()[0] / np.exp(-1j * kappa * xs[j])
        return float(r.real)

    grid = np.linspace(1e-9, np.pi / h - 1e-9, 1201)
    vals = np.array([resid(g) for g in grid])
    flips = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
    if flips.size == 0:
        return None
    i0 = int(flips[0])
    return float(brentq(resid, grid[i0], grid[i0 + 1], xtol=1e-14))


def solve_wavenumber(m, b, A, L):
    f = np.zeros(A.shape[0], dtype=complex)
    D = b.get_dofs("left").flatten()
    x = np.zeros(A.shape[0], dtype=complex)
    x[D] = 1.0
    u = solve(*condense(A, f, x=x, D=D))
    sel = m.p[1] < 1e-14
    xs = m.p[0][sel]
    us = u[:m.p.shape[1]][sel]
    o = np.argsort(xs)
    ph = np.unwrap(np.angle(us[o]))
    return -(ph[-1] - ph[0]) / (xs[o][-1] - xs[o][0])


def analytic_p1(kh: float):
    c = (1.0 - kh ** 2 / 3.0) / (1.0 + kh ** 2 / 6.0)
    return None if abs(c) > 1.0 else float(np.arccos(c))


def report(nppw: float):
    m, b, A_int, A, L, h = strip(nppw)
    kh_stencil = stencil_wavenumber(m, A_int, L, h)
    kh_solve = solve_wavenumber(m, b, A, L)
    return dict(nppw=nppw, h=h, kh_mesh=K_WAVE * h, ndofs=b.N,
                k_stencil=kh_stencil, k_solve=kh_solve,
                err=(None if kh_stencil is None
                     else abs(kh_stencil - K_WAVE) / K_WAVE))


def main() -> int:
    ok = True

    # --- WRONG variant: 3 elements per wavelength ---------------------------
    c = report(NPPW_COARSE)
    print(f"mesh_class=MeshQuad1 element_class=ElementQuad1")
    print(f"k={K_WAVE} n_wavelengths={N_WAVE}")
    print(f"coarse_elements_per_wavelength={NPPW_COARSE:g} "
          f"coarse_n_dofs={c['ndofs']}")
    print(f"coarse_kh={c['kh_mesh']:.4f} coarse_kh_gt_1={c['kh_mesh'] > 1.0}")
    print(f"coarse_k_ratio={c['k_stencil'] / K_WAVE:.6f}")
    print(f"coarse_phase_velocity_error_pct={c['err'] * 100:.3f}")
    print(f"coarse_phase_velocity_error_gt_10pct={c['err'] > 0.10}")
    # phase velocity c_h = omega/k_h -> arrival time error = k_h/k - 1
    t_err = abs(c['k_stencil'] / K_WAVE - 1.0)
    print("coarse_arrival_time_error_in_10_to_30pct="
          f"{0.10 <= t_err <= 0.30}")
    if not c['err'] > 0.10:
        print("FAIL: 3 elements per wavelength did not produce a >10% phase "
              f"velocity error (got {c['err'] * 100:.3f}%)", file=sys.stderr)
        ok = False

    # both estimators must agree, otherwise neither is trustworthy
    agree = abs(c['k_solve'] - c['k_stencil']) / K_WAVE < 0.01
    print(f"solve_and_stencil_estimates_agree={agree}")
    if not agree:
        print("FAIL: the driven-solve wavenumber and the interior stencil "
              "root disagree", file=sys.stderr)
        ok = False

    # the assembled operator really does obey the textbook P1 relation
    th = analytic_p1(c['kh_mesh'])
    print("assembled_stencil_matches_p1_dispersion_relation="
          f"{th is not None and abs(c['k_stencil'] * c['h'] - th) < 1e-9}")
    if th is None or abs(c['k_stencil'] * c['h'] - th) >= 1e-9:
        print("FAIL: the assembled stencil does not obey the analytic P1 "
              "dispersion relation", file=sys.stderr)
        ok = False

    # --- RIGHT variant: 20 elements per wavelength ---------------------------
    f = report(NPPW_FINE)
    print(f"fine_elements_per_wavelength={NPPW_FINE:g} fine_n_dofs={f['ndofs']}")
    print(f"fine_phase_velocity_error_pct={f['err'] * 100:.3f}")
    print(f"fine_phase_velocity_error_lt_1pct={f['err'] < 0.01}")
    print(f"refinement_improves_by_gt_10x={c['err'] / f['err'] > 10.0}")
    if not f['err'] < 0.01:
        print("FAIL: 20 elements per wavelength did not resolve the phase "
              f"velocity to 1% (got {f['err'] * 100:.3f}%)", file=sys.stderr)
        ok = False

    # --- O(k h)^2 rate ------------------------------------------------------
    seq = [report(n) for n in NPPW_RATE]
    errs = [s['err'] for s in seq]
    rates = [float(np.log2(errs[i] / errs[i + 1]))
             for i in range(len(errs) - 1)]
    print("dispersion_rates=" + ",".join(f"{r:.3f}" for r in rates))
    print(f"dispersion_rate_ge_1p5={min(rates) >= 1.5}")
    print(f"dispersion_rate_le_2p5={max(rates) <= 2.5}")
    if not min(rates) >= 1.5:
        print(f"FAIL: the dispersion error is not O(h^2): rates={rates}",
              file=sys.stderr)
        ok = False

    # --- catastrophic regime: no propagating discrete mode ------------------
    n = report(NPPW_NO_MODE)
    print(f"no_mode_kh={n['kh_mesh']:.4f} kh_above_2sqrt3="
          f"{n['kh_mesh'] > 2.0 * np.sqrt(3.0)}")
    print(f"no_propagating_mode_at_kh_above_2sqrt3={n['k_stencil'] is None}")
    print(f"analytic_relation_also_has_no_root={analytic_p1(n['kh_mesh']) is None}")
    if n['k_stencil'] is not None:
        print("FAIL: a propagating discrete mode was found at k*h = "
              f"{n['kh_mesh']:.3f}, where the P1 relation has no root",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
