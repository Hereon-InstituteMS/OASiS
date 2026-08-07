"""Tier-2: a too-thin PML back-reflects, and too-large sigma reflects instead of absorbing.

Claim: skfem helmholtz#3 — a PML extends the MeshTri/MeshQuad domain with a
complex stretch factor ``s = 1 - i*sigma(x)/k`` inside the BilinearForm; a
too-thin PML (< lambda/2 thick) gives visible back-reflection in the interior,
and too-large sigma creates a numerical impedance jump and reflects more than it
absorbs.  Standard choices: PML thickness 1-2 lambda, sigma_max such that
|R| < 1e-3 at normal incidence.

Wrong variants, both executed: (a) a PML only lambda/8 thick with a sensible
sigma; (b) a PML of proper thickness with sigma_max = 2e4 * k.  The reflection
coefficient is measured from the interior standing-wave ratio,
R = (SWR-1)/(SWR+1), sampled at the nodes between 0.5 lambda and Li-0.25 lambda
so neither the driven face nor the PML edge contaminates it.  The stretched
operator is assembled as ``(1/s) u_x v_x + s u_y v_y - k^2 s u v`` with
``@BilinearForm(dtype=complex)`` and a spatially varying quadratic
sigma(x) = sigma_max * ((x-Li)/d)^2.

Scaled-down configuration: a one-element-thick MeshQuad strip, k=8, 20 elements
per wavelength, 4 wavelengths of interior (222 DOFs for the tuned case) — normal
incidence in 1D, which is the case the claim quantifies.

Observed on skfem 12.0.1 (2026-08-06): no PML -> R = 0.97; lambda/8 PML ->
R = 0.375; tuned 1.5 lambda / sigma_max = 2k -> R = 3.9e-5; sigma_max = 2e4 k at
the same thickness -> R = 0.34, i.e. ~4 orders of magnitude worse.  Note the literal
wording "reflects more than it absorbs" (R > 0.5) is reached in the
too-SMALL-sigma direction (sigma_max = 0.05k -> R = 0.72), not in the
too-large-sigma direction within any sane range; what too-large sigma does is
degrade R monotonically by orders of magnitude.

Mutation control: with T2_MUTATE=1 both mistakes are replaced by the claim's
own standard choices at their pathology sites — the lambda/8 layer thickness
THIN_THICK_LAM becomes the 1.5 lambda TUNED_THICK_LAM, and the impedance-jump
HUGE_SIGMA becomes the tuned TUNED_SIGMA = 2k.  Both wrong variants then
reproduce the tuned |R| < 1e-3, so 'thin_thickness_lam=0.125 ...',
'thin_n_dofs=166', 'thin_R_gt_1e-2=True',
'thin_R_worse_than_tuned_by_gt_100x=True', 'huge_sigma_R_gt_1e-2=True',
'huge_sigma_worse_than_tuned_by_gt_1000x=True' and
'sigma_too_large_is_worse_than_optimum=True' disappear and the fixture goes
red.  Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from skfem import (Basis, BilinearForm, ElementQuad1, MeshQuad, asm, condense,
                   solve)

MUTATE = os.environ.get("T2_MUTATE") == "1"

K_WAVE = 8.0
NPPW = 20
N_WAVE_INTERIOR = 4
TUNED_THICK_LAM = 1.5      # the claim's 1-2 lambda band
TUNED_SIGMA = 2.0          # sigma_max / k
# the two mistakes; under mutation each is replaced by the tuned value the
# claim documents as the fix
THIN_THICK_LAM = 0.125 if not MUTATE else TUNED_THICK_LAM
HUGE_SIGMA = 2.0e4 if not MUTATE else TUNED_SIGMA


def reflection(thick_lam: float, sigma_over_k: float):
    lam = 2.0 * np.pi / K_WAVE
    Li = N_WAVE_INTERIOR * lam
    d = max(thick_lam, 1e-12) * lam
    L = Li + d
    h = lam / NPPW
    nx = max(int(round(L / h)), 4)
    m = (MeshQuad.init_tensor(np.linspace(0.0, L, nx + 1), np.array([0.0, h]))
         .with_boundaries({"left": lambda x: x[0] < 1e-9 * L,
                           "right": lambda x: x[0] > L - 1e-9 * L}))
    e = ElementQuad1()
    b = Basis(m, e)
    smax = sigma_over_k * K_WAVE

    @BilinearForm(dtype=complex)
    def stretched_helmholtz(u, v, w):
        t = np.clip((w.x[0] - Li) / d, 0.0, None)
        s = 1.0 - 1j * smax * t ** 2 / K_WAVE
        return (u.grad[0] * v.grad[0] / s + s * u.grad[1] * v.grad[1]
                - K_WAVE ** 2 * s * u * v)

    A = asm(stretched_helmholtz, b)
    f = np.zeros(A.shape[0], dtype=complex)
    left = b.get_dofs("left").flatten()
    D = np.concatenate([left, b.get_dofs("right").flatten()])
    x = np.zeros(A.shape[0], dtype=complex)
    x[left] = 1.0
    u = solve(*condense(A, f, x=x, D=D))

    sel = m.p[1] < 1e-14
    xs = m.p[0][sel]
    us = u[:m.p.shape[1]][sel]
    o = np.argsort(xs)
    xs, us = xs[o], us[o]
    win = (xs > 0.5 * lam) & (xs < Li - 0.25 * lam)
    a = np.abs(us[win])
    swr = float(a.max() / a.min())
    return (swr - 1.0) / (swr + 1.0), b.N, str(A.dtype)


def main() -> int:
    ok = True

    r_none, _, _ = reflection(1e-9, 0.0)                 # no PML at all
    r_thin, n_thin, _ = reflection(THIN_THICK_LAM, TUNED_SIGMA)
    r_tuned, n_tuned, dtype = reflection(TUNED_THICK_LAM, TUNED_SIGMA)
    r_huge, _, _ = reflection(TUNED_THICK_LAM, HUGE_SIGMA)

    print(f"mesh_class=MeshQuad1 element_class=ElementQuad1 "
          f"stretched_operator_dtype={dtype}")
    print(f"k={K_WAVE:g} elements_per_wavelength={NPPW} "
          f"interior_wavelengths={N_WAVE_INTERIOR}")
    print(f"no_pml_R={r_none:.4f} no_pml_R_gt_0p9={r_none > 0.9}")
    print(f"tuned_thickness_lam={TUNED_THICK_LAM} tuned_n_dofs={n_tuned} "
          f"tuned_sigma_over_k={TUNED_SIGMA:g}")
    print(f"tuned_R={r_tuned:.3e} tuned_R_lt_1e-3={r_tuned < 1e-3}")
    if not r_tuned < 1e-3:
        print(f"FAIL: the tuned PML did not reach |R| < 1e-3 ({r_tuned:.3e})",
              file=sys.stderr)
        ok = False

    # --- WRONG variant (a): PML thinner than lambda/2 ------------------------
    print(f"thin_thickness_lam={THIN_THICK_LAM} "
          f"thin_is_under_half_wavelength={THIN_THICK_LAM < 0.5}")
    print(f"thin_n_dofs={n_thin}")
    print(f"thin_R={r_thin:.3e} thin_R_gt_1e-2={r_thin > 1e-2}")
    print(f"thin_R_worse_than_tuned_by_gt_100x={r_thin / r_tuned > 100.0}")
    if not (r_thin > 1e-2 and r_thin / r_tuned > 100.0):
        print(f"FAIL: the lambda/8 PML did not back-reflect ({r_thin:.3e} vs "
              f"tuned {r_tuned:.3e})", file=sys.stderr)
        ok = False

    # --- WRONG variant (b): sigma so large it becomes an impedance jump ------
    print(f"huge_sigma_over_k={HUGE_SIGMA:g}")
    print(f"huge_sigma_R={r_huge:.3e} huge_sigma_R_gt_1e-2={r_huge > 1e-2}")
    print(f"huge_sigma_worse_than_tuned_by_gt_1000x={r_huge / r_tuned > 1000.0}")
    if not (r_huge > 1e-2 and r_huge / r_tuned > 1000.0):
        print(f"FAIL: sigma_max = {HUGE_SIGMA:g}*k did not degrade the PML "
              f"({r_huge:.3e} vs tuned {r_tuned:.3e})", file=sys.stderr)
        ok = False

    # --- thickness sweep: sub-lambda/2 layers all miss the 1e-3 target -------
    thicknesses = (0.125, 0.25, 0.5, 1.0, 1.5)
    Rs = [reflection(t, TUNED_SIGMA)[0] for t in thicknesses]
    print("thickness_sweep_lam=" + ",".join(f"{t:g}" for t in thicknesses))
    print("thickness_sweep_R=" + ",".join(f"{r:.2e}" for r in Rs))
    mono = all(Rs[i] > Rs[i + 1] for i in range(len(Rs) - 1))
    print(f"thickness_sweep_monotone_decreasing={mono}")
    under = [r for t, r in zip(thicknesses, Rs) if t < 0.5]
    print(f"all_sub_half_wavelength_R_gt_1e-3={all(r > 1e-3 for r in under)}")
    print("thickness_1_to_2_lam_R_lt_1e-3="
          f"{all(r < 1e-3 for t, r in zip(thicknesses, Rs) if t >= 1.0)}")
    if not mono:
        print(f"FAIL: reflection is not monotone in PML thickness: {Rs}",
              file=sys.stderr)
        ok = False

    # --- sigma sweep: there is an interior optimum ---------------------------
    sigmas = (0.05, 0.5, 2.0, 100.0, HUGE_SIGMA)
    Rs_s = [reflection(TUNED_THICK_LAM, s)[0] for s in sigmas]
    print("sigma_sweep_over_k=" + ",".join(f"{s:g}" for s in sigmas))
    print("sigma_sweep_R=" + ",".join(f"{r:.2e}" for r in Rs_s))
    imin = int(np.argmin(Rs_s))
    print(f"sigma_optimum_is_interior={0 < imin < len(sigmas) - 1}")
    print(f"sigma_too_small_reflects_more_than_it_absorbs={Rs_s[0] > 0.5}")
    print(f"sigma_too_large_is_worse_than_optimum={Rs_s[-1] > Rs_s[imin]}")
    if not (0 < imin < len(sigmas) - 1):
        print("FAIL: no interior optimum in sigma_max, so the impedance-jump "
              f"trade-off did not appear: {Rs_s}", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
