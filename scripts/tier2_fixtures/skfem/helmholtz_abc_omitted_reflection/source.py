"""Tier-2: omitting the ABC FacetBasis term makes the boundary reflect (standing wave).

Claim: skfem helmholtz#2 — the absorbing BC is ``+i*k*u`` on the boundary,
assembled via a ``FacetBasis`` ``BilinearForm``; omitting that term produces
standing-wave reflection off the domain boundary, with |u| showing an
interference pattern whose peaks are spaced at lambda/2, while adding the
``+i*k*u`` term absorbs the outgoing wave.

Scaled-down configuration: the claim's 2D MeshTri picture is reduced to a
one-element-thick MeshQuad strip (k=8, 20 elements per wavelength, 4 wavelengths
long, 162 DOFs) driven by u=1 at x=0.  The strip carries the same 1D outgoing
wave as a 2D domain's normal direction at a fraction of the cost, and the
standing-wave pattern is measurable node by node instead of by eye.

Wrong variant: assemble the interior Helmholtz operator only, leave the right
face natural, and measure the standing-wave ratio max|u|/min|u| and the spacing
of the |u| maxima.  A third variant assembles the ABC with the SHIPPED
helmholtz_2d template's default-dtype ``@BilinearForm`` and shows it is
bit-identical to having no ABC at all.

Observed on skfem 12.0.1 (2026-08-06): no ABC -> SWR ~1.6e2 with maxima spaced
at exactly lambda/2; correct dtype=complex ABC -> SWR ~1.004 and |u| ~ 1
everywhere.

Mutation control: with T2_MUTATE=1 the wrong variant's ABC selector flips from
"none" to "complex", i.e. the documented fix (add the @BilinearForm(dtype=complex)
1j*k*u*v term on FacetBasis('right')) is applied at the pathology site.  The
reflection then disappears, so 'no_abc_swr_gt_10=True', 'n_peaks_ge_6=True',
'peak_spacing_matches_half_wavelength=True',
'default_dtype_abc_identical_to_no_abc=True' and
'abc_improves_swr_by_gt_50x=True' are all printed =False and the fixture goes
red.  Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from skfem import (Basis, BilinearForm, ElementQuad1, FacetBasis, MeshQuad,
                   asm, condense, solve)

MUTATE = os.environ.get("T2_MUTATE") == "1"

K_WAVE = 8.0
NPPW = 20         # elements per wavelength
N_WAVE = 4        # domain length in wavelengths
# the mistake: no FacetBasis ABC term at all (the documented fix under
# mutation is the dtype=complex +i*k*u term)
ABC = "none" if not MUTATE else "complex"


def strip_solution(abc: str):
    """u=1 at x=0, outgoing wave to the right; returns nodal |u| along y=0."""
    lam = 2.0 * np.pi / K_WAVE
    L = N_WAVE * lam
    nx = int(round(NPPW * N_WAVE))
    hy = L / nx
    m = (MeshQuad.init_tensor(np.linspace(0.0, L, nx + 1),
                             np.array([0.0, hy]))
         .with_boundaries({"left": lambda x: x[0] < 1e-9 * L,
                           "right": lambda x: x[0] > L - 1e-9 * L}))
    e = ElementQuad1()
    b = Basis(m, e)
    fb = FacetBasis(m, e, facets="right")

    @BilinearForm(dtype=complex)
    def helmholtz(u, v, w):
        return (u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1]
                - K_WAVE ** 2 * u * v)

    @BilinearForm(dtype=complex)
    def abc_complex(u, v, w):
        return 1j * K_WAVE * u * v

    @BilinearForm                      # the shipped template's decorator
    def abc_default_dtype(u, v, w):
        return 1j * K_WAVE * u * v

    A = asm(helmholtz, b)
    n_abc = 0
    if abc == "complex":
        A_abc = asm(abc_complex, fb)
        n_abc = A_abc.nnz
        A = A + A_abc
    elif abc == "default":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A_abc = asm(abc_default_dtype, fb)
        n_abc = A_abc.nnz
        A = A + A_abc.astype(complex)

    f = np.zeros(A.shape[0], dtype=complex)
    D = b.get_dofs("left").flatten()
    x = np.zeros(A.shape[0], dtype=complex)
    x[D] = 1.0
    u = solve(*condense(A, f, x=x, D=D))

    sel = m.p[1] < 1e-14
    xs = m.p[0][sel]
    us = u[:m.p.shape[1]][sel]
    order = np.argsort(xs)
    return xs[order], us[order], lam, b.N, n_abc


def local_maxima(x: np.ndarray, a: np.ndarray) -> np.ndarray:
    idx = [i for i in range(1, a.size - 1)
           if a[i] > a[i - 1] and a[i] >= a[i + 1]]
    return x[idx]


def main() -> int:
    ok = True

    # --- WRONG variant: no ABC term -----------------------------------------
    x_w, u_w, lam, ndofs, _ = strip_solution(ABC)
    a_w = np.abs(u_w)
    swr_w = float(a_w.max() / a_w.min())
    peaks = local_maxima(x_w, a_w)
    spacing = float(np.diff(peaks).mean()) if peaks.size > 1 else float("nan")
    print(f"mesh_class=MeshQuad1 element_class=ElementQuad1 n_dofs={ndofs}")
    print(f"k={K_WAVE} elements_per_wavelength={NPPW} n_wavelengths={N_WAVE}")
    print(f"no_abc_swr={swr_w:.3f}")
    print(f"no_abc_swr_gt_10={swr_w > 10.0}")
    print(f"no_abc_n_peaks={peaks.size} n_peaks_ge_6={peaks.size >= 6}")
    print(f"peak_spacing={spacing:.5f} half_wavelength={lam / 2:.5f}")
    print("peak_spacing_matches_half_wavelength="
          f"{abs(spacing - lam / 2) < 0.02 * lam}")
    if not swr_w > 10.0:
        print(f"FAIL: no reflection without the ABC term (SWR={swr_w:.3f})",
              file=sys.stderr)
        ok = False
    if not abs(spacing - lam / 2) < 0.02 * lam:
        print("FAIL: the |u| maxima are not spaced at lambda/2 "
              f"({spacing:.5f} vs {lam / 2:.5f})", file=sys.stderr)
        ok = False

    # --- the SHIPPED template's ABC block is the same as no ABC -------------
    x_d, u_d, _, _, n_abc_default = strip_solution("default")
    same = float(np.abs(u_d - u_w).max())
    print(f"default_dtype_abc_nnz={n_abc_default}")
    print(f"default_dtype_abc_identical_to_no_abc={same == 0.0}")
    if same != 0.0 or n_abc_default != 0:
        print("FAIL: the default-dtype ABC assembly is no longer equivalent "
              "to omitting the term", file=sys.stderr)
        ok = False

    # --- RIGHT variant: +i*k*u on the FacetBasis -----------------------------
    x_g, u_g, _, _, n_abc_complex = strip_solution("complex")
    a_g = np.abs(u_g)
    swr_g = float(a_g.max() / a_g.min())
    dev = float(np.abs(a_g - 1.0).max())
    print(f"complex_abc_nnz={n_abc_complex}")
    print(f"with_abc_swr={swr_g:.4f}")
    print(f"with_abc_swr_lt_1p05={swr_g < 1.05}")
    print(f"with_abc_amplitude_dev={dev:.4f} with_abc_is_travelling_wave={dev < 0.05}")
    print(f"abc_improves_swr_by_gt_50x={swr_w / swr_g > 50.0}")
    if not (swr_g < 1.05 and dev < 0.05):
        print("FAIL: the +i*k*u FacetBasis term did not absorb the outgoing "
              f"wave (SWR={swr_g:.4f})", file=sys.stderr)
        ok = False

    # phase check: the absorbed solution really is the outgoing plane wave
    ph = np.unwrap(np.angle(u_g))
    kh = -(ph[-1] - ph[0]) / (x_g[-1] - x_g[0])
    print(f"with_abc_phase_matches_outgoing_wave={abs(kh - K_WAVE) / K_WAVE < 0.02}")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
