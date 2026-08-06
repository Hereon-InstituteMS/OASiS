"""Tier-2: at k=40 with 10 elements/wavelength, P1 drifts out of phase; P2 recovers it.

Claim: skfem helmholtz#4 — for high k (k > 20) use higher-order elements (P2, P3)
or DG to reduce pollution error; standard P1 at k=40 on a mesh with 10 elements
per wavelength shows the computed solution drifting out of phase across the
domain, the trailing-edge crest shifted by ~1/4 wavelength relative to the
analytic plane wave, while P2 on the SAME mesh recovers the phase.

Wrong variant: ElementQuad1 at exactly the claim's k=40 and 10 elements per
wavelength; the accumulated phase error against the analytic plane wave
exp(-i k x) is measured with ``Basis.probes`` along the propagation direction,
in wavelengths.  ElementQuad2 is then run on the identical mesh.

Scaled-down configuration: the claim's 2D domain is replaced by a
one-element-thick MeshQuad strip so the claim's own k=40 and 10
elements/wavelength can be kept verbatim (322 P1 DOFs / 963 P2 DOFs instead of
the ~10^5 a 2D square at that resolution would need).  The claim does not state
a domain size; the shift reaches the claimed ~1/4 wavelength after 16
wavelengths of propagation (on a unit square, 6.4 wavelengths, it is ~1/10).

Observed on skfem 12.0.1 (2026-08-06): P1 drift 0.252 wavelengths and 84%
relative L2 error; P2 on the same mesh 0.0017 wavelengths and 0.6%.
"""
from __future__ import annotations

import sys

import numpy as np
from skfem import (Basis, BilinearForm, ElementQuad1, ElementQuad2, FacetBasis,
                   MeshQuad, asm, condense, solve)

K_WAVE = 40.0        # the claim's high-k regime
NPPW = 10            # the claim's 10 elements per wavelength
N_WAVE = 16          # wavelengths of propagation
WRONG_ELEMENT = "P1"  # the mistake: standard P1 at high k

ELEMENTS = {"P1": ElementQuad1, "P2": ElementQuad2}


def run(kind: str):
    e = ELEMENTS[kind]()
    lam = 2.0 * np.pi / K_WAVE
    L = N_WAVE * lam
    nx = int(round(NPPW * N_WAVE))
    hy = L / nx
    m = (MeshQuad.init_tensor(np.linspace(0.0, L, nx + 1), np.array([0.0, hy]))
         .with_boundaries({"left": lambda x: x[0] < 1e-9 * L,
                           "right": lambda x: x[0] > L - 1e-9 * L}))
    b = Basis(m, e)

    @BilinearForm(dtype=complex)
    def helmholtz(u, v, w):
        return (u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1]
                - K_WAVE ** 2 * u * v)

    @BilinearForm(dtype=complex)
    def abc(u, v, w):
        return 1j * K_WAVE * u * v

    A = asm(helmholtz, b) + asm(abc, FacetBasis(m, e, facets="right"))
    f = np.zeros(A.shape[0], dtype=complex)
    D = b.get_dofs("left").flatten()
    x = np.zeros(A.shape[0], dtype=complex)
    x[D] = 1.0
    u = solve(*condense(A, f, x=x, D=D))

    xq = np.linspace(1e-9, L - 1e-9, 2001)
    yq = np.full_like(xq, 0.5 * hy)
    uq = b.probes(np.vstack([xq, yq])) @ u
    exact = np.exp(-1j * K_WAVE * xq)
    err = (np.unwrap(np.angle(uq)) - np.unwrap(np.angle(uq))[0]
           - (-K_WAVE * xq + K_WAVE * xq[0]))
    return dict(element=type(e).__name__, ndofs=b.N, nelements=m.nelements,
                kh=K_WAVE * (L / nx), drift_lam=abs(err[-1]) / (2 * np.pi),
                l2=float(np.linalg.norm(uq - exact) / np.linalg.norm(exact)))


def main() -> int:
    ok = True

    bad = run(WRONG_ELEMENT)
    good = run("P2")
    print(f"mesh_class=MeshQuad1 k={K_WAVE:g} k_gt_20={K_WAVE > 20.0}")
    print(f"elements_per_wavelength={NPPW} kh={bad['kh']:.4f} "
          f"n_wavelengths={N_WAVE}")
    print(f"p1_element={bad['element']} p1_n_dofs={bad['ndofs']}")
    print(f"p2_element={good['element']} p2_n_dofs={good['ndofs']}")
    print(f"same_mesh_for_p1_and_p2={bad['nelements'] == good['nelements']}")
    print(f"n_elements={bad['nelements']}")

    # --- WRONG variant: P1 at k=40 ------------------------------------------
    print(f"p1_phase_drift_wavelengths={bad['drift_lam']:.4f}")
    print(f"p1_drift_gt_0p2_wavelengths={bad['drift_lam'] > 0.2}")
    print("p1_drift_is_about_a_quarter_wavelength="
          f"{0.15 <= bad['drift_lam'] <= 0.35}")
    print(f"p1_l2_rel_gt_0p2={bad['l2'] > 0.2}")
    if not bad['drift_lam'] > 0.2:
        print("FAIL: P1 at k=40 with 10 elements/wavelength did not drift out "
              f"of phase ({bad['drift_lam']:.4f} wavelengths)", file=sys.stderr)
        ok = False
    if not bad['l2'] > 0.2:
        print(f"FAIL: P1 relative L2 error stayed small ({bad['l2']:.3e})",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant: P2 on the same mesh ---------------------------------
    print(f"p2_phase_drift_wavelengths={good['drift_lam']:.5f}")
    print(f"p2_drift_lt_0p01_wavelengths={good['drift_lam'] < 0.01}")
    print(f"p2_l2_rel_lt_0p05={good['l2'] < 0.05}")
    print("p2_recovers_phase_by_gt_20x="
          f"{bad['drift_lam'] / good['drift_lam'] > 20.0}")
    if not (good['drift_lam'] < 0.01 and good['l2'] < 0.05):
        print("FAIL: p-refinement to ElementQuad2 did not recover the phase "
              f"({good['drift_lam']:.5f} wavelengths, L2 {good['l2']:.3e})",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
