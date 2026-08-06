"""Tier-2: the scikit-fem complex trap is at ASSEMBLY, not at matrix arithmetic.

Claim: skfem helmholtz#0 — a plain ``@BilinearForm`` defaults to
``dtype=np.float64``, so a kernel returning ``1j * k * u * v`` is written into a
real buffer: the imaginary part is DISCARDED, the assembled matrix has dtype
float64, nnz 0 and max|A| exactly 0.0, and the only diagnostic is a numpy
ComplexWarning raised inside ``skfem/assembly/form/bilinear_form.py``.
``@BilinearForm(dtype=complex)`` on the SAME kernel gives complex128 with a
non-zero imaginary part; ``.astype(complex)`` afterwards is too late.
The claim also CORRECTS an older catalog text: sparse float + complex arithmetic
does NOT raise — ``K + 1j*M`` and in-place ``K += 1j*M`` both promote to
complex128; the only TypeError is scalar assignment into a float lil matrix.

Wrong variant: this fixture assembles the shipped ``helmholtz_2d`` template's
absorbing-BC block verbatim (``@BilinearForm`` / ``return 1j * k * u * v`` on a
FacetBasis over 'right') and shows the resulting system matrix is EXACTLY the
real symmetric ``K - k^2 M`` — i.e. the template has no absorbing BC at all,
while still exiting rc=0.

Configuration is the claim's own: MeshQuad.init_tensor(9x9), ElementQuad1,
FacetBasis on 'right', k=5.  max|A| of the correct complex assembly is checked
against the exact facet-mass value 2*k*h/3 (two facets share each interior
boundary node) rather than a pinned measurement.

Observed on skfem 12.0.1 / scipy 1.15.3 / numpy 1.26.4 (2026-08-06).
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
from skfem import (Basis, BilinearForm, ElementQuad1, FacetBasis, MeshQuad,
                   asm)
from skfem.models.poisson import laplace, mass

K_WAVE = 5.0
NX = 8  # 9x9 tensor grid -> h = 1/8


def build() -> tuple:
    m = (MeshQuad.init_tensor(np.linspace(0.0, 1.0, NX + 1),
                              np.linspace(0.0, 1.0, NX + 1))
         .with_boundaries({
             "left": lambda x: x[0] < 1e-10,
             "right": lambda x: x[0] > 1.0 - 1e-10,
             "bottom": lambda x: x[1] < 1e-10,
             "top": lambda x: x[1] > 1.0 - 1e-10,
         }))
    e = ElementQuad1()
    return m, e, Basis(m, e), FacetBasis(m, e, facets="right")


def main() -> int:
    ok = True
    k = K_WAVE
    m, e, ib, fb_right = build()
    print(f"mesh_class={type(m).__name__} element_class={type(e).__name__}")
    print(f"n_dofs={ib.N} n_elements={m.nelements}")

    # --- WRONG variant: the shipped template's absorbing-BC block ------------
    @BilinearForm  # default dtype is np.float64 -> imaginary part discarded
    def absorbing_bc_default(u, v, w):
        return 1j * k * u * v

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        A_bad = asm(absorbing_bc_default, fb_right)
    cplx_warn = [r for r in rec
                 if r.category.__name__ == "ComplexWarning"]
    msg = str(cplx_warn[0].message) if cplx_warn else ""
    origin = cplx_warn[0].filename.rsplit("/", 1)[-1] if cplx_warn else ""

    print(f"abc_default_dtype={A_bad.dtype}")
    print(f"abc_default_nnz={A_bad.nnz}")
    max_bad = float(np.abs(A_bad.toarray()).max())
    print(f"abc_default_max_abs_is_exactly_zero={max_bad == 0.0}")
    print(f"complexwarning_raised={bool(cplx_warn)}")
    print(f"complexwarning_msg={msg}")
    print(f"complexwarning_origin={origin}")
    if A_bad.dtype != np.float64 or A_bad.nnz != 0 or max_bad != 0.0:
        print("FAIL: the default-dtype ABC form did not assemble to an "
              f"all-zero float64 matrix (dtype={A_bad.dtype} "
              f"nnz={A_bad.nnz} max={max_bad})", file=sys.stderr)
        ok = False
    if not cplx_warn:
        print("FAIL: no ComplexWarning was raised by the discarding cast",
              file=sys.stderr)
        ok = False

    # .astype(complex) afterwards cannot resurrect the discarded imaginary part
    A_late = A_bad.astype(complex)
    print(f"astype_after_dtype={A_late.dtype}")
    print(f"astype_after_nnz={A_late.nnz}")
    late_useless = (A_late.nnz == 0
                    and float(np.abs(A_late.toarray()).max()) == 0.0)
    print(f"astype_after_recovers_nothing={late_useless}")
    if not late_useless:
        print("FAIL: .astype(complex) after assembly recovered information",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant: same kernel, dtype=complex ---------------------------
    @BilinearForm(dtype=complex)
    def absorbing_bc_complex(u, v, w):
        return 1j * k * u * v

    A_good = asm(absorbing_bc_complex, fb_right)
    imag_max = float(np.abs(A_good.imag.toarray()).max())
    real_max = float(np.abs(A_good.real.toarray()).max())
    print(f"abc_complex_dtype={A_good.dtype}")
    print(f"abc_complex_nnz={A_good.nnz}")
    print(f"abc_complex_imag_nonzero={imag_max > 0.0}")
    print(f"abc_complex_real_part_is_zero={real_max == 0.0}")
    # exact facet-mass value: an interior boundary node is shared by two
    # facets of length h, diag = 2 * h/3, times k.
    exact = 2.0 * k * (1.0 / NX) / 3.0
    print(f"abc_complex_max_abs={imag_max:.7f} exact_2kh_over_3={exact:.7f}")
    matches = abs(imag_max - exact) < 1e-12
    print(f"abc_complex_max_abs_matches_2kh_over_3={matches}")
    if A_good.dtype != np.complex128 or not (imag_max > 0.0) or not matches:
        print("FAIL: @BilinearForm(dtype=complex) did not produce a genuinely "
              "complex ABC matrix", file=sys.stderr)
        ok = False

    # --- Consequence: the template's system has NO absorbing BC -------------
    Kmat = laplace.assemble(ib)
    Mmat = mass.assemble(ib)
    A_template = (Kmat.astype(complex) - k ** 2 * Mmat.astype(complex)
                  + A_bad.astype(complex))
    A_correct = (Kmat.astype(complex) - k ** 2 * Mmat.astype(complex)
                 + A_good)
    tmpl_real_sym = (float(np.abs(A_template.imag).max()) == 0.0
                     and float(abs(A_template - A_template.T).max()) == 0.0)
    same_as_no_abc = float(abs(A_template
                               - (Kmat.astype(complex)
                                  - k ** 2 * Mmat.astype(complex))).max())
    corr_non_herm = float(abs(A_correct - A_correct.getH()).max()) > 0.0
    print(f"system_default_abc_is_real_symmetric={tmpl_real_sym}")
    print(f"system_default_abc_equals_no_abc={same_as_no_abc == 0.0}")
    print(f"system_complex_abc_is_non_hermitian={corr_non_herm}")
    if not tmpl_real_sym or same_as_no_abc != 0.0:
        print("FAIL: the default-dtype system was not identical to the "
              "no-ABC system", file=sys.stderr)
        ok = False
    if not corr_non_herm:
        print("FAIL: the dtype=complex system stayed Hermitian, so the ABC "
              "was not present there either", file=sys.stderr)
        ok = False

    # --- CORRECTION: float sparse + complex arithmetic does NOT raise -------
    try:
        S = Kmat + 1j * Mmat
        add_dtype, add_err = str(S.dtype), ""
    except TypeError as exc:            # pragma: no cover - documented absent
        add_dtype, add_err = "", str(exc)
    print(f"float_plus_complex_sparse_dtype={add_dtype}")
    print(f"float_plus_complex_sparse_raised={bool(add_err)}")

    K2 = Kmat.copy()
    try:
        K2 += 1j * Mmat
        iadd_dtype, iadd_err = str(K2.dtype), ""
    except TypeError as exc:            # pragma: no cover - documented absent
        iadd_dtype, iadd_err = "", str(exc)
    print(f"inplace_iadd_complex_dtype={iadd_dtype}")
    print(f"inplace_iadd_raised={bool(iadd_err)}")
    if add_dtype != "complex128" or iadd_dtype != "complex128":
        print("FAIL: sparse float+complex promotion no longer works, so the "
              "catalog CORRECTION is stale", file=sys.stderr)
        ok = False

    # ...the one place the cast TypeError really appears
    Kl = Kmat.tolil()
    try:
        Kl[0, 0] = 1j
        lil_msg = ""
    except TypeError as exc:
        lil_msg = str(exc)
    print(f"lil_scalar_assign_typeerror={bool(lil_msg)}")
    print(f"lil_msg={lil_msg}")
    if not lil_msg:
        print("FAIL: scalar complex assignment into a float lil matrix did "
              "not raise TypeError", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
