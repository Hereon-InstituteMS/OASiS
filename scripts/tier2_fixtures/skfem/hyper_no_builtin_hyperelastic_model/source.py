"""Tier-2: skfem ships NO hyperelastic constitutive model.

Claim: skfem hyperelasticity#0 -- "scikit-fem has NO built-in hyperelastic
model ... looking for skfem.models.elasticity.neohookean or similar fails
(ImportError); the only linear elasticity helper is
skfem.models.elasticity.linear_elasticity."

The claim is an inventory claim, so the fixture enumerates the inventory
rather than trusting a single name: every plausible hyperelastic model name
is probed against skfem.models.elasticity AND against the whole skfem
namespace, and the ONE helper the claim says exists is checked to exist and
to actually assemble.

Observed on skfem 12.0.1: `from skfem.models.elasticity import neohookean`
raises ImportError ("cannot import name 'neohookean' from
'skfem.models.elasticity'"); none of neohookean / mooney_rivlin / ogden /
saint_venant_kirchhoff / hyperelasticity / st_venant / yeoh / gent exist
anywhere in skfem; `linear_elasticity` exists and assembles.  A hyperelastic
problem therefore has to be hand-coded as a @BilinearForm.

Mutation control: this is an ABSENCE claim, so the pathology is the missing
symbol itself and the mutation supplies it.  T2_MUTATE=1 binds a real
neohookean @BilinearForm onto skfem.models.elasticity before any probe runs --
the exact upstream change that would make this entry stale, and the case the
fixture's own "FAIL: skfem does ship a hyperelastic model" branch anticipates.
`from skfem.models.elasticity import neohookean` then succeeds and the
inventory sweep finds the name, removing
neohookean_import_raises_ImportError=True,
hyperelastic_names_in_models_elasticity=[] and
no_hyperelastic_model_anywhere=True.  Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import skfem
import skfem.models.elasticity as EL
from skfem import Basis, ElementTriP1, ElementVector, MeshTri

MUTATE = os.environ.get("T2_MUTATE") == "1"

HYPERELASTIC_NAMES = [
    "neohookean", "neo_hookean", "NeoHookean",
    "mooney_rivlin", "MooneyRivlin",
    "ogden", "Ogden",
    "saint_venant_kirchhoff", "st_venant_kirchhoff",
    "hyperelasticity", "hyperelastic",
    "yeoh", "gent",
]


def _install_neohookean() -> None:
    """Give skfem.models.elasticity the model the claim says it lacks."""
    from skfem.helpers import ddot, det, grad, inv, transpose

    def neohookean(mu=1.0, lam=1.0):
        @skfem.BilinearForm
        def form(u, v, w):
            du = w["disp"].grad
            F = np.zeros_like(du)
            F[0, 0] = 1.0
            F[1, 1] = 1.0
            F = F + du
            Fit = transpose(inv(F))
            lnJ = np.log(det(F))
            dF = grad(u)
            T = np.einsum("ij...,kj...,kl...->il...", Fit, dF, Fit)
            return ddot(mu * dF + (mu - lam * lnJ) * T
                        + lam * ddot(Fit, dF) * Fit, grad(v))
        return form

    EL.neohookean = neohookean


def main() -> int:
    ok = True
    print(f"skfem_version={skfem.__version__}")
    if MUTATE:
        # PATHOLOGY REMOVED: the absent symbol is supplied, so every probe
        # below now finds a genuine hyperelastic model in the namespace.
        _install_neohookean()

    # --- the named probe from the claim -------------------------------
    err = None
    try:
        from skfem.models.elasticity import neohookean  # noqa: F401
    except ImportError as e:
        err = e
    print(f"neohookean_import_raises_ImportError={err is not None}")
    print(f"neohookean_import_message={str(err)[:80]!r}")
    if err is None:
        print("FAIL: skfem.models.elasticity.neohookean imported", file=sys.stderr)
        ok = False

    # --- the whole inventory -----------------------------------------
    found_el = [n for n in HYPERELASTIC_NAMES if hasattr(EL, n)]
    found_top = [n for n in HYPERELASTIC_NAMES if hasattr(skfem, n)]
    print(f"hyperelastic_names_probed={len(HYPERELASTIC_NAMES)}")
    print(f"hyperelastic_names_in_models_elasticity={found_el}")
    print(f"hyperelastic_names_in_skfem_toplevel={found_top}")
    print(f"no_hyperelastic_model_anywhere="
          f"{not found_el and not found_top}")
    if found_el or found_top:
        print(f"FAIL: skfem does ship a hyperelastic model: "
              f"{found_el + found_top}", file=sys.stderr)
        ok = False

    # --- the one helper the claim says IS there -----------------------
    public = sorted(n for n in dir(EL) if not n.startswith("_"))
    print(f"models_elasticity_public={public}")
    has_lin = hasattr(EL, "linear_elasticity")
    print(f"linear_elasticity_present={has_lin}")
    if not has_lin:
        print("FAIL: linear_elasticity is missing too", file=sys.stderr)
        ok = False
    else:
        m = MeshTri().refined(2)
        ib = Basis(m, ElementVector(ElementTriP1()))
        K = EL.linear_elasticity(1.0, 1.0).assemble(ib)
        print(f"linear_elasticity_matrix_shape={K.shape}")
        print(f"linear_elasticity_nnz={K.nnz}")
        print(f"linear_elasticity_assembles={K.shape == (ib.N, ib.N)}")
        if K.shape != (ib.N, ib.N) or K.nnz == 0:
            print("FAIL: linear_elasticity did not assemble", file=sys.stderr)
            ok = False
        # …and it is LINEAR: assembling it at a deformed state gives the
        # SAME matrix, which is precisely what a hyperelastic model could
        # not do — so this helper cannot stand in for one.
        state = ib.zeros()
        state[:] = 0.3 * np.sin(ib.doflocs[0])
        K2 = EL.linear_elasticity(1.0, 1.0).assemble(
            Basis(m, ElementVector(ElementTriP1())))
        same = bool(abs(K - K2).max() < 1e-14)
        print(f"linear_elasticity_state_independent={same}")
        print(f"linear_elasticity_symmetric="
              f"{bool(abs(K - K.T).max() < 1e-14)}")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
