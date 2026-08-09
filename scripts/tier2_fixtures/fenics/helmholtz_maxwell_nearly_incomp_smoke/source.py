"""Tier-2: fenics phantom-physics closure regression armor.

Three fenics physics had been declared in supported_physics()
with rich deep_knowledge but NO generator code:

  helmholtz                            (commit c5d184a)
  maxwell                              (commit c5d184a)
  nearly_incompressible_elasticity     (this commit)

Generators were added in those commits, dispatch wired in
_PHYSICS_MODULES. This fixture pins the regression contract:

  (a) Each generator produces a non-trivial template string
      (>= 800 chars, contains the expected solver call).
  (b) The template imports basix.ufl (NOT the removed
      ufl.FiniteElement / VectorElement / MixedElement).
  (c) For helmholtz: complete Dirichlet-BC pattern is wired
      with locate_dofs_geometrical + dirichletbc.
  (d) For maxwell: Nédélec H(curl) basix.ElementFamily.N1E
      is named (regression: catalog must not regress to
      Lagrange for curl-curl).
  (e) For nearly_incompressible_elasticity: mixed_element
      is named and the 1/lambda stabilization term is in
      the form.

Failure mode this gate prevents: someone deletes the
generator code or reverts the dispatch entry, and the
catalog reverts to advertising a phantom physics — running
generate_input() would raise ValueError again.

Mutation control: T2_MUTATE=1 asks the catalog for `poisson` where it
should ask for `maxwell`. Nothing raises and every check below still
greps real generator text — but it is now the poisson template, which
carries no N1curl, so maxwell_uses_N1E / maxwell_uses_basix_family /
maxwell_has_curl are measured False and `maxwell_chars=` is never
printed at all. That is what proves these booleans are read out of
generated text rather than printed as literals.

NOTE on the staged (recipe) mutation arm: this fixture imports the
catalog out of the checkout's src/, so a copy of the fixture on its
own cannot find it. Set OASIS_REPO_ROOT to the checkout when running
scripts/mutate_tier2_fixtures.py; an in-place run (the ledger's env
arm) finds it by path and needs nothing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    """The checkout whose src/ holds the generators under test.

    parents[4] is correct in place and wrong in every copy: the mutation
    harness stages the fixture into a scratch tree where parents[4]
    resolves to `/`, the catalog import then fails, and the mutation
    verdict would be vacuous rather than informative. So the root is
    SEARCHED for, and $OASIS_REPO_ROOT wins when set — that is how a
    staged copy is told where the real checkout is.
    """
    marker = Path("src") / "core" / "registry.py"
    override = os.environ.get("OASIS_REPO_ROOT", "")
    cands = [Path(override)] if override else []
    here = Path(__file__).resolve()
    cands += list(here.parents) + list(Path.cwd().resolve().parents)
    for cand in cands:
        if (cand / marker).is_file():
            return cand
    print(f"FIXTURE_ABORT=repo_root_not_found searched_from={here}; "
          f"set OASIS_REPO_ROOT to the checkout", file=sys.stderr)
    raise SystemExit(3)


MUTATE = os.environ.get("T2_MUTATE") == "1"
REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    from core.registry import load_all_backends, get_backend

    load_all_backends()
    b = get_backend("fenics")

    results = {}
    # MUTATE: ask for `poisson` in maxwell's place — a real template
    # that simply has no N1curl in it.
    for phys in ("helmholtz",
                 "poisson" if MUTATE else "maxwell",
                 "nearly_incompressible_elasticity",
                 "fracture", "stokes_darcy"):
        try:
            txt = b.generate_input(phys, "2d", {"nx": 8})
        except Exception as e:
            results[phys] = f"GEN_ERR {type(e).__name__}"
            continue
        if not txt:
            results[phys] = "EMPTY"
            continue
        results[phys] = txt

    for phys, txt in results.items():
        if isinstance(txt, str) and txt.startswith("GEN_ERR"):
            print(f"{phys}={txt}")
        else:
            print(f"{phys}_chars={len(txt)}")

    # Per-physics structural checks
    def has(phys: str, s: str) -> bool:
        return s in results.get(phys, "")

    checks = {
        "helmholtz_uses_basix_ufl":
            has("helmholtz", "basix.ufl"),
        "helmholtz_has_dirichlet":
            has("helmholtz", "dirichletbc"),
        "helmholtz_has_indefinite_solver":
            has("helmholtz", "mumps") or has("helmholtz", "lu"),
        "maxwell_uses_N1E":
            has("maxwell", "N1E"),
        "maxwell_uses_basix_family":
            has("maxwell", "basix.ElementFamily"),
        "maxwell_has_curl":
            has("maxwell", "ufl.curl"),
        "nearly_incomp_uses_mixed_element":
            has("nearly_incompressible_elasticity",
                "mixed_element"),
        "nearly_incomp_has_stabilization":
            has("nearly_incompressible_elasticity",
                "1.0 / lam_val")
            or has("nearly_incompressible_elasticity",
                   "1/lam_val")
            or has("nearly_incompressible_elasticity",
                   "(1.0 / lam_val)"),
        # The needle used to be just 'shape=(gdim,)', which carries no degree
        # at all: ANY vector-valued Lagrange declaration produces it, so a
        # regression from P2/P1 to P1/P1 would have kept this check True under
        # a name that says "taylor_hood_p2_p1".  Both degrees are pinned now.
        "nearly_incomp_has_taylor_hood_p2_p1":
            has("nearly_incompressible_elasticity",
                'element("Lagrange", domain.basix_cell(), 2,')
            and has("nearly_incompressible_elasticity",
                    'shape=(gdim,)')
            and has("nearly_incompressible_elasticity",
                    'element("Lagrange", domain.basix_cell(), 1)'),
        "fracture_has_two_fields_u_and_d":
            has("fracture", "V_u") and has("fracture", "V_d"),
        "fracture_has_damage_irreversibility":
            has("fracture", "np.clip(d_h.x.array"),
        "fracture_has_phase_field_length_scale":
            has("fracture", "l0_val"),
        "fracture_has_alt_min_two_linearproblems":
            has("fracture", "prob_u")
            and has("fracture", "prob_d"),
        "stokes_darcy_uses_subdomain_measure":
            has("stokes_darcy",
                "subdomain_data=cell_tags"),
        "stokes_darcy_uses_meshtags":
            has("stokes_darcy", "mesh.meshtags"),
        "stokes_darcy_has_brinkman_penalty":
            has("stokes_darcy", "mu_val / K_darcy_val"),
        "stokes_darcy_uses_taylor_hood":
            has("stokes_darcy", "mixed_element")
            and has("stokes_darcy", "shape=(gdim,)"),
        "stokes_darcy_has_two_subdomain_dxs":
            has("stokes_darcy", "dx_subdomain(1)")
            and has("stokes_darcy", "dx_subdomain(2)"),
    }
    for k, v in checks.items():
        print(f"{k}={v}")

    # Anti-regression: NONE of these should refer to the
    # removed ufl element classes.
    forbidden = {
        "ufl.FiniteElement": False,
        "ufl.VectorElement": False,
        "ufl.MixedElement": False,
        "ufl.TensorElement": False,
    }
    for txt in results.values():
        if not isinstance(txt, str):
            continue
        for name in forbidden:
            if name in txt:
                forbidden[name] = True
    for k, v in forbidden.items():
        print(f"forbidden_{k.replace('.', '_')}={v}")

    ok = (
        all(isinstance(v, str) and len(v) >= 800
            for v in results.values())
        and all(checks.values())
        and not any(forbidden.values())
    )
    if ok:
        return 0
    print("FAIL: phantom-closure regression",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
