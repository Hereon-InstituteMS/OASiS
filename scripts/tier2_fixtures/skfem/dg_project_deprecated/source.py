"""Tier-2: module-level skfem.project / skfem.projection are deprecated.

Claim: skfem dg_methods#8 -- skfem.project() and skfem.projection() are
DEPRECATED and emit DeprecationWarning('will be removed in the next release').
skfem/__init__.py flags both in __all__ with `# TODO remove due to
deprecation`.  Existing DG-to-P1 visualization snippets
``skfem.project(u, basis_from=ib_dg, basis_to=ib_p1)`` warn now and will raise
AttributeError on upgrade.  Replacement: ``f = ib_dg.interpolator(u);
u_p1 = ib_p1.project(f)`` -- the Basis.project INSTANCE method.

Wrong variant: call the module-level helpers under
``warnings.simplefilter('always')`` and capture what they emit.

Observed on skfem 12.0.1:
  * skfem.project(...)    -> DeprecationWarning
        'project is deprecated in favor of Basis.project (will be removed in
         the next release).'
    and then a second one from the projection() it delegates to.
  * skfem.projection(...) -> DeprecationWarning
        'projection is deprecated in favor of Basis.project.'
  * the Basis.project instance method emits NOTHING and agrees with the legacy
    result to 1.3e-15 on a DG-P1 field.
  * skfem/__init__.py lines 27-28 carry the literal
        'project',      # TODO remove due to deprecation
        'projection',   # TODO remove due to deprecation

Mutation control: T2_MUTATE=1 replaces the two deprecated module-level calls
with the recommended ib_p1.project(ib_dg.interpolator(u_dg)) -- the documented
fix. Nothing warns any more, so the quoted texts
"project is deprecated in favor of Basis.project (will be removed in the next
release)." and "projection is deprecated in favor of Basis.project.", together
with project_emits_deprecationwarning=True, project_warning_mentions_removal=True,
project_warning_names_basis_project=True and
projection_emits_deprecationwarning=True, all vanish and the fixture goes red.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import inspect
import os
import sys
import warnings

import numpy as np
import skfem
from skfem import Basis, ElementDG, ElementTriP1, MeshTri

MUTATE = os.environ.get("T2_MUTATE") == "1"

REMOVAL_PHRASE = "will be removed in the next release"
TODO_FLAG = "# TODO remove due to deprecation"


def catch(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn()
        records = [(type(c.message).__name__, str(c.message)) for c in caught]
    return value, records


def main() -> int:
    ok = True
    m = MeshTri().refined(2)
    ib_dg = Basis(m, ElementDG(ElementTriP1()))
    ib_p1 = Basis(m, ElementTriP1())
    u_dg = ib_dg.project(lambda x: x[0] + 2.0 * x[1])

    print(f"module_level_project_exists={hasattr(skfem, 'project')}")
    print(f"module_level_projection_exists={hasattr(skfem, 'projection')}")
    print(f"project_in_dunder_all={'project' in skfem.__all__}")
    print(f"projection_in_dunder_all={'projection' in skfem.__all__}")

    # --- WRONG variant: the deprecated module-level spellings --------------
    # Pathology: the module-level skfem.project / skfem.projection helpers.
    # Mutation: the documented replacement -- the Basis.project instance method.
    u_legacy, recs_project = catch(
        (lambda: ib_p1.project(ib_dg.interpolator(u_dg))) if MUTATE else
        (lambda: skfem.project(u_dg, basis_from=ib_dg, basis_to=ib_p1)))
    kinds = {k for k, _ in recs_project}
    texts = [t for _, t in recs_project]
    print(f"project_emits_deprecationwarning={'DeprecationWarning' in kinds}")
    print(f"project_warning_texts={texts!r}")
    print(f"project_warning_mentions_removal="
          f"{any(REMOVAL_PHRASE in t for t in texts)}")
    print(f"project_warning_names_basis_project="
          f"{any('in favor of Basis.project' in t for t in texts)}")
    if "DeprecationWarning" not in kinds:
        print("FAIL: skfem.project did not emit a DeprecationWarning",
              file=sys.stderr)
        ok = False
    if not any(REMOVAL_PHRASE in t for t in texts):
        print("FAIL: the removal phrase quoted by the catalog is gone",
              file=sys.stderr)
        ok = False

    _, recs_projection = catch(
        (lambda: ib_p1.project(ib_dg.interpolator(u_dg))) if MUTATE else
        (lambda: skfem.projection(u_dg, ib_p1, ib_dg)))
    ptexts = [t for _, t in recs_projection]
    print(f"projection_emits_deprecationwarning="
          f"{'DeprecationWarning' in {k for k, _ in recs_projection}}")
    print(f"projection_warning_texts={ptexts!r}")
    if not any("projection is deprecated in favor of Basis.project" in t
               for t in ptexts):
        print("FAIL: skfem.projection wording changed", file=sys.stderr)
        ok = False

    # --- the documented source-level evidence -----------------------------
    src = inspect.getsource(skfem)
    n_todo = src.count(TODO_FLAG)
    flags_project = "'project',  " + TODO_FLAG in src
    flags_projection = "'projection',  " + TODO_FLAG in src
    print(f"init_py_todo_flag_count={n_todo}")
    print(f"init_py_flags_project={flags_project}")
    print(f"init_py_flags_projection={flags_projection}")
    if not (flags_project and flags_projection):
        print("FAIL: the __all__ TODO markers quoted by the catalog are gone",
              file=sys.stderr)
        ok = False
    if n_todo < 2:
        print("FAIL: skfem/__init__.py no longer flags both helpers",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant: Basis.project instance method, no warning ---------
    u_new, recs_new = catch(lambda: ib_p1.project(ib_dg.interpolator(u_dg)))
    print(f"basis_project_emits_no_warning={not recs_new}")
    agree = float(np.abs(np.asarray(u_legacy) - np.asarray(u_new)).max())
    print(f"legacy_and_instance_method_agree_lt_1e-12={agree < 1e-12}")
    print(f"legacy_vs_instance_max_abs_diff={agree:.3e}")
    if recs_new:
        print("FAIL: the recommended Basis.project also warned", file=sys.stderr)
        ok = False
    if agree >= 1e-12:
        print("FAIL: the replacement does not reproduce the legacy result",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
