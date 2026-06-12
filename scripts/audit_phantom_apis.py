"""Phantom-API audit: do fenics Signal: clauses cite real dolfinx /
ufl / basix.ufl / petsc4py / slepc4py attributes?

Extracts every `<module>.<attr>` pattern from Signal: text via
regex, imports the module in the live ofa-fenicsx conda env,
and reports which attrs are REAL vs PHANTOM. Some phantoms are
intentional (the Signal documents a deprecated dolfin API
that no longer exists in dolfinx — the LLM should be warned).
Manual review of the PHANTOM list confirms each one is either:

  (a) intentional deprecation doc — Signal text contains a
      "does not exist", "no attribute", "(deprecated)",
      "removed", "no longer", etc. marker, OR
  (b) a submodule that needs explicit import (dolfinx.fem.petsc)
      — false positive, real but not loaded by hasattr().

Run this script periodically (or before catalog edits). It
audits THREE backends in one pass: fenics, skfem, ngsolve.
Modules that aren't importable in the current env are reported
as UNRESOLVABLE (not flagged as phantoms) — to cover every
module you need to invoke in BOTH the ofa-fenicsx conda env
AND the repo .venv:

  # Pass 1: fenics — resolves dolfinx / ufl / basix / petsc4py /
  # slepc4py; skfem / ngsolve modules are unresolvable here.
  /home/hermann/miniconda3/envs/ofa-fenicsx/bin/python \\
    scripts/audit_phantom_apis.py

  # Pass 2: skfem + ngsolve — resolves skfem / ngsolve /
  # netgen modules; dolfinx is unresolvable here.
  .venv/bin/python scripts/audit_phantom_apis.py

Exit code 0 = all phantoms are in INTENTIONAL_PHANTOMS.
Exit code 1 = a new UNEXPECTED phantom appeared; manual
review needed — fix the Signal: clause to use the real
API or add the (module, attr) to INTENTIONAL_PHANTOMS.

History (audit dates + actions taken):
  2026-06-02: surfaced ufl.errornorm phantom in
    deep_knowledge.py and generators/elasticity.py;
    replaced with the dolfinx
    assemble_scalar(form(inner(...)*dx)) pattern.
  2026-06-02: extended audit from fenics-only to also cover
    skfem + ngsolve. Both backends already clean — 2 skfem
    phantoms (NewtonSolver, neohookean) and 3 skfem
    submodule false-positives (helpers, io, models) added
    to INTENTIONAL_PHANTOMS as documented deprecation
    warnings.
"""
import importlib
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from verify_signal_clauses import verify_backend  # noqa: E402


FENICS_PATTERNS = [
    # dolfinx.<submodule>.<attr> (one level)
    re.compile(r"\b(dolfinx)\.(fem(?:\.petsc)?)\.([A-Za-z_][A-Za-z0-9_]+)"),
    re.compile(r"\b(dolfinx)\.([a-zA-Z_][a-zA-Z0-9_]*)\b"),
    re.compile(r"\b(ufl)\.([A-Za-z_][A-Za-z0-9_]+)"),
    re.compile(r"\b(basix\.ufl)\.([A-Za-z_][A-Za-z0-9_]+)"),
    re.compile(r"\b(petsc4py\.PETSc)\.([A-Za-z_][A-Za-z0-9_]+)"),
    re.compile(r"\b(slepc4py\.SLEPc)\.([A-Za-z_][A-Za-z0-9_]+)"),
]

SKFEM_PATTERNS = [
    re.compile(r"\b(skfem)\.([A-Za-z_][A-Za-z0-9_]+)"),
    re.compile(r"\b(skfem\.helpers)\.([A-Za-z_][A-Za-z0-9_]+)"),
    re.compile(r"\b(skfem\.models\.poisson)\.([A-Za-z_][A-Za-z0-9_]+)"),
    re.compile(r"\b(skfem\.models\.elasticity)\.([A-Za-z_][A-Za-z0-9_]+)"),
]

NGSOLVE_PATTERNS = [
    re.compile(r"\b(ngsolve)\.([A-Za-z_][A-Za-z0-9_]+)"),
    re.compile(r"\b(netgen\.occ)\.([A-Za-z_][A-Za-z0-9_]+)"),
    re.compile(r"\b(netgen\.csg)\.([A-Za-z_][A-Za-z0-9_]+)"),
]

# Default — used by collect_refs when no patterns arg.
PATTERNS = FENICS_PATTERNS


def collect_refs(backend: str,
                 patterns: list = None) -> set[tuple[str, str]]:
    if patterns is None:
        patterns = PATTERNS
    refs: set[tuple[str, str]] = set()
    for r in verify_backend(backend):
        sig = r.signal_text
        for pat in patterns:
            for m in pat.finditer(sig):
                groups = m.groups()
                if len(groups) == 3:
                    refs.add((f"{groups[0]}.{groups[1]}", groups[2]))
                else:
                    refs.add((groups[0], groups[1]))
    return refs


def classify_refs(
    refs: set[tuple[str, str]]
) -> tuple[list, list, list]:
    real: list[tuple[str, str]] = []
    phantom: list[tuple[str, str]] = []
    unresolvable: list[tuple[str, str, str]] = []
    for mod_path, attr in sorted(refs):
        try:
            mod = importlib.import_module(mod_path)
        except ImportError as e:
            unresolvable.append((mod_path, attr, str(e)[:80]))
            continue
        if hasattr(mod, attr):
            real.append((mod_path, attr))
        else:
            phantom.append((mod_path, attr))
    return real, phantom, unresolvable


INTENTIONAL_PHANTOMS = {
    # fenics: dolfin → dolfinx deprecation citations
    ("dolfinx.fem", "NonlinearVariationalProblem"),
    ("dolfinx.fem", "petsc"),  # submodule false positive
    ("ufl", "MixedElement"),
    ("ufl", "VectorElement"),
    ("ufl", "element"),
    ("ufl", "mixed_element"),
    ("ufl", "errornorm"),  # documented dolfin→dolfinx deprecation
    # skfem: deliberate warnings about non-existent helpers
    ("skfem", "NewtonSolver"),  # documented as missing
    ("skfem", "helpers"),       # submodule false positive
    ("skfem", "io"),            # submodule false positive
    ("skfem", "models"),        # submodule false positive
    ("skfem.models.elasticity", "neohookean"),  # documented as missing
}


def audit_backend(backend: str, patterns: list) -> tuple[int, list]:
    """Returns (n_unexpected_phantoms, unexpected_list)."""
    print(f"\n{'=' * 60}")
    print(f"Phantom-API audit — {backend} catalog")
    print(f"{'=' * 60}")
    refs = collect_refs(backend, patterns)
    print(f"Distinct refs cited: {len(refs)}")

    real, phantom, unresolvable = classify_refs(refs)
    print(f"REAL ({len(real)})  PHANTOM ({len(phantom)})  "
          f"UNRESOLVABLE ({len(unresolvable)})")
    if phantom:
        print("PHANTOM:")
        for m, a in phantom:
            tag = "(intentional)" if (m, a) in INTENTIONAL_PHANTOMS else "(UNEXPECTED)"
            print(f"  {m}.{a} {tag}")
    if unresolvable:
        print("UNRESOLVABLE:")
        for m, a, why in unresolvable:
            print(f"  {m}.{a} -- {why}")

    unexpected = [
        (m, a) for m, a in phantom
        if (m, a) not in INTENTIONAL_PHANTOMS
    ]
    return len(unexpected), unexpected


def main() -> int:
    # Run audits across the 3 backends with introspectable
    # live libraries. fourc / dealii / kratos / febio / dune
    # are not Python-attribute-introspectable in the same way
    # (compiled or YAML-driven) — covered by separate audits.
    fenics_n, fenics_un = audit_backend("fenics", FENICS_PATTERNS)
    skfem_n, skfem_un = audit_backend("skfem", SKFEM_PATTERNS)
    ngsolve_n, ngsolve_un = audit_backend("ngsolve", NGSOLVE_PATTERNS)

    total = fenics_n + skfem_n + ngsolve_n
    print(f"\n{'=' * 60}")
    print(f"Summary: {total} UNEXPECTED phantom citations")
    print(f"{'=' * 60}")
    if total:
        print("\nReview each unexpected phantom — does the Signal: text")
        print("document the phantom as a deprecation/warning, or does")
        print("it cite the symbol AS IF real? If the latter, fix the")
        print("Signal: clause to use the actual idiom, then add the")
        print("(module, attr) to INTENTIONAL_PHANTOMS so the audit")
        print("does not re-flag it.")
        for backend, un in [("fenics", fenics_un), ("skfem", skfem_un),
                            ("ngsolve", ngsolve_un)]:
            for m, a in un:
                print(f"  {backend}: {m}.{a}")
        return 1
    print("\nAll phantoms accounted for in INTENTIONAL_PHANTOMS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
