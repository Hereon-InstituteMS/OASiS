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

Run this script periodically (or before catalog edits) under
the ofa-fenicsx conda env:

  /home/hermann/miniconda3/envs/ofa-fenicsx/bin/python \\
    scripts/audit_phantom_apis.py

Exit code 0 always. Output is the audit report.

History (audit dates + actions taken):
  2026-06-02: surfaced ufl.errornorm phantom in
    deep_knowledge.py and generators/elasticity.py;
    replaced with the dolfinx
    assemble_scalar(form(inner(...)*dx)) pattern.
"""
import importlib
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from verify_signal_clauses import verify_backend  # noqa: E402


PATTERNS = [
    # dolfinx.<submodule>.<attr> (one level)
    re.compile(r"\b(dolfinx)\.(fem(?:\.petsc)?)\.([A-Za-z_][A-Za-z0-9_]+)"),
    re.compile(r"\b(dolfinx)\.([a-zA-Z_][a-zA-Z0-9_]*)\b"),
    re.compile(r"\b(ufl)\.([A-Za-z_][A-Za-z0-9_]+)"),
    re.compile(r"\b(basix\.ufl)\.([A-Za-z_][A-Za-z0-9_]+)"),
    re.compile(r"\b(petsc4py\.PETSc)\.([A-Za-z_][A-Za-z0-9_]+)"),
    re.compile(r"\b(slepc4py\.SLEPc)\.([A-Za-z_][A-Za-z0-9_]+)"),
]


def collect_refs(backend: str) -> set[tuple[str, str]]:
    refs: set[tuple[str, str]] = set()
    for r in verify_backend(backend):
        sig = r.signal_text
        for pat in PATTERNS:
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


def main() -> int:
    print("=" * 60)
    print("Phantom-API audit — fenics catalog")
    print("=" * 60)
    refs = collect_refs("fenics")
    print(f"Distinct refs cited: {len(refs)}")

    real, phantom, unresolvable = classify_refs(refs)

    print(f"\nREAL ({len(real)}):")
    for m, a in real:
        print(f"  {m}.{a}")
    print(f"\nPHANTOM ({len(phantom)}):")
    for m, a in phantom:
        print(f"  {m}.{a}")
    print(f"\nUNRESOLVABLE ({len(unresolvable)}):")
    for m, a, why in unresolvable:
        print(f"  {m}.{a} -- {why}")

    # Heuristic: known-intentional phantoms (dolfin
    # deprecations documented as Signal: text). If new
    # phantoms show up, they need manual review.
    INTENTIONAL_PHANTOMS = {
        ("dolfinx.fem", "NonlinearVariationalProblem"),
        ("dolfinx.fem", "petsc"),  # submodule false positive
        ("ufl", "MixedElement"),
        ("ufl", "VectorElement"),
        ("ufl", "element"),
        ("ufl", "mixed_element"),
        ("ufl", "errornorm"),
    }
    unexpected = [
        (m, a) for m, a in phantom
        if (m, a) not in INTENTIONAL_PHANTOMS
    ]
    if unexpected:
        print("\n" + "!" * 60)
        print(f"UNEXPECTED phantom citations: {len(unexpected)}")
        for m, a in unexpected:
            print(f"  {m}.{a}")
        print("Review each — does the Signal: text document the")
        print("phantom as a deprecation/warning, or does it cite")
        print("the symbol AS IF real? If the latter, fix the Signal:")
        print("clause to use the actual dolfinx idiom, then add the")
        print("(module, attr) to INTENTIONAL_PHANTOMS so the audit")
        print("does not re-flag it.")
        return 1
    print("\nAll phantoms accounted for in INTENTIONAL_PHANTOMS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
