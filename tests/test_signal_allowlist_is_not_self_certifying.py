"""The Signal gate must not declare a foreign library's diagnostics real.

`verify_signal_clauses.py` tier 0 checks that a Signal names a "real" entity,
where real means "present in a hand-maintained per-backend allowlist". That
makes the gate circular: a Signal cites a name, the name gets added so the gate
goes green, and the entry then validates the Signal. Nothing ever consults the
software.

Measured consequences, both found by adversarial audit rather than by the gate:

  * FEBio's allowlist declared `NOX`, `DIVERGED_LINE_SEARCH` and
    `DIVERGED_FNORM_NAN` real, annotated as "tokens that real FEBio logs emit".
    They are Trilinos and PETSc names. FEBio links NEITHER — confirmed against
    febio4 and all twelve of its shared libraries. An audit found 25 fabricated
    FEBio Signals; this entry is why they passed tier 0.
  * NGSolve's allowlist declared `KSPSolve`, `DIVERGED_BREAKDOWN` and
    `DIVERGED_INDEFINITE_PC` real, annotated as tokens "NGSolve also surfaces
    via krylovspace bindings". Its shared library contains no PETSc KSP strings
    at all.

A pitfall whose Signal cannot occur is worse than no pitfall: it tells an agent
that never seeing a message it could never have seen means the problem is
absent. So this guard refuses the specific move that produced both — allowlisting
another project's diagnostic vocabulary for a backend that does not link it.

It is deliberately narrow. It does not try to verify the whole allowlist against
every install, because most entries are ordinary API names and the corpora are
not always available. `scripts/verify_signal_literals.py` does that check where a
binary can be inspected. What this stops is the one pattern that has actually
caused fabrications twice.
"""
from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
GATE = REPO / "scripts" / "verify_signal_clauses.py"

# Diagnostic vocabularies that belong to a specific external library. A backend
# may legitimately allowlist these ONLY if it links that library.
_FOREIGN = {
    "PETSc": re.compile(r'^(KSP\w*|SNES\w*|DIVERGED_\w+|CONVERGED_[AR]TOL)$'),
    "Trilinos": re.compile(r'^(NOX\w*|Epetra\w*|Teuchos\w*|Tpetra\w*)$'),
}

# Which backends genuinely link which library, established by inspection of the
# installed artefacts rather than by assumption. deal.II names PETSc and
# Trilinos in its own wrapper namespaces (`PETScWrappers`, `TrilinosWrappers`),
# so those are real deal.II symbols regardless of whether a given build enables
# the feature — a Signal relying on the FEATURE still needs build scoping, which
# is a separate concern from whether the NAME exists.
_LINKS = {
    "fenics": {"PETSc"},        # dolfinx is built on petsc4py
    "dealii": {"PETSc", "Trilinos"},
    "fourc": {"Trilinos"},      # 4C is a Trilinos application
    "kratos": {"Trilinos"},     # TrilinosApplication is a real Kratos module
}


def _blocks() -> dict[str, str]:
    """Per-backend allowlist bodies, keyed by backend name."""
    src = GATE.read_text()
    parts = re.split(r'(?:if|elif)\s+backend\s*==\s*"(\w+)"', src)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}


@pytest.mark.parametrize("backend", sorted(_blocks()))
def test_no_backend_allowlists_a_library_it_does_not_link(backend):
    body = _blocks()[backend]
    quoted = set(re.findall(r'"([A-Za-z_]\w*)"', body))
    offenders = []
    for library, pattern in _FOREIGN.items():
        if library in _LINKS.get(backend, set()):
            continue
        offenders += [f"{name} ({library})" for name in sorted(quoted)
                      if pattern.match(name)]
    assert not offenders, (
        f"the Signal gate's allowlist for '{backend}' declares another "
        f"library's diagnostics to be real symbols, so any Signal quoting them "
        f"passes tier 0 without the software being consulted. {backend} does "
        f"not link that library.\n  " + "\n  ".join(offenders)
        + "\n\nThis is how 25 fabricated FEBio Signals reached agents. If the "
          "install genuinely links it, add the backend to _LINKS with the "
          "evidence; do not add the names.")


def test_the_guard_would_have_caught_the_known_cases():
    """Calibration: the predicate must fire on the exact entries that shipped.

    A guard whose discrimination is untested is a guard nobody should trust,
    and both real cases came with a confident comment asserting the opposite.
    """
    for library, pattern in _FOREIGN.items():
        pass
    febio_shipped = {"NOX", "DIVERGED_LINE_SEARCH", "DIVERGED_FNORM_NAN"}
    ngsolve_shipped = {"KSPSolve", "DIVERGED_BREAKDOWN", "DIVERGED_INDEFINITE_PC"}
    for name in febio_shipped | ngsolve_shipped:
        assert any(p.match(name) for p in _FOREIGN.values()), (
            f"the guard does not recognise {name}, which really shipped")
    # …and it must not fire on ordinary API names.
    for name in ("FullNewton", "BFGS", "SolverCG", "dirichletbc", "MeshTri"):
        assert not any(p.match(name) for p in _FOREIGN.values()), (
            f"the guard would falsely accuse {name}")
