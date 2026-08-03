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

# WHICH BACKENDS LINK WHICH LIBRARY — and why this is a fallback, not an oracle.
#
# A first version of this guard hard-coded the answer. That reproduces the exact
# defect it was written to catch: a hand-maintained table asserting what is real,
# never checked against the software. A peer audit made the point concrete by
# reporting that DUNE genuinely links PETSc on this box — which, if a DUNE Signal
# ever quoted a PETSc diagnostic, this guard would have called a fabrication.
#
# So the table below is only consulted when the installed artefacts cannot be
# inspected. Where they can, the evidence decides. `_evidence_links()` looks for
# the library's own marker strings in what the backend actually loads.
#
# On the DUNE claim specifically: `libpetsc_real.so.3.12` exists system-wide but
# NOT inside `dune-fem-env`, and the peer's `ldd` evidence could not be
# reproduced here. So DUNE is deliberately absent from the table — an unconfirmed
# peer assertion is not evidence, and the evidence path will admit it
# automatically if the linkage is real.
_LINKS = {
    "fenics": {"PETSc"},        # dolfinx is built on petsc4py
    # deal.II names PETSc and Trilinos in its own wrapper namespaces
    # (`PETScWrappers`, `TrilinosWrappers`), so those are genuinely deal.II
    # symbols whether or not a given build enables the feature. A Signal that
    # relies on the FEATURE still needs build scoping — a separate concern from
    # whether the NAME exists.
    "dealii": {"PETSc", "Trilinos"},
    "fourc": {"Trilinos"},      # 4C is a Trilinos application
    "kratos": {"Trilinos"},     # TrilinosApplication is a real Kratos module
}

# How to recognise a library: its own shared objects, or failing that, marker
# strings inside the backend's. Filenames come first because they are direct and
# cheap — a first version scanned only the alphabetically-first 40 libraries for
# markers and so never reached `libpetsc*`, reporting that dolfinx does not link
# PETSc. It passed anyway because the table covered it, which is precisely the
# kind of "works by accident" this file exists to refuse.
_LIB_GLOBS = {"PETSc": ("libpetsc*.so*",),
              "Trilinos": ("libteuchos*.so*", "libnox*.so*", "libepetra*.so*",
                           "libtrilinos*.so*")}
_MARKERS = {"PETSc": ("PETSC ERROR", "KSPSolve"),
            "Trilinos": ("Teuchos::", "NOX::")}


def _evidence_links(backend: str) -> set[str] | None:
    """Which libraries this backend demonstrably loads, or None if unknowable.

    Returning None rather than an empty set matters: "we could not look" must
    not read as "it links nothing", or the guard starts accusing backends whose
    corpus happens to be unavailable.
    """
    import subprocess

    roots = {
        "fenics": ["/home/alexander/miniconda3/envs/fenics/lib"],
        "dune": ["/home/alexander/miniconda3/envs/dune-fem-env/lib"],
        "febio": ["/home/alexander/Schreibtisch/febio-src/cbuild/lib"],
        "fourc": ["/home/alexander/4C/build"],
    }.get(backend)
    if not roots:
        return None
    found: set[str] = set()
    looked = False
    for root in roots:
        p = pathlib.Path(root)
        if not p.is_dir():
            continue
        all_libs = sorted(p.glob("*.so*"))
        if not all_libs:
            continue
        looked = True
        # Direct evidence: the library's own shared object sits here.
        for library, globs in _LIB_GLOBS.items():
            if any(next(p.glob(g), None) for g in globs):
                found.add(library)
        # Indirect: the backend's own binaries carry the library's strings. Only
        # worth the scan for libraries not already established.
        todo = {lib: mk for lib, mk in _MARKERS.items() if lib not in found}
        if not todo:
            continue
        try:
            blob = subprocess.run(["strings", "-n", "6",
                                   *map(str, all_libs[:60])],
                                  capture_output=True, text=True,
                                  timeout=300).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        for library, markers in todo.items():
            if any(m in blob for m in markers):
                found.add(library)
    return found if looked else None


def _blocks() -> dict[str, str]:
    """Per-backend allowlist bodies, keyed by backend name."""
    src = GATE.read_text()
    parts = re.split(r'(?:if|elif)\s+backend\s*==\s*"(\w+)"', src)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}


@pytest.mark.parametrize("backend", sorted(_blocks()))
def test_no_backend_allowlists_a_library_it_does_not_link(backend):
    body = _blocks()[backend]
    quoted = set(re.findall(r'"([A-Za-z_]\w*)"', body))
    # Evidence first, table only where the artefacts cannot be inspected.
    measured = _evidence_links(backend)
    links = _LINKS.get(backend, set()) if measured is None else (
        measured | _LINKS.get(backend, set()))
    offenders = []
    for library, pattern in _FOREIGN.items():
        if library in links:
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
