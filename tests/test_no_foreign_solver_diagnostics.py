"""A backend must not quote diagnostics from a solver stack it does not use.

THE CASE FOR A GATE RATHER THAN A SWEEP
---------------------------------------
On 2026-08-03 someone ran a PETSc-vocabulary sweep over the NGSolve knowledge
and corrected two entries. Both now say so explicitly:

    mixed_poisson#2  "NOTE: `KSPSolve: DIVERGED_INDEFINITE_PC` is a PETSc
                      message and is NOT emitted by NGSolve..."
    surface_pde#2    "NOTE: `KSPSolve: DIVERGED_BREAKDOWN` is a PETSc string
                      and is never emitted by..."

The same sweep missed `time_dependent_ns#4`, which still reads "Signal: PETSc
reports `KSPSolve: DIVERGED_BREAKDOWN`". NGSolve does not route through PETSc;
umfpack emits nothing and returns a finite vector.

That is the whole argument for gating this class of defect. The pattern was
known, someone went looking for exactly it, and a third instance survived
because a manual sweep depends on the sweeper's diligence at that moment.
Measured now across every backend that does not use PETSc: **six uncorrected
entries** — three NGSolve, three FEBio, which uses NOX.

WHY IT MATTERS TO AN AGENT
--------------------------
This is the project's recurring shape: the caution is sound, the mechanism is
sound, the observable is not produced. An agent told to watch for
`DIVERGED_BREAKDOWN` from a solver that cannot emit it watches forever, sees
nothing, and reads the silence as success. The guard does not fail to help — it
manufactures confidence, which is worse than no entry at all.

WHAT IS ALLOWED
---------------
Naming a foreign string in order to say it is NOT emitted is exactly the right
correction, and the two entries above are the model. So an entry that carries a
retraction cue near the string passes. So does a backend that genuinely uses the
stack: FEniCSx really does route through PETSc and its KSP/SNES strings are
real, so it is not checked here.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BACKENDS_DIR = REPO / "src" / "backends"

# Which foreign solver stacks to look for, and which backends legitimately use
# them. Deliberately conservative: only stacks whose diagnostic strings are
# distinctive enough that a false positive is implausible.
FOREIGN_STACKS = {
    "PETSc": {
        "pattern": re.compile(
            r"KSPSolve|SNESSolve|KSP_DIVERGED|SNES_DIVERGED"
            r"|DIVERGED_(?:BREAKDOWN|INDEFINITE_PC|FNORM_NAN|LINE_SEARCH|ITS"
            r"|MAX_IT|LINEAR_SOLVE|PC_FAILED)"
            r"|PETSc reports"),
        # Backends that really do route through PETSc, so these strings are
        # theirs to quote.
        "legitimate": {"fenics", "dealii"},
    },
}

# Naming a foreign string to say it is NOT emitted is the correct fix, not a
# violation. Same idea as the retraction exemption in the quoted-diagnostics
# screen: a checker that punishes the correction deletes the correction.
_RETRACTION = re.compile(
    r"is (?:a )?(?:PETSc|foreign|another library.s)|never emitted|not emitted"
    r"|does not (?:use|route|emit)|do not (?:use|route|emit)"
    r"|is NOT emitted|no such (?:string|message)|comes from PETSc"
    r"|signal-text correction",
    re.IGNORECASE)


def _entries(backend: str) -> list[tuple[Path, str]]:
    be_dir = BACKENDS_DIR / backend
    out: list[tuple[Path, str]] = []
    if not be_dir.is_dir():
        return out
    seen: set[str] = set()
    for py in sorted(be_dir.rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(errors="ignore"))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and "Signal:" in node.value
                    and node.value not in seen):
                seen.add(node.value)
                out.append((py, node.value))
    return out


BACKENDS = sorted(p.name for p in BACKENDS_DIR.iterdir()
                  if p.is_dir() and not p.name.startswith("_")) \
    if BACKENDS_DIR.is_dir() else []


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_diagnostics_from_a_solver_stack_this_backend_lacks(backend):
    """Quoting a message the backend cannot emit builds a guard that never fires."""
    offenders = []
    for stack, spec in FOREIGN_STACKS.items():
        if backend in spec["legitimate"]:
            continue
        for path, entry in _entries(backend):
            m = spec["pattern"].search(entry)
            if not m:
                continue
            if _RETRACTION.search(entry):
                continue  # correctly says the string is NOT emitted
            offenders.append(
                f"{path.relative_to(REPO)}\n      [{stack}] "
                f"...{entry[max(0, m.start() - 55):m.start() + 70].strip()}...")

    assert not offenders, (
        f"{backend}: {len(offenders)} entries quote diagnostics from a solver "
        f"stack this backend does not use, so an agent following them watches "
        f"for a message that can never appear:\n  " + "\n  ".join(offenders[:10])
        + "\n\nReplace the quoted string with what this backend ACTUALLY prints "
          "— or, if the point is that the foreign string is a common "
          "misconception, say so explicitly the way ngsolve/mixed_poisson#2 "
          "does: 'NOTE: `KSPSolve: DIVERGED_INDEFINITE_PC` is a PETSc message "
          "and is NOT emitted by NGSolve'. That phrasing is exempt from this "
          "gate and is the right correction.\n\n"
          "This gate exists because a manual sweep on 2026-08-03 corrected two "
          "NGSolve entries for exactly this and missed a third.")
