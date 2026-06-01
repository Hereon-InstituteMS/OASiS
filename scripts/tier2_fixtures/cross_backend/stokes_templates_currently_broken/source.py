"""Tier-2 Layer-F companion: known-broken Stokes
templates STAY broken until task #26/#27/#28 ships fixes.

Layer F (catalog_template_executes) caught on 2026-06-01
that skfem / ngsolve / fenics Stokes templates fail
end-to-end:

  skfem::stokes::2d
    Generator template uses homogeneous BC with no
    pressure pin → singular saddle-point; sciipy
    spsolve / condense fails. Layer-C MMS gate fixed
    this (see scripts/tier2_fixtures/skfem/
    stokes_mms_convergence/) but the GENERATOR template
    in src/backends/skfem/generators/stokes.py still
    ships the broken version.

  ngsolve::stokes::2d
    Generator uses a.mat.Inverse(fes.FreeDofs()) without
    pinning the pressure null space → Pardiso phase-33
    error -4 (zero pivot). Same root cause as the bug
    caught in commit cac2ce1's MMS gate.

  fenics::stokes::2d
    XDMFFile.write_function fails on a mixed-element
    sub-component (sol.sub(0).collapse() returns a
    Function whose XDMF write path expects a different
    DofMap layout).

This fixture INVERTS the Layer-F semantics: it asserts
each broken template fails as expected. When task #26
(skfem stokes), #27 (ngsolve stokes), or #28 (fenics
stokes) lands a fix, the inverted assertion flips and
the fixture FAILS — signalling that the template now
runs cleanly and should move to the main Layer-F gate.

Inverted-gate pattern: a fixture that PASSES when the
catalog is broken-as-expected, and FAILS when the
catalog improves. Provides regression armor against
PARTIAL fixes that change the failure mode without
fully fixing.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

logging.disable(logging.CRITICAL)

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))


KNOWN_BROKEN = [
    # (backend, physics, variant, expected_failure_token,
    #  task_id)
    ("skfem", "stokes", "2d",
     "ValueError",
     "task #26 — skfem stokes template ships broken BC"),
    ("ngsolve", "stokes", "2d",
     "Pardiso",
     "task #27 — ngsolve stokes lacks pressure pin"),
    ("fenics", "stokes", "2d",
     "write_function",
     "task #28 — fenics stokes XDMF write path"),
]


def _resolve_python(backend_name: str) -> str | None:
    repo_venv = sys.executable
    fenicsx_env = (Path.home() / "miniconda3" / "envs"
                    / "ofa-fenicsx" / "bin" / "python")
    if backend_name in ("skfem", "kratos", "ngsolve"):
        return repo_venv
    if backend_name == "fenics":
        return (str(fenicsx_env) if fenicsx_env.is_file()
                else None)
    return None


def _exec_template(backend, physics, variant, py
                    ) -> tuple[int, str, str]:
    from core.registry import (
        load_all_backends, get_backend)
    load_all_backends()
    b = get_backend(backend)
    template = b.generate_input(physics, variant, {})
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
            dir="/tmp") as f:
        f.write(template)
        path = f.name
    try:
        r = subprocess.run([py, path], capture_output=True,
                            text=True, timeout=60,
                            cwd="/tmp")
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    finally:
        try:
            Path(path).unlink()
        except Exception:
            pass


def main() -> int:
    unexpected_pass = []
    wrong_failure_mode = []
    correct_failures = 0
    for backend, physics, variant, token, task_id in KNOWN_BROKEN:
        py = _resolve_python(backend)
        if py is None:
            print(
                f"{backend}::{physics}::{variant} "
                f"SKIPPED (no env)")
            continue
        rc, out, err = _exec_template(
            backend, physics, variant, py)
        print(f"{backend}::{physics}::{variant} "
              f"rc={rc} task=\"{task_id}\"")
        if rc == 0:
            unexpected_pass.append(
                f"{backend}::{physics}::{variant} "
                f"PASSED — bug fixed? Update Layer F "
                f"main gate and remove this row from "
                f"KNOWN_BROKEN. ({task_id})")
            continue
        # Failure as expected — check the failure mode
        # still matches the documented token.
        combined = (out + err).lower()
        if token.lower() not in combined:
            wrong_failure_mode.append(
                f"{backend}::{physics}::{variant} "
                f"failed but mode differs from documented "
                f"'{token}': "
                f"got {err[:200]!r}")
        else:
            correct_failures += 1

    print(f"correct_failures={correct_failures}")
    print(f"unexpected_pass_count={len(unexpected_pass)}")
    print(f"wrong_failure_mode_count="
          f"{len(wrong_failure_mode)}")
    for u in unexpected_pass:
        print(f"  unexpected_pass: {u}")
    for w in wrong_failure_mode:
        print(f"  wrong_mode: {w}")

    if unexpected_pass:
        print("FAIL: known-broken template unexpectedly "
              "passed — move to main Layer F gate",
              file=sys.stderr)
        return 2
    if wrong_failure_mode:
        print("FAIL: known-broken template failed via a "
              "DIFFERENT mode — investigate",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
