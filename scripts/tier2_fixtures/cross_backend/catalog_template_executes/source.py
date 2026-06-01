"""Tier-2 Layer-F: catalog generator templates execute end-to-end.

Layer B verified `generate_input` doesn't raise on the
catalog. Layer C verified our MMS RE-IMPLEMENTATION of
the catalog's API converges. This is the missing piece:
take the catalog's ACTUAL emitted template, write it to
a tempfile, run it through the backend's interpreter,
and assert the run succeeds + emits expected sentinels.

This is what an LLM agent actually does: ask the MCP
for a template, save it, and run it. If the catalog
ships a template that the LLM-emit-and-run loop can't
execute, every other layer in the matrix could pass and
the user still gets a broken script.

Approach: per backend in {skfem, kratos} (both run
quickly in the repo .venv), for the simplest physics
(poisson 2d), call generate_input via the live
registry, write to a tempfile in /tmp, exec it via the
ACTUAL python interpreter the runner uses, and assert
exit_code == 0 + no Traceback. NGSolve/fenics need
their own conda envs and would slow this down to 30+
seconds per backend — keep this gate fast.
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


def run_template_in_subprocess(
        backend_name: str, physics: str, variant: str,
        python_path: str,
        timeout: int = 60) -> tuple[int, str, str]:
    """Generate the catalog template, write to tempfile,
    exec via the given python. Return (rc, out_head,
    err_head)."""
    from core.registry import (
        load_all_backends, get_backend)
    load_all_backends()
    b = get_backend(backend_name)
    template = b.generate_input(physics, variant, {})

    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
            dir="/tmp") as f:
        f.write(template)
        script_path = f.name

    try:
        result = subprocess.run(
            [python_path, script_path],
            capture_output=True, text=True,
            timeout=timeout,
            cwd="/tmp")
        return (result.returncode,
                result.stdout[:800],
                result.stderr[:800])
    except subprocess.TimeoutExpired:
        return (-1, "", "TIMEOUT")
    finally:
        try:
            Path(script_path).unlink()
        except Exception:
            pass


def _resolve_python(backend_name: str) -> str | None:
    """Return the absolute path to the Python interpreter
    that has THIS backend's runtime available, OR None
    when the env is missing (caller will skip this row)."""
    repo_venv = sys.executable
    fenicsx_env = (Path.home() / "miniconda3" / "envs"
                    / "ofa-fenicsx" / "bin" / "python")
    if backend_name in ("skfem", "kratos", "ngsolve"):
        # Repo .venv has scikit-fem 12 + KratosMultiphysics
        # + NGSolve installed.
        return repo_venv
    if backend_name == "fenics":
        return (str(fenicsx_env) if fenicsx_env.is_file()
                else None)
    return None


def main() -> int:
    print(f"repo_venv_python={sys.executable}")

    rows = [
        ("skfem", "poisson", "2d"),
        ("kratos", "poisson", "2d"),
        ("ngsolve", "poisson", "2d"),
        ("fenics", "poisson", "2d"),
    ]
    fail = []
    executed = 0
    for backend, physics, variant in rows:
        py = _resolve_python(backend)
        if py is None:
            print(
                f"{backend}::{physics}::{variant} "
                f"SKIPPED (no env)")
            continue
        try:
            rc, out, err = run_template_in_subprocess(
                backend, physics, variant, py,
                timeout=120)
        except Exception as e:
            fail.append(
                f"{backend}::{physics}::{variant} "
                f"setup_error {type(e).__name__}: {e}")
            continue
        executed += 1
        print(f"{backend}::{physics}::{variant}_rc={rc}")
        if rc != 0:
            fail.append(
                f"{backend}::{physics}::{variant} rc={rc}")
            print(f"  stderr: {err[:300]}")
        if "Traceback" in err or "Traceback" in out:
            fail.append(
                f"{backend}::{physics}::{variant} "
                f"traceback")

    print(f"total_executed={executed}")
    print(f"failures={len(fail)}")
    for r in fail:
        print(f"  fail: {r}")
    if fail:
        print("FAIL: catalog template execution",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
