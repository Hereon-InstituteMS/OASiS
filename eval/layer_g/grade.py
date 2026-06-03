"""Layer G grader — execute Claude's code and pass/fail it.

Two grading modes:
  - compile_and_run : write code to tmp file, exec in target conda env,
                      check exit code + expected_substring / forbid_substring.
  - mms             : same as compile_and_run, plus parse a tagged numerical
                      result and compare against mms_reference within
                      tolerance. (v0: tag is "MMS_L2=<value>" stdout line.)

Each grade returns a Verdict with structured pass/fail + reason.
"""
from __future__ import annotations
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Verdict:
    passed: bool
    exit_code: int | None
    stdout: str
    stderr: str
    reason: str
    artifacts: dict[str, Any] = field(default_factory=dict)


CONDA_RUN = ["conda", "run", "-n"]
TIMEOUT_SEC = int(os.environ.get("LAYER_G_TIMEOUT", "180"))


def grade(task: dict[str, Any], code: str) -> Verdict:
    grade_mode = task.get("grade", "compile_and_run")
    env = task["conda_env"]

    with tempfile.TemporaryDirectory(prefix="layer_g_") as tmp:
        tmp_path = Path(tmp)
        ext = _ext_for_backend(task["backend"])
        script = tmp_path / f"solution{ext}"
        script.write_text(code)

        run_result = _execute(script, env, task["backend"])

    if run_result.exit_code is None:
        return Verdict(
            passed=False,
            exit_code=None,
            stdout=run_result.stdout,
            stderr=run_result.stderr,
            reason="timeout",
        )

    if run_result.exit_code != 0:
        return Verdict(
            passed=False,
            exit_code=run_result.exit_code,
            stdout=run_result.stdout,
            stderr=run_result.stderr,
            reason=f"nonzero exit: {run_result.exit_code}",
        )

    combined = run_result.stdout + "\n" + run_result.stderr

    if forbid := task.get("forbid_substring"):
        if forbid in combined:
            return Verdict(
                passed=False,
                exit_code=run_result.exit_code,
                stdout=run_result.stdout,
                stderr=run_result.stderr,
                reason=f"forbidden substring found: {forbid!r}",
            )

    if expected := task.get("expected_substring"):
        if expected not in combined:
            return Verdict(
                passed=False,
                exit_code=run_result.exit_code,
                stdout=run_result.stdout,
                stderr=run_result.stderr,
                reason=f"expected substring missing: {expected!r}",
            )

    if grade_mode == "mms":
        mms_verdict = _grade_mms(task, combined)
        if not mms_verdict.passed:
            return mms_verdict

    return Verdict(
        passed=True,
        exit_code=run_result.exit_code,
        stdout=run_result.stdout,
        stderr=run_result.stderr,
        reason="passed",
        artifacts=run_result.artifacts,
    )


@dataclass
class _RunResult:
    exit_code: int | None
    stdout: str
    stderr: str
    artifacts: dict[str, Any] = field(default_factory=dict)


def _execute(script: Path, env: str, backend: str) -> _RunResult:
    interpreter = _interpreter_for_backend(backend)
    if interpreter is None:
        # Non-executable backend (e.g., fourc consumes YAML input).
        # Just verify the file was created and is non-empty.
        if script.stat().st_size == 0:
            return _RunResult(exit_code=1, stdout="", stderr="empty file")
        return _RunResult(
            exit_code=0,
            stdout=f"GENERATED_INPUT_FILE={script}\n",
            stderr="",
            artifacts={"input_file": str(script)},
        )

    cmd = [*CONDA_RUN, env, interpreter, str(script)]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
            cwd=script.parent,
        )
    except subprocess.TimeoutExpired as e:
        return _RunResult(
            exit_code=None,
            stdout=_as_str(e.stdout),
            stderr=_as_str(e.stderr) or "TimeoutExpired",
        )
    except FileNotFoundError as e:
        return _RunResult(
            exit_code=127,
            stdout="",
            stderr=f"conda not found or env missing: {e}",
        )

    return _RunResult(
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def _interpreter_for_backend(backend: str) -> str | None:
    return {
        "skfem": "python",
        "fenics": "python",
        "ngsolve": "python",
        "kratos": "python",
        "dune": "python",
        "dealii": None,  # dealii needs cmake+compile cycle, v0 verifies file only
        "fourc": None,   # fourc consumes YAML input, v0 verifies file only
        "febio": None,   # febio consumes XML, v0 verifies file only
    }.get(backend)


def _ext_for_backend(backend: str) -> str:
    return {
        "skfem": ".py",
        "fenics": ".py",
        "ngsolve": ".py",
        "kratos": ".py",
        "dune": ".py",
        "dealii": ".cc",
        "fourc": ".4C.yaml",
        "febio": ".feb",
    }.get(backend, ".txt")


def _as_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return x


_MMS_L2_RE = re.compile(r"MMS_L2\s*=\s*([0-9.eE+-]+)")


def _grade_mms(task: dict[str, Any], combined_output: str) -> Verdict:
    ref = task.get("mms_reference", {})
    expected = ref.get("expected_l2")
    tol = ref.get("tolerance", 1e-2)
    if expected is None:
        return Verdict(
            passed=False,
            exit_code=0,
            stdout=combined_output,
            stderr="",
            reason="mms grade requested but mms_reference.expected_l2 missing",
        )
    m = _MMS_L2_RE.search(combined_output)
    if not m:
        return Verdict(
            passed=False,
            exit_code=0,
            stdout=combined_output,
            stderr="",
            reason="MMS_L2=<value> tag not found in output",
        )
    observed = float(m.group(1))
    if abs(observed - expected) > tol:
        return Verdict(
            passed=False,
            exit_code=0,
            stdout=combined_output,
            stderr="",
            reason=f"MMS L2 mismatch: observed={observed} expected={expected} tol={tol}",
        )
    return Verdict(
        passed=True,
        exit_code=0,
        stdout=combined_output,
        stderr="",
        reason="mms passed",
    )
