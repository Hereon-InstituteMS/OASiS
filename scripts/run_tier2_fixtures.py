#!/usr/bin/env python3
"""
Tier-2 fixture runner — operationally verify Signal: clauses.

Each fixture is a directory under
``scripts/tier2_fixtures/<backend>/<id>/`` containing:

  * ``fixture.json`` — metadata: target backend, physics name,
    pitfall_index it verifies, the Signal substring or regex
    that the captured output must contain, and how to run the
    fixture (compile_only / compile_and_run / python).
  * ``source.cpp`` / ``source.py`` — the intentional-failure
    code. Compiling and/or running it must produce output that
    matches the Signal.
  * Optional ``cmd.sh`` for shell-level fixtures.

The runner:
  1. Walks every fixture under scripts/tier2_fixtures/
  2. Executes per the fixture's ``mode``
  3. Captures stderr + stdout
  4. Checks the captured text against the fixture's
     ``expect_in_output`` (substring; case-insensitive) and
     ``forbid_in_output`` lists.
  5. Writes a row to ``scripts/scan_results/tier2_results.json``
     keyed ``<backend>::<physics>::<pitfall_index>``.

The verify_signal_clauses.py harness reads that JSON; the
test_signal_verification.py merge gate enforces non-regression
on Tier-2 pass counts as well.

Senior-AI-scientist critic (2026-05-31 round 2) called Tier-2
the actual top risk after the Tier-0+1 fix — "every encoded
pitfall is a claim the project cannot defend without
operational verification". This runner exists so the project
CAN defend, one fixture at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "scripts" / "tier2_fixtures"
OUTPUT = REPO_ROOT / "scripts" / "scan_results" / "tier2_results.json"

# Prefer the Debug-built deal.II at ~/Schreibtisch/dealii-debug
# (Assert macros enabled — unlocks the Assert-gated Signal families
# like ExcDimensionMismatch). Falls back to the conda Release install
# (~/miniconda3/envs/ofa-dealii) which leaves Assert as a no-op.
_DEBUG_PREFIX = Path.home() / "Schreibtisch" / "dealii-debug"
_RELEASE_PREFIX = Path.home() / "miniconda3" / "envs" / "ofa-dealii"
DEFAULT_DEALII_PREFIX = (
    _DEBUG_PREFIX if (_DEBUG_PREFIX / "lib" / "libdeal_II.g.so").is_file()
    else _RELEASE_PREFIX
)


@dataclass
class FixtureResult:
    key: str                              # backend::physics::index
    backend: str
    physics: str
    pitfall_index: int
    fixture_id: str
    mode: str                              # compile_only / compile_and_run / python
    status: str = "harness_pending"        # passed / failed / harness_pending / skipped
    expect_matched: list = field(default_factory=list)
    forbid_violated: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    captured_head: str = ""                # first 800 bytes of captured output (sanitised)


def _run(cmd: list[str], cwd: Path, env: dict | None = None,
         timeout: int = 60) -> tuple[int, str]:
    """Run cmd, return (exit_code, combined_stderr_stdout). Best-effort
    sanitisation strips $HOME prefix to keep results portable."""
    try:
        r = subprocess.run(
            cmd, cwd=str(cwd), env=env, timeout=timeout,
            capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired as e:
        return 124, f"(timeout after {timeout}s) {e}"
    except FileNotFoundError as e:
        return 127, f"(command not found: {cmd[0]}) {e}"
    combined = (r.stdout or "") + (r.stderr or "")
    # Sanitise $HOME so committed results don't leak usernames.
    home = str(Path.home())
    combined = combined.replace(home, "~")
    return r.returncode, combined


def _eval_fixture(fixture_dir: Path,
                  meta: dict) -> FixtureResult:
    backend = str(meta.get("backend", "")).strip()
    physics = str(meta.get("physics", "")).strip()
    idx = int(meta.get("pitfall_index", -1))
    mode = str(meta.get("mode", "")).strip()
    key = f"{backend}::{physics}::{idx}"
    result = FixtureResult(
        key=key, backend=backend, physics=physics,
        pitfall_index=idx, fixture_id=fixture_dir.name,
        mode=mode,
    )
    expect = [str(s) for s in meta.get("expect_in_output", [])]
    forbid = [str(s) for s in meta.get("forbid_in_output", [])]
    if not expect:
        result.status = "harness_pending"
        result.notes.append("no expect_in_output set; cannot match")
        return result

    # Build env (per-backend defaults).
    env = os.environ.copy()
    extra_env = meta.get("env", {})
    if isinstance(extra_env, dict):
        for k, v in extra_env.items():
            env[str(k)] = str(v)

    # Fixture metadata can flag that it requires the Debug-built
    # deal.II — without it, Assert() macros are compiled out and
    # the expected exception never fires.
    requires_debug = bool(meta.get("requires_debug", False))
    has_debug_lib = (_DEBUG_PREFIX / "lib" / "libdeal_II.g.so").is_file()
    if requires_debug and not has_debug_lib and backend == "dealii":
        result.status = "skipped"
        result.notes.append(
            "fixture requires Debug-built deal.II at "
            "~/Schreibtisch/dealii-debug (Assert macros enabled); "
            "current install is Release-only — skip until rebuilt "
            "(task #30)")
        return result

    if mode == "compile_only" and backend == "dealii":
        prefix = Path(env.get("DEAL_II_DIR",
                              str(DEFAULT_DEALII_PREFIX)))
        if not prefix.is_dir():
            result.status = "skipped"
            result.notes.append(
                f"deal.II install root {prefix} not present; skip")
            return result
        src = fixture_dir / "source.cpp"
        if not src.is_file():
            result.status = "harness_pending"
            result.notes.append("source.cpp not present")
            return result
        # Build via CMake using the deal.II package config —
        # raw g++ -I<prefix>/include doesn't work because the
        # conda install relies on its CMake macros to surface
        # the bundled-TBB include path and link flags. The
        # CMake config also has a hard-coded compiler path
        # from the conda build farm, so we override
        # CMAKE_CXX_COMPILER explicitly with whatever compiler
        # is on PATH.
        import shutil as _shutil
        cxx = (env.get("CXX")
               or _shutil.which("g++")
               or _shutil.which("c++") or "g++")
        build_dir = fixture_dir / "_build"
        build_dir.mkdir(exist_ok=True)
        # Wipe any stale CMake cache so re-runs always pick up
        # the current source.cpp.
        for stale in ("CMakeCache.txt", "CMakeFiles"):
            target = build_dir / stale
            if target.exists():
                if target.is_file():
                    target.unlink()
                else:
                    import shutil as _sh
                    _sh.rmtree(target, ignore_errors=True)
        # Generate a minimal CMakeLists.txt next to the source.
        cmakelists = fixture_dir / "CMakeLists.txt"
        if not cmakelists.is_file():
            cmakelists.write_text(
                "cmake_minimum_required(VERSION 3.13)\n"
                "find_package(deal.II 9.0 REQUIRED HINTS "
                f"{prefix})\n"
                "deal_ii_initialize_cached_variables()\n"
                "project(tier2_fixture CXX)\n"
                "add_executable(prog source.cpp)\n"
                "deal_ii_setup_target(prog)\n"
            )
        # Force Debug build so deal.II Assert(...) macros fire
        # (the conda install was built Release, which compiles
        # them out — that's why some ExcDimensionMismatch
        # fixtures silently succeed when they should raise).
        cmake_configure = ["cmake",
                           f"-DCMAKE_CXX_COMPILER={cxx}",
                           "-DCMAKE_BUILD_TYPE=Debug",
                           f"-S", str(fixture_dir),
                           f"-B", str(build_dir)]
        rc, out = _run(cmake_configure, cwd=fixture_dir,
                       env=env, timeout=120)
        if rc != 0:
            # Configure failed — that's environmental, not a
            # bug in the fixture. Mark skipped so the merge
            # gate doesn't punish a broken deal.II install.
            result.status = "skipped"
            result.notes.append(
                "cmake configure failed (environmental); "
                "fixture cannot be evaluated until deal.II "
                "is installed and discoverable")
            result.captured_head = out[:800]
            return result
        cmake_build = ["cmake", "--build", str(build_dir)]
        rc, out = _run(cmake_build, cwd=fixture_dir,
                       env=env, timeout=180)
        result.captured_head = out[:800]

        # Detect environmental failures that have nothing to do
        # with the fixture's intended bug. The deal.II 9.1.1
        # conda install (per task #30) is missing bundled-TBB
        # headers, so the very first #include in any deal.II
        # program triggers a "tbb/task.h: No such file or
        # directory" error. Mark such cases as `skipped`, not
        # failed — punishing the merge gate for a broken
        # install would be wrong.
        env_blockers = (
            ("tbb/task.h: No such file or directory",
             "deal.II install missing TBB headers (task #30)"),
            ("CMAKE_CXX_COMPILER", "compiler discovery failure"),
        )
        for marker, reason in env_blockers:
            if marker in out and rc != 0:
                result.status = "skipped"
                result.notes.append(
                    f"environmental: {reason}. Fixture cannot "
                    f"be evaluated until the install is "
                    f"repaired; the Signal: clause itself "
                    f"may still be correct.")
                return result

        # Compile-only: success means the bug DID NOT
        # reproduce (the fixture is supposed to fail to
        # compile). Failure means the bug DID reproduce —
        # which is what we want to verify via the Signal match
        # below.
        if rc == 0:
            result.status = "failed"
            result.notes.append(
                "compile succeeded but the fixture is designed "
                "to FAIL compilation — the bug it claims to "
                "reproduce is not present in this deal.II "
                "version")
            return result
    elif mode == "compile_and_run" and backend == "dealii":
        prefix = Path(env.get("DEAL_II_DIR",
                              str(DEFAULT_DEALII_PREFIX)))
        if not prefix.is_dir():
            result.status = "skipped"
            result.notes.append(
                f"deal.II install root {prefix} not present; skip")
            return result
        src = fixture_dir / "source.cpp"
        if not src.is_file():
            result.status = "harness_pending"
            result.notes.append("source.cpp not present")
            return result
        import shutil as _shutil
        cxx = (env.get("CXX")
               or _shutil.which("g++")
               or _shutil.which("c++") or "g++")
        build_dir = fixture_dir / "_build"
        build_dir.mkdir(exist_ok=True)
        for stale in ("CMakeCache.txt", "CMakeFiles"):
            target = build_dir / stale
            if target.exists():
                if target.is_file():
                    target.unlink()
                else:
                    import shutil as _sh
                    _sh.rmtree(target, ignore_errors=True)
        cmakelists = fixture_dir / "CMakeLists.txt"
        if not cmakelists.is_file():
            cmakelists.write_text(
                "cmake_minimum_required(VERSION 3.13)\n"
                "find_package(deal.II 9.0 REQUIRED HINTS "
                f"{prefix})\n"
                "deal_ii_initialize_cached_variables()\n"
                "project(tier2_fixture CXX)\n"
                "add_executable(prog source.cpp)\n"
                "deal_ii_setup_target(prog)\n"
            )
        # Configure + build
        rc, out = _run(
            ["cmake", f"-DCMAKE_CXX_COMPILER={cxx}",
             "-S", str(fixture_dir), "-B", str(build_dir)],
            cwd=fixture_dir, env=env, timeout=120)
        if rc != 0:
            env_blockers = (
                ("tbb/task.h: No such file or directory",
                 "deal.II install missing TBB headers"),
            )
            for marker, reason in env_blockers:
                if marker in out:
                    result.status = "skipped"
                    result.notes.append(f"environmental: {reason}")
                    result.captured_head = out[:800]
                    return result
            result.status = "skipped"
            result.notes.append(
                "cmake configure failed (environmental)")
            result.captured_head = out[:800]
            return result
        rc, out = _run(
            ["cmake", "--build", str(build_dir)],
            cwd=fixture_dir, env=env, timeout=180)
        if rc != 0:
            # Compilation failed for compile_and_run — that's
            # NOT what we want; this mode expects the program
            # to build successfully and then exhibit the bug
            # at run time. Mark as failed with the captured
            # build output.
            result.status = "failed"
            result.notes.append(
                "compile_and_run: build failed; this mode "
                "requires the program to compile and run, then "
                "fail at run-time matching the Signal")
            result.captured_head = out[:800]
            return result
        # Run the executable. Force the conda env's lib dir to
        # the front of LD_LIBRARY_PATH so the MKL components
        # match — without this, base conda's MKL gets loaded
        # alongside the env-built deal.II and dlsym fails
        # finding mkl_blas_dgemm.
        exe = build_dir / "prog"
        if not exe.is_file():
            result.status = "failed"
            result.notes.append(
                "compile_and_run: build produced no `prog` "
                "executable")
            result.captured_head = out[:800]
            return result
        run_env = dict(env)
        env_lib = (Path.home() / "miniconda3" / "envs"
                   / "ofa-dealii" / "lib")
        debug_lib = prefix / "lib"
        ld_paths = [str(debug_lib), str(env_lib),
                    run_env.get("LD_LIBRARY_PATH", "")]
        run_env["LD_LIBRARY_PATH"] = ":".join(p for p in ld_paths if p)
        rc, out = _run([str(exe)], cwd=fixture_dir,
                       env=run_env, timeout=60)
        # Combine build + run output; the Signal may appear in
        # either. (Build-time warnings sometimes carry the
        # Signal text — e.g. deprecation warnings.)
        result.captured_head = out[:800]
    elif mode == "python":
        # Python-side runtime check. Each backend lives in its
        # own python env. FEniCSx and DUNE-fem are in their own
        # conda envs; Kratos / NGSolve / scikit-fem all live in
        # the repo's local .venv (per task #18 wiring). When
        # invoked from a different python (e.g. `python3
        # scripts/...` from the user shell), sys.executable
        # cannot import any of these — must route per-backend.
        src = fixture_dir / "source.py"
        if not src.is_file():
            result.status = "harness_pending"
            result.notes.append("source.py not present")
            return result
        python = sys.executable
        REPO_VENV = REPO_ROOT / ".venv" / "bin" / "python"

        def _route_to_repo_venv(env_var: str, label: str) -> str | None:
            cand = env.get(env_var) or (
                str(REPO_VENV) if REPO_VENV.is_file() else None)
            if cand and Path(cand).is_file():
                return cand
            result.status = "skipped"
            result.notes.append(
                f"{label} env python not found; set "
                f"{env_var} or install repo .venv (task #18)")
            return None

        if backend == "fenics":
            cand = (env.get("FENICS_PYTHON")
                    or str(Path.home() / "miniconda3" / "envs"
                           / "ofa-fenicsx" / "bin" / "python"))
            if Path(cand).is_file():
                python = cand
            else:
                result.status = "skipped"
                result.notes.append(
                    "FEniCSx env python not found; set "
                    "FENICS_PYTHON or install ofa-fenicsx conda env")
                return result
        elif backend == "dune":
            cand = (env.get("DUNE_PYTHON")
                    or str(Path.home() / "miniconda3" / "envs"
                           / "ofa-dune" / "bin" / "python"))
            if Path(cand).is_file():
                python = cand
            else:
                result.status = "skipped"
                result.notes.append(
                    "DUNE-fem env python not found; set "
                    "DUNE_PYTHON or install ofa-dune conda env")
                return result
        elif backend == "kratos":
            cand = _route_to_repo_venv("KRATOS_PYTHON", "Kratos")
            if cand is None:
                return result
            python = cand
        elif backend == "ngsolve":
            cand = _route_to_repo_venv("NGSOLVE_PYTHON", "NGSolve")
            if cand is None:
                return result
            python = cand
        elif backend == "skfem":
            cand = _route_to_repo_venv("SKFEM_PYTHON", "scikit-fem")
            if cand is None:
                return result
            python = cand
        cmd = [python, str(src)]
        # Per-fixture timeout override (default 120s). Some
        # fixtures spawn many backend subprocesses (e.g. the
        # cross_backend/catalog_template_executes fixture) and
        # legitimately need more wall-time. Audit pass 4 fix
        # (2026-06-02).
        per_fixture_timeout = int(meta.get("timeout_seconds", 120))
        rc, out = _run(
            cmd, cwd=fixture_dir, env=env,
            timeout=per_fixture_timeout)
        result.captured_head = out[:800]
    elif mode == "cmd":
        src = fixture_dir / "cmd.sh"
        if not src.is_file():
            result.status = "harness_pending"
            result.notes.append("cmd.sh not present")
            return result
        cmd = ["bash", str(src)]
        per_fixture_timeout = int(meta.get("timeout_seconds", 120))
        rc, out = _run(
            cmd, cwd=fixture_dir, env=env,
            timeout=per_fixture_timeout)
        result.captured_head = out[:800]
    else:
        result.status = "harness_pending"
        result.notes.append(
            f"mode {mode!r} not recognised; expected one of "
            f"compile_only / compile_and_run / python / cmd")
        return result

    # Match expectations on captured output.
    low_out = out.lower()
    for needle in expect:
        if needle.lower() in low_out:
            result.expect_matched.append(needle)
    for needle in forbid:
        if needle.lower() in low_out:
            result.forbid_violated.append(needle)

    all_expected = (len(result.expect_matched) == len(expect))
    no_forbidden = (len(result.forbid_violated) == 0)
    if all_expected and no_forbidden:
        result.status = "passed"
    else:
        result.status = "failed"
        if not all_expected:
            missing = sorted(set(expect) - set(result.expect_matched))
            result.notes.append(
                f"missing in output: {missing}")
        if result.forbid_violated:
            result.notes.append(
                f"forbidden tokens appeared: {result.forbid_violated}")
    return result


def run() -> dict:
    out_map: dict[str, dict] = {}
    if not FIXTURES_DIR.is_dir():
        return out_map
    for backend_dir in sorted(FIXTURES_DIR.iterdir()):
        if not backend_dir.is_dir():
            continue
        for fixture_dir in sorted(backend_dir.iterdir()):
            if not fixture_dir.is_dir():
                continue
            meta_path = fixture_dir / "fixture.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError) as e:
                print(f"  {fixture_dir.name}: meta parse failed: {e}")
                continue
            r = _eval_fixture(fixture_dir, meta)
            # Detect silent key collisions — two fixtures with
            # the same (backend, physics, pitfall_index) tuple
            # would otherwise be invisible because the second
            # overwrites the first in the per-key dict.
            # Found 2026-06-01: interpolation_points_is_property
            # collided with nonlinear_problem_signature_kwargs
            # on fenics::nonlinear_pde::0; only caught because
            # the post-add total stayed flat instead of +1.
            if r.key in out_map:
                prior = out_map[r.key]
                prior_fixture = prior.get("fixture_id", "?")
                r.notes.append(
                    f"KEY COLLISION: this fixture's key "
                    f"{r.key!r} is already used by fixture "
                    f"{prior_fixture!r}. Pick a distinct "
                    f"pitfall_index in fixture.json — the "
                    f"first-written result will otherwise be "
                    f"silently overwritten."
                )
                r.status = "failed"
            out_map[r.key] = asdict(r)
            status_glyph = {
                "passed": "✓", "failed": "✗",
                "harness_pending": "⏳", "skipped": "↷",
            }.get(r.status, "?")
            print(f"  {status_glyph} {fixture_dir.name} "
                  f"({r.mode}): {r.status}")
            if r.notes:
                for note in r.notes:
                    print(f"      {note}")
    return out_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--write-results", action="store_true",
        help="persist results to scan_results/tier2_results.json")
    args = ap.parse_args()

    results = run()

    summary = {
        "n_fixtures": len(results),
        "passed": sum(1 for r in results.values()
                      if r["status"] == "passed"),
        "failed": sum(1 for r in results.values()
                      if r["status"] == "failed"),
        "harness_pending": sum(1 for r in results.values()
                               if r["status"] == "harness_pending"),
        "skipped": sum(1 for r in results.values()
                       if r["status"] == "skipped"),
    }
    print()
    print(f"Tier-2 summary: {summary}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "summary": summary,
        "results": results,
    }, indent=2))
    print(f"results written to "
          f"{OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
