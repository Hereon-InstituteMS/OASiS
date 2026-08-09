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
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "scripts" / "tier2_fixtures"
OUTPUT = REPO_ROOT / "scripts" / "scan_results" / "tier2_results.json"


# A `key=value` expectation must match the WHOLE value, not a prefix of it.
#
# The matcher is plain case-insensitive substring, so `same_name_files=1` stayed
# matched when the fixture's own mutation turned the value into `10`. A DUNE pass
# found that live and named two more of the same shape
# (`umfpack_linear_iterations=1`, `k10_linear_iterations=1`); both survive only
# because a boolean guard happens to sit beside the number. Measured across the
# corpus, 2931 expectations have the `key=<int>` form and are prefixes of any
# longer value.
#
# Fixing 2931 strings would be the wrong repair — the defect is in the matcher.
# So for a needle that looks like an assertion (`key=value`, no spaces), the
# character following the match must not continue the value: end of output, a
# newline, a space, or punctuation is fine; another digit or letter is not.
# Everything else — prose fragments, quoted diagnostics, multi-word phrases —
# keeps plain substring matching, which is what those need.
_ASSERTION = re.compile(r"^[A-Za-z_][\w.\[\]]*=\S+$")


def _needle_present(needle: str, low_out: str) -> bool:
    """True if `needle` appears in `low_out` without being truncated."""
    low = needle.lower()
    if not _ASSERTION.match(needle.strip()):
        return low in low_out
    start = 0
    while True:
        i = low_out.find(low, start)
        if i == -1:
            return False
        j = i + len(low)
        # The KEY must start where the needle starts. Guarding only the right
        # edge left the mirror-image hole open: a needle that is a SUFFIX of a
        # longer key matched that longer key's line, so an expectation could be
        # satisfied by a sibling expectation it never mentions.
        #
        #     masked_max=1.000000            matched  unmasked_max=1.000000
        #     displacement_criterion_satisfied=true   matched
        #         tight_gamma_displacement_criterion_satisfied=true
        #
        # 11 expectations in the corpus were guaranteed true this way, including
        # both control expectations of dune/unmasked_ds_flux_covers_whole_boundary
        # — whose own docstring says the mutation drives unmasked_max to
        # 1.000000, which then satisfies the masked_max control. Same defect as
        # the value-prefix one this function was written to fix, one character to
        # the left.
        if i > 0 and (low_out[i - 1].isalnum() or low_out[i - 1] == "_"):
            start = i + 1
            continue
        # A value continues only through word characters, a dot, or a sign;
        # anything else (newline, space, comma, quote, EOF) ends it.
        if j >= len(low_out) or not (low_out[j].isalnum()
                                     or low_out[j] in "._-+"):
            return True
        start = i + 1

def fixture_inventory_fingerprint() -> str:
    """Identity of the fixture set: which fixtures exist and what they contain.

    Deliberately NOT a claim about whether they pass — it records what was
    summarised, so a summary can be told apart from a stale one. Every file in
    every fixture directory is hashed, so editing a fixture's expectation or its
    source counts as a change, not just adding or deleting one.
    """
    import hashlib

    h = hashlib.sha256()
    for d in sorted(p.parent for p in FIXTURES_DIR.glob("*/*/fixture.json")):
        h.update(f"{d.parent.name}/{d.name}\n".encode())
        for f in sorted(d.iterdir()):
            if f.is_file():
                h.update(f.name.encode())
                h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()

# Which deal.II do the deal.II fixtures compile against?
#
# Prefer a DEBUG build wherever one exists, because Assert is compiled out in
# Release and the Assert-gated Signal families (ExcDimensionMismatch,
# ExcInvalidFEIndex, ExcDivideByZero, …) simply cannot be produced by a Release
# library. A fixture asserting such a Signal against a Release deal.II is not
# testing anything.
#
# This used to be two hard-coded paths from one developer's machine
# (~/Schreibtisch/dealii-debug and ~/miniconda3/envs/ofa-dealii). Neither
# exists on this host, and the consequence was not an error but a SKIP: every
# deal.II fixture reported "install root not present", which reads identically
# to "no deal.II here" on a machine that has two working builds.
#
# TWO BRANCHES FIXED THIS INDEPENDENTLY AND BOTH FIXES ARE KEPT:
#   knowledge/dealii-verify        replaced each single path with a CANDIDATE
#                                  LIST probed by the library file itself, and
#                                  found the real build locations on this host.
#   knowledge/setup-and-portability delegated to the deal.II BACKEND's own
#                                  discovery — the same code path the MCP
#                                  server uses, so the fixtures and the server
#                                  cannot disagree about which install is in
#                                  play.
# Order: explicit env override, then a Debug build from the candidate list,
# then the backend's own discovery, then the Release candidates, then the
# historical guesses.
def _first_existing(cands, marker) -> Optional[Path]:
    for c in cands:
        try:
            if (Path(c) / marker).is_file():
                return Path(c)
        except OSError:
            continue
    return None


_DEBUG_CANDIDATES = [
    os.environ.get("DEAL_II_DEBUG_DIR", ""),
    Path.home() / "Schreibtisch" / "dealii-debug",
    Path("/media/alexander/PortableSSD/dealii-verify-r2/dbgbuild"),
    Path.home() / "dealii" / "dbgbuild",
]
_RELEASE_CANDIDATES = [
    os.environ.get("DEAL_II_DIR", ""),
    Path.home() / "dealii" / "build",
    Path.home() / "dealii" / "install",
    Path.home() / "miniconda3" / "envs" / "ofa-dealii",
]
# A build tree keeps its package config under cmake/config/, an install tree
# under lib/cmake/deal.II/ — accept either, and key the Debug probe on the
# debug library itself since that is what Assert needs.
_DEBUG_PREFIX = (_first_existing(_DEBUG_CANDIDATES, "lib/libdeal_II.g.so")
                 or Path.home() / "Schreibtisch" / "dealii-debug")
_RELEASE_PREFIX = (
    _first_existing(_RELEASE_CANDIDATES, "lib/libdeal_II.so")
    or _first_existing(_RELEASE_CANDIDATES, "cmake/config/deal.IIConfig.cmake")
    or Path.home() / "miniconda3" / "envs" / "ofa-dealii")


def _has_debug_library(p: Path) -> bool:
    return any((p / "lib").glob("libdeal_II.g.so*")) or \
        any((p / "lib").glob("libdeal.ii.g.so*"))


def _discover_dealii_prefix() -> Path:
    env = os.environ.get("DEALII_PREFIX") or os.environ.get("DEAL_II_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    if _has_debug_library(_DEBUG_PREFIX):
        return _DEBUG_PREFIX
    try:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from backends.dealii.backend import _find_dealii  # type: ignore
        found = _find_dealii()
        if found is not None:
            # A source tree keeps its libraries under build/.
            for candidate in (Path(found) / "build", Path(found)):
                if candidate.is_dir() and any(
                        (candidate / "lib").glob("libdeal_II*.so*")):
                    return candidate
            return Path(found)
    except Exception:
        pass
    return _RELEASE_PREFIX


DEFAULT_DEALII_PREFIX = _discover_dealii_prefix()


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
        # BACKEND-QUALIFIED. The bare directory name is not unique across the
        # tree — elasticity_mms_convergence, poisson_mms_convergence and
        # stokes_mms_convergence each exist under several backends — and the
        # results record is read row-wise by fixture_id, so a bare name made
        # each backend's row overwrite the previous one's and ten runs
        # collapsed to three. Qualifying it keeps every run its own row.
        pitfall_index=idx, fixture_id=f"{backend}/{fixture_dir.name}",
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
    # WHERE THE CHECKOUT IS, for fixtures that audit the catalog rather than a
    # solver. In place they find it by walking up from __file__, but the
    # mutation harness stages a copy into a scratch tree that has no such
    # ancestor, so the walk fails and the fixture aborts with
    # FIXTURE_ABORT=no_oasis_checkout — which the harness scores
    # VACUOUS_BASELINE, i.e. "this verdict would mean nothing".
    #
    # Measured before this line: 7 of the 11 Kratos fixtures that ship a
    # `_mutation` block reported VACUOUS_BASELINE on every run, so the ledger
    # carried NO machine discrimination evidence for them even though each had
    # been proved KILLED by hand with OASIS_REPO exported on the command line.
    # The runner knows where the checkout is; the staged fixture cannot. It is
    # the runner's job to say so.
    #
    # Deliberately does NOT override an OASIS_REPO the caller already set, and
    # deliberately points at the REAL checkout rather than the scratch copy:
    # these fixtures audit the SHIPPED catalog, and their mutation lives in
    # their own source, not in the catalog.
    env.setdefault("OASIS_REPO", str(REPO_ROOT))
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
            f"fixture requires a Debug-built deal.II (Assert macros live); "
            f"none of {[str(c) for c in _DEBUG_CANDIDATES if c]} has "
            f"lib/libdeal_II.g.so — set DEAL_II_DEBUG_DIR to one")
        return result
    # Point the build at the Debug tree when the fixture asked for it, so
    # `requires_debug` actually changes which library gets linked instead of
    # merely gating the run. Without this the flag was a no-op whenever a
    # Release install happened to be found first.
    if requires_debug and has_debug_lib and backend == "dealii":
        env.setdefault("DEAL_II_DIR", str(_DEBUG_PREFIX))
        env["DEAL_II_DIR"] = str(_DEBUG_PREFIX)

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

        def _route_to_repo_venv(env_var: str, label: str,
                               probe_module: str = "") -> str | None:
            """Find an interpreter that can run this backend's fixture.

            The first version looked only at `$ENV_VAR` and `REPO_ROOT/.venv`.
            That fails in every git worktree and in any clone whose environment
            lives elsewhere: measured here, 96 of 108 fixtures skipped and the
            run then OVERWROTE `tier2_results.json` with `passed: 1`. A sibling
            audit hit the same thing from the other side, rewriting 112 to 6.

            So the search now widens, cheapest first, and — the part that
            actually matters — it CHECKS the candidate can import the package
            instead of assuming a path implies a working environment.
            """
            def _readable_file(p) -> bool:
                # `is_file()` RAISES PermissionError on a path whose parent is
                # unreadable — it does not return False. Scanning sibling
                # directories walked into one such venv and killed the whole
                # run before a single fixture reported.
                try:
                    return Path(p).is_file()
                except OSError:
                    return False

            candidates = [env.get(env_var),
                          str(REPO_VENV) if _readable_file(REPO_VENV) else None]
            # A sibling checkout's venv: worktrees share one environment, and
            # REPO_ROOT is the worktree, not the checkout that owns the venv.
            try:
                siblings = sorted(REPO_ROOT.parent.iterdir())
            except OSError:
                siblings = []
            for sibling in siblings:
                try:
                    if sibling.is_dir():
                        candidates.append(
                            str(sibling / ".venv" / "bin" / "python"))
                except OSError:
                    continue
            # The interpreter running this script, last: if it can import the
            # package it is as good as any path we could guess.
            candidates.append(sys.executable)

            for cand in candidates:
                if not cand or not _readable_file(cand):
                    continue
                if probe_module:
                    try:
                        probe = subprocess.run(
                            [cand, "-c", f"import {probe_module}"],
                            capture_output=True, timeout=120,
                            stdin=subprocess.DEVNULL)
                    except (OSError, subprocess.SubprocessError):
                        continue
                    if probe.returncode != 0:
                        continue
                return cand

            result.status = "skipped"
            result.notes.append(
                f"{label}: no interpreter found that can import "
                f"{probe_module or 'the package'}; set {env_var} to one")
            return None

        def _conda_python(env_var: str, env_names: list[str], label: str,
                          probe_module: str) -> str | None:
            """Find an interpreter for a conda-hosted backend.

            The hard-coded env name (`ofa-fenicsx`, `ofa-dune`) is a GUESS,
            and on this host it is wrong: the dolfinx env is called `fenics`
            (plus `fenicsc` for the complex build). A wrong guess made every
            fenics fixture report `skipped` — which a reader cannot tell apart
            from "dolfinx is not installed". So: try the override, then each
            known env name, then the running interpreter, and in every case
            CHECK the candidate can import the package rather than inferring
            that from the path.
            """
            cands = [env.get(env_var)]
            for name in env_names:
                cands.append(str(Path.home() / "miniconda3" / "envs" / name
                                 / "bin" / "python"))
                cands.append(str(Path.home() / "anaconda3" / "envs" / name
                                 / "bin" / "python"))
                cands.append(str(Path.home() / "mambaforge" / "envs" / name
                                 / "bin" / "python"))
            cands.append(sys.executable)
            tried = []
            for c in cands:
                if not c:
                    continue
                try:
                    if not Path(c).is_file():
                        continue
                except OSError:
                    continue
                tried.append(c)
                try:
                    probe = subprocess.run(
                        [c, "-c", f"import {probe_module}"],
                        capture_output=True, timeout=180,
                        stdin=subprocess.DEVNULL)
                except (OSError, subprocess.SubprocessError):
                    continue
                if probe.returncode == 0:
                    return c
            result.status = "skipped"
            result.notes.append(
                f"{label}: no interpreter found that can import "
                f"{probe_module}; tried {tried or cands}; set {env_var}")
            return None

        # A cross_backend fixture is not a single-backend fixture, and its
        # `backend` field does not say what it needs. All four are filed as
        # `backend: kratos` "for runner-routing purposes" (their own _comment
        # says so) because the kratos slot used to land in the repo .venv, which
        # is where the MCP stack lives. That stopped being true: the kratos
        # branch now PROBES `import KratosMultiphysics`, the repo venv's Kratos
        # wheel fails on this host with `GLIBC_2.32 not found`, and the only
        # interpreter that does import Kratos (/mnt/kratos-tier2/kv) has no
        # `mcp`. So all four reported `skipped` — indistinguishable, to a
        # reader, from "no Kratos installed here", while their recorded status
        # stayed `passed` from an older run.
        #
        # They are routed like `coupling` instead, for the same reason: what
        # they actually import is the MCP stack and the live registry. Probing
        # `mcp` is the honest requirement; probing any one backend is a claim
        # about what the fixture runs that is not true.
        if fixture_dir.parent.name == "cross_backend":
            cand = _route_to_repo_venv("OASIS_PYTHON", "OASiS server", "mcp")
            if cand is None:
                return result
            python = cand
        elif backend == "fenics":
            cand = _conda_python("FENICS_PYTHON",
                                 ["fenics", "ofa-fenicsx", "fenicsx",
                                  "dolfinx"], "FEniCSx", "dolfinx")
            if cand is None:
                return result
            python = cand
        elif backend == "dune":
            cand = _conda_python("DUNE_PYTHON",
                                 ["ofa-dune", "dune-fem-env", "dune311"],
                                 "DUNE-fem", "dune.fem")
            if cand is None:
                return result
            python = cand
        elif backend == "kratos":
            cand = _route_to_repo_venv("KRATOS_PYTHON", "Kratos", "KratosMultiphysics")
            if cand is None:
                return result
            python = cand
        elif backend == "ngsolve":
            cand = _route_to_repo_venv("NGSOLVE_PYTHON", "NGSolve", "ngsolve")
            if cand is None:
                return result
            python = cand
        elif backend == "skfem":
            cand = _route_to_repo_venv("SKFEM_PYTHON", "scikit-fem", "skfem")
            if cand is None:
                return result
            python = cand
        elif backend == "coupling":
            # A coupling fixture is not a single-backend fixture. It DRIVES
            # several backends through the registered `couple` tool, so the
            # interpreter it needs is one that can import the MCP stack — each
            # participant is then launched in its own backend's interpreter, by
            # the driver, exactly as a real coupling does it. Probing `mcp` is
            # the honest requirement; probing any one backend would be a lie
            # about what the fixture runs.
            cand = _route_to_repo_venv("OASIS_PYTHON", "OASiS server", "mcp")
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
        if _needle_present(needle, low_out):
            result.expect_matched.append(needle)
    for needle in forbid:
        # Forbidden strings keep PLAIN substring matching, deliberately. A
        # forbid is a tripwire, and a tripwire should fire on the widest
        # reading — if "Traceback" appears in any form we want the failure.
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


def run(backend: str | None = None, fixture: str | None = None) -> dict:
    """Evaluate the fixtures, optionally narrowed to one backend and/or one
    fixture id.

    The filters exist because a whole-tree run is now hours: the coupling
    fixtures alone drive two solvers through tens to hundreds of iterations
    each. Without them, checking one fixture means either running everything or
    invoking its `source.py` by hand, and running it by hand bypasses the
    routing and the expectation matching — which is to say it does not check
    the fixture, only the script inside it.
    """
    out_map: dict[str, dict] = {}
    if not FIXTURES_DIR.is_dir():
        return out_map
    for backend_dir in sorted(FIXTURES_DIR.iterdir()):
        if not backend_dir.is_dir():
            continue
        if backend and backend_dir.name != backend:
            continue
        for fixture_dir in sorted(backend_dir.iterdir()):
            if not fixture_dir.is_dir():
                continue
            if fixture and fixture_dir.name != fixture:
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
    ap.add_argument(
        "--backend", default=None,
        help="only fixtures under scripts/tier2_fixtures/<backend>/")
    ap.add_argument(
        "--fixture", default=None,
        help="only the fixture directory with this name")
    args = ap.parse_args()

    # A PARTIAL run must never become the whole-tree record. That is the same
    # defect d17ac19 fixed from the other side: there, `--write-results` was
    # parsed and never read, so any exploratory run silently replaced the
    # snapshot. Adding filters re-opens it — `--backend coupling
    # --write-results` would write 18 rows over 108 and the summary would read
    # as a catastrophic regression. So the two are refused together.
    if (args.backend or args.fixture) and args.write_results:
        print("refusing to write results from a filtered run: the snapshot is "
              "a whole-tree record, and 18 rows written over 108 reads as a "
              "regression rather than as a partial run. Re-run without "
              "--backend/--fixture to persist.")
        return

    results = run(backend=args.backend, fixture=args.fixture)
    if not results:
        where = " ".join(x for x in (args.backend, args.fixture) if x)
        print(f"no fixtures matched {where!r}")
        return

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

    # `--write-results` was PARSED AND NEVER READ, so the write was
    # unconditional and the flag was decoration. Every exploratory run destroyed
    # the recorded snapshot: a DUNE audit reported it rewriting 112 passes to 6,
    # and it overwrote 108 with 1 under me before I noticed. A summary that any
    # incidental run can clobber is not a record.
    if not args.write_results:
        print("\nresults NOT written (pass --write-results to persist). "
              f"Recorded snapshot at {OUTPUT.relative_to(REPO_ROOT)} is "
              f"untouched.")
        return

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "summary": summary,
        # Identity of the fixture set these results describe. Without it a
        # committed snapshot cannot be told from a current one: an audit found
        # `tier2_results.json` 18 commits stale, recording `passed: 113` on a
        # host where 7 passed and 10 failed — and the floor test that reads it
        # was green, including with the backend's binary hidden. The number
        # travels with the repo, so every user gets our result reported as
        # theirs. Comparing this fingerprint catches the fixture set having moved
        # underneath a stale summary.
        "fixture_fingerprint": fixture_inventory_fingerprint(),
        "results": results,
    }, indent=2))
    print(f"results written to "
          f"{OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
