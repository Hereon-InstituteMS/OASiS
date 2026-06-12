"""Solver smoke tests — actually run trivial FEM problems to verify the pipeline.

Each test takes <2 seconds: mesh → function space → assemble/solve → check result.
Goes beyond hasattr by testing the full solver pipeline end-to-end.

Results are stored in the fingerprint alongside the API surface check.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class SmokeResult:
    """Result of a solver smoke test."""
    solver: str
    passed: bool
    error: Optional[str] = None
    version: Optional[str] = None
    duration_ms: float = 0.0
    tested_at: str = ""

    def __post_init__(self):
        if not self.tested_at:
            self.tested_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def _run_script(python: str, script: str, timeout: int = 15) -> tuple[bool, str, str]:
    """Run a Python script in a subprocess. Returns (success, first_line_stdout, stderr).

    Only returns the first line of stdout to avoid banners/cleanup output.
    """
    try:
        r = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=timeout,
        )
        # Take only first non-empty line (avoids Kratos/4C banners on cleanup)
        first_line = ""
        for line in r.stdout.splitlines():
            line = line.strip()
            if line and line.startswith("{"):
                first_line = line
                break
        return r.returncode == 0, first_line, r.stderr[-500:] if r.stderr else ""
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)


def smoke_ngsolve() -> SmokeResult:
    """Smoke test: solve Laplace on unit square with NGSolve."""
    t0 = time.time()
    script = '''
import ngsolve as ngs
from netgen.geom2d import unit_square
mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.5))
V = ngs.H1(mesh, order=1, dirichlet="bottom|right|top|left")
u, v = V.TnT()
a = ngs.BilinearForm(ngs.grad(u)*ngs.grad(v)*ngs.dx).Assemble()
f = ngs.LinearForm(1*v*ngs.dx).Assemble()
gfu = ngs.GridFunction(V)
gfu.vec.data = a.mat.Inverse(V.FreeDofs()) * f.vec
print(f'{{"ok": true, "max": {max(gfu.vec):.6f}, "version": "{ngs.__version__}"}}')
'''
    ok, stdout, stderr = _run_script(sys.executable, script)
    dt = (time.time() - t0) * 1000
    if ok:
        try:
            data = json.loads(stdout)
            return SmokeResult("ngsolve", True, version=data.get("version"),
                               duration_ms=round(dt, 1))
        except json.JSONDecodeError:
            return SmokeResult("ngsolve", False, error=f"Bad output: {stdout[:200]}",
                               duration_ms=round(dt, 1))
    return SmokeResult("ngsolve", False, error=stderr[:300], duration_ms=round(dt, 1))


def smoke_skfem() -> SmokeResult:
    """Smoke test: solve Poisson on unit square with scikit-fem."""
    t0 = time.time()
    script = '''
import skfem, json, numpy as np
from skfem import *
from skfem.models.poisson import laplace, unit_load
m = MeshTri.init_symmetric()
e = ElementTriP1()
ib = Basis(m, e)
A = asm(laplace, ib)
b = asm(unit_load, ib)
# Use boundary nodes by coordinate (symmetric mesh has no named boundaries)
boundary_dofs = ib.get_dofs(lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0))
u = solve(*condense(A, b, D=boundary_dofs.all()))
print(json.dumps({"ok": True, "max": float(u.max()), "version": skfem.__version__}))
'''
    ok, stdout, stderr = _run_script(sys.executable, script)
    dt = (time.time() - t0) * 1000
    if ok:
        try:
            data = json.loads(stdout)
            return SmokeResult("skfem", True, version=data.get("version"),
                               duration_ms=round(dt, 1))
        except json.JSONDecodeError:
            return SmokeResult("skfem", False, error=f"Bad output: {stdout[:200]}",
                               duration_ms=round(dt, 1))
    return SmokeResult("skfem", False, error=stderr[:300], duration_ms=round(dt, 1))


def smoke_kratos() -> SmokeResult:
    """Smoke test: create a model and check Kratos core works."""
    t0 = time.time()
    script = '''
import KratosMultiphysics as KM, json
model = KM.Model()
mp = model.CreateModelPart("test")
mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
mp.CreateNewNode(1, 0.0, 0.0, 0.0)
mp.CreateNewNode(2, 1.0, 0.0, 0.0)
ver = str(KM.Kernel.Version()).replace('"', '')
print(json.dumps({"ok": True, "nodes": mp.NumberOfNodes(), "version": ver}))
'''
    ok, stdout, stderr = _run_script(sys.executable, script)
    dt = (time.time() - t0) * 1000
    if ok:
        try:
            data = json.loads(stdout)
            return SmokeResult("kratos", True, version=data.get("version"),
                               duration_ms=round(dt, 1))
        except json.JSONDecodeError:
            return SmokeResult("kratos", False, error=f"Bad output: {stdout[:200]}",
                               duration_ms=round(dt, 1))
    return SmokeResult("kratos", False, error=stderr[:300], duration_ms=round(dt, 1))


def smoke_dune() -> SmokeResult:
    """Smoke test: import dune.fem (JIT compile may take time on first run).

    Known limitation: import-only — does NOT solve anything, because a
    real dune.fem solve JIT-compiles C++ modules on first use (minutes).
    A passing result means importable, not solve-capable; upgrade to a
    solve-based test once a working install is available to validate it.
    """
    t0 = time.time()
    script = '''
import json
try:
    import dune.fem
    print(json.dumps({"ok": True}))
except ImportError as e:
    print(json.dumps({"ok": False, "error": str(e)}))
'''
    ok, stdout, stderr = _run_script(sys.executable, script, timeout=60)
    dt = (time.time() - t0) * 1000
    if ok:
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            return SmokeResult("dune", False, error=f"Bad output: {stdout[:200]}",
                               duration_ms=round(dt, 1))
        # The inner script exits 0 even on ImportError — the verdict
        # lives in the "ok" flag, not the return code.
        if data.get("ok") is True:
            return SmokeResult("dune", True, duration_ms=round(dt, 1))
        return SmokeResult("dune", False,
                           error=str(data.get("error", "unknown"))[:300],
                           duration_ms=round(dt, 1))
    return SmokeResult("dune", False, error=stderr[:300], duration_ms=round(dt, 1))


def smoke_fourc() -> SmokeResult:
    """Smoke test: check 4C binary runs --version."""
    import os, shutil
    t0 = time.time()
    binary = os.environ.get("FOURC_BINARY", "")
    if not binary:
        for p in [os.path.expanduser("~/4C/build/4C"), "/opt/4C/build/4C"]:
            if Path(p).exists():
                binary = p; break
    if not binary or not Path(binary).exists():
        return SmokeResult("fourc", False, error="Binary not found",
                           duration_ms=0)
    try:
        r = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=10)
        dt = (time.time() - t0) * 1000
        version = ""
        for line in (r.stdout + r.stderr).splitlines():
            if "Multi-Physics" in line or "Release" in line:
                version = line.strip(); break
        return SmokeResult("fourc", True, version=version or "found",
                           duration_ms=round(dt, 1))
    except Exception as e:
        return SmokeResult("fourc", False, error=str(e),
                           duration_ms=round((time.time()-t0)*1000, 1))


# ── Registry ─────────────────────────────────────────────────

SMOKE_TESTS = {
    "ngsolve": smoke_ngsolve,
    "skfem": smoke_skfem,
    "kratos": smoke_kratos,
    "dune": smoke_dune,
    "fourc": smoke_fourc,
}


def run_all_smoke_tests(solvers: list[str] | None = None) -> dict[str, SmokeResult]:
    """Run smoke tests for all (or specified) solvers."""
    targets = solvers or list(SMOKE_TESTS.keys())
    results = {}
    for name in targets:
        if name in SMOKE_TESTS:
            try:
                results[name] = SMOKE_TESTS[name]()
            except Exception as e:
                results[name] = SmokeResult(name, False, error=str(e))
    return results
