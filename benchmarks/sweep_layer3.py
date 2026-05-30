#!/usr/bin/env python3
"""
Per-physics layer-3 sweep across every available backend.

Drives each backend through its `generate_input` + `run` path with the same
canonical reference problem per physics row, captures pass/fail, elapsed
wall-clock, and a key scalar (max(u), tip displacement, ...) from the VTU
output.  Emits a markdown table at the end and a JSON dump for downstream
plotting.

Env vars required for the full sweep (set automatically if matching install
exists on disk):
    DEAL_II_DIR     deal.II cmake config dir (.../lib/cmake/deal.II)
    FOURC_BINARY    Path to the built 4C binary
    FOURC_ROOT      Source root for FOURC (some templates depend on it)

Run:
    python benchmarks/sweep_layer3.py [--only POISSON,ELASTICITY]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

# ── repo paths ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "data"))

# ── env discovery (so users do not need to pre-export anything) ───────────
def _discover_env():
    """Best-effort discovery of solver locations on this machine."""
    home = Path.home()
    # deal.II — both the deal.II-style underscore env var (DEAL_II_DIR, used by
    # the dealii backend's HINTS line) and CMAKE_PREFIX_PATH (which is what
    # find_package(deal.II) actually consumes — the env var spelling
    # `deal.II_DIR` with a dot is not portable across shells).
    dealii_env_root = home / "miniconda3/envs/ofa-dealii"
    if "DEAL_II_DIR" not in os.environ:
        for c in [
            dealii_env_root / "lib/cmake/deal.II",
            Path("/usr/lib/x86_64-linux-gnu/cmake/deal.II"),
        ]:
            if c.is_dir():
                os.environ["DEAL_II_DIR"] = str(c)
                break
    if dealii_env_root.is_dir():
        # Best-effort discovery: append the conda env as a fallback, not as
        # the highest-priority entry, so an explicit user-provided
        # CMAKE_PREFIX_PATH still resolves first.  Use os.pathsep so this
        # remains correct on Windows (`;`) too.
        existing = os.environ.get("CMAKE_PREFIX_PATH", "")
        prefix_parts = [p for p in existing.split(os.pathsep) if p]
        if str(dealii_env_root) not in prefix_parts:
            prefix_parts.append(str(dealii_env_root))
            os.environ["CMAKE_PREFIX_PATH"] = os.pathsep.join(prefix_parts)
    # The conda-forge deal.II 9.1.1 package bakes a feedstock-only
    # compiler path into deal.IIConfig.cmake (/home/conda/feedstock_root/...)
    # which does not exist at runtime.  Override CC/CXX so CMake uses the
    # host's working compiler instead.
    import shutil as _shutil
    if "CXX" not in os.environ:
        for cxx in ("g++", "c++", "clang++"):
            if _shutil.which(cxx):
                os.environ["CXX"] = cxx
                break
    if "CC" not in os.environ:
        for cc in ("gcc", "cc", "clang"):
            if _shutil.which(cc):
                os.environ["CC"] = cc
                break
    # 4C
    if "FOURC_BINARY" not in os.environ:
        for c in [
            home / "Schreibtisch/4C-src/4C/build/4C",
            home / "4C/build/4C",
        ]:
            if c.is_file():
                os.environ["FOURC_BINARY"] = str(c)
                os.environ.setdefault("FOURC_ROOT", str(c.parent.parent))
                break


# NOTE: `_discover_env()` mutates os.environ.  It is invoked at the top of
# main() (not at import time) so that importing this module as a library —
# e.g. for unit-testing the matrix definition or `Cell` dataclass — does not
# silently rewrite CC, CXX, CMAKE_PREFIX_PATH, DEAL_II_DIR, FOURC_BINARY in
# the calling process.

# ── imports that need the registry available at module scope ──────────────
from core.registry import load_all_backends, get_backend  # noqa: E402
from core.backend import BackendStatus  # noqa: E402
from core.post_processing import post_process_file  # noqa: E402


RESULTS_DIR = REPO_ROOT / "benchmarks" / "sweep_results"


@dataclass
class Cell:
    """One (backend, physics, variant, params) cell of the matrix."""

    backend: str
    physics: str
    variant: str
    params: dict
    # The scalar field whose max(abs) we extract from the VTU output for
    # cross-cell comparison.  None ⇒ no scalar extraction (just pass/fail).
    field: str | None = None
    # Human-readable reference if there is one.
    expected: float | None = None
    # Relative tolerance for the expected value.
    rtol: float = 0.05


@dataclass
class CellResult:
    cell: Cell
    status: str
    elapsed_s: float | None
    scalar: float | None = None
    field_used: str | None = None
    error: str | None = None
    note: str = ""


# ═══════════════════════════════════════════════════════════════════════════
#   MATRIX
# ═══════════════════════════════════════════════════════════════════════════
# Reference problem per row:
#
#   POISSON           -Δu = 1 on [0,1]², u = 0 on ∂Ω.  max(u) ≈ 0.0737.
#   ELASTICITY        Cantilever beam under tip load — sign + order check.
#
# More rows can be added; sweep accepts --only to filter.

_KAPPA = 1.0

MATRIX: dict[str, list[Cell]] = {
    "POISSON": [
        # Reference: -Δu = 1 on [0,1]², u = 0 on ∂Ω.
        # Analytic max(u) ≈ 0.07367 (Fourier series, ~5-digit accuracy).
        Cell("skfem",   "poisson", "2d",         {"kappa": _KAPPA, "nx": 32, "ny": 32},
             field="phi", expected=0.0737, rtol=0.05),
        Cell("ngsolve", "poisson", "2d",         {"kappa": _KAPPA, "nx": 32, "ny": 32},
             field="phi", expected=0.0737, rtol=0.05),
        Cell("fenics",  "poisson", "2d",         {"kappa": _KAPPA, "nx": 32, "ny": 32},
             field="u", expected=0.0737, rtol=0.05),
        # Kratos `poisson_2d` is a known-fake scipy stub — see F-013 in
        # ~/.claude/memory.  The backend's validate_input correctly rejects
        # it; kept here as documentation of the catalog drift.
        Cell("kratos",  "poisson", "2d",         {"kappa": _KAPPA, "nx": 32, "ny": 32},
             field="TEMPERATURE", expected=0.0737, rtol=0.05),
        # deal.II 9.1.1 in the conda-forge ofa-dealii env requires the
        # legacy tbb/task.h header which was removed in TBB 2020.x — see
        # F-014.  Conda only ships dealii up to 9.3.2 against python<=3.10,
        # and ofa-dealii is pinned to 3.12.  Needs a separate env to fix.
        # deal.II's Poisson template writes the result vector as "solution"
        # (data_out.add_data_vector(solution, "solution") in the generated
        # main.cpp), so the expected field name here is "solution", not "u".
        Cell("dealii",  "poisson", "2d",         {"refinements": 5},
             field="solution", expected=0.0737, rtol=0.05),
        Cell("fourc",   "poisson", "poisson_2d", {},
             field="phi_1", expected=0.0737, rtol=0.05),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
#   CELL RUNNER
# ═══════════════════════════════════════════════════════════════════════════


async def run_cell(cell: Cell, work_dir: Path) -> CellResult:
    backend = get_backend(cell.backend)
    if backend is None:
        return CellResult(cell, status="not_registered", elapsed_s=None,
                          error=f"no backend registered with name {cell.backend!r}")

    status, msg = backend.check_availability()
    if status != BackendStatus.AVAILABLE:
        return CellResult(cell, status=f"unavailable", elapsed_s=None,
                          error=msg[:200] if msg else "unavailable",
                          note=str(status))

    t0 = time.time()
    try:
        content = backend.generate_input(cell.physics, cell.variant, cell.params)
    except Exception as e:
        return CellResult(cell, status="generate_failed", elapsed_s=None,
                          error=f"{type(e).__name__}: {e!s:.300}")

    errors = backend.validate_input(content)
    if errors:
        return CellResult(cell, status="validation_failed", elapsed_s=None,
                          error="; ".join(errors)[:300])

    # Clear any artefacts from a previous sweep run in this cell's directory
    # before we ask the backend to run — otherwise stale .vtu files from a
    # different revision could be picked up by `backend.get_result_files(job)`
    # and a non-VTU-producing run would silently report the previous run's
    # scalar as a pass.
    if work_dir.exists():
        import shutil
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        job = await backend.run(content, work_dir, np=1, timeout=600)
    except Exception as e:
        return CellResult(cell, status="run_threw", elapsed_s=time.time() - t0,
                          error=f"{type(e).__name__}: {e!s:.300}")

    elapsed = time.time() - t0

    if job.status != "completed":
        return CellResult(cell, status=job.status, elapsed_s=elapsed,
                          error=(job.error or "")[:300])

    # ── extract scalar from the LAST VTU (final time step / converged state)
    vtu_files = sorted([f for f in backend.get_result_files(job) if f.suffix == ".vtu"])
    scalar = None
    field_used = None
    if cell.field is not None:
        # A cell that asked for a scalar but received no .vtu is a failure
        # of the run even if the backend exited 0 — surface it instead of
        # quietly returning status=completed with scalar=None.
        if not vtu_files:
            return CellResult(
                cell, status="no_vtu_output", elapsed_s=elapsed,
                error=("backend reported completed but produced no .vtu in "
                       f"{work_dir}; expected field {cell.field!r}"),
            )
        try:
            pp = post_process_file(vtu_files[-1], plot_dir=work_dir, plot_fields=False)
            available = [f.name for f in pp.fields]
            for f in pp.fields:
                if f.name == cell.field or f.name.lower() == cell.field.lower():
                    scalar = float(f.max)
                    field_used = f.name
                    break
            # No fallback.  A silent fallback to `pp.fields[0]` would happily
            # report max(Owner) (a cell-ID integer) as if it were `u`, which
            # is exactly the kind of stealth-wrong result this sweep exists
            # to catch.  If the expected field is absent we report the gap
            # explicitly so the matrix shows it, leaving the cell scalar=None.
            if scalar is None:
                return CellResult(
                    cell, status="field_not_found", elapsed_s=elapsed,
                    error=f"expected field {cell.field!r} absent; available: {available[:8]}",
                )
        except Exception as e:
            return CellResult(cell, status="postproc_failed", elapsed_s=elapsed,
                              error=f"{type(e).__name__}: {e!s:.300}")

    return CellResult(cell, status="completed", elapsed_s=elapsed,
                      scalar=scalar, field_used=field_used)


# ═══════════════════════════════════════════════════════════════════════════
#   ROW / MATRIX DRIVERS
# ═══════════════════════════════════════════════════════════════════════════


async def run_row(row_name: str, cells: list[Cell]) -> list[CellResult]:
    print(f"\n=== {row_name} ({len(cells)} cells) ===")
    out: list[CellResult] = []
    for cell in cells:
        work_dir = RESULTS_DIR / row_name.lower() / cell.backend
        result = await run_cell(cell, work_dir)
        out.append(result)
        # one-line summary per cell
        s = result.status
        e = f"{result.elapsed_s:.1f}s" if result.elapsed_s else "    "
        sc = f"{result.scalar:.5f}" if result.scalar is not None else "       "
        f_ = result.field_used or ""
        ref = f" ref={cell.expected:.5f}" if cell.expected is not None else ""
        print(f"  {cell.backend:8s}  {s:18s}  {e:8s}  {sc} {f_:18s}{ref}")
        if result.error:
            print(f"           -> {result.error[:200]}")
    return out


def emit_markdown_table(rows: dict[str, list[CellResult]]) -> str:
    lines: list[str] = []
    lines.append("# Per-physics layer-3 sweep — results\n")
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} on this machine._\n")
    for row_name, results in rows.items():
        lines.append(f"\n## {row_name}\n")
        lines.append("| Backend | Status | Elapsed | Scalar | Field | Ref | Within tol |")
        lines.append("|---------|--------|---------|--------|-------|-----|------------|")
        for r in results:
            ok_tol = ""
            if r.cell.expected is not None and r.scalar is not None:
                rel = abs(r.scalar - r.cell.expected) / max(abs(r.cell.expected), 1e-30)
                ok_tol = "✓" if rel <= r.cell.rtol else f"✗ ({rel:.2%})"
            elif r.scalar is None and r.status == "completed":
                ok_tol = "—"
            lines.append(
                f"| {r.cell.backend} | {r.status} | "
                f"{f'{r.elapsed_s:.1f}s' if r.elapsed_s else '—'} | "
                f"{f'{r.scalar:.5f}' if r.scalar is not None else '—'} | "
                f"{r.field_used or '—'} | "
                f"{f'{r.cell.expected:.5f}' if r.cell.expected is not None else '—'} | "
                f"{ok_tol} |"
            )
    return "\n".join(lines)


def cell_result_to_dict(r: CellResult) -> dict:
    d = asdict(r)
    d["cell"] = asdict(r.cell)
    return d


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="",
                   help="comma-separated row names (POISSON,ELASTICITY,...)")
    args = p.parse_args()

    _discover_env()  # mutate os.environ here, not at module-import time
    load_all_backends()
    print("loaded backends:")
    from core.registry import available_backends
    for b in available_backends():
        print(f"  - {b.name():8s}  {b.display_name()}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    requested = set(s.strip().upper() for s in args.only.split(",") if s.strip())
    # Fail fast on a typo: an empty result set (silently writing an empty
    # sweep.json) would look like a successful validation run.
    unknown = requested - MATRIX.keys()
    if unknown:
        raise SystemExit(
            f"--only references rows that are not in the matrix: "
            f"{sorted(unknown)} — known rows: {sorted(MATRIX.keys())}"
        )
    rows_to_run = {n: c for n, c in MATRIX.items() if not requested or n in requested}

    results: dict[str, list[CellResult]] = {}
    for row_name, cells in rows_to_run.items():
        results[row_name] = await run_row(row_name, cells)

    # ── persist
    json_path = RESULTS_DIR / "sweep.json"
    json_path.write_text(json.dumps(
        {row: [cell_result_to_dict(r) for r in res] for row, res in results.items()},
        indent=2, default=str,
    ))

    md = emit_markdown_table(results)
    md_path = RESULTS_DIR / "sweep.md"
    md_path.write_text(md)

    print("\n" + "=" * 72)
    print(md)
    print("=" * 72)
    print(f"\nJSON: {json_path}")
    print(f"  MD: {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
