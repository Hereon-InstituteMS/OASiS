#!/usr/bin/env python3
"""
Run all benchmarks and generate paper-ready results.

Usage:
    python benchmarks/run_all_benchmarks.py

Outputs to benchmarks/results/ with JSON data and PNG plots.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.registry import load_all_backends, get_backend, available_backends
from core.backend import BackendStatus
from core.post_processing import post_process_file, compare_results

RESULTS_DIR = Path(__file__).parent / "results"


async def run_benchmark(backend, physics, variant, params, work_dir):
    """Run a single benchmark and return results."""
    content = backend.generate_input(physics, variant, params)
    errors = backend.validate_input(content)
    if errors:
        return {"solver": backend.name(), "status": "validation_failed", "errors": errors}

    job = await backend.run(content, work_dir, np=1, timeout=600)

    result = {
        "solver": backend.name(),
        "display_name": backend.display_name(),
        "status": job.status,
        "elapsed_s": round(job.elapsed, 3) if job.elapsed else None,
    }

    if job.status == "completed":
        vtu_files = [f for f in backend.get_result_files(job) if f.suffix == ".vtu"]
        if vtu_files:
            pp = post_process_file(vtu_files[0], plot_dir=work_dir, plot_fields=True)
            result["mesh"] = pp.mesh.to_dict()
            result["fields"] = [f.to_dict() for f in pp.fields]
            result["plots"] = pp.plots
        else:
            result["output_files"] = [f.name for f in backend.get_result_files(job)]
    elif job.error:
        result["error"] = job.error[:500]

    return result


async def benchmark_poisson():
    """Cross-solver Poisson benchmark: -Δu=1 on [0,1]², u=0 on ∂Ω."""
    print("\n" + "="*60)
    print("BENCHMARK 1: Poisson equation (cross-solver)")
    print("="*60)

    bench_dir = RESULTS_DIR / "poisson_cross_solver"
    bench_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("fenics", "poisson", "2d", {"kappa": 1.0, "nx": 32, "ny": 32}),
        ("dealii", "poisson", "2d", {"refinements": 5}),
        ("fourc", "poisson", "poisson_2d", {}),
    ]

    results = []
    vtu_paths = []
    for solver_name, physics, variant, params in configs:
        backend = get_backend(solver_name)
        if not backend:
            continue
        status, _ = backend.check_availability()
        if status != BackendStatus.AVAILABLE:
            continue

        work_dir = bench_dir / solver_name
        work_dir.mkdir(parents=True, exist_ok=True)
        r = await run_benchmark(backend, physics, variant, params, work_dir)
        results.append(r)
        print(f"  {r['display_name']}: {r['status']} ({r.get('elapsed_s', '?')}s)")

        if r["status"] == "completed":
            vtus = list(work_dir.glob("*.vtu"))
            if vtus:
                vtu_paths.append(vtus[0])

    # Cross-solver comparison
    comparison = {}
    if len(vtu_paths) >= 2:
        comparison = compare_results(vtu_paths)

    output = {
        "benchmark": "Poisson -Δu=1 on [0,1]², u=0 on ∂Ω",
        "expected_max_u": 0.07373,
        "results": results,
        "cross_solver_comparison": comparison,
    }

    (bench_dir / "benchmark.json").write_text(json.dumps(output, indent=2, default=str))
    print(f"  Results saved to {bench_dir}")
    return output


async def benchmark_elasticity():
    """Cross-solver elasticity benchmark: cantilever beam."""
    print("\n" + "="*60)
    print("BENCHMARK 2: Linear elasticity (cantilever beam)")
    print("="*60)

    bench_dir = RESULTS_DIR / "elasticity_cross_solver"
    bench_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("fenics", "linear_elasticity", "2d", {"E": 1000, "nu": 0.3}),
        ("dealii", "linear_elasticity", "2d", {}),
    ]

    results = []
    for solver_name, physics, variant, params in configs:
        backend = get_backend(solver_name)
        if not backend:
            continue
        status, _ = backend.check_availability()
        if status != BackendStatus.AVAILABLE:
            continue

        work_dir = bench_dir / solver_name
        work_dir.mkdir(parents=True, exist_ok=True)
        r = await run_benchmark(backend, physics, variant, params, work_dir)
        results.append(r)
        print(f"  {r['display_name']}: {r['status']} ({r.get('elapsed_s', '?')}s)")

    output = {"benchmark": "Linear elasticity cantilever", "results": results}
    (bench_dir / "benchmark.json").write_text(json.dumps(output, indent=2, default=str))
    print(f"  Results saved to {bench_dir}")
    return output


async def benchmark_navier_stokes():
    """Navier-Stokes lid-driven cavity at Re=100."""
    print("\n" + "="*60)
    print("BENCHMARK 3: Navier-Stokes lid-driven cavity (Re=100)")
    print("="*60)

    bench_dir = RESULTS_DIR / "navier_stokes"
    bench_dir.mkdir(parents=True, exist_ok=True)

    backend = get_backend("fenics")
    if not backend or backend.check_availability()[0] != BackendStatus.AVAILABLE:
        print("  FEniCS not available, skipping")
        return None

    work_dir = bench_dir / "fenics"
    work_dir.mkdir(parents=True, exist_ok=True)
    r = await run_benchmark(backend, "navier_stokes", "2d",
                            {"Re": 100, "nx": 32, "ny": 32}, work_dir)
    print(f"  {r['display_name']}: {r['status']} ({r.get('elapsed_s', '?')}s)")

    output = {"benchmark": "Navier-Stokes cavity Re=100", "results": [r]}
    (bench_dir / "benchmark.json").write_text(json.dumps(output, indent=2, default=str))
    print(f"  Results saved to {bench_dir}")
    return output


async def benchmark_thermal_structural():
    """Coupled thermal-structural analysis."""
    print("\n" + "="*60)
    print("BENCHMARK 4: Coupled thermal-structural (multi-physics)")
    print("="*60)

    bench_dir = RESULTS_DIR / "thermal_structural"
    bench_dir.mkdir(parents=True, exist_ok=True)

    backend = get_backend("fenics")
    if not backend or backend.check_availability()[0] != BackendStatus.AVAILABLE:
        print("  FEniCS not available, skipping")
        return None

    work_dir = bench_dir / "fenics"
    work_dir.mkdir(parents=True, exist_ok=True)
    r = await run_benchmark(backend, "thermal_structural", "2d",
                            {"E": 200e3, "nu": 0.3, "alpha": 12e-6,
                             "T_hot": 100, "T_cold": 0,
                             "nx": 40, "ny": 40}, work_dir)
    print(f"  {r['display_name']}: {r['status']} ({r.get('elapsed_s', '?')}s)")

    output = {"benchmark": "Coupled thermal-structural", "results": [r]}
    (bench_dir / "benchmark.json").write_text(json.dumps(output, indent=2, default=str))
    print(f"  Results saved to {bench_dir}")
    return output


async def benchmark_hyperelasticity():
    """Hyperelasticity (Neo-Hookean) — nonlinear mechanics."""
    print("\n" + "="*60)
    print("BENCHMARK 5: Hyperelasticity (Neo-Hookean, large deformation)")
    print("="*60)

    bench_dir = RESULTS_DIR / "hyperelasticity"
    bench_dir.mkdir(parents=True, exist_ok=True)

    backend = get_backend("fenics")
    if not backend or backend.check_availability()[0] != BackendStatus.AVAILABLE:
        print("  FEniCS not available, skipping")
        return None

    work_dir = bench_dir / "fenics"
    work_dir.mkdir(parents=True, exist_ok=True)
    r = await run_benchmark(backend, "hyperelasticity", "3d",
                            {"E": 1000, "nu": 0.3}, work_dir)
    print(f"  {r['display_name']}: {r['status']} ({r.get('elapsed_s', '?')}s)")

    output = {"benchmark": "Hyperelasticity Neo-Hookean", "results": [r]}
    (bench_dir / "benchmark.json").write_text(json.dumps(output, indent=2, default=str))
    print(f"  Results saved to {bench_dir}")
    return output


async def main():
    print("OASiS — Full Benchmark Suite")
    print(f"{'='*60}")

    load_all_backends()

    # Show available backends
    avail = available_backends()
    print(f"\nAvailable backends: {[b.display_name() for b in avail]}")

    start = time.time()
    all_results = {}

    all_results["poisson"] = await benchmark_poisson()
    all_results["elasticity"] = await benchmark_elasticity()
    all_results["navier_stokes"] = await benchmark_navier_stokes()
    all_results["thermal_structural"] = await benchmark_thermal_structural()
    all_results["hyperelasticity"] = await benchmark_hyperelasticity()

    total = time.time() - start

    # Summary
    print(f"\n{'='*60}")
    print(f"All benchmarks complete in {total:.1f}s")
    print(f"Results saved to: {RESULTS_DIR}")

    # Write summary
    summary = {
        "total_time_s": round(total, 1),
        "benchmarks": list(all_results.keys()),
        "backends_used": [b.name() for b in avail],
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # Print table
    print("\n## Results Summary")
    print("| Benchmark | Solver | Status | DOFs | Time |")
    print("|-----------|--------|--------|------|------|")
    for bname, bdata in all_results.items():
        if bdata and "results" in bdata:
            for r in bdata["results"]:
                dofs = r.get("mesh", {}).get("n_points", "-")
                t = r.get("elapsed_s", "-")
                print(f"| {bname} | {r.get('display_name', r['solver'])} | {r['status']} | {dofs} | {t}s |")


if __name__ == "__main__":
    asyncio.run(main())
