"""Tier-2: quad4 shear locking, and the numeric floor it establishes.

Catalog claim under test (KNOWLEDGE['structural_dynamics']):

    "Linear hex8 (SmallDisplacementElement3D8N) shear-locks in bending-dominated
     problems ... Same applies to linear quad4 in 2D — use quad8/quad9.
     Signal: cantilever tip deflection ... 20-40% smaller than analytic;
     switching to [quadratic] recovers it."

Real Kratos StructuralMechanicsAnalysis, plane-stress cantilever
(L = 10, h = 1, t = 1, E = 2e11, nu = 0, tip shear 1000 N,
LinearElasticPlaneStress2DLaw, sparse_lu). Timoshenko reference
w = P L^3 / (3 E I) + P L / (k G A), k = 5/6, so w = -2.012e-5.

Measured 2026-08-03 on Kratos 10.4.0:
    quad4 10x1 -> -1.340e-5  = 66.6% of Timoshenko  (33.4% too stiff)
    quad8 10x1 -> -2.010e-5  = 99.9%                (recovered)
    quad4 20x2 -> 88.8% , 40x4 -> 96.9% , 80x8 -> 99.2%

This fixture also doubles as the CORRECTNESS gate for the Kratos structural
path: it does not merely check that the run exits 0, it checks the converged
answer against the analytic beam solution.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEAM = HERE / "_beam.py"

CASES = [("quad4", 10, 1), ("quad8", 10, 1), ("quad4", 80, 8)]
results = {}
for etype, nx, ny in CASES:
    proc = subprocess.run([sys.executable, str(BEAM), etype, str(nx), str(ny)],
                          cwd=str(HERE), capture_output=True, text=True, timeout=600)
    line = [ln for ln in proc.stdout.splitlines() if "JSONRESULT" in ln]
    if not line:
        print(f"FAIL: no JSONRESULT for {etype} {nx}x{ny}")
        print(proc.stdout[-800:])
        print(proc.stderr[-800:])
        sys.exit(1)
    results[f"{etype}_{nx}x{ny}"] = json.loads(line[0].split("JSONRESULT", 1)[1])

q4 = results["quad4_10x1"]
q8 = results["quad8_10x1"]
q4f = results["quad4_80x8"]

print(f"timoshenko_reference={q4['timoshenko']:.6e}")
print(f"quad4_10x1_uy={q4['uy']:.6e}")
print(f"quad8_10x1_uy={q8['uy']:.6e}")
print(f"quad4_80x8_uy={q4f['uy']:.6e}")
print(f"quad4_10x1_ratio={q4['ratio_to_timoshenko']:.4f}")
print(f"quad8_10x1_ratio={q8['ratio_to_timoshenko']:.4f}")
print(f"quad4_80x8_ratio={q4f['ratio_to_timoshenko']:.4f}")

locks = 0.60 <= q4["ratio_to_timoshenko"] <= 0.80
recovers = abs(q8["ratio_to_timoshenko"] - 1.0) <= 0.02
converges = abs(q4f["ratio_to_timoshenko"] - 1.0) <= 0.02
print(f"quad4_locks_20_to_40_percent={locks}")
print(f"quad8_recovers_within_2_percent={recovers}")
print(f"quad4_converges_under_refinement={converges}")

if not (locks and recovers and converges):
    print("FAIL: fixture expectations not met")
    sys.exit(1)
