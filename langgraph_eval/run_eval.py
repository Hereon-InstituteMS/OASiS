"""Driver for the open-weight model-size ablation.

Iterates: (model size) × (BARE | MCP) × tasks × seeds. Each cell gets its
own sandbox workdir and a deterministic result path. After all cells run,
calls grade.py to produce summary.csv and scaling.pdf.

Resumable: cells whose result.txt already exists are skipped unless --force.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from pathlib import Path

from agent import build_bare_agent, build_mcp_agent
from tasks_subset import TASKS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "runs" / "openweight"


def cell_dir(out_root: Path, model: str, cond: str, task_id: str, seed: int) -> Path:
    return out_root / f"{model}_{cond}_{task_id}_seed{seed}"


def run_one(model: str, cond: str, task, seed: int, out_root: Path,
            force: bool) -> dict:
    work = cell_dir(out_root, model, cond, task.id, seed) / "work"
    work.mkdir(parents=True, exist_ok=True)
    result_path = work / "result.txt"
    if result_path.exists() and not force:
        return {"model": model, "condition": cond, "task_id": task.id,
                "seed": seed, "status": "skipped",
                "result_path": str(result_path)}
    factory = build_bare_agent if cond == "BARE" else build_mcp_agent
    agent = factory(size=model, seed=seed, workdir=work)

    prompt = task.prompt.replace("{OUT}", str(result_path))
    t0 = time.time()
    err = None
    try:
        agent.invoke(
            {"messages": [("user", prompt)]},
            config={"recursion_limit": 80},
        )
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    dt = time.time() - t0

    return {
        "model": model, "condition": cond, "task_id": task.id, "seed": seed,
        "wall_clock_s": round(dt, 1),
        "status": "ok" if err is None else "agent_error",
        "agent_error": err,
        "result_path": str(result_path),
        "result_exists": result_path.exists(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["7b", "14b", "32b"])
    p.add_argument("--conditions", nargs="+", default=["BARE", "MCP"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--tasks", nargs="*", default=None,
                   help="task IDs to run (default: all in tasks_subset.TASKS)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "cells.jsonl"

    selected = (TASKS if not args.tasks
                else [t for t in TASKS if t.id in args.tasks])
    print(f"Cells: {len(args.models)} × {len(args.conditions)} × "
          f"{len(selected)} × {len(args.seeds)} = "
          f"{len(args.models) * len(args.conditions) * len(selected) * len(args.seeds)}")

    with log_path.open("a") as logf:
        for model in args.models:
            for cond in args.conditions:
                for task in selected:
                    for seed in args.seeds:
                        rec = run_one(model, cond, task, seed, args.out, args.force)
                        logf.write(json.dumps(rec) + "\n")
                        logf.flush()
                        print(f"  {rec['model']:>4} {rec['condition']:<4} "
                              f"{rec['task_id']:<3} s{rec['seed']}  "
                              f"{rec.get('wall_clock_s', '-'):>6} s  "
                              f"{rec['status']}")

    print("Done. Now run: python grade.py", args.out)


if __name__ == "__main__":
    main()
