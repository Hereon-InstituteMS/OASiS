"""Apply HOE-v2 grading gates to the open-weight ablation outputs.

Produces:
* ``summary.csv`` — one row per cell with passed / reason / wall_clock
* ``scaling.json`` — pass rate per (model, condition) for the scaling plot
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from tasks_subset import TASKS, _parse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    args = p.parse_args()

    tasks = {t.id: t for t in TASKS}
    rows = []
    pass_counts = defaultdict(lambda: [0, 0])  # (model,cond) -> [pass, total]

    log = args.out / "cells.jsonl"
    for line in log.read_text().splitlines():
        rec = json.loads(line)
        rp = Path(rec["result_path"])
        passed, reason = False, "no result.txt"
        if rp.exists():
            parsed = _parse(rp.read_text())
            t = tasks.get(rec["task_id"])
            if t is not None:
                passed, reason = t.grade(parsed)
        key = (rec["model"], rec["condition"])
        pass_counts[key][1] += 1
        if passed:
            pass_counts[key][0] += 1
        rows.append({**rec, "passed": passed, "grade_reason": reason})

    with (args.out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    scaling = {f"{m}_{c}": {"pass": p_, "total": t_, "rate": p_ / t_ if t_ else 0}
               for (m, c), (p_, t_) in pass_counts.items()}
    (args.out / "scaling.json").write_text(json.dumps(scaling, indent=2))

    print("Pass rates:")
    for k, v in scaling.items():
        print(f"  {k:<14} {v['pass']:>3}/{v['total']:<3}  {v['rate']*100:5.1f}%")


if __name__ == "__main__":
    main()
