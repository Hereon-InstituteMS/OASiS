"""Layer G orchestrator — run one (task, mode) pair end-to-end and log.

Usage:
  python -m eval.layer_g.run --task <task-id> --mode <full|stripped|minus:TOKEN>
  python -m eval.layer_g.run --all              # run every task in held_out_tasks.yaml in all default modes

Default mode set:
  full
  stripped
  minus:<task.target_pitfall>

Logs JSONL to eval/layer_g/results/<timestamp>.jsonl.

Reads ANTHROPIC_API_KEY from environment. Dry-run mode (--dry-run) skips the
Claude call and uses a fixed stub response — for harness-shape testing.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

# yaml is an optional dep — fall back to a simple loader if absent.
try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    yaml = None  # type: ignore[assignment]

from eval.layer_g.ablation_modes import apply_ablation, count_pitfalls
from eval.layer_g.grade import grade


TASKS_FILE = Path(__file__).resolve().parent / "held_out_tasks.yaml"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_tasks() -> list[dict[str, Any]]:
    if yaml is None:
        raise SystemExit("PyYAML required: pip install pyyaml")
    with open(TASKS_FILE) as f:
        data = yaml.safe_load(f)
    return data["tasks"]


_BACKENDS_LOADED = False


def _ensure_backends_loaded() -> None:
    global _BACKENDS_LOADED
    if _BACKENDS_LOADED:
        return
    try:
        from core.registry import load_all_backends  # type: ignore[import-not-found]
    except ImportError as e:
        raise SystemExit(
            f"could not import core.registry from {REPO_ROOT}/src: {e}"
        ) from e
    load_all_backends()
    _BACKENDS_LOADED = True


def get_knowledge(backend_name: str, physics: str) -> dict[str, Any]:
    """Pull the catalog dict for (backend, physics) using the same path
    the MCP server uses internally."""
    _ensure_backends_loaded()
    from core.registry import get_backend  # type: ignore[import-not-found]
    backend = get_backend(backend_name)
    if backend is None:
        raise RuntimeError(
            f"backend {backend_name!r} not registered. Check "
            f"src/backends/{backend_name}/backend.py is importable in this env."
        )
    knowledge = backend.get_knowledge(physics)
    if not isinstance(knowledge, dict):
        return {"raw": knowledge}
    return knowledge


def default_modes_for_task(task: dict[str, Any]) -> list[str]:
    target = task.get("target_pitfall")
    modes = ["full", "stripped"]
    if target:
        modes.append(f"minus:{target}")
    return modes


def run_one(
    task: dict[str, Any],
    mode: str,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run a single (task, mode) cell and return a result record."""
    record: dict[str, Any] = {
        "task_id": task["id"],
        "mode": mode,
        "backend": task["backend"],
        "physics": task["physics"],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    try:
        base_knowledge = get_knowledge(task["backend"], task["physics"])
        ablated = apply_ablation(base_knowledge, mode)
        record["pitfalls_full"] = count_pitfalls(base_knowledge)
        record["pitfalls_used"] = count_pitfalls(ablated)
    except Exception as e:
        record["error"] = f"knowledge load failed: {e}"
        record["passed"] = False
        record["traceback"] = traceback.format_exc()
        return record

    if dry_run:
        from eval.layer_g.grade import Verdict
        verdict = Verdict(
            passed=False,
            exit_code=None,
            stdout="(dry-run)",
            stderr="",
            reason="dry-run; no Claude call made",
        )
        record["claude"] = {"dry_run": True}
    else:
        try:
            from eval.layer_g.claude_runner import run_claude
            resp = run_claude(task, mode, ablated)
            record["claude"] = {
                "response_id": resp.response_id,
                "stop_reason": resp.stop_reason,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "code_block_count": len(resp.code_blocks),
            }
        except Exception as e:
            record["error"] = f"claude call failed: {e}"
            record["passed"] = False
            record["traceback"] = traceback.format_exc()
            return record

        if not resp.code_blocks:
            record["passed"] = False
            record["error"] = "claude returned no code block"
            return record

        code = resp.code_blocks[0]
        record["code_chars"] = len(code)

        verdict = grade(task, code)

    record["passed"] = verdict.passed
    record["reason"] = verdict.reason
    record["exit_code"] = verdict.exit_code
    record["stdout_tail"] = (verdict.stdout or "")[-1500:]
    record["stderr_tail"] = (verdict.stderr or "")[-1500:]
    record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Layer G ablation runner")
    parser.add_argument("--task", help="single task id")
    parser.add_argument("--mode", help="single mode override")
    parser.add_argument("--all", action="store_true",
                        help="run every task in default modes")
    parser.add_argument("--dry-run", action="store_true",
                        help="skip Claude call; verify harness")
    parser.add_argument("--out", help="output JSONL path (default: timestamped)")
    args = parser.parse_args(argv)

    if not (args.task or args.all):
        parser.error("specify --task or --all")

    tasks = load_tasks()
    if args.task:
        tasks = [t for t in tasks if t["id"] == args.task]
        if not tasks:
            print(f"no task with id={args.task!r}", file=sys.stderr)
            return 2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"run_{stamp}.jsonl"

    n_pass = n_run = 0
    with open(out_path, "w") as f:
        for task in tasks:
            modes = [args.mode] if args.mode else default_modes_for_task(task)
            for mode in modes:
                print(f"[{task['id']}][{mode}] running...", flush=True)
                rec = run_one(task, mode, dry_run=args.dry_run)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n_run += 1
                if rec.get("passed"):
                    n_pass += 1
                print(
                    f"[{task['id']}][{mode}] -> "
                    f"{'PASS' if rec.get('passed') else 'FAIL'} "
                    f"({rec.get('reason', '?')})",
                    flush=True,
                )

    print(f"\n=== {n_pass}/{n_run} passed; results in {out_path} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
