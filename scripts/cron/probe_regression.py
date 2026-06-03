#!/usr/bin/env python3
"""Probe-regression cron — daily rerun of the live falsification probes.

Catalog pitfalls only stay honest if their predicted error mode still
fires when the buggy code path runs. Upstream releases can fix a defect
(making the pitfall obsolete) or break a guarantee (making the
catalog claim wrong). Without a rerun loop, the catalog rots silently.

This script runs `tests/test_pitfall_falsification_live.py` across all
backend conda envs that have the required packages installed, diffs
the per-test verdicts against the prior run's baseline, and emits a
Markdown report flagging:

  - NEWLY-FAILING  : probe was passing, now fails. Likely env regression
                     (numpy bump, conda upgrade) — investigate first.
  - NEWLY-PASSING  : probe was failing or skipped, now passes. Could
                     mean an upstream fix landed and the pitfall is now
                     stale (consider deletion).
  - NEWLY-SKIPPED  : probe ran before but now skips. Probably a missing
                     dependency — install issue.

It NEVER mutates the catalog. Output is purely advisory; the user
decides which pitfalls to update based on the report.

Wire-up:
  # nightly at 03:30
  30 3 * * * cd /home/hermann/Schreibtisch/Open-FEM-agent && \
    /home/hermann/miniconda3/bin/python3 scripts/cron/probe_regression.py

The script is idempotent (re-runs produce the same baseline if no
state changed) and safe to invoke from any cwd.
"""
from __future__ import annotations
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
TEST_FILE = REPO / "tests" / "test_pitfall_falsification_live.py"
BASELINE = REPO / "scripts" / "cron" / "baseline_probes.json"
REPORTS_DIR = REPO / "scripts" / "cron" / "reports"


ENVS = [
    {
        "name": "venv",
        "python": str(REPO / ".venv" / "bin" / "python"),
        "label": "default venv (skfem / kratos / ngsolve)",
    },
    {
        "name": "ofa-fenicsx",
        "python": str(Path.home() / "miniconda3/envs/ofa-fenicsx/bin/python"),
        "label": "conda ofa-fenicsx (dolfinx)",
    },
    {
        "name": "ofa-dealii",
        "python": str(Path.home() / "miniconda3/envs/ofa-dealii/bin/python"),
        "label": "conda ofa-dealii",
    },
    {
        "name": "ofa-dune",
        "python": str(Path.home() / "miniconda3/envs/ofa-dune/bin/python"),
        "label": "conda ofa-dune",
    },
]


_VERDICT_RE = re.compile(
    r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+"
    r"tests/test_pitfall_falsification_live\.py::"
    r"TestPitfallFalsificationLive::(\S+)"
)


def run_probes(env: dict) -> dict[str, str]:
    """Return {test_name: verdict} for every probe in this env.

    Tests not exercised (skipped at collection time) appear with
    verdict 'SKIPPED'. If the python executable is missing we
    return {} so the env contributes no signal.
    """
    py = env["python"]
    if not Path(py).is_file():
        print(f"[{env['name']}] python not found at {py}, skipping env",
              file=sys.stderr)
        return {}

    cmd = [
        py, "-m", "pytest",
        str(TEST_FILE),
        "-v", "--tb=no", "--no-header",
        "-r", "fEsxXp",  # report all verdict types
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(REPO),
        )
    except subprocess.TimeoutExpired:
        print(f"[{env['name']}] timeout after 10min", file=sys.stderr)
        return {}

    verdicts: dict[str, str] = {}
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        m = _VERDICT_RE.match(line.strip())
        if m:
            verdict, name = m.group(1), m.group(2)
            # Multi-env: same probe might run in multiple envs;
            # pick the most-informative verdict (any non-SKIPPED beats
            # SKIPPED, FAILED is worse than PASSED so it wins for
            # surfacing regressions).
            existing = verdicts.get(name)
            if existing is None:
                verdicts[name] = verdict
            elif verdict in ("FAILED", "ERROR") and existing == "PASSED":
                verdicts[name] = verdict
            elif existing == "SKIPPED" and verdict != "SKIPPED":
                verdicts[name] = verdict
    return verdicts


def collect_all_envs() -> dict[str, str]:
    """Merge probe verdicts across every available env."""
    merged: dict[str, str] = {}
    for env in ENVS:
        results = run_probes(env)
        for name, verdict in results.items():
            existing = merged.get(name)
            if existing is None:
                merged[name] = verdict
            elif existing == "SKIPPED" and verdict != "SKIPPED":
                merged[name] = verdict
            elif verdict in ("FAILED", "ERROR") and existing == "PASSED":
                merged[name] = verdict
    return merged


def diff_against_baseline(current: dict[str, str],
                          baseline: dict[str, str]) -> dict[str, list[str]]:
    """Return categorised diffs: newly_failing / newly_passing / newly_skipped."""
    newly_failing: list[str] = []
    newly_passing: list[str] = []
    newly_skipped: list[str] = []
    appeared: list[str] = []

    for name, curr in current.items():
        prev = baseline.get(name)
        if prev is None:
            appeared.append(f"{name} (now: {curr})")
            continue
        if prev == curr:
            continue
        if curr in ("FAILED", "ERROR") and prev not in ("FAILED", "ERROR"):
            newly_failing.append(f"{name}: {prev} -> {curr}")
        elif curr == "PASSED" and prev != "PASSED":
            newly_passing.append(f"{name}: {prev} -> {curr}")
        elif curr == "SKIPPED" and prev != "SKIPPED":
            newly_skipped.append(f"{name}: {prev} -> {curr}")

    return {
        "newly_failing": newly_failing,
        "newly_passing": newly_passing,
        "newly_skipped": newly_skipped,
        "appeared": appeared,
    }


def write_report(stamp: str, diff: dict[str, list[str]],
                 current: dict[str, str]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"probes_{stamp}.md"

    pass_n = sum(1 for v in current.values() if v == "PASSED")
    fail_n = sum(1 for v in current.values() if v in ("FAILED", "ERROR"))
    skip_n = sum(1 for v in current.values() if v == "SKIPPED")
    total = len(current)

    lines = [
        f"# Probe regression report — {stamp}",
        "",
        f"Total probes: {total}  •  PASS: {pass_n}  •  "
        f"FAIL: {fail_n}  •  SKIP: {skip_n}",
        "",
    ]

    if diff["newly_failing"]:
        lines.append("## NEWLY FAILING (likely env regression — investigate first)")
        lines.append("")
        for entry in diff["newly_failing"]:
            lines.append(f"- {entry}")
        lines.append("")

    if diff["newly_passing"]:
        lines.append(
            "## NEWLY PASSING (possible upstream fix — pitfall may be stale)")
        lines.append("")
        for entry in diff["newly_passing"]:
            lines.append(f"- {entry}")
        lines.append("")

    if diff["newly_skipped"]:
        lines.append("## NEWLY SKIPPED (dependency lost?)")
        lines.append("")
        for entry in diff["newly_skipped"]:
            lines.append(f"- {entry}")
        lines.append("")

    if diff["appeared"]:
        lines.append("## New probes (first run — added to baseline)")
        lines.append("")
        for entry in diff["appeared"]:
            lines.append(f"- {entry}")
        lines.append("")

    if not any(diff.values()):
        lines.append("No changes since the last run. Catalog probes are stable.")
        lines.append("")

    path.write_text("\n".join(lines))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-baseline-update", action="store_true",
        help="don't overwrite baseline_probes.json (dry-run mode)",
    )
    args = parser.parse_args(argv)

    print(f"[probe-regression] running {len(ENVS)} envs...")
    current = collect_all_envs()
    print(f"[probe-regression] collected {len(current)} probe verdicts")

    if BASELINE.exists():
        baseline = json.loads(BASELINE.read_text())
    else:
        print("[probe-regression] no baseline yet — first run, will seed.")
        baseline = {}

    diff = diff_against_baseline(current, baseline)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = write_report(stamp, diff, current)
    print(f"[probe-regression] report: {report_path}")

    print(
        f"[probe-regression] newly_failing={len(diff['newly_failing'])} "
        f"newly_passing={len(diff['newly_passing'])} "
        f"newly_skipped={len(diff['newly_skipped'])} "
        f"appeared={len(diff['appeared'])}"
    )

    if not args.no_baseline_update:
        BASELINE.write_text(json.dumps(current, indent=2, sort_keys=True))
        print(f"[probe-regression] baseline updated -> {BASELINE}")

    # Exit non-zero if there are newly failing probes — useful for
    # surfacing in CI / cron mail notification.
    return 1 if diff["newly_failing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
