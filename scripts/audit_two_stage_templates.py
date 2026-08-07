#!/usr/bin/env python3
"""Find templates whose "it executes" evidence stops one step short of the solver.

THE DEFECT
----------
`catalog_template_executes` generates a template, runs it, and records the exit
code. For most backends that is genuinely end to end: the template IS the script
that builds the problem and solves it.

For some it is not. The Kratos DEM template is a FILE WRITER — running it emits
`input.py`, `ProjectParameters.json` and an `.mdpa` mesh, then exits 0. The
emitted `input.py` is the thing that calls Kratos, and it was never run. The
gate has been recording `kratos::dem::2d_rc=0` for the half that cannot fail;
run the emitted file and it dies at `entry string : strategy`. The DEM generator
has never produced a working deck, and a green gate said otherwise.

Verified here rather than taken on report: the template's own text contains
`input.py`, `ProjectParameters`, `open(` and `write(`.

WHAT THIS DOES
--------------
Runs each template in an isolated working directory, then looks at what it left
behind. If it wrote something runnable, that artefact is run too, and the
two exit codes are reported separately. A template that exits 0 while the file
it wrote exits non-zero is the shape being hunted.

WHAT IT WILL NOT CLAIM
----------------------
A backend with no usable interpreter is SKIPPED WITH A REASON, never passed.
The Kratos wheel in the repo venv has 3 of ~40 applications and does not import
at all (GLIBC); the 28-application source build at /mnt/kratos-tier2/kv does.
Pointing this at the wrong one produces a confident wrong answer in either
direction, so the interpreter actually used is recorded in every row.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Artefacts that are themselves runnable, in the order we would try them.
_RUNNABLE = ("input.py", "run.py", "main.py", "MainKratos.py")

INTERPRETERS = {
    # The 28-application build. The repo venv's 3-app wheel cannot import
    # Kratos at all, so using it would report every Kratos row as broken.
    "kratos": os.environ.get(
        "KRATOS_PYTHON", "/mnt/kratos-tier2/kv/bin/python"),
    "skfem": os.environ.get("SKFEM_PYTHON", sys.executable),
    "ngsolve": os.environ.get("NGSOLVE_PYTHON", sys.executable),
    "fenics": os.environ.get(
        "FENICS_PYTHON",
        str(Path.home() / "miniconda3" / "envs" / "fenics" / "bin" / "python")),
    "dune": os.environ.get("DUNE_PYTHON", ""),
}


def _interp(backend: str) -> str:
    p = INTERPRETERS.get(backend, "")
    return p if p and Path(p).is_file() else ""


def check(backend: str, physics: str, variant: str, timeout: int = 120) -> dict:
    row = {"backend": backend, "physics": physics, "variant": variant}
    py = _interp(backend)
    if not py:
        row["status"] = "SKIPPED"
        row["reason"] = (f"no interpreter for {backend}; set "
                         f"{backend.upper()}_PYTHON. Not a pass.")
        return row
    row["interpreter"] = py

    gen = (
        "import sys;sys.path.insert(0,%r)\n"
        "from core.registry import load_all_backends,get_backend\n"
        "load_all_backends()\n"
        "print(get_backend(%r).generate_input(%r,%r,{}))\n"
        % (str(REPO / "src"), backend, physics, variant))
    try:
        g = subprocess.run([py, "-c", gen], capture_output=True, text=True,
                           timeout=timeout, stdin=subprocess.DEVNULL)
    except (subprocess.TimeoutExpired, OSError) as exc:
        row["status"] = "ERROR"
        row["reason"] = f"could not generate: {exc}"[:160]
        return row
    if g.returncode != 0:
        row["status"] = "GENERATE_FAILED"
        row["reason"] = (g.stderr or "")[-300:]
        return row
    template = g.stdout

    with tempfile.TemporaryDirectory(prefix="tmpl_") as td:
        script = Path(td) / "template.py"
        script.write_text(template)
        before = {p.name for p in Path(td).iterdir()}
        try:
            r = subprocess.run([py, str(script)], capture_output=True,
                               text=True, timeout=timeout, cwd=td,
                               stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            row["status"] = "TIMEOUT"
            return row
        row["template_rc"] = r.returncode
        made = sorted({p.name for p in Path(td).iterdir()} - before)
        row["emitted"] = made

        runnable = [m for m in made if m in _RUNNABLE]
        if not runnable:
            # Single-stage: the template did the work itself. Its exit code is
            # the whole answer, which is what the existing gate assumes.
            row["status"] = "SINGLE_STAGE"
            return row

        # Two-stage: the template wrote a script. THAT is the deck the user
        # would run, and its exit code is the one that matters.
        row["status"] = "TWO_STAGE"
        art = runnable[0]
        row["ran"] = art
        try:
            r2 = subprocess.run([py, art], capture_output=True, text=True,
                                timeout=timeout, cwd=td,
                                stdin=subprocess.DEVNULL)
            row["emitted_rc"] = r2.returncode
            row["emitted_tail"] = ((r2.stdout or "") + (r2.stderr or ""))[-400:]
        except subprocess.TimeoutExpired:
            row["emitted_rc"] = 124
            row["emitted_tail"] = "TIMEOUT"
        # The finding: green template, red artefact.
        row["gate_is_shallow"] = (row["template_rc"] == 0
                                  and row["emitted_rc"] != 0)
        return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("targets", nargs="*",
                    help="backend:physics:variant (default: a known set)")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    targets = args.targets or ["kratos:dem:2d"]
    rows = []
    for t in targets:
        parts = t.split(":")
        if len(parts) != 3:
            print(f"  skipping malformed target {t!r}")
            continue
        row = check(*parts, timeout=args.timeout)
        rows.append(row)
        head = f"{t:<28} {row['status']:<16}"
        if row["status"] == "TWO_STAGE":
            verdict = ("SHALLOW GATE — template rc=0 but the file it wrote "
                       "fails" if row.get("gate_is_shallow") else "both run")
            print(f"  {head} template_rc={row['template_rc']} "
                  f"{row['ran']}_rc={row.get('emitted_rc')}  {verdict}")
            if row.get("gate_is_shallow"):
                print(f"        emitted: {', '.join(row['emitted'])}")
                for ln in (row.get("emitted_tail") or "").strip().splitlines()[-3:]:
                    print(f"        | {ln[:110]}")
        else:
            print(f"  {head} {row.get('reason', '') or row.get('emitted', '')}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
