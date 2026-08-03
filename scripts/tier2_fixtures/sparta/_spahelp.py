"""Shared helper for the SPARTA tier-2 fixtures.

Locates the installed binary and the distribution's data/ tree, writes a deck
into a scratch directory with the data files it needs, and runs it.

Every fixture that imports this SKIPS cleanly (exit 0, a line starting with
'SKIP:') when no SPARTA binary is present, so the suite stays green on a host
without SPARTA.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_CANDIDATES = [
    os.environ.get("SPARTA_BINARY"),
    shutil.which("spa_serial"),
    shutil.which("spa_mpi"),
    str(Path.home() / "sparta" / "src" / "spa_serial"),
    "/home/alexander/Schreibtisch/sparta/src/spa_serial",
]
BINARY = next((c for c in _CANDIDATES if c and Path(c).is_file()), None)


def _roots() -> list[Path]:
    out = []
    env = os.environ.get("SPARTA_ROOT")
    if env and Path(env).is_dir():
        out.append(Path(env))
    if BINARY:
        out.append(Path(BINARY).resolve().parent.parent)
    out.append(Path.home() / "sparta")
    return [p for p in out if p.is_dir()]


def find_data(name: str) -> Path | None:
    """Resolve a distribution data file by name, data/ before examples/."""
    for root in _roots():
        cand = root / "data" / name
        if cand.is_file():
            return cand
        hits = sorted((root / "examples").glob(f"*/{name}"))
        if hits:
            return hits[0]
    return None


def require(*names: str) -> dict[str, Path] | None:
    found = {}
    for n in names:
        p = find_data(n)
        if p is None:
            return None
        found[n] = p
    return found


def run(deck: str, data: dict[str, Path] | None = None, timeout: int = 600):
    """Run a deck; return (returncode, combined_output)."""
    work = Path(tempfile.mkdtemp(prefix="spa_fx_"))
    for name, src in (data or {}).items():
        shutil.copy(src, work / name)
    (work / "in.case").write_text(deck)
    try:
        proc = subprocess.run([BINARY, "-in", "in.case"], cwd=str(work),
                              capture_output=True, text=True, timeout=timeout)
        rc, txt = proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc = -99
        txt = (exc.stdout or b"").decode(errors="replace") if exc.stdout else ""
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return rc, txt


def errors(txt: str) -> list[str]:
    return [l for l in txt.splitlines() if l.startswith("ERROR")]


def stats_rows(txt: str) -> tuple[list[str], list[list[float]]]:
    """Parse the LAST stats table: returns (header tokens, numeric rows)."""
    header, rows = [], []
    for line in txt.splitlines():
        if line.startswith("Step "):
            header, rows = line.split(), []
            continue
        if header and re.match(r"^\s*[\d.eE+-]+(\s|$)", line):
            parts = line.split()
            if len(parts) == len(header):
                try:
                    rows.append([float(p) for p in parts])
                    continue
                except ValueError:
                    pass
            if rows:
                header_done = True  # noqa: F841
    return header, rows


def col(header: list[str], rows: list[list[float]], name: str) -> list[float]:
    idx = header.index(name)
    return [r[idx] for r in rows]


def skip_if_unavailable(*data_names: str):
    """Exit 0 with a SKIP line if the binary or any data file is missing."""
    if BINARY is None:
        print("SKIP: no SPARTA binary found (set SPARTA_BINARY)")
        sys.exit(0)
    data = require(*data_names) if data_names else {}
    if data is None:
        print("SKIP: SPARTA distribution data files not found (set SPARTA_ROOT)")
        sys.exit(0)
    return data
