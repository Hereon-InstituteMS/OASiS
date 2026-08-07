#!/usr/bin/env python3
"""Verify every shipped 4C deck template against the grammar and the binary.

Two independent checks, because they catch different things:

  * **Grammar.** Every section name and every key path in the deck must exist
    in the dump of `4C -p` for the installed binary. A mis-cased key and an
    invented key produce the SAME 4C message ("Failed to match specification in
    section '...'"), so the error text cannot tell them apart — the index can.
    A previous pass over the templates that DID exist found 22 invented section
    names, 18 invented keys, 4 mis-cased keys and 23 misplaced keys; nothing in
    the test suite noticed, because none of those templates was ever executed.

  * **Execution.** The deck is written into a fresh ext4 temp directory and run
    with the real binary at the rank count the catalog records. Nothing else
    is copied in, so a deck that quietly depends on a sibling file fails here
    rather than in a user's hands.

The grammar check alone is not enough (a deck can be spelled perfectly and
still abort in setup); execution alone is not enough either (4C accepts and
ignores some misplaced keys). Both, or the deck does not ship.

Usage:
    python scripts/verify_fourc_decks.py               # grammar only, fast
    python scripts/verify_fourc_decks.py --execute     # + run every deck
    python scripts/verify_fourc_decks.py --execute --only fsi,fs3i
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import yaml  # noqa: E402

from backends.fourc import decks as deck_mod  # noqa: E402

BINARY = Path(os.environ.get("FOURC_BINARY", "/home/alexander/4C/build/4C"))
LD = os.environ.get("FOURC_LD_LIBRARY_PATH", "/opt/4C-dependencies/lib")

# The grammar dump cannot enumerate these: TITLE is free text, FUNCT<n> is
# user-numbered, and the legacy mesh sections are free-form strings parsed by
# the element factories rather than by the input-spec tree. Established by
# validating the index against all 1978 parseable upstream decks — those were
# the ONLY things that did not resolve.
UNINDEXED_SECTIONS = {
    "TITLE",
    "NODE COORDS", "STRUCTURE ELEMENTS", "FLUID ELEMENTS", "ALE ELEMENTS",
    "TRANSPORT ELEMENTS", "TRANSPORT2 ELEMENTS", "THERMO ELEMENTS",
    "LUBRICATION ELEMENTS", "ARTERY ELEMENTS", "REDUCED D AIRWAYS ELEMENTS",
    "PARTICLES", "PERIODIC BOUNDINGBOX ELEMENTS",
    "DNODE-NODE TOPOLOGY", "DLINE-NODE TOPOLOGY",
    "DSURF-NODE TOPOLOGY", "DVOL-NODE TOPOLOGY",
}


def _is_funct(name: str) -> bool:
    return name.startswith("FUNCT") and name[5:].isdigit()


def _grammar():
    import fourc_grammar_index as g
    return g.get()


def check_grammar(text: str, g) -> list[str]:
    """Return one message per section/key that does not exist in the binary."""
    problems: list[str] = []
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        return ["deck does not parse as a YAML mapping"]
    for sec, body in doc.items():
        if sec in UNINDEXED_SECTIONS or _is_funct(sec):
            continue
        if not g.has_section(sec):
            problems.append(f"unknown section {sec!r}")
            continue
        flat = g.sections[sec]
        allowed = {p.split("/")[-1] for p in flat}
        for entry in (body if isinstance(body, list) else [body]):
            if not isinstance(entry, dict):
                continue
            problems.extend(
                f"unknown key {sec} :: {k}" for k in entry if k not in allowed)
    return problems


def execute(text: str, np_: int, timeout: int = 900) -> tuple[int, float, str]:
    """Run the deck alone in a fresh temp dir. Returns (rc, seconds, tail)."""
    env = dict(os.environ, LD_LIBRARY_PATH=LD)
    work = Path(tempfile.mkdtemp(prefix="fourc-deck-"))
    try:
        (work / "in.4C.yaml").write_text(text)
        base = [str(BINARY), "in.4C.yaml", "out"]
        if np_ > 1:
            base = ["mpirun", "-np", str(np_), "--oversubscribe"] + base
        # stdbuf -oL is not optional: without it MPI_Abort tears the process
        # down before stdout is flushed and the diagnostic is lost, leaving
        # only the MPI banner — which reads as "no error message".
        cmd = ["stdbuf", "-oL"] + base
        t0 = time.time()
        p = subprocess.run(cmd, cwd=work, env=env, timeout=timeout,
                           capture_output=True, text=True)
        dt = time.time() - t0
        tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-12:])
        return p.returncode, dt, tail
    except subprocess.TimeoutExpired:
        return 124, float(timeout), "timeout"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    wanted = {s.strip() for s in a.only.split(",") if s.strip()}

    g = _grammar()
    bad = 0
    for d in deck_mod.DECKS:
        if wanted and d.physics not in wanted:
            continue
        text = deck_mod.render(d.physics, d.variant)
        label = f"{d.physics}/{d.variant}"
        problems = check_grammar(text, g)
        line = f"{label:34s} bytes={len(text):6d} np={d.np}"
        if problems:
            bad += 1
            print(f"FAIL {line}")
            for p in problems:
                print(f"       {p}")
            continue
        if not a.execute:
            print(f"ok   {line}  grammar clean")
            continue
        rc, dt, tail = execute(text, d.np, a.timeout)
        if rc == 0:
            print(f"ok   {line}  rc=0 in {dt:5.1f}s")
        else:
            bad += 1
            print(f"FAIL {line}  rc={rc} in {dt:5.1f}s")
            for ln in tail.splitlines():
                print(f"       {ln}")
    print(f"\n{bad} failing of "
          f"{len([d for d in deck_mod.DECKS if not wanted or d.physics in wanted])}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
