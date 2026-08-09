"""Tier-2: every SPARTA generator the catalog advertises actually runs.

For each physics row and each template variant, generate the deck through the
backend, stage the data files it references, and run it with the installed
binary. A generator that raises, that references a data file the stager cannot
resolve, or that exits non-zero is a catalog claim the project cannot defend.

Also checks that the payload served by get_knowledge() carries no absolute host
path — knowledge handed to a model must not describe the machine it was
recorded on.

Mutation control. This fixture is a GATE: its two expectations assert the
ABSENCE of a defect, so the control is the mirror of the usual one — instead of
removing a pathology it INJECTS the two defects the gate exists to catch, and
the gate must fire. T2_MUTATE=1 appends one absolute host path of exactly the
kind the old 'installed_build' block hard-coded to the _general blob the leak
scan reads, and masks the 'Signal:' marker on the first pitfall the scan sees.
`knowledge_has_no_host_paths` and `all_pitfalls_carry_a_signal` both go False.
Neither the served payload nor any file on disk is modified: the injection is
local to the two strings this script scans.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import BINARY  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from core.registry import get_backend, load_all_backends  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=host_path_and_missing_signal_injected_into_the_scanned_"
          "payload_strings")

load_all_backends()
backend = get_backend("sparta")

# ---- payload hygiene (runs with or without the binary) --------------------
HOST_PATH = re.compile(r"(?:/home/[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+"
                       r"|/opt/[A-Za-z0-9_.-]+/(?:lib|bin))")
leaks = []
for p in backend.supported_physics():
    blob = json.dumps(backend.get_knowledge(p.name), default=str)
    for m in HOST_PATH.findall(blob):
        leaks.append((p.name, m))
blob_general = json.dumps(backend.get_knowledge("_general"), default=str)
if MUTATE:
    # Inject the defect the scan exists to catch, into the scanned STRING only.
    blob_general += ' {"binary": "/home/someuser/sparta/src/spa_serial"}'
for m in HOST_PATH.findall(blob_general):
    leaks.append(("_general", m))
print(f"absolute_host_paths_in_knowledge={sorted(set(leaks))}")
print(f"knowledge_has_no_host_paths={not leaks}")

sizes = {}
for p in backend.supported_physics():
    sizes[p.name] = len(json.dumps(backend.get_knowledge(p.name), default=str))
print(f"knowledge_payload_max_bytes={max(sizes.values())}")
print(f"knowledge_payload_mean_bytes={sum(sizes.values()) // len(sizes)}")

# every pitfall must carry a Signal: clause
missing_signal = []
_masked = False
for p in backend.supported_physics():
    for i, pit in enumerate(backend.get_knowledge(p.name).get("pitfalls", [])):
        text = str(pit)
        if MUTATE and not _masked:
            # Mask the marker on the FIRST pitfall scanned, again in the
            # scanned string only, so the scan has exactly one defect to find.
            text = text.replace("Signal:", "S1gnal:")
            _masked = True
        if "Signal:" not in text:
            missing_signal.append((p.name, i))
print(f"pitfalls_without_signal={missing_signal}")
print(f"all_pitfalls_carry_a_signal={not missing_signal}")

if BINARY is None:
    print("SKIP: no SPARTA binary found — generator execution not attempted")
    ok = (not leaks) and (not missing_signal)
    if not ok:
        print("FAIL: payload hygiene checks failed")
        sys.exit(1)
    sys.exit(0)

from backends.sparta.backend import stage_deck_data_files  # noqa: E402

failed = []
ran = []
for p in backend.supported_physics():
    for variant in p.template_variants:
        tag = f"{p.name}/{variant}"
        try:
            deck = backend.generate_input(p.name, variant, {})
        except Exception as exc:                       # noqa: BLE001
            print(f"{tag}: GENERATOR RAISED {exc!r}")
            failed.append(tag)
            continue
        problems = backend.validate_input(deck)
        work = Path(tempfile.mkdtemp(prefix="spa_gen_"))
        (work / "in.case").write_text(deck)
        staged = stage_deck_data_files(deck, work, binary=BINARY)
        try:
            proc = subprocess.run([BINARY, "-in", "in.case"], cwd=str(work),
                                  capture_output=True, text=True, timeout=1200)
            rc = proc.returncode
            err = [l for l in (proc.stdout + proc.stderr).splitlines()
                   if l.startswith("ERROR")]
        except subprocess.TimeoutExpired:
            rc, err = -99, ["timeout"]
        shutil.rmtree(work, ignore_errors=True)
        print(f"{tag}: rc={rc} validate={problems} "
              f"missing_data={staged['missing']} err={err[:1]}")
        ran.append(tag)
        if rc != 0 or staged["missing"] or problems:
            failed.append(tag)

print(f"generators_executed={len(ran)}")
print(f"generators_failed={failed}")
print(f"all_generators_execute={not failed}")

ok = (not failed) and (not leaks) and (not missing_signal)
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
