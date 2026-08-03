"""Tier-2: SPARTA documentation-page names are not input-script commands.

sparta_knowledge.json['commands'] is the SPARTA DOC INDEX — one entry per doc
page. 55 of its 121 keys are filenames, not things you may type in a deck:
every compute_* and fix_* page, plus dump_image, surf_react_adsorb and suffix.
Typing the underscore form is a hard error from the parser.

This fixture (a) feeds each form to the installed binary and checks the error,
(b) checks the correct 'compute <ID> <style> ...' form is accepted, and
(c) checks the backend's own validate_input now rejects the underscore form —
before 2026-08-03 it validated against the doc index and waved them through.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# .../scripts/tier2_fixtures/sparta/<id>/source.py -> repo root is parents[4].
# Insert THIS checkout's src at the very front so an installed copy of the
# package elsewhere on the machine cannot shadow it.
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

CANDIDATES = [
    os.environ.get("SPARTA_BINARY"),
    shutil.which("spa_serial"),
    shutil.which("spa_mpi"),
    "/home/alexander/Schreibtisch/sparta/src/spa_serial",
    str(Path.home() / "sparta" / "src" / "spa_serial"),
]
BINARY = next((c for c in CANDIDATES if c and Path(c).is_file()), None)

PREAMBLE = ("seed 12345\ndimension 2\nboundary p p p\n"
            "create_box 0 1 0 1 -0.5 0.5\ncreate_grid 2 2 1\n")
DOC_PAGE_FORMS = ["compute_grid all all n", "fix_ave_surf all 1 1 1",
                  "dump_image all 100 img.ppm", "surf_react_adsorb x y", "suffix kk"]
REAL_FORMS = ["compute cg grid all all n", "balance_grid rcb cell"]


def probe(line: str) -> tuple[int, str]:
    work = Path(tempfile.mkdtemp(prefix="sparta_cmd_"))
    (work / "in.case").write_text(PREAMBLE + line + "\n")
    proc = subprocess.run([BINARY, "-in", "in.case"], cwd=str(work),
                          capture_output=True, text=True, timeout=120)
    txt = proc.stdout + proc.stderr
    shutil.rmtree(work, ignore_errors=True)
    err = next((l for l in txt.splitlines() if l.startswith("ERROR")), "")
    return proc.returncode, err


if BINARY is None:
    print("SKIP: no SPARTA binary found (set SPARTA_BINARY)")
else:
    rejected, accepted = [], []
    for line in DOC_PAGE_FORMS:
        rc, err = probe(line)
        if rc != 0 and "Unknown command" in err:
            rejected.append(line.split()[0])
        else:
            print(f"UNEXPECTED: {line!r} rc={rc} err={err!r}")
    for line in REAL_FORMS:
        rc, err = probe(line)
        if rc == 0:
            accepted.append(line.split()[0])
        else:
            print(f"UNEXPECTED: {line!r} rc={rc} err={err!r}")
    print(f"docpage_forms_rejected_by_binary={sorted(rejected)}")
    print(f"all_docpage_forms_rejected={len(rejected) == len(DOC_PAGE_FORMS)}")
    print(f"real_forms_accepted={sorted(accepted)}")
    print(f"all_real_forms_accepted={len(accepted) == len(REAL_FORMS)}")
    if len(rejected) != len(DOC_PAGE_FORMS) or len(accepted) != len(REAL_FORMS):
        print("FAIL: binary probe expectations not met")
        sys.exit(1)

# ---- backend-side gate (runs with or without the binary) -------------------
from core.registry import get_backend, load_all_backends  # noqa: E402

load_all_backends()
backend = get_backend("sparta")
kn = backend.get_knowledge("rarefied_flow")
surface = kn.get("command_surface", {})
print(f"n_true_commands={surface.get('n_true_commands')}")
print(f"n_docpage_only={len(surface.get('not_commands_doc_page_names_only') or [])}")
print(f"installed_version={kn.get('installed_build', {}).get('version')}")

good = PREAMBLE + "run 10\n"
bad = PREAMBLE + "compute_grid all all n\nrun 10\n"
good_errs = backend.validate_input(good)
bad_errs = backend.validate_input(bad)
print(f"validate_accepts_good_deck={good_errs == []}")
print(f"validate_rejects_docpage_form={bool(bad_errs)}")

ok = (surface.get("n_true_commands") == 66
      and len(surface.get("not_commands_doc_page_names_only") or []) == 55
      and good_errs == [] and bool(bad_errs))
if not ok:
    print("FAIL: backend gate expectations not met")
    sys.exit(1)
