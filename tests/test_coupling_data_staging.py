"""Data-file staging for the COUPLED path.

The coupled path used to do no data staging at all: SpartaBackend.run()
staged species/surf files for single runs (#42), but couple()/run_coupling
executed raw participant commands in fresh work_dirs, so SPARTA died with
'Cannot open species file ar.species' (../particle.cpp:711) unless every
file had been hand-copied first.

Fixes under test:
  1. Participant.data_files — the driver stages listed files into work_dir
     before the iteration loop, and a missing file is a LOUD setup error.
  2. stage_deck_data_files (sparta backend) — resolves deck references
     (species / collide vss / read_surf / read_grid / react) with the
     task's own data dirs taking precedence over the SPARTA distribution.
  3. _stage_sparta_deck_refs (couple() tool layer) — scans participant
     work_dirs for SPARTA decks and auto-stages their references.
  4. Live (skipped without a SPARTA binary): a real coupled DSMC run
     through run_coupling with auto-staged data files converges.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.coupling_driver import Participant, run_coupling
from backends.sparta.backend import (
    stage_deck_data_files, _find_sparta_binary, _sparta_data_dirs,
)
from tools.coupling import _stage_sparta_deck_refs

_ECHO_EXPORT = (
    'import json\n'
    'json.dump({"field_name":"x","n_points":1,"coordinates":[[0.0]],'
    '"values":[1.0]},open("exports.json","w"))\n'
)


# ── 1. driver-level data_files staging ─────────────────────────────────

def test_driver_stages_data_files(tmp_path):
    """Files listed in Participant.data_files must be in work_dir before
    the participant command first runs."""
    src = tmp_path / "srcdata"
    src.mkdir()
    (src / "ar.species").write_text("# species data\n")
    (src / "circle.surf").write_text("# surf data\n")

    wd_a = tmp_path / "A"
    wd_a.mkdir()
    (wd_a / "run.py").write_text(
        'import json, os\n'
        'assert os.path.exists("ar.species"), "ar.species not staged"\n'
        'assert os.path.exists("circle.surf"), "circle.surf not staged"\n'
        + _ECHO_EXPORT)
    wd_b = tmp_path / "B"
    wd_b.mkdir()
    (wd_b / "run.py").write_text(_ECHO_EXPORT)

    pa = Participant("A", [sys.executable, "run.py"], wd_a,
                     imports_from=["B"],
                     data_files=[str(src / "ar.species"),
                                 str(src / "circle.surf")])
    pb = Participant("B", [sys.executable, "run.py"], wd_b,
                     imports_from=["A"])
    r = run_coupling([pa, pb], max_iter=5, tol=1e-6)
    assert r.converged, f"coupling failed: {r.error}"
    assert (wd_a / "ar.species").read_text() == "# species data\n"


def test_driver_missing_data_file_is_loud_failure(tmp_path):
    """A listed-but-absent data file must fail the coupling BEFORE any
    iteration, naming the file — not die mid-run with 'Cannot open'."""
    wd = tmp_path / "A"
    wd.mkdir()
    (wd / "run.py").write_text(_ECHO_EXPORT)
    pa = Participant("A", [sys.executable, "run.py"], wd,
                     data_files=[str(tmp_path / "does_not_exist.surf")])
    pb = Participant("B", [sys.executable, "run.py"], wd)
    r = run_coupling([pa, pb], max_iter=5)
    assert not r.converged
    assert r.iterations == 0
    assert "does_not_exist.surf" in (r.error or "")


# ── 2. sparta deck-reference resolution ────────────────────────────────

_DECK = """# minimal DSMC deck
seed        12345
species     ar.species Ar
collide     vss gas ar.vss
read_surf   circle.surf clip
run         10
"""


def test_stage_deck_refs_from_extra_dirs(tmp_path):
    data = tmp_path / "taskdata"
    data.mkdir()
    for n in ("ar.species", "ar.vss", "circle.surf"):
        (data / n).write_text(f"task {n}\n")
    wd = tmp_path / "wd"
    wd.mkdir()
    res = stage_deck_data_files(_DECK, wd, binary=None,
                                extra_dirs=[str(data)])
    assert res["missing"] == []
    assert sorted(res["staged"]) == ["ar.species", "ar.vss", "circle.surf"]
    assert (wd / "circle.surf").read_text() == "task circle.surf\n"


def test_task_data_dir_beats_distribution(tmp_path, monkeypatch):
    """A task-specific circle.surf must win over a distribution file of
    the same name — they are different geometries."""
    dist = tmp_path / "dist"
    (dist / "data").mkdir(parents=True)
    (dist / "data" / "circle.surf").write_text("distribution circle\n")
    task = tmp_path / "task"
    task.mkdir()
    (task / "circle.surf").write_text("task circle\n")
    monkeypatch.setenv("SPARTA_ROOT", str(dist))
    wd = tmp_path / "wd"
    wd.mkdir()
    res = stage_deck_data_files("read_surf circle.surf\nrun 1\n", wd,
                                binary=None, extra_dirs=[str(task)])
    assert (wd / "circle.surf").read_text() == "task circle\n"
    assert res["staged"]["circle.surf"] == str(task / "circle.surf")


def test_sparta_data_dir_env_is_searched(tmp_path, monkeypatch):
    envdir = tmp_path / "envdata"
    envdir.mkdir()
    (envdir / "ar.species").write_text("env species\n")
    monkeypatch.setenv("SPARTA_DATA_DIR", str(envdir))
    dirs = _sparta_data_dirs(binary=None)
    assert envdir in dirs
    wd = tmp_path / "wd"
    wd.mkdir()
    res = stage_deck_data_files("species ar.species Ar\nrun 1\n", wd,
                                binary=None)
    assert res["staged"].get("ar.species") == str(envdir / "ar.species")


def test_existing_workdir_file_never_overwritten(tmp_path):
    data = tmp_path / "d"
    data.mkdir()
    (data / "ar.species").write_text("source version\n")
    wd = tmp_path / "wd"
    wd.mkdir()
    (wd / "ar.species").write_text("workdir version\n")
    stage_deck_data_files("species ar.species Ar\n", wd, binary=None,
                          extra_dirs=[str(data)])
    assert (wd / "ar.species").read_text() == "workdir version\n"


# ── 3. couple()-layer auto-staging ─────────────────────────────────────

def test_couple_layer_scans_decks_and_stages(tmp_path):
    data = tmp_path / "taskdata"
    data.mkdir()
    for n in ("ar.species", "ar.vss", "circle.surf"):
        (data / n).write_text(f"task {n}\n")
    wd = tmp_path / "gas"
    wd.mkdir()
    (wd / "in.gas").write_text(_DECK)
    notes = _stage_sparta_deck_refs(
        wd, {"name": "gas", "data_dir": str(data)})
    for n in ("ar.species", "ar.vss", "circle.surf"):
        assert (wd / n).exists(), f"{n} not auto-staged"
    assert any("staged" in n for n in notes)


def test_couple_layer_reports_missing_loudly(tmp_path):
    wd = tmp_path / "gas"
    wd.mkdir()
    (wd / "in.gas").write_text("read_surf no_such_geometry.surf\nrun 1\n")
    notes = _stage_sparta_deck_refs(wd, {"name": "gas"})
    assert any("MISSING" in n and "no_such_geometry.surf" in n
               for n in notes)


def test_couple_layer_data_files_parents_searched(tmp_path):
    """Passing data_files should also make their parent dir a search dir
    for OTHER references in the same deck."""
    data = tmp_path / "d"
    data.mkdir()
    (data / "ar.species").write_text("s\n")
    (data / "ar.vss").write_text("v\n")
    wd = tmp_path / "gas"
    wd.mkdir()
    (wd / "in.gas").write_text("species ar.species Ar\n"
                               "collide vss gas ar.vss\nrun 1\n")
    _stage_sparta_deck_refs(
        wd, {"name": "gas", "data_files": [str(data / "ar.species")]})
    assert (wd / "ar.vss").exists()


# ── 4. live coupled DSMC run (needs a SPARTA binary) ───────────────────

_SPARTA_BIN = _find_sparta_binary()
_HAVE_DIST = bool(_SPARTA_BIN) and any(
    (d / "data" / "air.species").is_file()
    for d in _sparta_data_dirs(_SPARTA_BIN))

_LIVE_DECK = """seed        12345
dimension   2
global      gridcut 0.0 comm/sort yes
boundary    p p p
create_box  0 1 0 1 -0.5 0.5
create_grid 10 10 1
global      nrho 1.0e20 fnum 1.0e17
species     air.species N2 O2
mixture     air N2 O2 vstream 0 0 0 temp 273.15
collide     vss air air.vss
create_particles air n 1000
timestep    7.0e-9
stats       50
run         100
"""

_LIVE_GAS_WRAPPER = """import json, subprocess
r = subprocess.run([{bin!r}, "-in", "in.gas"], capture_output=True, text=True)
out = r.stdout
assert "Cannot open" not in out, out[-1500:]
assert r.returncode == 0 and "ERROR" not in out, out[-1500:]
np_final = None
grab = False
for line in out.splitlines():
    t = line.split()
    if grab and len(t) == 3 and t[0].isdigit():
        np_final = float(t[2])
    if line.strip().startswith("Step"):
        grab = True
json.dump({{"field_name": "np", "n_points": 1, "coordinates": [[0.0]],
           "values": [np_final if np_final is not None else -1.0]}},
          open("exports.json", "w"))
"""


@pytest.mark.skipif(not _HAVE_DIST,
                    reason="SPARTA binary or distribution data not found")
def test_live_coupled_sparta_run_with_auto_staging(tmp_path):
    """End-to-end through the FIXED coupled path: the deck references
    air.species / air.vss which are NOT in the fresh work_dir; the
    couple()-layer auto-staging must resolve them (distribution search)
    and the coupled fixed-point loop must converge on a real DSMC run."""
    gas = tmp_path / "gas"
    gas.mkdir()
    (gas / "in.gas").write_text(_LIVE_DECK)
    (gas / "gas.py").write_text(_LIVE_GAS_WRAPPER.format(bin=_SPARTA_BIN))
    solid = tmp_path / "solid"
    solid.mkdir()
    (solid / "solid.py").write_text(
        'import json, os\n'
        'v = 0.0\n'
        'if os.path.exists("imports.json"):\n'
        '    imp = json.load(open("imports.json"))\n'
        '    if "gas" in imp and imp["gas"].get("n_points", 0) > 0:\n'
        '        v = float(imp["gas"]["values"][0])\n'
        'json.dump({"field_name": "resp", "n_points": 1,'
        '"coordinates": [[0.0]], "values": [273.15 + 1e-6 * v]},'
        'open("exports.json", "w"))\n')

    notes = _stage_sparta_deck_refs(gas, {"name": "gas"})
    assert (gas / "air.species").exists(), f"auto-staging failed: {notes}"
    assert (gas / "air.vss").exists(), f"auto-staging failed: {notes}"

    parts = [
        Participant("gas", [sys.executable, "gas.py"], gas,
                    imports_from=["solid"], timeout=600),
        Participant("solid", [sys.executable, "solid.py"], solid,
                    imports_from=["gas"], timeout=60),
    ]
    r = run_coupling(parts, max_iter=6, tol=1e-4)
    assert r.converged, f"live coupled DSMC run failed: {r.error}"
    # the DSMC deck seeds 1000 particles; the export is the final count
    assert r.exports["gas"]["values"][0] > 0
