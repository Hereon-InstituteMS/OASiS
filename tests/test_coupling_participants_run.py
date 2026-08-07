"""EXECUTE the shipped participant scripts — do not merely parse them.

`knowledge(topic='coupling', solver=...)` serves these files verbatim and tells
the agent to copy them and run them. A syntax check does not establish that,
and an audit found the claim "the artefact that gets executed in the test suite"
resting on nothing but `ast.parse`. So this module actually runs them.

Nothing here pins a measured number. The assertions are structural (a file was
written, the required keys are present, the values are finite, the interface is
non-empty) plus one cross-side CONSISTENCY check: the two participants must
agree with EACH OTHER at the shared interface once the coupling has converged.
That is a property of a correct coupling, not a result read off a run.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PART_DIR = REPO / "data" / "coupling_participants"

# The complementary side, exactly as the served payload prescribes it.
RIGHT_EDITS = {
    "SIDE": '"neumann"', "PARTNER": '"left"', "X0, X1": "0.6, 1.1",
    "Y0, Y1": "0.0, 0.4", "IFACE_X": "0.6", "K": "1.5", "T_OUTER": "300.0",
    "NX, NY": "18, 12", "T_INIT": "310.0", "Q_INIT": "0.0",
}


def _edit(text: str, edits: dict) -> str:
    import re
    for k, v in edits.items():
        pat = re.compile(rf"^({re.escape(k)}\s*=\s*)(.*?)(\s*(?:#.*)?)$", re.M)
        assert pat.search(text), f"edit key {k!r} not found in the shipped script"
        text = pat.sub(lambda m: m.group(1) + v + m.group(3), text, count=1)
    return text


def _interpreter(backend: str):
    """Whatever this install actually uses for that backend, or None."""
    if backend == "fenics":
        from backends.fenics.backend import _find_fenics_python
        p = _find_fenics_python()
        return str(p) if p else None
    if backend in ("skfem", "ngsolve", "fourc", "dealii", "febio", "kratos"):
        return sys.executable                    # the wrapper runs here
    if backend == "dune":
        from backends.dune.backend import _find_dune_python
        return _find_dune_python()
    return None


def _fourc_edits() -> dict:
    from backends.fourc.backend import _find_fourc_binary
    b = _find_fourc_binary()
    if not b:
        return {}
    e = {"FOURC_BIN": json.dumps(str(b))}
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if ld:
        e["FOURC_LD"] = json.dumps(ld.split(os.pathsep)[0])
    return e


def _available(name: str) -> bool:
    from core.registry import get_backend, load_all_backends
    load_all_backends()
    b = get_backend(name)
    return bool(b) and b.check_availability()[0].value == "available"


def _shipped(name: str) -> Path:
    return PART_DIR / f"participant_{name}.py"


def _prepare(tmp: Path, backend: str, side: str) -> Path:
    """Write one participant, edited for `side`, into its own work dir."""
    wd = tmp / f"{backend}_{side}"
    wd.mkdir(parents=True, exist_ok=True)
    text = _shipped(backend).read_text()
    edits = dict(_fourc_edits()) if backend == "fourc" else {}
    if backend == "dealii":
        exe = _dealii_exe()
        if exe is None:
            pytest.skip("no deal.II build environment on this host")
        edits["DEALII_EXE"] = json.dumps(str(exe))
    if side == "right":
        edits.update(RIGHT_EDITS)
    if edits:
        text = _edit(text, edits)
    dst = wd / f"participant_{side}.py"
    dst.write_text(text)
    return dst


def _check_export(wd: Path):
    ep = wd / "exports.json"
    assert ep.is_file(), f"no exports.json in {wd}"
    d = json.loads(ep.read_text())
    for key in ("field_name", "coordinates", "values"):
        assert key in d, f"exports.json is missing the required key {key!r}"
    assert len(d["coordinates"]) > 0, "empty interface: nothing is shared"
    assert len(d["values"]) == len(d["coordinates"])
    import math
    assert all(math.isfinite(v) for v in d["values"])
    if "normal_fluxes" in d:
        assert len(d["normal_fluxes"]) == len(d["coordinates"])
        assert all(math.isfinite(q) for q in d["normal_fluxes"])
    return d


BACKENDS_WITH_SCRIPTS = sorted(
    p.name[len("participant_"):-3] for p in PART_DIR.glob("participant_*.py"))

# The backends whose shipped script implements the SAME canonical conduction
# problem, so the documented right-side edit block applies verbatim to them.
# FEBio (elasticity — FEBio 4 has no heat module), SPARTA (DSMC on a surface
# mesh) and Kratos (Dirichlet-only script) solve different problems and are
# covered by the contract tests, not by this parametrisation.
DN_HEAT_BACKENDS = [b for b in ("fenics", "fourc", "ngsolve", "skfem", "dune",
                                "dealii") if b in BACKENDS_WITH_SCRIPTS]


def _dealii_exe():
    """Build the shipped deal.II solver once; None if this host cannot."""
    import shutil as _sh
    if not _sh.which("cmake"):
        return None
    cand = [os.environ.get("DEAL_II_DIR", "")] + [
        str(Path.home() / "dealii" / "build"), str(Path.home() / "dealii"),
        "/usr/local", "/usr"]
    root = next((c for c in cand if c and
                 (Path(c) / "lib/cmake/deal.II/deal.IIConfig.cmake").is_file()), None)
    if not root:
        return None
    build = Path(os.environ.get("TMPDIR", "/tmp")) / "oasis_dealii_participant_build"
    exe = build / "heat_iface_dealii"
    if exe.is_file():
        return exe
    build.mkdir(parents=True, exist_ok=True)
    for cmd in (["cmake", "-S", str(PART_DIR), "-B", str(build),
                 f"-DDEAL_II_DIR={root}", "-DCMAKE_BUILD_TYPE=Release"],
                ["make", "-C", str(build), "-j4"]):
        if subprocess.run(cmd, capture_output=True, text=True,
                          timeout=1800).returncode != 0:
            return None
    return exe if exe.is_file() else None


@pytest.mark.parametrize("backend", DN_HEAT_BACKENDS)
@pytest.mark.parametrize("side", ["left", "right"])
def test_shipped_participant_runs_standalone(tmp_path, backend, side):
    """Iteration-1 path: no imports.json, must still write a usable export."""
    if not _available(backend):
        pytest.skip(f"{backend} not available on this install")
    py = _interpreter(backend)
    if not py:
        pytest.skip(f"no interpreter resolved for {backend}")
    script = _prepare(tmp_path, backend, side)
    r = subprocess.run([py, script.name], cwd=str(script.parent),
                       capture_output=True, text=True, timeout=900)
    assert (script.parent / "exports.json").is_file(), (
        f"{backend} ({side}) wrote no exports.json, rc={r.returncode}\n"
        f"stdout: {r.stdout[-1500:]}\nstderr: {r.stderr[-1500:]}")
    _check_export(script.parent)


@pytest.mark.parametrize("pair", [("fenics", "fourc"), ("fourc", "fenics")])
def test_shipped_participants_couple_across_two_codes(tmp_path, pair):
    """A REAL cross-code coupling through OASiS's own driver, both ways round.

    Asserts convergence and that the two independent codes agree with each
    other at the shared interface — a consistency property, not a stored value.
    """
    left, right = pair
    for b in pair:
        if not _available(b):
            pytest.skip(f"{b} not available on this install")
        if not _interpreter(b):
            pytest.skip(f"no interpreter resolved for {b}")
    if not _shipped(left).is_file() or not _shipped(right).is_file():
        pytest.skip("participant script missing")

    from core.coupling_driver import Participant, run_coupling
    from core.quality_checks import check_interface_balance

    sl = _prepare(tmp_path, left, "left")
    sr = _prepare(tmp_path, right, "right")
    parts = [
        Participant("left", [_interpreter(left), sl.name], sl.parent,
                    imports_from=["right"], timeout=900),
        Participant("right", [_interpreter(right), sr.name], sr.parent,
                    imports_from=["left"], timeout=900),
    ]
    res = run_coupling(parts, max_iter=90, tol=1e-7,
                       accelerator="constant", theta0=0.7)
    assert res.converged, f"{left}->{right} coupling failed: {res.error}"

    ex = res.exports
    tl = sum(ex["left"]["values"]) / len(ex["left"]["values"])
    tr = sum(ex["right"]["values"]) / len(ex["right"]["values"])
    span = 320.0 - 300.0                       # the placeholder problem's range
    assert abs(tl - tr) < 1e-3 * span, (
        f"the two codes disagree at the interface: {tl} vs {tr}")
    assert not check_interface_balance(ex["left"], ex["right"], "left", "right")
    # non-matching interface discretisation is the normal case, not a special one
    assert ex["left"]["n_points"] != ex["right"]["n_points"]
