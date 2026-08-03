"""Regression tests for result attestation.

Every test here corresponds to a fabrication route that was demonstrated to
work against an earlier version of this module. They exist so those routes
cannot silently reopen.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

meshio = pytest.importorskip("meshio")

from core.attestation import (Attestation, AttestationError, attest_quantity,
                              find_data_artefacts, require_attested,
                              verify_attestation)


# ── helpers ───────────────────────────────────────────────────────────────
def _grid(n: int = 8):
    pts = np.array([[i / n, j / n, 0.0] for i in range(n + 1)
                    for j in range(n + 1)], float)
    tris = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            tris.append([a, a + 1, a + n + 1])
            tris.append([a + 1, a + n + 2, a + n + 1])
    return pts, [("triangle", np.array(tris))]


def _write_field(directory: Path, value, name: str = "u", fname: str = "sol.vtu"):
    directory.mkdir(parents=True, exist_ok=True)
    pts, cells = _grid()
    vals = (np.full(len(pts), value) if np.isscalar(value)
            else np.asarray(value, float))
    meshio.write_points_cells(str(directory / fname), pts, cells,
                              point_data={name: vals})
    return directory / fname


# ── the core property ─────────────────────────────────────────────────────
def test_honest_claim_is_accepted(tmp_path):
    _write_field(tmp_path, 2.0)
    att = attest_quantity(tmp_path, "j1", "l2_norm")
    ok, why = require_attested(att.value, att)
    assert ok, why


def test_fabricated_claim_is_refused(tmp_path):
    _write_field(tmp_path, 2.0)
    att = attest_quantity(tmp_path, "j1", "l2_norm")
    ok, why = require_attested(att.value * 10, att)
    assert not ok and "not the value OASiS computed" in why


def test_claim_without_attestation_is_refused():
    ok, why = require_attested(1.234, None)
    assert not ok and "no attestation" in why


# ── routes that previously worked ─────────────────────────────────────────
def test_value_printed_by_the_solver_script_is_not_evidence(tmp_path):
    """The agent authors the script, so printed values are narration.

    A real run whose script printed a hard-coded 'L2_error = 1e-12' once
    attested that fabricated number. Narration must never be admissible.
    """
    (tmp_path / "stdout.log").write_text("run complete\nL2_error = 1.0000e-12\n")
    (tmp_path / "results_summary.json").write_text('{"L2_error": 1e-12}')
    assert find_data_artefacts(tmp_path) == []
    with pytest.raises(AttestationError, match="no solver data output"):
        attest_quantity(tmp_path, "j1", "l2_norm")


def test_small_values_are_compared_relatively(tmp_path):
    """A tolerance floor of 1.0 once made every pair of tiny numbers match:
    a claim eight times the attested value passed."""
    _write_field(tmp_path, 1.25e-13)
    att = attest_quantity(tmp_path, "j1", "l2_norm")
    assert att.value < 1e-10
    ok, _ = require_attested(att.value * 8, att)
    assert not ok
    ok_honest, _ = require_attested(att.value, att)
    assert ok_honest


def test_probe_outside_the_mesh_is_refused(tmp_path):
    """probe_field extrapolates rather than failing, so the domain must be
    checked explicitly."""
    _write_field(tmp_path, 1.0)
    with pytest.raises(AttestationError, match="outside the mesh"):
        attest_quantity(tmp_path, "j1", "probe_values",
                        probe_points=[[9.0, 9.0, 0.0]])


def test_probe_inside_the_mesh_works(tmp_path):
    _write_field(tmp_path, 3.0)
    att = attest_quantity(tmp_path, "j1", "probe_values",
                          probe_points=[[0.5, 0.5, 0.0]])
    assert att.values and abs(att.values[0] - 3.0) < 1e-9


def test_artefact_outside_the_run_window_is_refused(tmp_path):
    _write_field(tmp_path, 1.0)
    long_ago = time.time() - 600
    with pytest.raises(AttestationError, match="during the run"):
        attest_quantity(tmp_path, "j1", "l2_norm",
                        run_started=long_ago, run_finished=long_ago + 60)


def test_artefact_inside_the_run_window_is_accepted(tmp_path):
    _write_field(tmp_path, 1.0)
    now = time.time()
    att = attest_quantity(tmp_path, "j1", "l2_norm",
                          run_started=now - 60, run_finished=now + 60)
    assert att.value > 0


def test_artefact_edited_after_attestation_is_refused(tmp_path):
    _write_field(tmp_path, 1.0)
    att = attest_quantity(tmp_path, "j1", "l2_norm")
    _write_field(tmp_path, 5.0)                      # tamper
    ok, why = verify_attestation(att)
    assert not ok and "changed after attestation" in why
    gated, why2 = require_attested(att.value, att)
    assert not gated


def test_non_finite_field_is_refused(tmp_path):
    _write_field(tmp_path, np.nan)
    with pytest.raises(AttestationError):
        attest_quantity(tmp_path, "j1", "l2_norm")


def test_unknown_quantity_is_refused(tmp_path):
    _write_field(tmp_path, 1.0)
    with pytest.raises(AttestationError, match="not attestable"):
        attest_quantity(tmp_path, "j1", "whatever_i_like")


def test_no_data_output_at_all_is_refused(tmp_path):
    (tmp_path / "notes.txt").write_text("I solved it, the answer is 42")
    with pytest.raises(AttestationError):
        attest_quantity(tmp_path, "j1", "l2_norm")


def test_attestation_records_provenance(tmp_path):
    p = _write_field(tmp_path, 2.0)
    att = attest_quantity(tmp_path, "job42", "l2_norm")
    assert att.job_id == "job42"
    assert Path(att.source_file).name == p.name
    assert len(att.source_sha256) == 64
    assert att.n_points > 0
    assert "l2" in att.computed_by.lower() or att.computed_by
