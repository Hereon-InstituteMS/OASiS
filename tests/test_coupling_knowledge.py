"""Guards for the coupling knowledge an agent is actually served.

The failure this file exists to prevent, measured on the live tools before the
rewrite: the whole coupling corpus for all nine backends was smaller than a
tenth of one backend's pitfall payload, named two backends out of nine, varied
not at all with `solver=`, and documented the DEPRECATED tool's enum while the
general tool's contract lived only inside that tool's own docstring.

Every assertion here is structural — a key that must be documented, a payload
that must differ per backend, a script that must compile. Nothing pins a number
produced by a run, because a knowledge test that pins a measurement makes the
measurement part of the shipped corpus.
"""
from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tools.coupling_knowledge import (            # noqa: E402
    _BACKEND_ORDER, _PARTICIPANT_DIR, coupling_core, coupling_knowledge,
    coupling_sides_table, precice_knowledge,
)


def _core() -> str:
    return coupling_core()


# ── the contract must be IN the knowledge, not only in a docstring ───────

@pytest.mark.parametrize("token", [
    "imports.json", "exports.json", "InterfaceData", "coordinates",
    "normal_fluxes", "imports_from", "work_dir", "field_name", "n_points",
])
def test_couple_contract_terms_are_documented(token):
    """The `couple` contract used to exist ONLY in the tool's docstring: the
    knowledge contained zero occurrences of any of these."""
    assert token in _core(), f"'{token}' missing from knowledge(topic='coupling')"


def test_knowledge_names_the_general_tool_not_the_deprecated_enum():
    core = _core()
    assert "couple(" in core
    # the legacy enum values must not be presented as the way to couple
    for legacy in ("heat_dd", "poisson_dd", "tsi_dd", "poisson_dd_study"):
        assert legacy not in core, (
            f"knowledge still advertises the deprecated coupled_solve enum "
            f"value '{legacy}'")


def test_knowledge_does_not_point_at_private_generators():
    """It used to instruct the agent to write 'a new subdomain script generator
    (analogous to `_fenics_heat_subdomain_script`)' — a private function that no
    agent can call."""
    core = _core()
    assert "_fenics_heat_subdomain_script" not in core
    assert "_generate_domain_b_input" not in core


def test_theta_guidance_matches_a_real_parameter():
    """Two knowledge sections used to give theta-selection tables while the tool
    had no theta parameter at all."""
    from tools import consolidated
    src = inspect.getsource(consolidated)
    sig = re.search(r"async def couple\((.*?)\) -> str:", src, re.S).group(1)
    assert "theta" in sig, "couple() has no theta parameter"
    assert "theta" in _core(), "knowledge does not mention theta"


def test_sign_convention_states_both_quantities():
    core = _core()
    assert "apply the same number, export opposite numbers" in core.lower()


# ── per-backend payloads must actually differ ────────────────────────────

def test_every_backend_has_a_named_payload_entry():
    core = _core()
    for name in _BACKEND_ORDER:
        assert f"solver='{name}'" in core, (
            f"knowledge(topic='coupling') never tells the agent to ask for "
            f"solver='{name}'")


def test_coupling_payload_varies_by_backend():
    """The measured bug: byte-identical output for every solver."""
    seen = {}
    for name in _BACKEND_ORDER:
        text = coupling_knowledge(name)
        assert text != coupling_core(), f"solver='{name}' returned the core payload"
        assert text not in seen.values(), (
            f"solver='{name}' is byte-identical to solver='{seen and [k for k,v in seen.items() if v==text][0]}'")
        seen[name] = text


def test_precice_payload_varies_by_backend():
    seen = {}
    for name in _BACKEND_ORDER:
        text = precice_knowledge(name)
        assert text not in seen.values(), (
            f"precice solver='{name}' is byte-identical to another backend")
        seen[name] = text


def test_unknown_solver_is_not_a_dead_end():
    out = coupling_knowledge("no-such-backend")
    assert "no-such-backend" in out
    # still hands over the general contract rather than nothing
    assert "imports.json" in out


# ── the shipped participant scripts must be real, runnable files ─────────

def _script_backends():
    return [n for n in _BACKEND_ORDER
            if (_PARTICIPANT_DIR / f"participant_{n}.py").is_file()]


def test_at_least_the_flagship_participants_ship():
    for name in ("fenics", "fourc"):
        assert (_PARTICIPANT_DIR / f"participant_{name}.py").is_file(), (
            f"participant script for {name} is missing")


@pytest.mark.parametrize("name", _script_backends())
def test_participant_script_parses_and_is_complete(name):
    p = _PARTICIPANT_DIR / f"participant_{name}.py"
    text = p.read_text()
    ast.parse(text)                                   # syntactically runnable
    assert "imports.json" in text and "exports.json" in text
    assert "EDIT THIS BLOCK" in text, (
        "the script must mark the block a user edits, or a weak model has to "
        "work out which constants are problem data")
    assert "PLACEHOLDER" in text, (
        "the edit block must say its numbers are placeholders — shipping a "
        "ready-made benchmark configuration is the anchoring the project bans")
    assert re.search(r"^PARTNER\b", text, re.M), f"{name}: no PARTNER constant"
    # a participant that can take both roles must expose which one it is
    if "SIDE ==" in text:
        assert re.search(r"^SIDE\b", text, re.M), f"{name}: no SIDE constant"


@pytest.mark.parametrize("name", _script_backends())
def test_served_payload_contains_the_whole_script(name):
    served = coupling_knowledge(name)
    text = (_PARTICIPANT_DIR / f"participant_{name}.py").read_text()
    assert text in served, (
        f"knowledge(topic='coupling', solver='{name}') does not serve the "
        f"participant script verbatim — it can drift from the file that is tested")
    assert "```python" in served


# ── nothing machine-specific may be served ───────────────────────────────

_HOST_PATH = re.compile(r"(?<![\w/])(?:/home/|/Users/|/opt/|/usr/(?:local/)?(?:bin|lib))")


@pytest.mark.parametrize("name", [""] + _BACKEND_ORDER)
def test_no_absolute_host_paths_in_served_coupling_knowledge(name):
    for text in (coupling_knowledge(name), precice_knowledge(name)):
        hits = _HOST_PATH.findall(text)
        assert not hits, (
            f"solver='{name}': served knowledge carries a path from the build "
            f"host ({hits[:3]}); it is wrong on every other machine. Tell the "
            f"agent to read the path from discover(query='list') instead.")


def test_discover_has_a_coupling_branch():
    """There was no path to the coupling knowledge unless you already knew the
    topic string existed."""
    from tools import consolidated
    src = inspect.getsource(consolidated)
    assert 'query == "coupling"' in src
    doc = re.search(r"def discover\(.*?\"\"\"(.*?)\"\"\"", src, re.S).group(1)
    assert "coupling" in doc, (
        "discover's docstring does not advertise query='coupling'")
    assert "knowledge(topic='coupling'" in src, (
        "discover(query='coupling') must name the follow-up knowledge call")


def test_sides_table_covers_every_backend():
    table = coupling_sides_table()
    for label in ("FEniCSx", "NGSolve", "scikit-fem", "DUNE", "deal.II",
                  "4C", "FEBio", "Kratos", "SPARTA"):
        assert label in table, f"{label} is absent from the side table"
