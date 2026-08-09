"""What the agent receives must always be parseable, however big the knowledge gets.

`prepare_simulation` serialised its knowledge dict and sliced the string at a
character count. Measured across every backend and physics, three SPARTA
payloads came back with the `## Knowledge` block cut mid-string, so `json.loads`
failed on exactly what an agent is handed. Not merely losing content — handing a
weak model something it cannot parse at all.

The failure mode also grows with the thing we are trying to improve. Every
expansion that adds knowledge pushes more payloads past the cap, so enriching a
backend was quietly making it less usable. That is the opposite of the point,
and it is why this is a test rather than a fix-and-move-on.

Whole keys are dropped now, load-bearing ones last, with the payload told to the
agent as incomplete and the missing sections named.
"""
from __future__ import annotations

import functools
import json
import re

import pytest

from tools.consolidated import _fit_json_payload


@functools.lru_cache(maxsize=1)
def _served():
    """Every (backend, physics) payload, generated ONCE.

    `prepare_simulation` builds templates and reads reference files, so driving
    it per assertion turns a guard into a ten-minute suite and the guard gets
    deleted. Two hundred payloads, collected once, shared by both checks.
    """
    from core.registry import get_backend, list_backends, load_all_backends
    from tools import consolidated as C

    load_all_backends()
    captured = {}

    class _Recorder:
        def tool(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn
            return deco

    C.register_consolidated_tools(_Recorder())
    prepare = captured["prepare_simulation"]

    # Only oversized payloads can truncate, and `prepare_simulation` is
    # expensive — it generates a template and reads reference files per call.
    # Driving all 206 took 10.6 minutes, which is how a guard becomes a thing
    # people delete. The raw knowledge dict is free to measure, so it decides
    # which ones are worth serving. The margin is generous because pitfalls are
    # carved out before the fit, so raw size overestimates what must fit.
    from tools.consolidated import _fit_json_payload  # noqa: F401  (same module)
    LIMIT = 16_000
    MARGIN = 0.5

    out = []
    for entry in list_backends():
        be = get_backend(entry["name"])
        if not be:
            continue
        for p in be.supported_physics():
            try:
                raw = be.get_knowledge(p.name) or {}
                size = len(json.dumps(raw, default=str))
            except Exception:
                size = LIMIT                     # unmeasurable: check it
            if size < LIMIT * MARGIN:
                continue
            try:
                text = prepare(entry["name"], p.name)
            except Exception as exc:             # a broken physics row
                text = f"__ERROR__ {exc}"
            out.append((entry["name"], p.name, text))
    return tuple(out)


# ── the unit that does the work ───────────────────────────────────────────
def test_a_payload_that_fits_is_returned_whole():
    payload = {"description": "short", "solver": "cg"}
    text, dropped = _fit_json_payload(payload, 10_000)
    assert not dropped
    assert json.loads(text) == payload


def test_an_oversized_payload_is_still_valid_json():
    payload = {"description": "keep me", "bulk": "x" * 50_000}
    text, dropped = _fit_json_payload(payload, 1_000)
    json.loads(text)                      # must not raise
    assert "bulk" in dropped


def test_load_bearing_keys_survive_and_bulk_goes_first():
    """A payload that keeps the manual and drops the element names would be
    worse than useless."""
    payload = {
        "relevant_commands": {"c%d" % i: "x" * 400 for i in range(50)},
        "description": "what this physics is",
        "required_sections": ["A", "B"],
        "element": "hex8",
    }
    text, dropped = _fit_json_payload(payload, 2_000)
    kept = json.loads(text)
    assert "description" in kept and "required_sections" in kept
    assert "element" in kept
    assert dropped == ["relevant_commands"]


def test_surviving_keys_keep_the_authors_order():
    payload = {"description": "d", "element": "e", "bulk": "x" * 40_000}
    text, _ = _fit_json_payload(payload, 500)
    assert list(json.loads(text)) == ["description", "element"]


def test_nothing_droppable_still_returns_valid_json():
    """A single key larger than the limit cannot be trimmed; the result must
    still parse rather than being a sliced fragment."""
    text, dropped = _fit_json_payload({"huge": "x" * 40_000}, 100)
    json.loads(text)
    assert dropped == ["huge"]


# ── through the real tool, across everything ──────────────────────────────
_BLOCK = re.compile(r"## Knowledge\n```json\n(.*?)\n```", re.S)


def test_every_served_payload_parses():
    """The one that matters: three SPARTA payloads used to fail this."""
    broken = []
    for backend, physics, text in _served():
        m = _BLOCK.search(text)
        if not m:
            continue
        try:
            json.loads(m.group(1))
        except json.JSONDecodeError as exc:
            broken.append(f"{backend}/{physics}: {exc}")
    assert not broken, (
        "the knowledge block an agent receives is not valid JSON — a weak "
        "model cannot parse this at all:\n  " + "\n  ".join(broken))


def test_an_incomplete_payload_says_so():
    """A silent hole reads as 'this backend has nothing to say'."""
    from core.registry import get_backend

    silent = []
    for backend, physics, text in _served():
        m = _BLOCK.search(text)
        if not m:
            continue
        be = get_backend(backend)
        full = be.get_knowledge(physics) or {}
        if not isinstance(full, dict):
            continue
        missing = {k for k in full if k != "pitfalls"} - set(json.loads(m.group(1)))
        if missing and "too large to serve whole" not in text:
            silent.append(f"{backend}/{physics} omits {sorted(missing)}")
    assert not silent, (
        "sections were dropped with no note to the agent:\n  "
        + "\n  ".join(silent))
