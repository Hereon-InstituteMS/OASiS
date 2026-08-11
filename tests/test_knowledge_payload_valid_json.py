"""Regression: a size-capped knowledge payload must still be VALID JSON.

WHY this exists
===============
`prepare_simulation` renders each backend's physics knowledge into a
```json fence. Until 2026-08-03 the size cap was applied with a raw
string slice, ``text[:LIMIT]``, which cuts in the middle of whatever
value happens to sit at that byte offset.

A sweep of all (backend, physics) payloads on this install found 12 rows
over the 16000-char cap, and ALL 12 came back as INVALID JSON under the
slice — 7 SPARTA rows and 5 FEniCSx rows — failing with
"Unterminated string starting at: ..." or "Expecting property name
enclosed in double quotes: ...". An agent that receives an unparseable
payload has nothing to fall back on, and a small model will not go
hunting for a second copy: this is strictly worse than serving less.

The fix is `tools.consolidated._fit_json_block`, which shrinks by
dropping or thinning WHOLE entries instead of slicing. This test pins
its three guarantees so they cannot silently regress as knowledge grows
— and knowledge growing is exactly what triggered the bug:

  1. the rendered block always parses as JSON,
  2. no load-bearing entry is ever removed (the description, the
     runnable example, the spaces / weak form / BCs / solver blocks),
  3. anything removed is NAMED, together with the call that fetches it.

The test is deliberately written against the LIVE catalog rather than a
fixture, so a newly added physics row that blows the cap is caught here
rather than in front of an agent.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

# The cap prepare_simulation applies to the knowledge block.
KNOWLEDGE_LIMIT = 16000


def _all_backends():
    """Register every backend package and return the live instances."""
    from core.registry import get_backend
    out = []
    root = _REPO / "src" / "backends"
    for name in sorted(os.listdir(root)):
        if not (root / name).is_dir() or name.startswith("_") or name == "__pycache__":
            continue
        try:
            importlib.import_module(f"backends.{name}.backend").register()
        except Exception:
            continue
        be = get_backend(name)
        if be is not None:
            out.append((name, be))
    return out


def _payloads():
    """Yield (backend_name, physics, payload_dict) for every catalog row.

    The payload is what prepare_simulation dumps: the knowledge dict minus
    the `pitfalls` list, which is rendered separately outside the fence.
    """
    for name, be in _all_backends():
        try:
            physics = [p.name for p in be.supported_physics()]
        except Exception:
            continue
        for phys in physics:
            try:
                k = be.get_knowledge(phys)
            except Exception:
                continue
            if not isinstance(k, dict) or not k:
                continue
            yield name, phys, {kk: vv for kk, vv in k.items() if kk != "pitfalls"}


class TestKnowledgePayloadValidJson(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import tools.consolidated as consolidated
        cls.C = consolidated
        cls.rows = list(_payloads())
        if not cls.rows:
            raise unittest.SkipTest("no backend catalogs importable in this env")

    def test_every_fitted_payload_parses(self) -> None:
        """Guarantee 1 — the rendered block is always valid JSON."""
        broken = []
        for name, phys, payload in self.rows:
            text, _note = self.C._fit_json_block(payload, KNOWLEDGE_LIMIT)
            try:
                json.loads(text)
            except Exception as exc:
                broken.append(f"{name}::{phys}: {exc}")
        self.assertFalse(
            broken,
            "knowledge payload does not parse as JSON after fitting "
            "(the agent receives this verbatim inside a ```json fence):\n  "
            + "\n  ".join(broken))

    def test_load_bearing_entries_survive(self) -> None:
        """Guarantee 2 — a shrink never removes setup-critical content."""
        lost = []
        for name, phys, payload in self.rows:
            text, _note = self.C._fit_json_block(payload, KNOWLEDGE_LIMIT)
            kept = json.loads(text)
            for key in payload:
                if self.C._is_load_bearing(key) and key not in kept:
                    lost.append(f"{name}::{phys} lost {key!r}")
        self.assertFalse(
            lost,
            "fitting dropped a load-bearing entry — a small model cannot "
            "reconstruct these:\n  " + "\n  ".join(lost))

    def test_removals_are_announced(self) -> None:
        """Guarantee 3 — whatever was removed is named for the caller."""
        silent = []
        for name, phys, payload in self.rows:
            text, note = self.C._fit_json_block(payload, KNOWLEDGE_LIMIT)
            kept = json.loads(text)
            removed = [k for k in payload if k not in kept]
            thinned = [k for k in kept
                       if isinstance(kept[k], (dict, list))
                       and any("_more_entries_omitted__" in str(x)
                               for x in (kept[k] if isinstance(kept[k], list)
                                         else kept[k].keys()))]
            if (removed or thinned) and not note.strip():
                silent.append(f"{name}::{phys} removed {removed + thinned} with no note")
            for k in removed:
                if k not in note:
                    silent.append(f"{name}::{phys} dropped {k!r} without naming it")
        self.assertFalse(
            silent,
            "content was removed without telling the caller what or how to "
            "fetch it:\n  " + "\n  ".join(silent))

    def test_under_cap_payloads_are_untouched(self) -> None:
        """A payload that fits must be passed through byte-identically."""
        for name, phys, payload in self.rows:
            raw = json.dumps(payload, indent=2, default=str)
            if len(raw) > KNOWLEDGE_LIMIT:
                continue
            text, note = self.C._fit_json_block(payload, KNOWLEDGE_LIMIT)
            self.assertEqual(text, raw, f"{name}::{phys} altered while under the cap")
            self.assertEqual(note, "", f"{name}::{phys} noted a trim it did not make")

    def test_slicing_would_have_broken_real_payloads(self) -> None:
        """The bug is real, not hypothetical — keep the evidence executable.

        If this ever finds ZERO sliceable payloads it means every catalog row
        now fits the cap; that is fine and the assertion below is skipped
        rather than failed, so the test does not become a growth requirement.
        """
        would_break = []
        for name, phys, payload in self.rows:
            raw = json.dumps(payload, indent=2, default=str)
            if len(raw) <= KNOWLEDGE_LIMIT:
                continue
            try:
                json.loads(raw[:KNOWLEDGE_LIMIT])
            except Exception:
                would_break.append(f"{name}::{phys}")
        if not would_break:
            self.skipTest("no catalog row currently exceeds the cap")
        # Same rows, through the real fitter — all must parse.
        for name, phys, payload in self.rows:
            if f"{name}::{phys}" not in would_break:
                continue
            text, _ = self.C._fit_json_block(payload, KNOWLEDGE_LIMIT)
            json.loads(text)   # raises and fails the test if the fix regressed

    def test_prepare_simulation_fence_parses(self) -> None:
        """End-to-end: the ```json fence inside the real tool output parses.

        This is the surface the agent actually reads, so it is checked
        separately from the helper.
        """
        import asyncio
        import re
        from tools.consolidated import register_consolidated_tools
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("payload-test")
        register_consolidated_tools(mcp)
        tools = asyncio.get_event_loop().run_until_complete(mcp.list_tools())
        if not any(t.name == "prepare_simulation" for t in tools):
            self.skipTest("prepare_simulation not registered")

        # Pick the biggest row per backend — the one most likely to be capped.
        biggest = {}
        for name, phys, payload in self.rows:
            n = len(json.dumps(payload, indent=2, default=str))
            if n > biggest.get(name, (0, None))[0]:
                biggest[name] = (n, phys)

        broken = []
        for name, (_n, phys) in sorted(biggest.items()):
            try:
                out = asyncio.get_event_loop().run_until_complete(
                    mcp.call_tool("prepare_simulation",
                                  {"solver": name, "physics": phys}))
            except Exception:
                continue
            text = "".join(getattr(c, "text", "") for c in
                           (out[0] if isinstance(out, tuple) else out))
            m = re.search(r"## Knowledge\n```json\n(.*?)\n```", text, re.S)
            if not m:
                continue
            try:
                json.loads(m.group(1))
            except Exception as exc:
                broken.append(f"{name}::{phys}: {exc}")
        self.assertFalse(
            broken,
            "the ```json fence in real prepare_simulation output does not "
            "parse:\n  " + "\n  ".join(broken))


if __name__ == "__main__":
    unittest.main()
