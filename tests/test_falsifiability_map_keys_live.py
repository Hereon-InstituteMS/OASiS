"""Regression: every key in data/postmortems/_falsifiability.json
points at a pitfall that still exists in the live catalog.

_falsifiability.json maps `backend::physics::pitfall_index`
keys to per-pitfall classification (falsifiable / cost). The
scoreboard in scripts/verify_signal_clauses.py reads this file
to compute the HONEST tier-2 ratio
(tier2_passed / n_realistically_verifiable) instead of the
naive tier2_passed / total.

When a pitfall is REMOVED from the catalog (rare but possible
during refactors), its entry in _falsifiability.json becomes
stale. The scoreboard would still count it as
'realistically_verifiable', misreporting the denominator.

Similarly, when a pitfall is REORDERED (new pitfall inserted
at index 2, shifting index 2+ entries to 3+), the index in
_falsifiability.json now points at a different pitfall — the
classification follows the wrong pitfall.

This gate validates that every non-_comment key in
_falsifiability.json resolves to a live pitfall. If a stale
key appears, the gate names it so the author can choose: drop
the stale entry, or re-classify the new pitfall at that index.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))


class TestFalsifiabilityMapKeysLive(unittest.TestCase):
    def test_all_keys_resolve(self) -> None:
        path = (_REPO / "data" / "postmortems"
                / "_falsifiability.json")
        self.assertTrue(path.is_file(),
                        f"_falsifiability.json not found at {path}")
        with open(path) as f:
            fmap = json.load(f)

        # Build the live pitfall-key set.
        from verify_signal_clauses import verify_backend
        live: set[str] = set()
        for backend in ("fenics", "ngsolve", "skfem", "dealii",
                        "kratos", "fourc", "febio", "dune"):
            for r in verify_backend(backend):
                live.add(f"{r.backend}::{r.physics}::"
                         f"{r.pitfall_index}")

        # Non-_comment keys must all resolve.
        keys = {k for k in fmap.keys() if not k.startswith("_")}
        stale = keys - live
        self.assertEqual(
            stale, set(),
            f"_falsifiability.json contains stale keys (pitfall "
            f"no longer in live catalog): {sorted(stale)}. Either "
            "drop the entry from _falsifiability.json, or update "
            "the pitfall_index to the current position if the "
            "pitfall was reordered. (The scoreboard in "
            "scripts/verify_signal_clauses.py uses this map to "
            "compute n_realistically_verifiable; stale keys "
            "inflate the denominator silently.)")


if __name__ == "__main__":
    unittest.main()
