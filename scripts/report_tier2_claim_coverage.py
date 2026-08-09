#!/usr/bin/env python3
"""Claim-attributed tier-2 coverage, counting only fixtures git already tracks.

The in-flight ones are not evidence yet: they have not been run here.
usage: t2coverage.py <worktree> <backend> [--all]
"""
import json, subprocess, sys
from pathlib import Path
REPO = Path(sys.argv[1]).resolve(); BACKEND = sys.argv[2]
ALL = "--all" in sys.argv
sys.path.insert(0, str(REPO / "src"))
from core.registry import load_all_backends, get_backend
try: load_all_backends()
except Exception: pass
b = get_backend(BACKEND)

by_text = {}


def _norm(key):
    """One canonical form for a claim key.

    Two spellings are in the tree and they mean the same thing:

        physics:index      declared by fixtures in their `covers` field
        physics::index     synthesised here from physics + pitfall_index

    Nothing reconciled them, so every backend whose fixtures USE `covers`
    scored zero: sparta 0/211 and dune 0/103, on a tree where the sparta
    pytest gate passes at ~98.7%. 4C looked healthy at 93.5% only because
    391 of its 418 fixtures omit `covers` entirely and fall through to the
    synthesised key; its 8 fixtures that do declare `covers` failed silently.

    A metric that reads 0% for a fully covered backend is not a low score,
    it is a broken instrument — and this one feeds the freeze criterion's
    coverage gate, i.e. the number that decides whether the corpus is done.
    """
    return str(key).replace("::", ":", 1) if "::" in str(key) else str(key)


claims = {}
for p in b.supported_physics():
    k = b.get_knowledge(p.name)
    if isinstance(k, dict):
        for i, e in enumerate(k.get("pitfalls", []) or []):
            if isinstance(e, str) and "Signal:" in e:
                claims[_norm(f"{p.name}::{i}")] = e
                # SPARTA attaches ten cross-cutting pitfalls to EVERY physics
                # area, so a positional walk sees the same claim ten times and
                # the denominator inflates: 211 slots where there are ~75
                # DISTINCT claims. Counting slots made a nearly-covered backend
                # read 47.4% while its own gate passed at ~98.7% of distinct
                # claims. Neither number was wrong; they were different
                # questions. Coverage is about claims, so the text is the
                # identity and duplicates collapse.
                by_text.setdefault(e, []).append(_norm(f"{p.name}::{i}"))
tracked = set()
out = subprocess.run(["git", "-C", str(REPO), "ls-files",
                      f"scripts/tier2_fixtures/{BACKEND}"],
                     capture_output=True, text=True).stdout.split()
for f in out:
    parts = f.split("/")
    if len(parts) > 3:
        tracked.add(parts[3])
root = REPO / "scripts" / "tier2_fixtures" / BACKEND
covered, dirs, synth = {}, 0, 0
for d in sorted(p.parent for p in root.glob("*/fixture.json")):
    if not ALL and d.name not in tracked:
        continue
    dirs += 1
    m = json.loads((d / "fixture.json").read_text())
    dec = m.get("covers")
    if dec is None:
        key = _norm(f"{m.get('physics')}::{m.get('pitfall_index')}")
        dec = [key] if key in claims else []
    if not dec:
        synth += 1
    for c in dec:
        covered.setdefault(_norm(c), []).append(d.name)
# Collapse aliases: one distinct claim text, however many positional slots.
distinct = {text: keys for text, keys in by_text.items()}
covered_texts = {text for text, keys in distinct.items()
                 if any(k in covered for k in keys)}
real = sorted(covered_texts)
claims = distinct
print(f"{BACKEND}: {'ALL' if ALL else 'TRACKED'} fixtures={dirs} "
      f"(synthetic/no-claim={synth})  claims={len(claims)}")
print(f"  claim-attributed coverage = {len(real)}/{len(claims)} = "
      f"{100*len(real)/len(claims):.1f}%")
print(f"  80% needs {-(-8*len(claims)//10)} claims; "
      f"{max(0,-(-8*len(claims)//10)-len(real))} to go")
dupes = {c: f for c, f in covered.items() if len(f) > 1}
if dupes: print(f"  DOUBLE-COUNTED: {dupes}")
bogus = {c: f for c, f in covered.items() if c not in claims}
if bogus: print(f"  COVERS NAMING NO CLAIM: {bogus}")
