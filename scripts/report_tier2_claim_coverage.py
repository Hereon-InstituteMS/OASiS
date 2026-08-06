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
claims = {}
for p in b.supported_physics():
    k = b.get_knowledge(p.name)
    if isinstance(k, dict):
        for i, e in enumerate(k.get("pitfalls", []) or []):
            if isinstance(e, str) and "Signal:" in e:
                claims[f"{p.name}::{i}"] = e
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
        key = f"{m.get('physics')}::{m.get('pitfall_index')}"
        dec = [key] if key in claims else []
    if not dec:
        synth += 1
    for c in dec:
        covered.setdefault(c, []).append(d.name)
real = [c for c in covered if c in claims]
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
