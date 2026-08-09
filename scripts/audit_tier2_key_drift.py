#!/usr/bin/env python3
"""Screen tier-2 fixtures for KEY DRIFT: a fixture whose runner key points at a
claim it does not actually verify.

The runner key (backend::physics::pitfall_index) is the ONLY link between a
piece of evidence and the claim it defends. Pitfall indices are POSITIONAL, so
inserting an entry earlier in a list silently re-points every fixture after it —
and the fixture still passes, because passing only means "the software printed
what I expected", not "I tested the right claim". FEBio had three of six
fixtures mis-keyed this way; Kratos had eighteen move under it.

This cannot be decided mechanically, so the script produces LEADS, ranked. For
each fixture it measures how much of the claim's distinctive vocabulary appears
in the fixture's own text (docstring / cmd.sh header, expectation strings,
_comment). A fixture that verifies its claim almost always shares the claim's
rare terms — API names, exception classes, the physics being measured. A low
score is a candidate for hand adjudication, nothing more.

usage: t2keydrift.py <worktree> <backend> [--top N]
"""
from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
BACKEND = sys.argv[2]
TOP = 15
if "--top" in sys.argv:
    TOP = int(sys.argv[sys.argv.index("--top") + 1])

sys.path.insert(0, str(REPO / "src"))
from core.registry import load_all_backends, get_backend  # noqa: E402

try:
    load_all_backends()
except Exception:
    pass
b = get_backend(BACKEND)
assert b is not None, BACKEND

claims: dict[str, str] = {}
for p in b.supported_physics():
    try:
        k = b.get_knowledge(p.name)
    except Exception:
        continue
    if not isinstance(k, dict):
        continue
    for i, entry in enumerate(k.get("pitfalls", []) or []):
        if isinstance(entry, str) and "Signal:" in entry:
            claims[f"{p.name}::{i}"] = entry

STOP = set("""a an the and or of to in is are was were be been for on at by with
from that this these those it its as not no if then than so such which what when
where how why do does did done can could should would may might must will shall
you your we our they their he she his her him them i me my mine one two three
into onto out up down over under again further once here there all any both each
few more most other some only own same too very s t just don now use uses used
using set sets setting get gets getting make makes made give gives given take
takes taken run runs running call calls called write writes written read reads
value values case cases work works working thing things way ways time times
first second third next last also however instead rather still even though while
because since after before during between about above below through during
signal verified measured empirically catalog entry claim claims wording quoted
previously previous version reproduce reproduces does not never always must
should raise raises raised error errors fail fails failed""".split())
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def terms(text: str) -> Counter:
    out = Counter()
    for w in WORD.findall(text):
        lw = w.lower()
        if lw in STOP:
            continue
        out[lw] += 1
    return out


# Inverse document frequency over the claim corpus: a term that appears in
# every claim ("dolfinx", "mesh") says nothing about WHICH claim.
df = Counter()
for text in claims.values():
    df.update(set(terms(text)))
N = max(1, len(claims))


def fixture_text(d: Path) -> str:
    parts = []
    meta_path = d / "fixture.json"
    meta = json.loads(meta_path.read_text())
    parts.append(str(meta.get("_comment", "")))
    parts.extend(str(s) for s in meta.get("expect_in_output", []))
    parts.extend(str(s) for s in meta.get("forbid_in_output", []))
    src = d / "source.py"
    if src.exists():
        try:
            doc = ast.get_docstring(ast.parse(src.read_text())) or ""
        except SyntaxError:
            doc = ""
        parts.append(doc)
        parts.append(src.read_text())
    for name in ("cmd.sh", "run.sh"):
        p = d / name
        if p.exists():
            text = p.read_text()
            parts.append("\n".join(
                l.lstrip("# ") for l in text.splitlines()
                if l.startswith("#")))
            # A deal.II fixture's cmd.sh is three lines; the evidence lives in
            # the shared translation unit it names. Without pulling in the
            # probe's own comment block and body the score measures how verbose
            # the wrapper is, not whether the fixture matches its claim — every
            # correctly-keyed deal.II fixture scored 0.000 before this.
            m = re.search(r"run\.sh\"?\s+(\w+)\s+(\w+)\s+(\w+)", text)
            if m:
                tu, probe = m.group(1), m.group(3)
                cc = d.parent / "_shared" / f"{tu}.cc"
                if cc.exists():
                    body = cc.read_text()
                    # the probe function plus the comment block above it
                    fn = re.search(
                        r"((?:^//[^\n]*\n)+)?^static int " + re.escape(probe)
                        + r"\(\)\s*\{(.*?)^\}", body,
                        re.S | re.M)
                    parts.append(fn.group(0) if fn else "")
    return "\n".join(parts)


root = REPO / "scripts" / "tier2_fixtures" / BACKEND
rows = []
for d in sorted(p.parent for p in root.glob("*/fixture.json")):
    meta = json.loads((d / "fixture.json").read_text())
    key = f"{meta.get('physics')}::{meta.get('pitfall_index')}"
    claim = claims.get(key)
    if claim is None:
        rows.append((None, d.name, key, "SYNTHETIC/absent key", ""))
        continue
    ct = terms(claim)
    ft = set(terms(fixture_text(d)))
    # Weight each claim term by rarity; score = share of claim weight matched.
    num = den = 0.0
    matched, missed = [], []
    for w, _ in ct.items():
        weight = 1.0 / df[w] if df[w] else 1.0
        den += weight
        if w in ft:
            num += weight
            matched.append(w)
        else:
            missed.append(w)
    score = num / den if den else 0.0
    rare_missed = sorted(missed, key=lambda w: df[w])[:6]
    rows.append((score, d.name, key, "", ",".join(rare_missed)))

absent = [r for r in rows if r[0] is None]
scored = sorted((r for r in rows if r[0] is not None), key=lambda r: r[0])
print(f"backend={BACKEND}  fixtures={len(rows)}  claims={len(claims)}")
print(f"\nkeys that name no existing claim ({len(absent)}):")
for _, name, key, note, _x in absent:
    print(f"  {name:55s} {key:32s} {note}")
print(f"\nlowest-overlap fixtures (candidates for hand adjudication):")
print(f"  {'score':>6}  {'fixture':55s} {'key':32s} rare claim terms absent")
for score, name, key, _n, missed in scored[:TOP]:
    print(f"  {score:6.3f}  {name:55s} {key:32s} {missed}")
med = scored[len(scored) // 2][0] if scored else 0.0
print(f"\nmedian overlap score = {med:.3f}  "
      f"(a fixture far below the median is the lead, not the verdict)")
