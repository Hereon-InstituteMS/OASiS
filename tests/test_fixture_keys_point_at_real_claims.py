"""A fixture's key must name the claim it actually exercises.

WHY, and it is not hypothetical
-------------------------------
The runner key is the ONLY link between a piece of evidence and the claim it
defends. Nothing else connects them, so when the key is wrong the failure is
completely silent: the claim looks covered, the evidence points elsewhere, and
every test still passes.

Both halves of that have happened here, in the same week:

  * FEBio — three of its six pre-existing fixtures were keyed to the wrong
    claim. `missing_control_silent_zero_result` executed the dropped-<Control>
    claim while pointing at the Poisson-ratio claim, and the catalog
    self-consistency gate sat on index 0, which made the real
    `linear_elasticity#0` claim appear covered by something that never touched
    it. Two claims defended by nothing, two others credited twice.
  * Kratos — moving 53 entries out of the pitfall lists SHIFTED INDICES, which
    would have repointed 18 fixtures at the wrong claims. They were renumbered
    atomically, and a second pass then caught that the prose in each fixture's
    `_comment` still quoted the OLD index — the human-readable evidence record
    disagreeing with the machine-readable field, which is the half that
    misleads a person reading the file.

So this gate answers one question per fixture: does the claim your key names
exist? It cannot tell whether the fixture exercises the RIGHT claim — only a
reader comparing the fixture to the entry can do that — but it catches every
key that points at nothing, which is the failure that made a claim look
defended when it was not.

It also protects the coverage metric. `covers` lists let one fixture attest to
several claims, which is correct (DUNE JIT-compiles per form, so splitting
claims that share a module buys build time and no evidence). But an unchecked
`covers` list is a coverage number anyone can raise by inventing keys, and the
freeze criterion reads that number.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

FIXTURES = REPO / "scripts" / "tier2_fixtures"
BACKENDS_DIR = REPO / "src" / "backends"

# A key may legitimately point past the end of a pitfall list: the MMS
# convergence harnesses use a high sentinel because they verify a whole family
# rather than one entry. Counting those as coverage overstated two backends by
# 9 fixtures, so they are recognised here and excluded from the claim count
# rather than being reported as broken keys.
SENTINEL_FLOOR = 90


def _claim_counts(backend: str) -> dict[str, int]:
    """How many pitfall entries each physics of this backend declares.

    THE REGISTRY IS THE AUTHORITY, and getting this wrong made the first
    version of this gate accuse 13 good deal.II fixtures. A static AST walk saw
    10 deal.II physics with pitfall lists and missed `dg_transport`, `poisson`
    and `helmholtz` — which have 7, 7 and 6 pitfalls respectively when asked
    through `get_knowledge()`. Backends assemble their knowledge dicts by
    merging, aliasing and importing across modules, and no static reader
    follows that; 4C and FEniCSx yielded zero physics at all.

    Accusing a correct fixture of pointing at a non-existent claim is the worst
    possible failure for this gate: it sends someone renumbering fixtures that
    were already right. So ask the backend, and fall back to the AST only when
    the backend cannot be loaded here — in which case a MISS is not reported as
    drift, because the reader is the thing that is blind.
    """
    counts: dict[str, int] = {}
    try:
        from core.registry import get_backend, load_all_backends
        load_all_backends()
        be = get_backend(backend)
    except Exception:
        be = None

    if be is not None:
        try:
            for cap in be.supported_physics():
                k = be.get_knowledge(cap.name)
                if isinstance(k, dict):
                    pit = k.get("pitfalls")
                    if isinstance(pit, (list, tuple)):
                        counts[cap.name.lstrip("_")] = len(pit)
        except Exception:
            counts = {}
    if counts:
        return counts

    # Fallback only. Marked so the caller can refuse to report drift from it.
    be_dir = BACKENDS_DIR / backend
    if not be_dir.is_dir():
        return counts
    for py in be_dir.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(errors="ignore"))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            # KNOWLEDGE = {"<physics>": {..., "pitfalls": [...]}}
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if not (isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                        and isinstance(value, ast.Dict)):
                    continue
                for k2, v2 in zip(value.keys, value.values):
                    if (isinstance(k2, ast.Constant) and k2.value == "pitfalls"
                            and isinstance(v2, (ast.List, ast.Tuple))):
                        n = len(v2.elts)
                        name = key.value.lstrip("_")
                        counts[name] = max(counts.get(name, 0), n)
    return counts


def _fixture_keys(backend: str) -> list[tuple[str, str, list[str]]]:
    """(fixture name, source of key, keys) for every fixture of a backend."""
    out = []
    fx = FIXTURES / backend
    if not fx.is_dir():
        return out
    for d in sorted(fx.iterdir()):
        manifest = d / "fixture.json"
        if not manifest.is_file():
            continue
        try:
            spec = json.loads(manifest.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        covers = spec.get("covers")
        if isinstance(covers, list) and covers:
            out.append((d.name, "covers", [str(c) for c in covers]))
        elif spec.get("physics") is not None and spec.get("pitfall_index") is not None:
            out.append((d.name, "physics/pitfall_index",
                        [f"{spec['physics']}:{spec['pitfall_index']}"]))
    return out


BACKENDS = sorted(p.name for p in BACKENDS_DIR.iterdir()
                  if p.is_dir() and not p.name.startswith("_")) \
    if BACKENDS_DIR.is_dir() else []


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_fixture_key_names_an_existing_claim(backend):
    """A key pointing at nothing credits coverage to no claim at all."""
    counts = _claim_counts(backend)
    keys = _fixture_keys(backend)
    if not keys:
        pytest.skip(f"{backend} has no fixtures with claim keys")
    if not counts:
        pytest.skip(f"{backend} declares no pitfall lists this can read")

    broken = []
    for name, source, ks in keys:
        for k in ks:
            m = re.match(r"^(.+?):(\d+)$", k)
            if not m:
                # Non-numeric keys (e.g. "_general:assemble.Signal") address
                # named fields rather than list indices; not checkable here.
                continue
            physics, idx = m.group(1).lstrip("_"), int(m.group(2))
            if idx >= SENTINEL_FLOOR:
                continue  # documented family-level sentinel
            if physics not in counts:
                # NOT reported as drift. A physics this reader cannot see is
                # far more likely to be a limit of the reader than a broken
                # key — that mistake accused 13 correct deal.II fixtures whose
                # physics the registry resolves perfectly well. Only an index
                # PAST A LIST WE CAN COUNT is evidence of drift.
                continue
            elif idx >= counts[physics]:
                broken.append(
                    f"{name} ({source}) -> {k}: {physics} declares only "
                    f"{counts[physics]} pitfalls (valid 0..{counts[physics]-1})")

    assert not broken, (
        f"{backend}: {len(broken)} fixture keys name a claim that does not "
        f"exist, so those fixtures defend nothing and the claims they were "
        f"meant to cover are uncovered:\n  " + "\n  ".join(broken[:20])
        + "\n\nThis is what index drift looks like. Moving or removing a "
          "pitfall renumbers the ones after it; renumber the fixtures in the "
          "same commit, and check the prose in each fixture's _comment too — "
          "Kratos found the machine-readable field fixed and the comment still "
          "quoting the old index.")


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_two_fixtures_claim_the_same_key(backend):
    """Double-counting one claim inflates coverage and hides an uncovered one.

    Exactly how FEBio's count was wrong: the self-consistency gate sat on
    index 0 alongside the real fixture for it, so one claim was credited twice
    while another had nothing.
    """
    keys = _fixture_keys(backend)
    if not keys:
        pytest.skip(f"{backend} has no fixtures with claim keys")

    owners: dict[str, list[str]] = {}
    for name, _src, ks in keys:
        for k in ks:
            owners.setdefault(k, []).append(name)
    clashes = {k: v for k, v in owners.items() if len(v) > 1}

    assert not clashes, (
        f"{backend}: {len(clashes)} claims are keyed by more than one fixture, "
        f"which credits them twice and leaves others uncounted:\n  "
        + "\n  ".join(f"{k}: {', '.join(v)}" for k, v in
                      list(clashes.items())[:12]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_fixture_records_what_it_did(backend):
    """The evidence record must not be a placeholder.

    `_comment` is where a fixture says which claim it defends, what was run, and
    what the software printed. It is the only part a person reads when deciding
    whether the fixture actually exercises the claim its key names — and that
    judgement cannot be automated, because a key can be syntactically valid and
    still point at the wrong claim. Kratos hit exactly that: when moving 53
    entries shifted 18 fixtures' indices, the machine-readable field was fixed
    while the prose still quoted the OLD claim. The prose is the half that
    misleads a human.

    So a fixture whose comment reads PLACEHOLDER is a fixture nobody can audit
    by reading. Measured when this was added: 63 FEniCSx and 25 deal.II
    fixtures, the two backends whose authoring runs were interrupted most often;
    every other backend was already at zero, which is why this is worth holding
    rather than waiving.
    """
    fx = FIXTURES / backend
    if not fx.is_dir():
        pytest.skip(f"{backend} has no fixtures")

    empty = []
    for d in sorted(fx.iterdir()):
        manifest = d / "fixture.json"
        if not manifest.is_file():
            continue
        try:
            spec = json.loads(manifest.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        comment = str(spec.get("_comment", "")).strip()
        if not comment:
            empty.append(f"{d.name}: no _comment")
        elif comment.upper().startswith("PLACEHOLDER"):
            empty.append(f"{d.name}: _comment is PLACEHOLDER")
        elif len(comment) < 40:
            empty.append(f"{d.name}: _comment too short to be a record "
                         f"({len(comment)} chars)")

    assert not empty, (
        f"{backend}: {len(empty)} fixtures do not record what they did, so "
        f"nobody can tell by reading whether they defend the claim their key "
        f"names:\n  " + "\n  ".join(empty[:20])
        + (f"\n  ... and {len(empty) - 20} more" if len(empty) > 20 else "")
        + "\n\nState the claim in one line, what was run, and what the software "
          "printed. Measured numbers belong here — the fixture is exactly "
          "where they are allowed, unlike the knowledge text.")
