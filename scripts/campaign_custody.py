#!/usr/bin/env python3
"""The campaign itself must be pre-registered, not merely declared to be.

``campaign3_blind`` was not under version control.  A frozen, pre-registered
design that can be silently edited is not pre-registered, and once results exist
there is no way to show afterwards that the grader which graded is the grader
that was designed.  That is the first thing a hostile referee asks, it is
answerable in an hour now, and never afterwards.

Three commands:

    sync        copy the version-controlled campaign into the live directory and
                verify byte equality afterwards
    commit      hash the campaign state -- design, runner, grader, phase gate,
                builder, every task text and public spec -- plus the legacy
                plaintext builders, into a timestamped manifest that goes into
                git BEFORE any run
    verify      re-check the live directory against the committed manifest

Why the legacy builders are hashed rather than committed
--------------------------------------------------------
``build_problems.py`` and ``build_extra.py`` hold every hidden field for B1-B7
as a Python literal, so they cannot go into a repository.  But "cannot be
committed" must not mean "can be silently changed": their SHA-256 goes into the
manifest, so an edit is detectable without the content ever being published.
``build_coupled_v2.py`` needs neither treatment -- it holds no solution, because
the fields are drawn from a CSPRNG at build time and only the seed reaches the
sealed key.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN_SRC = REPO / "campaign3_blind"
CAMPAIGN_LIVE = Path("/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind")
MANIFEST = REPO / "data" / "blind_campaign_commitment.json"

# Tracked: everything that decides how a run is produced or graded.
TRACKED_FILES = ("DESIGN.md", "run_blind.py", "grade_blind.py", "phase.py",
                 "build_coupled_v2.py", "shield_keys.sh")
TRACKED_TREES = ("problems",)

# Held out of git because they hold hidden fields as literals, but hashed so an
# edit is still detectable.
HASH_ONLY = ("build_problems.py", "build_extra.py", "build_coupled.py")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _tracked(root: Path):
    for name in TRACKED_FILES:
        p = root / name
        if p.is_file():
            yield p
    for tree in TRACKED_TREES:
        d = root / tree
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix in (".txt", ".json"):
                    yield p


def cmd_sync(args):
    """Repo is authoritative; the live directory is a copy of it."""
    if not CAMPAIGN_LIVE.is_dir():
        print(f"no live campaign at {CAMPAIGN_LIVE}")
        return 1
    copied, diffs = [], []
    for src in _tracked(CAMPAIGN_SRC):
        rel = src.relative_to(CAMPAIGN_SRC)
        dst = CAMPAIGN_LIVE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not args.check:
            shutil.copy2(src, dst)
            copied.append(str(rel))
        if not dst.is_file() or sha256(src) != sha256(dst):
            diffs.append(str(rel))
    if args.check:
        print(f"{len(diffs)} file(s) differ between repo and live campaign")
        for d in diffs:
            print("   ", d)
        print("VERDICT:", "FAIL — the live campaign is not the committed one"
              if diffs else "PASS — live campaign matches the repo")
        return 1 if diffs else 0
    print(f"synced {len(copied)} file(s) to {CAMPAIGN_LIVE}")
    if diffs:
        print("WARNING: still differing after copy:", diffs)
        return 1
    return 0


def cmd_commit(args):
    entries = []
    for p in _tracked(CAMPAIGN_SRC):
        entries.append({"path": str(p.relative_to(CAMPAIGN_SRC)),
                        "bytes": p.stat().st_size, "sha256": sha256(p),
                        "in_git": True})
    for name in HASH_ONLY:
        p = CAMPAIGN_LIVE / name
        if not p.exists():
            continue
        import os
        restore = None
        if not os.access(p, os.R_OK):
            restore = p.stat().st_mode & 0o777
            os.chmod(p, 0o600)
        try:
            entries.append({"path": name, "bytes": p.stat().st_size,
                            "sha256": sha256(p), "in_git": False,
                            "why_not_in_git": "holds hidden fields as Python "
                                              "literals; hashed so an edit is "
                                              "detectable without publishing it"})
        finally:
            if restore is not None:
                os.chmod(p, restore)
    man = {
        "schema": "oasis-blind-campaign-commitment/1",
        "campaign": "campaign3_blind",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "Committed BEFORE any evaluation run. Proves that the grader "
                "which graded is the grader that was designed, and that the "
                "task texts were not edited after results were seen.",
        "entries": sorted(entries, key=lambda e: e["path"]),
    }
    man["manifest_sha256"] = hashlib.sha256(
        json.dumps(man["entries"], sort_keys=True).encode()).hexdigest()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man, indent=2))
    print(f"wrote {MANIFEST}")
    print(f"  {len(entries)} file(s); manifest_sha256 = {man['manifest_sha256']}")
    print("\nCommit this file to git BEFORE running the campaign.")
    return 0


def cmd_verify(args):
    if not MANIFEST.is_file():
        print(f"no campaign commitment at {MANIFEST}")
        return 1
    man = json.loads(MANIFEST.read_text())
    root = Path(args.root) if args.root else CAMPAIGN_SRC
    ok, bad, missing = [], [], []
    for e in man["entries"]:
        p = root / e["path"]
        if not p.is_file():
            p = CAMPAIGN_LIVE / e["path"]
        if not p.is_file():
            missing.append(e["path"])
            continue
        try:
            (ok if sha256(p) == e["sha256"] else bad).append(e["path"])
        except PermissionError:
            missing.append(f"{e['path']} (unreadable — sealed)")
    recomputed = hashlib.sha256(
        json.dumps(man["entries"], sort_keys=True).encode()).hexdigest()
    self_ok = recomputed == man.get("manifest_sha256")
    print(f"committed at   : {man.get('generated_utc')}")
    print(f"self-consistent: {self_ok}")
    print(f"matched        : {len(ok)}")
    print(f"CHANGED        : {bad or 'none'}")
    print(f"missing        : {missing or 'none'}")
    verdict = "PASS" if self_ok and not bad and not missing else "FAIL"
    print(f"VERDICT        : {verdict}")
    return 0 if verdict == "PASS" else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync")
    s.add_argument("--check", action="store_true",
                   help="report differences without copying")
    s.set_defaults(fn=cmd_sync)
    sub.add_parser("commit").set_defaults(fn=cmd_commit)
    v = sub.add_parser("verify")
    v.add_argument("--root", default=None)
    v.set_defaults(fn=cmd_verify)
    a = ap.parse_args()
    raise SystemExit(a.fn(a))


if __name__ == "__main__":
    main()
