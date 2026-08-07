#!/usr/bin/env python3
"""Consume audit_leaks.py output; invalidate tainted runs; extend the rerun queue.

Fixed here: the previous version did ``d["outcome_pre_leak_audit"] = d["outcome"]``
on a ledger that may not have an ``outcome`` key at all.  Campaign-3 ledgers are
exactly that case — ``run_blind.py`` writes ``outcome`` only when the run failed
with an infrastructure error, because grading is offline — so the first tainted
campaign-3 run would have raised ``KeyError`` and stopped the sweep before it
invalidated anything.  A leak-invalidation tool that crashes on the campaign it
was extended to cover leaves the tainted runs marked clean.

It also used ``pathlib.Path(t["run"]).parent.parent``, which assumed the audit
reported a trajectory path.  The audit now reports the ledger explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def invalidate(report: dict, queue: Path | None = None, dry_run: bool = False):
    done, skipped = [], []
    for t in report.get("tainted", []):
        lp = Path(t.get("ledger") or (Path(t["run"]) / "ledger.json"))
        if not lp.exists():
            skipped.append((str(lp), "no ledger"))
            continue
        d = json.loads(lp.read_text())
        if d.get("outcome") == "INVALID_INFRA":
            skipped.append((str(lp), "already invalid"))
            continue
        # .get, not [], and only record a prior outcome if one existed.
        prior = d.get("outcome")
        if prior is not None:
            d["outcome_pre_leak_audit"] = prior
        d["outcome"] = "INVALID_INFRA"
        d["error"] = f"LEAK(auto): {'; '.join(t['findings'])}"
        d["leak_findings"] = t["findings"]
        if not dry_run:
            lp.write_text(json.dumps(d, indent=2))
            if queue:
                with open(queue, "a") as q:
                    q.write(f"{lp.parent.name}\n")
        done.append((lp.parent.name, t["findings"]))
    return done, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--queue", default="rerun_queue.txt")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rep = json.loads(Path(a.report).read_text())
    done, skipped = invalidate(rep, Path(a.queue) if a.queue else None,
                               a.dry_run)
    for name, f in done:
        print(f"{'WOULD INVALIDATE' if a.dry_run else 'AUTO-INVALIDATED'} "
              f"{name}: {f}")
    for p, why in skipped:
        print(f"skipped {p}: {why}")
    print(f"{len(done)} invalidated, {len(skipped)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
