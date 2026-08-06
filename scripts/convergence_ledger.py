#!/usr/bin/env python3
"""Is the knowledge converged yet? Round over round, as a number.

The freeze rule is "keep running verification rounds until a round stops
finding things". That is only actionable if each round records what it checked
and what it found, so the next one can be compared against it. Otherwise
"converged" becomes a feeling, and the feeling arrives exactly when everyone is
tired of looking — which is the failure this whole effort exists to avoid.

WHAT A ROUND IS
A round is a full pass over a backend by an agent that EXECUTES claims rather
than reading them, followed by an independent adversarial audit of that pass.
Round 1 was the extraction/verification pass plus the seven audits.

THE METRIC MUST BE COMPUTED HERE, NOT REPORTED BY THE AGENT
An audit of the deal.II pass established why. That pass reported 566 of 817
claims executed — 69.3%. Its own two ledger scripts printed 449/817 = 55.0% and
184, and neither yielded 566. Worse than the disagreement, the metric was a
category error: it summed EXECUTION counts against a denominator of CLAIM
STRINGS. None of the 817 claim ids mentioned "essentials" though 55 leaves were
credited; none were step citations though 79 were credited; several entries were
experiment counts. The one defensible figure was the one the script itself
printed for pitfalls: 43/133 = 32.3%.

So a self-reported percentage is not comparable between backends, and a freeze
criterion built on incomparable numbers measures nothing. `surface_covered` is
therefore defined as ONE thing that this script computes from the tree:

    pitfall claims with an execution fixture behind them / pitfall claims total

Pitfalls are the right unit because they are what the paper sells, they are
countable without judgement, and a fixture is unambiguous evidence of execution.
An agent's own richer accounting is welcome in its report; it does not enter the
criterion.

THE METRIC
For each backend, per round:

  checked      — claims actually executed this round (not claims that exist)
  falsified    — executed claims that did not hold
  rate         — falsified / checked

and separately, because they are different failure modes:

  audit_defects       — errors the ADVERSARIAL PASS found in that round's own
                        corrections. This is the number that matters most: it
                        says whether the people fixing things are getting them
                        right. Round 1 was non-zero on every single backend.
  unmatchable_signals — Signal clauses quoting strings the software cannot emit
  contamination_hits  — measured answers, machine paths, campaign references
                        reachable by an agent
  surface_covered     — executed / total claims. A 2% rate on 5% of the surface
                        is not convergence, it is a small sample.

FREEZE CRITERION — all four, per backend:
  * rate               <= 0.02   (a round barely finds anything)
  * audit_defects      == 0      (the corrections are right)
  * unmatchable_signals== 0      (no invented diagnostics)
  * contamination_hits == 0      (nothing readable as an answer)
  * surface_covered    >= 0.80   (and it looked at most of the surface)

The last one is the guard against declaring victory by checking less. Round 1
found the damage concentrated in the fraction nobody executed: FEBio's 25
invented error messages and 11 non-existent material names were all in the 73%
its pass never touched. A round that checks a smaller surface will find fewer
problems and mean nothing by it.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "data" / "convergence"

THRESHOLDS = {
    "rate": 0.02,
    "audit_defects": 0,
    "unmatchable_signals": 0,
    "contamination_hits": 0,
    "surface_covered": 0.80,
}


def load_rounds() -> dict[int, dict]:
    rounds = {}
    if LEDGER.exists():
        for p in sorted(LEDGER.glob("round_*.json")):
            n = int(p.stem.split("_")[1])
            rounds[n] = json.loads(p.read_text())
    return rounds


def backend_verdict(entry: dict) -> tuple[bool, list[str]]:
    """Would this backend pass the freeze criterion on this round's numbers?"""
    fails = []
    checked = entry.get("checked") or 0
    total = entry.get("total_claims") or 0
    falsified = entry.get("falsified")
    if checked and falsified is not None:
        rate = falsified / checked
        if rate > THRESHOLDS["rate"]:
            fails.append(f"falsification rate {rate:.1%} > {THRESHOLDS['rate']:.0%}")
    else:
        fails.append("no falsification rate recorded")
    for key in ("audit_defects", "unmatchable_signals", "contamination_hits"):
        v = entry.get(key)
        if v is None:
            fails.append(f"{key} not measured")
        elif v > THRESHOLDS[key]:
            fails.append(f"{key}={v}")
    if total:
        cov = checked / total
        if cov < THRESHOLDS["surface_covered"]:
            fails.append(f"only {cov:.0%} of the surface executed "
                         f"(need {THRESHOLDS['surface_covered']:.0%})")
    else:
        fails.append("total claim count unknown, so coverage cannot be judged")
    return (not fails), fails


def main() -> int:
    rounds = load_rounds()
    if not rounds:
        print(f"No rounds recorded under {LEDGER}.")
        return 1

    latest = max(rounds)
    print(f"Rounds recorded: {', '.join(str(r) for r in sorted(rounds))}\n")

    hdr = (f"{'backend':<10} {'checked':>8} {'falsified':>10} {'rate':>7} "
           f"{'audit':>6} {'sig':>5} {'contam':>7} {'cover':>7}  verdict")
    print(hdr)
    print("-" * len(hdr))

    ready = []
    for be, e in sorted(rounds[latest].get("backends", {}).items()):
        checked = e.get("checked") or 0
        fals = e.get("falsified")
        total = e.get("total_claims") or 0
        rate = f"{fals / checked:.1%}" if (checked and fals is not None) else "?"
        cov = f"{checked / total:.0%}" if total else "?"
        ok, why = backend_verdict(e)
        ready.append(ok)
        print(f"{be:<10} {checked:>8} {str(fals):>10} {rate:>7} "
              f"{str(e.get('audit_defects')):>6} "
              f"{str(e.get('unmatchable_signals')):>5} "
              f"{str(e.get('contamination_hits')):>7} {cov:>7}  "
              + ("READY" if ok else "; ".join(why)[:60]))

    print()
    if all(ready) and ready:
        print(f"Round {latest}: every backend meets the freeze criterion. "
              f"Run one more round to confirm it finds nothing, then freeze.")
        return 0
    print(f"Round {latest}: NOT converged — "
          f"{sum(1 for r in ready if not r)}/{len(ready)} backends fail the "
          f"criterion. Another round is required.")
    return 2


# ── computed coverage, so no backend can self-report an incomparable number ──
def measured_pitfall_coverage(repo: Path | None = None) -> dict[str, dict]:
    """Pitfall claims and their tier-2 fixtures, counted from the tree.

    Deliberately crude and uniform. An agent that walks its own knowledge dict
    can produce a defensible-looking percentage from any denominator it likes —
    one pass reported 69.3% from a metric that mixed execution counts with claim
    strings, while its own scripts printed 55.0% and a bare 184. Comparability
    beats sophistication when the number decides whether we freeze.
    """
    import ast
    import json

    repo = repo or REPO
    backends_dir = repo / "src" / "backends"
    fixtures_dir = repo / "scripts" / "tier2_fixtures"

    out: dict[str, dict] = {}
    for be_dir in sorted(p for p in backends_dir.iterdir() if p.is_dir()):
        name = be_dir.name
        if name.startswith("_"):
            continue
        # Pitfall claims: every string carrying a `Signal:` clause, deduplicated
        # by value so a shared sub-dict attached by reference to many physics
        # entries counts once. A DUNE audit found a naive walk inflating a count
        # 111 -> 127 exactly that way.
        signals: set[str] = set()
        for py in be_dir.rglob("*.py"):
            try:
                tree = ast.parse(py.read_text(errors="ignore"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if "Signal:" in node.value:
                        signals.add(node.value)
        fx = fixtures_dir / name
        n_fx = 0
        if fx.is_dir():
            n_fx = len([d for d in fx.iterdir() if (d / "fixture.json").is_file()])
        out[name] = {
            "pitfall_claims": len(signals),
            "fixtures": n_fx,
            "covered": round(n_fx / len(signals), 4) if signals else None,
        }
    return out


def print_measured_coverage() -> None:
    rows = measured_pitfall_coverage()
    print(f"\n{'backend':<10} {'pitfalls':>9} {'fixtures':>9} {'covered':>8}")
    print("-" * 40)
    tp = tf = 0
    for be, d in sorted(rows.items()):
        tp += d["pitfall_claims"]
        tf += d["fixtures"]
        cov = "—" if d["covered"] is None else f"{d['covered']:.1%}"
        print(f"{be:<10} {d['pitfall_claims']:>9} {d['fixtures']:>9} {cov:>8}")
    print("-" * 40)
    print(f"{'TOTAL':<10} {tp:>9} {tf:>9} "
          f"{(tf / tp if tp else 0):>7.1%}")
    print("\nsurface_covered for the freeze criterion is this column — computed "
          "from the tree, identical in meaning for every backend.")


if __name__ == "__main__":
    import sys as _sys
    # `--coverage` prints the computed, cross-backend-comparable figure. A first
    # version appended this dispatch AFTER `raise SystemExit(main())`, so it was
    # dead code that silently did nothing — the same shape as the
    # `--write-results` flag that was parsed and never read.
    if "--coverage" in _sys.argv:
        print_measured_coverage()
        raise SystemExit(0)
    raise SystemExit(main())
