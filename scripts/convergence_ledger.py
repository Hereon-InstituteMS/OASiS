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


if __name__ == "__main__":
    raise SystemExit(main())
