"""The decoy keys used to exercise the leak auditor must never be real answers.

WHY THIS EXISTS
---------------
`tests/fixtures/blind_leaks/` holds files named `B1_key.json`, `B2_key.json`
and `D3_key.json`, each carrying an `exact_solution` and a `source_term`. Those
IDs are live blind-campaign problems, and the codes match too (B1 NGSolve, B2
deal.II, D3 FEniCSx+Kratos). They exist so the leak auditor has something
shaped like a key to detect.

They are decoys — checked, and none of their source terms is the one its live
task states. But nothing enforced that. The campaign generator draws a fresh
problem each run, and the sealed keys deliberately live outside the repository;
if a redraw ever landed on the same manufactured solution as a decoy, or if
someone refreshed these fixtures by copying a real key "so the test is
realistic", the exact answer to a live blind problem would be sitting in a
public repository, in a file whose name says it is the key.

That is the one failure this campaign cannot absorb. The whole design rests on
the agent never seeing the exact solution — the task text gives it the domain,
the coefficients and the right-hand side, and nothing else. An answer reachable
by grepping the repo the agent runs inside is not a blind evaluation, and no
result measured afterwards would be worth reporting.

WHAT THIS CHECKS
----------------
For every decoy that shares an ID with a live problem, the decoy's source term
must NOT appear in that problem's task text, and the task's stated source term
must not appear in the decoy. Source terms are compared rather than exact
solutions because the task text publishes f and withholds u — so f is the field
the two can legitimately be compared on, and a decoy whose f matches the live f
is a decoy describing the live problem.

It deliberately does not require the decoys to be absent. Testing a leak
detector needs something to detect, and deleting them would leave the auditor
unexercised — which is how the auditing gap this repository keeps finding gets
created in the first place.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DECOYS = REPO / "tests" / "fixtures" / "blind_leaks"
PROBLEMS = REPO / "campaign3_blind" / "problems"

# Compare on a substantial prefix: expressions are long, and an accidental
# collision on a short head (e.g. "2*x") means nothing.
_MIN_MATCH = 40


def _terms(value) -> list[str]:
    if isinstance(value, dict):
        return [str(v) for v in value.values() if v]
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [str(value)] if value else []


def test_no_decoy_key_carries_a_live_problems_answer() -> None:
    if not DECOYS.is_dir():
        return
    leaks = []
    for kf in sorted(DECOYS.glob("*_key.json")):
        tid = kf.name.split("_")[0]
        task = PROBLEMS / tid / "task.txt"
        if not task.is_file():
            continue                      # decoy for a problem that is not live
        text = task.read_text(encoding="utf-8")
        try:
            key = json.loads(kf.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        for src in _terms(key.get("source_term")):
            probe = src.strip()[:_MIN_MATCH]
            if len(probe) >= _MIN_MATCH and probe in text:
                leaks.append(
                    f"{kf.name}: its source term appears in the LIVE task "
                    f"{tid}/task.txt — this decoy describes the live problem, "
                    f"so its exact_solution is that problem's answer")

        for m in re.finditer(r"SOURCE TERM[^\n]*", text):
            live = m.group(0)
            for src in _terms(key.get("source_term")):
                probe = src.strip()[:_MIN_MATCH]
                if len(probe) >= _MIN_MATCH and probe in live:
                    leaks.append(
                        f"{kf.name}: the live task's stated source term "
                        f"contains this decoy's source term")

    assert not leaks, (
        "a decoy key in tests/fixtures/blind_leaks/ matches a LIVE blind "
        "problem, so the exact solution to a problem an agent is about to be "
        "evaluated on is committed to this repository:\n    "
        + "\n    ".join(sorted(set(leaks)))
        + "\n\nRe-draw the decoy. Never refresh these files by copying a real "
          "key: the sealed keys stay outside the repository on purpose, and a "
          "campaign whose answers are greppable from the agent's own checkout "
          "is not blind."
    )
