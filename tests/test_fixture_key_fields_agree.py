"""A fixture that carries both `covers` and `physics`/`pitfall_index` must not
have them disagree.

WHY THIS EXISTS
---------------
Two things in this tree resolve a fixture to the claim it defends, and until
now they used different fields:

  * `tests/test_fixture_keys_point_at_real_claims.py` prefers `covers` when a
    fixture declares it, falling back to `physics:pitfall_index`.
  * `scripts/run_tier2_fixtures.py` builds `<backend>::<physics>::<index>` and
    never looks at `covers`.

Three Kratos DEM fixtures had a correct `covers` (`dem::13`, `dem::14`,
`dem::15`) and a stale `pitfall_index` left behind at 1, 2 and 3 — the values
from before the DEM pitfall list grew. The consequences pointed in opposite
directions and neither was obviously wrong:

  * The gate saw no clash, because `covers` is distinct. It passed.
  * The runner saw `dem::1`, `dem::2` and `dem::3` already taken by the
    fixtures that legitimately own them, reported KEY COLLISION, and marked
    all three FAILED — in a results file that was about to be committed as the
    execution record.

So one check said the corpus was fine and the other said three fixtures were
broken, about the same property, and both were reading a field the other
ignored. The fixtures themselves were fine; the two key definitions were not
the same definition.

This test makes the disagreement impossible rather than adjudicating it. If a
fixture declares a single `covers` entry of the form `<physics>::<index>`, then
its `physics` and `pitfall_index` must say the same thing.

WHAT IT DOES NOT DO
-------------------
It does not require `covers` to exist — most fixtures carry only the pair, and
that is fine. It does not check multi-entry `covers` against the pair, because
a fixture that legitimately defends several claims cannot encode all of them in
one `pitfall_index`; those are the case the pair cannot express and `covers`
exists for.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "scripts" / "tier2_fixtures"
_ENTRY = re.compile(r"([A-Za-z0-9_]+)::(\d+)$")


def _disagreements() -> list[str]:
    out = []
    for fj in sorted(FIXTURES.glob("*/*/fixture.json")):
        try:
            spec = json.loads(fj.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        covers = spec.get("covers")
        physics, idx = spec.get("physics"), spec.get("pitfall_index")
        if not (isinstance(covers, list) and len(covers) == 1):
            continue
        if physics is None or idx is None:
            continue
        m = _ENTRY.fullmatch(str(covers[0]).strip())
        if not m:
            continue
        # A leading underscore is a deliberate convention, not a mismatch: the
        # catalog key is `_auxiliary_overview` and it is exposed to users as
        # `auxiliary_overview`. test_fixture_keys_point_at_real_claims strips
        # it in four places; strip it here too, or this gate reports seven
        # Kratos fixtures as broken for spelling their own physics correctly.
        c_physics, c_idx = m.group(1).lstrip("_"), int(m.group(2))
        if c_physics != str(physics).lstrip("_") or c_idx != int(idx):
            out.append(
                f"{fj.parent.parent.name}/{fj.parent.name}: covers says "
                f"{covers[0]}, physics/pitfall_index says {physics}::{idx}")
    return out


def test_covers_and_pitfall_index_say_the_same_thing() -> None:
    bad = _disagreements()
    assert not bad, (
        f"{len(bad)} fixture(s) name one claim in `covers` and a different one "
        f"in `physics`/`pitfall_index`. The claim-coverage gate reads the "
        f"first and the fixture runner reads the second, so the corpus looks "
        f"correct to one and broken to the other:\n    "
        + "\n    ".join(bad)
        + "\n\nUpdate `pitfall_index` to match `covers`. Renumbering a pitfall "
          "list moves every index after it; both fields have to move together."
    )
