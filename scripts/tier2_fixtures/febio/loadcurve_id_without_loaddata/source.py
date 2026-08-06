"""Tier-2: lc="1" with no <LoadData> reads SUCCESS and then solves nothing.

Verifies febio::linear_elasticity#4. This is the shape that defeats a
wrapper checking only the reader line: `Reading file ...SUCCESS!` is
printed, and the failure — `Invalid load curve ID` followed by
`Model initialization failed` — arrives afterwards.

The fixture asserts the reader really did say SUCCESS on the broken
deck, so the misleading half is pinned and not just the error.

MUTATION CONTROL. T2_MUTATE=1 keeps the <LoadData> section in the
"wrong" slot — the pathology removed. The load-curve id resolves, the
run terminates normally and
'missing_load_controller=reproduced' is no longer printed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

def main() -> int:
    if MUTATE:
        print("mutation=the_wrong_slot_keeps_its_loaddata_section")
    return L.init_error(
        "missing_load_controller",
        wrong=L.solid_deck() if MUTATE
        else L.solid_deck(loaddata=""),
        right=L.solid_deck(),
        message="Invalid load curve ID",
        also=("Model initialization failed",))


if __name__ == "__main__":
    sys.exit(main())
