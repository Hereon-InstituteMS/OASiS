"""Every input key OASiS names must exist in the backend that would consume it.

A wrong warning misleads a user. A wrong KEY produces a deck the solver refuses
to parse, for every user, every time — and the user has no way to tell whether
the key is wrong or their problem is. Two of these were live in the corpus when
this gate was written:

  * `SOUNDSPEED` and `SMOOTHING_LENGTH`, served as required 4C SPH material
    parameters beside the real `DYN_VISCOSITY` and `BULK_MODULUS`, one of them
    carrying the numeric tuning rule "c >= 10 * v_max". Neither appears in any
    file of 4C's source or in any of its 2171 input decks.
  * `AREA0`, in the arterial-network DECK TEMPLATE — the YAML actually written
    into the input file. 4C's `MAT_CNST_ART` has no such parameter; the
    reference geometry comes from `DIAM`. The same generator already carried a
    warning saying exactly that, and the template was never updated.

WHY THIS IS A BASELINE AND NOT A HARD FAIL
------------------------------------------
The corpus has candidates under active triage, and a hard fail would block every
unrelated change until the last one is resolved. So the gate ratchets: it fails
on any key NOT in the recorded baseline, which catches a newly invented key the
moment it is written, while the known set is worked down.

The baseline is deliberately a readable JSON file listing every key by name, not
a count and not a hash. An allowlist you cannot read is a place for things to
hide. It is expected to shrink to empty; entries leave it as they are fixed or
reclassified as correct negatives.

WHAT THIS GATE REFUSES TO ASSERT
--------------------------------
It renders no verdict at all for a backend whose corpus it cannot see. Kratos
does not import on this host, and only 3 of its ~40 applications are installed,
so a key from the DEM or Poromechanics application is invisible here no matter
how real it is — reported UNVERIFIABLE, never "invented". An audit that judged
Kratos against scipy once reported 121 of 139 keys as fabricated; every one was
an artefact of not being able to see the software.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

BASELINE_PATH = REPO / "scripts" / "scan_results" / "named_key_baseline.json"
BACKENDS = ["fourc", "fenics", "dealii", "ngsolve", "skfem", "kratos", "dune",
            "febio", "sparta"]


def _baseline() -> dict:
    if not BASELINE_PATH.is_file():
        return {}
    return json.loads(BASELINE_PATH.read_text()).get("baseline", {})


@pytest.mark.parametrize("backend", BACKENDS)
def test_named_keys_resolve_in_the_backend(backend: str) -> None:
    from audit_named_input_keys import audit

    res = audit(backend)

    if res["status"] in ("UNKNOWN", "NO_ENTRIES"):
        pytest.skip(f"no verdict possible: {res['reason']}")
    if res["status"] == "PARTIAL_CORPUS":
        pytest.skip(
            f"{backend}: corpus is a known subset, so a key that does not "
            f"resolve is not thereby shown to be invented — "
            f"{res['corpus_partial']}. "
            f"{len(res.get('unverifiable', []))} keys unverifiable here.")

    known = set(_baseline().get(backend, []))
    found = {u["key"] for u in res["unresolved"]}
    new = sorted(found - known)

    assert not new, (
        f"{backend}: {len(new)} input key(s) named by the knowledge do not "
        f"exist anywhere in the backend's corpus, and are not in the triage "
        f"baseline:\n"
        + "\n".join(
            f"    {k}  <- {next(u['signal'] for u in res['unresolved'] if u['key'] == k)}"
            for k in new)
        + f"\n\nEach is either an invented key — a user following it writes a "
          f"deck the solver rejects — or a correct negative whose phrasing the "
          f"absence matcher did not recognise. Check it against the backend's "
          f"source AND its example inputs (grep -r -a -F; the -a matters, "
          f"without it grep silently skips the compiled libraries where many "
          f"backends register their variables). Fix real ones; add correct "
          f"negatives to {BASELINE_PATH.name} with a note saying why."
    )


def test_baseline_only_shrinks() -> None:
    """Keys resolved since the baseline was taken must be removed from it.

    Without this the baseline becomes a graveyard: a key gets fixed, its entry
    stays, and the entry silently re-permits the same fabrication if it is
    reintroduced later. The list has to track reality in both directions.
    """
    from audit_named_input_keys import audit

    stale: list[str] = []
    for backend, keys in _baseline().items():
        res = audit(backend)
        if res["status"] not in ("OK", "CANDIDATES"):
            continue
        still = {u["key"] for u in res["unresolved"]}
        for k in keys:
            if k not in still:
                stale.append(f"{backend}:{k}")
    assert not stale, (
        "these keys now resolve and must be dropped from the baseline, so it "
        "cannot re-permit them later:\n    " + "\n    ".join(stale))
