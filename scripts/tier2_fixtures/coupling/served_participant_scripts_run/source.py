"""The nine participant scripts, taken out of what the TOOL returns, and run.

THE CLAIM UNDER TEST. `knowledge(topic='coupling')` ends with a section headed
"PER-BACKEND PARTICIPANT SCRIPTS — one call each, complete and runnable", and
each per-backend payload hands over a script under "COMPLETE PARTICIPANT SCRIPT
— copy verbatim, edit the marked block only". The module that serves them says
why they are files rather than string literals: "the file is the artefact that
gets executed in the test suite, so the text an agent is served and the text
that was proven to run cannot drift apart."

That reasoning is only sound if something actually executes the SERVED text. A
test that reads `data/coupling_participants/participant_x.py` off disk proves
the file runs; it proves nothing about what the tool returns, and the two are
separated by an extraction function, a payload template and nine call sites. So
this fixture goes through the registered `knowledge` tool, cuts the script out
of the fenced block in the reply, and runs THAT.

It also checks the launch text, because that is where the drift actually
happened. Four `.replace()` calls were meant to tell the four WRAPPER backends —
4C, FEBio, SPARTA, deal.II — that `command` runs a Python wrapper and the solver
binary goes into a constant inside the script. All four were silent no-ops: the
literals they searched for wrapped their lines at a different point than the
text did, so every wrapper backend served the generic "get the interpreter from
discover(query='list')" instead, and an agent following it verbatim would put a
compiled solver binary where a Python interpreter belongs. `str.replace` cannot
fail, so nothing reported it for as long as it was wrong.
"""
from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _couplinglib as L                                    # noqa: E402

# The order the served index lists them in.
BACKENDS = ["fenics", "fourc", "ngsolve", "skfem", "dune", "dealii",
            "febio", "kratos", "sparta"]

# The four whose participant is a Python WRAPPER around a separate binary.
WRAPPERS = {"fourc": "FOURC_BIN", "febio": "FEBIO", "sparta": "SPARTA",
            "dealii": "DEALII_EXE"}

# Backends whose shipped script runs standalone with no staged data files.
# SPARTA's deck opens surf / species / vss files that `couple` does not copy —
# its payload says so in capitals — and Kratos is not importable here, which is
# exactly what the starred row in the sides table records.
NOT_EXECUTED = {"sparta": "its deck opens surf/species/vss files that must be "
                          "staged into work_dir first",
                "kratos": "KratosMultiphysics does not import in OASiS's "
                          "interpreter on this install"}


def served(solver: str) -> str:
    return L.call_tool("knowledge", {"topic": "coupling", "solver": solver})


def extract(payload: str) -> str:
    """The fenced python block the payload calls the complete script."""
    blocks = re.findall(r"```python\n(.*?)```", payload, re.S)
    if not blocks:
        raise AssertionError("no ```python fenced block in the served payload")
    return max(blocks, key=len)


def run_extracted(backend: str, text: str) -> None:
    """Write the SERVED text into a fresh work_dir and run it there."""
    edits = L.BINARY_EDITS[backend]() if backend in L.BINARY_EDITS else {}
    if edits:
        text = L.edit(text, edits)
    wd = L.workroot(f"served_{backend}")
    script = wd / "participant_left.py"
    script.write_text(text)
    r = subprocess.run([L.interpreter(backend), script.name], cwd=str(wd),
                       capture_output=True, text=True, timeout=900)
    ep = wd / "exports.json"
    if not L.check(ep.is_file(), f"{backend}_served_script_wrote_no_exports",
                   f"rc={r.returncode} stderr={r.stderr[-400:]}"):
        return
    d = json.loads(ep.read_text())
    for key in ("field_name", "coordinates", "values"):
        L.check(key in d, f"{backend}_export_missing_{key}", "")
    n = len(d.get("coordinates", []))
    print(f"{backend}_served_script_interface_points={n}")
    L.check(n > 0, f"{backend}_empty_interface",
            "an empty interface set gives a well-behaved solution of the "
            "wrong problem")
    L.check(len(d.get("values", [])) == n, f"{backend}_values_length_mismatch",
            f"{len(d.get('values', []))} values for {n} points")
    L.check(all(math.isfinite(float(v)) for v in d.get("values", [])),
            f"{backend}_non_finite_values", "")


def body() -> None:
    verbatim = 0
    executed = 0
    for backend in BACKENDS:
        payload = served(backend)
        print(f"{backend}_payload_chars={len(payload)}")
        L.check(len(payload) > 2000, f"{backend}_payload_too_thin",
                f"{len(payload)} chars — a payload this short cannot contain a "
                f"complete participant script")

        # ── served VERBATIM: the shipped file, byte for byte ──────────────
        text = extract(payload)
        shipped = L.PARTICIPANT_DIR / f"participant_{backend}.py"
        L.check(shipped.is_file(), f"{backend}_no_shipped_script", str(shipped))
        if shipped.is_file():
            same = text == shipped.read_text()
            print(f"{backend}_served_verbatim={same}")
            if L.check(same, f"{backend}_served_text_differs_from_the_file",
                       "the served payload and the file the test suite executes "
                       "have drifted apart, which is the exact failure the "
                       "file-not-literal design was chosen to prevent"):
                verbatim += 1

        # ── it is a complete Python program, not a fragment ───────────────
        try:
            ast.parse(text)
            print(f"{backend}_parses=True")
        except SyntaxError as e:
            L.check(False, f"{backend}_served_script_does_not_parse", str(e))
            continue

        # ── the launch text has to be right for THIS backend ──────────────
        if backend in WRAPPERS:
            const = WRAPPERS[backend]
            says_wrapper = "runs the WRAPPER" in payload
            names_const = f"`{const}`" in payload
            print(f"{backend}_launch_says_wrapper={says_wrapper}")
            print(f"{backend}_launch_names_{const}={names_const}")
            L.check(says_wrapper, f"{backend}_launch_text_is_the_generic_one",
                    "this backend's participant is a Python wrapper, so "
                    "`command` must run Python and not the solver binary — the "
                    "four dead `.replace()` calls dropped exactly this "
                    "sentence for all four wrapper backends")
            L.check(names_const, f"{backend}_launch_does_not_name_{const}",
                    f"the payload must say the binary path goes into {const} "
                    f"inside the script")
            L.check(const in text, f"{backend}_script_has_no_{const}",
                    f"the served script does not define {const}, so the launch "
                    f"text points at a constant that does not exist")
        else:
            L.check("discover(query='list')" in payload,
                    f"{backend}_launch_does_not_send_you_to_discover",
                    "a non-wrapper backend's `command` is that backend's own "
                    "interpreter, which the payload must resolve through "
                    "discover rather than hard-coding")

        # ── and it RUNS ───────────────────────────────────────────────────
        if backend in NOT_EXECUTED:
            print(f"{backend}_not_executed_because: {NOT_EXECUTED[backend]}")
            continue
        L.require_available(backend)
        if backend == "dealii":
            L.dealii_exe()
        run_extracted(backend, text)
        executed += 1

    print(f"payloads_served={len(BACKENDS)}")
    print(f"scripts_served_verbatim={verbatim}")
    print(f"scripts_executed={executed}")
    L.check(verbatim == len(BACKENDS), "not_every_script_served_verbatim",
            f"{verbatim} of {len(BACKENDS)}")


L.main(body)
