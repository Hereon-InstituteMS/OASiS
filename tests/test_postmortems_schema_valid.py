"""Regression: every data/postmortems/*.json validates against
the canonical schema in data/postmortems/_schema.json.

Catches schema-drift bugs like the two surfaced 2026-06-02:

  - cross-backend-table1-promotion-ngsolve-poc.json — used a
    descriptive backend name 'ngsolve (proof of concept; pattern
    applies to ...)' instead of the bare enum 'ngsolve'.
  - dealii-cheap-bucket-closed-and-verify-determinism-fix.json —
    used 'dealii (+ harness)' instead of 'dealii'.

Both also used a `Process` category that isn't in the schema
enum (Syntax|Physics|Numerical|API|Integration), and prefixed
pitfall_db_entries with `[Process]` correspondingly. The schema
gate caught all 6 violations once invoked.

This pytest gate keeps the post-mortem corpus structurally
canonical so the retrieval tool (`_load_matching_postmortems`)
can rely on the documented contract.

Skipped under-test files that begin with `_` (schema itself,
`_falsifiability.json` index, etc.) — those are not records.
"""
from __future__ import annotations

import glob
import json
import os
import unittest
from pathlib import Path

import jsonschema

_REPO = Path(__file__).resolve().parent.parent


class TestPostmortemsSchemaValid(unittest.TestCase):
    def test_all_postmortems_validate(self) -> None:
        schema_path = _REPO / "data" / "postmortems" / "_schema.json"
        self.assertTrue(schema_path.is_file(),
                        f"schema not found at {schema_path}")
        with open(schema_path) as f:
            schema = json.load(f)

        files = sorted(glob.glob(
            str(_REPO / "data" / "postmortems" / "*.json")))
        files = [f for f in files
                 if not os.path.basename(f).startswith("_")]

        self.assertGreater(
            len(files), 0,
            "no postmortem records found — data/postmortems/ "
            "should contain *.json record files plus _schema.json")

        all_errors: list[tuple[str, list[str]]] = []
        for fname in files:
            with open(fname) as f:
                try:
                    doc = json.load(f)
                except json.JSONDecodeError as e:
                    all_errors.append(
                        (os.path.basename(fname), [f"JSON: {e}"]))
                    continue
            errors = [
                f"{'.'.join(str(p) for p in e.path)}: {e.message[:200]}"
                for e in jsonschema.Draft7Validator(schema).iter_errors(doc)
            ]
            if errors:
                all_errors.append((os.path.basename(fname), errors))

        if all_errors:
            msg = (
                "Post-mortem schema validation failed:\n"
                + "\n".join(
                    f"  {fname}:\n    " + "\n    ".join(errs)
                    for fname, errs in all_errors)
                + "\nFix the record OR update data/postmortems/"
                "_schema.json (the latter only if the canonical "
                "contract is actually changing — bump the schema "
                "version + update the retrieval tool in lock-step).")
            self.fail(msg)


if __name__ == "__main__":
    unittest.main()
