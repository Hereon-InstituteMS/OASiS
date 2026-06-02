"""Post-mortem schema validation.

Every fix that ships a pitfall-DB entry must also ship a post-mortem
under data/postmortems/<id>.json (task #46). This test refuses to let
those records drift away from the schema in `_schema.json`.

We avoid a hard dependency on jsonschema by validating the small set of
required fields explicitly — both jsonschema and ajv are heavyweight
adds for one schema file.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path


POSTMORTEMS = Path(__file__).resolve().parent.parent / "data" / "postmortems"

ALLOWED_BACKENDS = {"kratos", "fenics", "fourc", "ngsolve", "skfem",
                    "dealii", "dune", "febio"}
ALLOWED_CATEGORIES = {"Syntax", "Physics", "Numerical", "API", "Integration"}

REQUIRED_FIELDS = (
    "id", "date", "backend", "physics", "surface_symptom",
    "root_cause", "categories", "pitfall_db_entries",
    "agent_detection_after_fix", "source_refs",
)


class TestPostmortemSchema(unittest.TestCase):

    def test_schema_file_exists(self):
        self.assertTrue((POSTMORTEMS / "_schema.json").exists(),
                        "data/postmortems/_schema.json is the canonical "
                        "format definition and must remain present")

    def test_each_postmortem_has_required_fields(self):
        failures = []
        for path in sorted(POSTMORTEMS.glob("*.json")):
            if path.name.startswith("_"):
                continue
            with path.open() as fp:
                doc = json.load(fp)
            for field in REQUIRED_FIELDS:
                if field not in doc:
                    failures.append(f"{path.name}: missing field '{field}'")
            if "backend" in doc and doc["backend"] not in ALLOWED_BACKENDS:
                failures.append(f"{path.name}: backend "
                                f"{doc['backend']!r} not in allow-list")
            if "categories" in doc:
                bad = [c for c in doc["categories"]
                       if c not in ALLOWED_CATEGORIES]
                if bad:
                    failures.append(f"{path.name}: categories {bad} "
                                    f"outside Table-1 set")
            entries = doc.get("pitfall_db_entries", [])
            if not entries:
                failures.append(
                    f"{path.name}: pitfall_db_entries is empty — a "
                    f"post-mortem without a pitfall entry defeats "
                    f"the discipline (task #46)")
            for entry in entries:
                stripped = entry.lstrip()
                if not any(stripped.startswith(f"[{c}]")
                           for c in ALLOWED_CATEGORIES):
                    failures.append(
                        f"{path.name}: pitfall entry missing "
                        f"[Category] prefix: {entry[:60]!r}")
        if failures:
            self.fail(f"{len(failures)} post-mortem issues:\n  - "
                      + "\n  - ".join(failures))


if __name__ == "__main__":
    unittest.main()
