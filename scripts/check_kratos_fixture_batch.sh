#!/usr/bin/env bash
# Last step of EVERY tier-2 fixture batch. Not optional, and not replaceable by
# checking the fixtures you just wrote.
#
# WHY THIS EXISTS
# ---------------
# Fixtures here are produced by throwaway generator scripts. A generator writes
# a whole batch. If you later find a defect in ONE fixture and correct it in a
# follow-up script, then edit and re-run the ORIGINAL generator to change a
# DIFFERENT fixture in the same batch, the correction is silently overwritten by
# the superseded version — and it gets committed in that state.
#
# That happened. poisson_convectiondiffusionsettings_on_processinfo was
# corrected in a follow-up script, then reverted by a re-run of the batch
# generator, and committed failing. Every fixture in that batch passed when it
# was written, so per-fixture verification could not see it. Only running the
# COMPLETE set caught it.
#
# The rule: verify the whole set, not the diff.
#
# Usage:
#   KRATOS_PYTHON=/path/to/python scripts/check_kratos_fixture_batch.sh
#
# Exits non-zero if any kratos fixture fails other than the two known
# host-path-dependent ones (see KNOWN_HOST_PATH below).

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTEST_PY="${PYTEST_PY:-/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python}"

# These two scan a hard-coded KratosMultiphysics/.libs path under a checkout
# that does not exist on every host. They are host-path-dependent, not wrong.
KNOWN_HOST_PATH="cosimulation_accelerator_mapper_names geomechanics_cl_naming"

if [ -z "${KRATOS_PYTHON:-}" ]; then
    echo "KRATOS_PYTHON is not set; a fixture run without a working Kratos" \
         "proves nothing." >&2
    exit 2
fi

echo "== 1/3  every kratos fixture, whole set =="
out="$(cd "$REPO" && "$PYTEST_PY" - <<'PY'
import json, os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent if False else Path.cwd()
sys.path.insert(0, str(REPO / "scripts"))
sys.argv = ["run_tier2_fixtures.py"]
import run_tier2_fixtures as R
for d in sorted((REPO / "scripts" / "tier2_fixtures" / "kratos").iterdir()):
    if not d.is_dir():
        continue
    meta = json.loads((d / "fixture.json").read_text())
    r = R._eval_fixture(d, meta)
    print(f"{r.status}\t{d.name}")
PY
)" || true

echo "$out" | awk -F'\t' '$1!="passed"{print "  " $1 "  " $2}'
bad=0
while IFS=$'\t' read -r status name; do
    [ "$status" = "passed" ] && continue
    [ -z "${name:-}" ] && continue
    case " $KNOWN_HOST_PATH " in
        *" $name "*) echo "  (known host-path failure, ignored: $name)" ;;
        *) bad=$((bad + 1)) ;;
    esac
done <<< "$out"

echo "== 2/3  a fixture must not pass with the backend absent =="
(cd "$REPO" && "$PYTEST_PY" -m pytest tests/test_fixtures_cannot_pass_vacuously.py -q) || bad=$((bad + 1))

echo "== 3/3  knowledge-catalog gates =="
(cd "$REPO" && "$PYTEST_PY" -m pytest \
    tests/test_pitfall_signal_coverage.py \
    tests/test_pitfall_categories.py \
    tests/test_catalog_pitfalls_nonempty.py \
    tests/test_no_orphan_pitfalls.py -q) || bad=$((bad + 1))

if [ "$bad" -ne 0 ]; then
    echo "BATCH NOT CLEAN: $bad unexpected failure(s). Do not commit." >&2
    exit 1
fi
echo "batch clean."
