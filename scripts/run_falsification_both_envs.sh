#!/usr/bin/env bash
# Run the pitfall-falsification gate in BOTH the project .venv
# (skfem/kratos/ngsolve) AND the ofa-fenicsx conda env (dolfinx).
#
# Why: each backend's falsifications run in the env where that
# backend is importable. The test file uses skipUnless decorators
# so each invocation skips the backends it can't see. Combined,
# every falsification is verified live exactly once.
#
# Usage:
#   bash scripts/run_falsification_both_envs.sh
#
# Exit codes:
#   0   both envs pass (skips are expected on per-env basis)
#   1   one or more envs report a failure (not just skipped)
#   2   one of the envs / pytest installs is unavailable
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

VENV_PY="$REPO/.venv/bin/python"
FENICSX_PY="$HOME/miniconda3/envs/ofa-fenicsx/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: $VENV_PY not found" >&2
    exit 2
fi
if [[ ! -x "$FENICSX_PY" ]]; then
    echo "ERROR: $FENICSX_PY not found" >&2
    exit 2
fi

echo "═════════════════════════════════════════════════════════"
echo "1/2  .venv  (skfem / kratos / ngsolve)"
echo "═════════════════════════════════════════════════════════"
"$VENV_PY" -m pytest tests/test_pitfall_falsification_live.py --tb=no
RC_VENV=$?
echo

echo "═════════════════════════════════════════════════════════"
echo "2/2  ofa-fenicsx  (dolfinx)"
echo "═════════════════════════════════════════════════════════"
"$FENICSX_PY" -m pytest tests/test_pitfall_falsification_live.py --tb=no
RC_FENICSX=$?
echo

echo "═════════════════════════════════════════════════════════"
echo "Combined result"
echo "═════════════════════════════════════════════════════════"
if [[ $RC_VENV -eq 0 ]] && [[ $RC_FENICSX -eq 0 ]]; then
    echo "✓ both envs pass — every falsification verified in its appropriate env"
    exit 0
elif [[ $RC_VENV -ne 0 ]] && [[ $RC_FENICSX -ne 0 ]]; then
    echo "✗ BOTH envs report failures (rc_venv=$RC_VENV rc_fenicsx=$RC_FENICSX)"
    exit 1
elif [[ $RC_VENV -ne 0 ]]; then
    echo "✗ .venv failed (rc=$RC_VENV); ofa-fenicsx passed"
    exit 1
else
    echo "✗ ofa-fenicsx failed (rc=$RC_FENICSX); .venv passed"
    exit 1
fi
