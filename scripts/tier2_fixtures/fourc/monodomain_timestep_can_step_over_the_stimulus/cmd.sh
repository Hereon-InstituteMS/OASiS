#!/bin/bash
# Tier-2 for fourc::cardiac_monodomain#0 — a too-large time step does not blur the
# upstroke, it deletes the stimulus.
#
# Claim: "dt too large gives MISSED ACTIVATION (a stimulus that should fire
#        produces no AP)".
# Observed, on upstream scatra_myocard_FHN_material, whose Neumann stimulus is a
# multifunction that is on only over t in [0,2], [350,352], [850,852] -- three
# 2-unit windows:
#   dt = 0.1 (upstream default)  -> phi = 0.778036, result test CORRECT
#   dt = 1.0                     -> phi = 0.798323, silently 2.6% off
#   dt = 5.0                     -> phi = exactly 0.0
# At dt = 5 every sample of the stimulus curve falls outside every window, so the
# source term is never applied and there is no action potential at all. Nothing is
# printed: the run converges, exits through the result test, and reports a
# perfectly clean solve of a problem with no stimulus in it.
#
# RESULTSEVERY/RESTARTEVERY are pushed out of range so the arms do not spend their
# time writing output; that does not touch the solution.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream scatra_myocard_FHN_material.4C.yaml) || exit 3
IFPACK=$(upstream xml/preconditioner/ifpack.xml) || exit 3
cd "$TMP" || exit 3
# IFPACK_XML_FILE is a relative path resolved against the WORKING DIRECTORY, not
# against the deck; without this the run dies in Teuchos with status 134.
mkdir -p xml/preconditioner && cp "$IFPACK" xml/preconditioner/
grep -q '  RESULTSEVERY: 20' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '  MAXTIME: 900' "$BASE"     || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q 'TIMES: \[0, 2, 350, 352, 850, 852, 500000\]' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

COARSE_DT=5.0

sed 's/  RESULTSEVERY: 20/  RESULTSEVERY: 1000000/; s/  RESTARTEVERY: 20/  RESTARTEVERY: 1000000/' \
    "$BASE" > quiet.yaml
sed 's/  MAXTIME: 900/  MAXTIME: 900\n  TIMESTEP: 0.1/'          quiet.yaml > dt01.yaml
sed 's/  MAXTIME: 900/  MAXTIME: 900\n  TIMESTEP: 1.0/'          quiet.yaml > dt1.yaml
sed "s/  MAXTIME: 900/  MAXTIME: 900\n  TIMESTEP: $COARSE_DT/"   quiet.yaml > dtbig.yaml

probe DT01  dt01.yaml
probe DT1   dt1.yaml
probe DTBIG dtbig.yaml

echo "DT01_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/DT01.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/DT01.log"
grep -m1 -F "phi      at node   1" "$TMP/DT1.log"
grep -m1 -F "phi      at node   1" "$TMP/DTBIG.log"
echo "DTBIG_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/DTBIG.log")"
if grep -q 'phi      at node   1.*actresult= 0.00000000000000000e+00' "$TMP/DTBIG.log"; then
  echo "VERDICT: COARSE_DT_PRODUCES_AN_ACTION_POTENTIAL=no"
else
  echo "VERDICT: COARSE_DT_PRODUCES_AN_ACTION_POTENTIAL=yes"
fi
echo "DTBIG_STIMULUS_WARNINGS=$(grep -ciE 'stimul|missed|upstroke|time step too' "$TMP/DTBIG.log")"
exit 0
