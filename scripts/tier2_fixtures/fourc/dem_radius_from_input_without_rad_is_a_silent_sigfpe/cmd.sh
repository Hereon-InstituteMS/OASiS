#!/bin/bash
# Tier-2 for fourc::particle_dem#4 — RadiusFromParticleInput with no RAD token
# is a silent divide-by-zero, not an error.
#
# Claimed: the radius stays 0, the MIN/MAX bounds check PASSES because
#          MIN_RADIUS defaults to 0.0, and the run dies on a raw SIGFPE with no
#          4C error block. Setting MIN_RADIUS above zero converts it into a
#          clean abort.
#
# T2_MUTATE=1 removes the pathology: INITIAL_RADIUS is left at its
# RadiusFromParticleMaterial default, so the radius comes from the material,
# there is no SIGFPE, EXIT_NORAD becomes 0 and the fixture must go red.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

BASE=$(upstream particle_dem_1d_normalcontact_linspring_stiffset.4C.yaml) || exit 3
grep -q 'TYPE phase1 POS' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q 'RAD ' "$BASE" && { echo "FIXTURE_ABORT=upstream_deck_already_has_RAD"; exit 3; }

python3 - "$BASE" "$TMP" "$MUTATE" <<'PY'
import sys
src, tmp, mutate = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3] == "1"
anchor = "  NORMAL_STIFF: 3.5e-05"
assert anchor in src
sel = anchor + '\n  INITIAL_RADIUS: "RadiusFromParticleInput"'
# the pathology: read radii from input while no line carries a RAD token
open(tmp + "/norad.yaml", "w").write(src if mutate else src.replace(anchor, sel))
# the same deck with a MIN_RADIUS floor, so the bounds check can see the zero
open(tmp + "/floor.yaml", "w").write(
    src if mutate else src.replace(anchor, sel + "\n  MIN_RADIUS: 0.001"))
# and the same selector done RIGHT, with RAD on every particle line
open(tmp + "/withrad.yaml", "w").write(
    src.replace(anchor, sel).replace("POS -0.015 0.0 0.0", "POS -0.015 0.0 0.0 RAD 0.01")
       .replace("POS 0.015 0.0 0.0", "POS 0.015 0.0 0.0 RAD 0.01"))
PY

probe NORAD   "$TMP/norad.yaml"
probe FLOOR   "$TMP/floor.yaml"
probe WITHRAD "$TMP/withrad.yaml"

# 136 = 128 + SIGFPE(8): the shell reports a killed process, not a 4C exit.
echo "NORAD_DIED_ON_A_SIGNAL=$(grep -qF 'Floating point exception' "$TMP/NORAD.log" && echo yes || echo no)"
grep -m1 -F "Signal: Floating point exception (8)" "$TMP/NORAD.log"
grep -m1 -F "Signal code: Floating point divide-by-zero (3)" "$TMP/NORAD.log"
# There is no 4C diagnostic of any kind: no error block, no mention of radius.
echo "NORAD_FOURC_ERROR_BLOCKS=$(grep -c 'PROC 0 ERROR' "$TMP/NORAD.log")"
# Which 4C radius diagnostics fired? Name them instead of grepping for the
# word, which also matches a demangled stack frame (set_initial_radius) and is
# a proxy for the wrong thing.
echo "NORAD_RADIUS_DIAGNOSTICS=$(grep -cE 'non-positive maximum allowed particle radius|minimum allowed particle radius|minimum particle radius smaller|maximum particle radius larger|RADIUSDISTRIBUTION_SIGMA is not set' "$TMP/NORAD.log")"
# Deliberately NOT a keyword grep. Two successive attempts to say "the word
# radius never appears" were environment-dependent -- the demangled backtrace
# carries a 4C symbol containing it, and which frames are printed depends on
# how the process was launched. Counting the named diagnostics is stable.
# It dies before the first time step.
echo "NORAD_STEPS_PRINTED=$(grep -c '^TIME:' "$TMP/NORAD.log")"
# A positive MIN_RADIUS turns the same deck into a clean, named abort.
grep -m1 -F "minimum particle radius smaller than minimum allowed particle radius!" "$TMP/FLOOR.log"   && echo "MIN_RADIUS_FLOOR_CATCHES_IT=yes" || echo "MIN_RADIUS_FLOOR_CATCHES_IT=no"
# Written correctly, the same selector runs.
grep -m1 -F "processor 0 finished normally" "$TMP/WITHRAD.log" && echo "WITH_RAD_RUNS=yes"
exit 0
