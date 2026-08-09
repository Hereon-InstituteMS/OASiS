#!/bin/bash

# Tier-2 for fourc::electrochemistry#6, and the execution that FALSIFIED its
# Signal.
#
# The entry said SUPG on a pure-diffusion ELCH "damps physical concentration
# gradients by 5-20%".  On the diffusion-conduction framework — which is what
# DIFFCOND_FORMULATION: true selects, and what the concentrated-solution ELCH
# decks use — you cannot get that far.  4C refuses:
#
#   "No stabilization is necessary for solving the ELCH diffusion-conduction
#    framework!!"   from src/scatra/4C_scatra_timint_elch.cpp
#
# So the advice "keep no_stabilization" is right, but it is enforced, not a
# quiet degradation to watch for.
#
# The second half of the fixture goes where the choice IS free — a dilute
# Nernst-Planck ELCH — and measures it rather than assuming: the two settings
# do not differ by a small percentage there, they differ by order unity,
# because that deck is convection-dominated and stabilisation is doing real
# work.  Either way, "a few per cent of quiet damping" is not what happens.
# --- self-contained preamble (deliberately NOT sourced from ../_lib) --------
# scripts/mutate_tier2_fixtures.py copies ONLY this directory into a scratch
# tree.  A fixture that sources ../_lib/preamble.sh therefore cannot even
# start there, its mutant dies for the wrong reason, and the KILLED verdict
# certifies nothing.  Everything this fixture needs is inline, so the
# mutation proof is real.  Same honesty rule as the shared preamble: when 4C
# is missing this prints FIXTURE_ABORT=no_binary and exits non-zero, and
# fixture.json forbids both strings, so an absent solver makes the fixture
# RED rather than green.
set -u
for _c in "${FOURC_BINARY:-}" "$HOME/4C/build/4C" \
          "$HOME/Schreibtisch/4C-src/4C/build/4C" "/usr/local/bin/4C"; do
  [ -n "${_c:-}" ] && [ -x "$_c" ] && BIN="$_c" && break
done
if [ -z "${BIN:-}" ]; then
  echo "FIXTURE_ABORT=no_binary (set FOURC_BINARY to a 4C executable)"
  exit 3
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/4C-dependencies/lib}"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
# stdbuf is not decoration: 4C writes result-test verdicts to raw std::cout
# and MPI_Abort discards a block-buffered stdout (pitfall input_format#18).
run4c() { stdbuf -oL -eL "$BIN" "$1" "$2" 2>&1; }
probe() { run4c "$2" "$TMP/o_$1" > "$TMP/$1.log" 2>&1; echo "EXIT_$1=$?"; }
DECKS=""
for _d in "${FOURC_INPUT_FILES:-}" "$HOME/4C/tests/input_files" \
          "$HOME/Schreibtisch/4C-src/4C/tests/input_files"; do
  [ -n "${_d:-}" ] && [ -d "$_d" ] && DECKS="$_d" && break
done
upstream() {  # <name.4C.yaml> -> its path, or abort loudly
  if [ -z "$DECKS" ]; then
    echo "FIXTURE_ABORT=no_upstream_decks (set FOURC_INPUT_FILES)"; exit 3; fi
  if [ ! -f "$DECKS/$1" ]; then
    echo "FIXTURE_ABORT=no_upstream_decks (missing $1)"; exit 3; fi
  printf '%s\n' "$DECKS/$1"
}
# ---------------------------------------------------------------------------

DIFF=$(upstream elch_2D_tertiary_twoEqu_ENC_varParams_ndb_2iter.4C.yaml) || exit 3
NP=$(upstream elch_1D_10ele_3ions_stab_fdcheck.4C.yaml) || exit 3
grep -q 'STABTYPE: "no_stabilization"'    "$DIFF" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q 'DIFFCOND_FORMULATION: true'      "$DIFF" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q 'DEFINITION_TAU: "Taylor_Hughes_Zarins"' "$NP" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

# 1. Diffusion-conduction ELCH: SUPG is not a 'damping' knob, it is refused.
cp "$DIFF" "$TMP/dc_nostab.yaml"
sed -e 's/  STABTYPE: "no_stabilization"/  STABTYPE: "SUPG"/' \
    -e 's/  DEFINITION_TAU: "Zero"/  DEFINITION_TAU: "Taylor_Hughes_Zarins"/' \
    "$DIFF" > "$TMP/dc_supg.yaml"

probe DC_NOSTAB "$TMP/dc_nostab.yaml"
probe DC_SUPG   "$TMP/dc_supg.yaml"
echo "DC_NOSTAB_PASSED=$(grep -c 'is CORRECT' "$TMP/DC_NOSTAB.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/DC_NOSTAB.log"
grep -m1 -F "No stabilization is necessary for solving the ELCH diffusion-conduction framework!!" "$TMP/DC_SUPG.log"
grep -m1 -F "4C_scatra_timint_elch.cpp" "$TMP/DC_SUPG.log"

# 2. Nernst-Planck ELCH, where the choice IS free: measure what it does.
#    (FDCHECK is stripped from both arms; it is a Jacobian check, not physics,
#    and it trips on values near zero in the unstabilised run.)
python3 - "$NP" "$TMP" <<'NPPY'
import sys, re
from pathlib import Path
t = open(sys.argv[1]).read(); T = Path(sys.argv[2])
t = t.replace('  FDCHECK: "local"\n', '').replace('  FDCHECKTOL: 0.002\n', '')
rec = re.sub(r"      VALUE: [-0-9.e+]+\n      TOLERANCE: [-0-9.e+]+\n",
             "      VALUE: 0.0\n      TOLERANCE: 1e-30\n", t)
(T/"np_supg.yaml").write_text(rec)
old = 'SCALAR TRANSPORT DYNAMIC/STABILIZATION:\n  DEFINITION_TAU: "Taylor_Hughes_Zarins"'
assert old in rec
(T/"np_nostab.yaml").write_text(rec.replace(
    old, 'SCALAR TRANSPORT DYNAMIC/STABILIZATION:\n  STABTYPE: "no_stabilization"\n  DEFINITION_TAU: "Zero"'))
NPPY

probe NP_SUPG   "$TMP/np_supg.yaml"
probe NP_NOSTAB "$TMP/np_nostab.yaml"
grep -oP 'actresult=\s*\K[-0-9.e+]+' "$TMP/NP_SUPG.log"   > "$TMP/a.txt"
grep -oP 'actresult=\s*\K[-0-9.e+]+' "$TMP/NP_NOSTAB.log" > "$TMP/b.txt"
echo "NP_VALUES_READ=$(wc -l < "$TMP/a.txt") $(wc -l < "$TMP/b.txt")"
python3 - "$TMP/a.txt" "$TMP/b.txt" <<'CMPPY'
import sys
a = [float(x) for x in open(sys.argv[1])]
b = [float(x) for x in open(sys.argv[2])]
worst = max(abs(x-y)/max(abs(x), 1e-30) for x, y in zip(a, b)) if a and b else 0.0
print(f"NP_SUPG_FIRST_VALUE={a[0]:.17g}" if a else "NP_SUPG_FIRST_VALUE=none")
print(f"NP_NOSTAB_FIRST_VALUE={b[0]:.17g}" if b else "NP_NOSTAB_FIRST_VALUE=none")
print("NP_STABILISATION_CHANGES_CONCENTRATIONS=" + ("yes" if worst > 0.01 else "no"))
print("NP_CHANGE_IS_A_SMALL_PERTURBATION=" + ("yes" if 0 < worst < 0.25 else "no"))
CMPPY
exit 0
