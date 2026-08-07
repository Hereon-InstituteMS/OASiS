#!/bin/bash

# Tier-2 for fourc::electrochemistry#0, and the execution that FALSIFIED its
# Signal.
#
# The entry said an 'ENC' run "produces ZERO potential drop at the interface,
# whereas 'divi' resolves the double-layer voltage".  Two things are wrong
# with that.
#
#  1. Where BOTH closures are legal, they agree.  The upstream decks
#     elch_2D_tertiary_twoEqu_ENC_varParams_ndb_2iter and
#     ..._divi_... differ in exactly one line — EQUPOT — and assert the SAME
#     reference values, potential included.  Both pass all ten of them at the
#     decks' own 1e-11.  The ENC potential is read out here too: it is
#     -6.5e-02 at node 1 and a different value at node 97, i.e. neither zero
#     nor flat.
#  2. Where the material demands 'divi', ENC is not a silent wrong answer, it
#     is a hard abort naming the requirement:
#     "Newman material must be combined with divi closing equation for
#     electric potential!" from
#     src/scatra_ele/4C_scatra_ele_calc_service_elch_diffcond.cpp
#
# So EQUPOT is not a free physics dial that quietly changes the answer; it is
# constrained by the material model, and 4C says so.
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

ENC=$(upstream elch_2D_tertiary_twoEqu_ENC_varParams_ndb_2iter.4C.yaml) || exit 3
DIVI=$(upstream elch_2D_tertiary_twoEqu_divi_varParams_ndb_2iter.4C.yaml) || exit 3
NEWMAN=$(upstream elch_2D_DLcap_linearKinetics.4C.yaml) || exit 3
grep -q 'EQUPOT: "ENC"'  "$ENC"    || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q 'EQUPOT: "divi"' "$DIVI"   || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q 'EQUPOT: "divi"' "$NEWMAN" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

# The two tertiary decks are the same problem posed both ways.  Establish that
# they differ in exactly one line before drawing any conclusion from it.
echo "ENC_VS_DIVI_DIFFERING_LINES=$(diff "$ENC" "$DIVI" | grep -c '^[<>]')"
echo "ENC_VS_DIVI_DIFF_IS_ONLY_EQUPOT=$(diff "$ENC" "$DIVI" | grep -c 'EQUPOT')"

probe ENC  "$ENC"
probe DIVI "$DIVI"
echo "ENC_TESTS_PASSED=$(grep -c 'is CORRECT' "$TMP/ENC.log")"
echo "DIVI_TESTS_PASSED=$(grep -c 'is CORRECT' "$TMP/DIVI.log")"

# Both decks assert the SAME reference values, including the potential phi4,
# so passing both means the two closures agree to the decks' own 1e-11.
echo "SHARED_REFERENCE_VALUES=$(grep -c 'VALUE:' "$ENC")"

# Read the ENC potential itself, to show it is neither zero nor flat.
python3 - "$ENC" "$TMP/enc_record.yaml" <<'RECPY'
import sys, re
t = open(sys.argv[1]).read()
for v in ("-0.06527147929184442", "-0.06458503922358104"):
    assert v in t, "upstream deck no longer carries the phi4 reference values"
    t = t.replace(f"VALUE: {v}", "VALUE: 0.0")
open(sys.argv[2], "w").write(t)
RECPY
probe ENC_RECORD "$TMP/enc_record.yaml"
grep -m2 -E 'phi4 +at node +(1|97).*is WRONG --> actresult=' "$TMP/ENC_RECORD.log"
ZEROS=$(grep -cE 'phi4 .*actresult= 0\.00000000000000000e\+00' "$TMP/ENC_RECORD.log")
echo "ENC_POTENTIAL_IS_IDENTICALLY_ZERO=$([ "$ZEROS" -gt 0 ] && echo yes || echo no)"

# Where the material demands 'divi', ENC is not a silent wrong answer at all.
sed 's/EQUPOT: "divi"/EQUPOT: "ENC"/' "$NEWMAN" > "$TMP/newman_enc.yaml"
probe NEWMAN_DIVI "$NEWMAN"
probe NEWMAN_ENC  "$TMP/newman_enc.yaml"
grep -m1 -F "Newman material must be combined with divi closing equation for electric potential!" "$TMP/NEWMAN_ENC.log"
grep -m1 -F "4C_scatra_ele_calc_service_elch_diffcond.cpp" "$TMP/NEWMAN_ENC.log"
exit 0
