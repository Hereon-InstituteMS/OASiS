# Shared preamble for the 4C tier-2 fixtures.  Source it as:
#
#     . "$(dirname "$0")/../_lib/preamble.sh"
#
# It provides $BIN, $TMP, run4c, upstream, and — the part that matters for
# honesty — it makes a fixture ABORT LOUDLY when 4C is not installed instead of
# quietly reporting a pass.
#
# WHY THE LOUD ABORT.  Five FEBio fixtures in this repo had their whole
# expectation be a bare key like `degenerate_hex8=`, while the absent-solver
# path printed `degenerate_hex8=skipped_no_binary`.  The expectation matched, no
# forbidden string appeared, and the fixtures reported PASS on machines with no
# solver — certifying nothing.  Every fixture that sources this file therefore
# prints `FIXTURE_ABORT=no_binary` (or `=no_upstream_decks`) and exits non-zero
# when it cannot do the real work, and every fixture.json lists
# "FIXTURE_ABORT=" and "no_binary" in forbid_in_output.  Absence of 4C then
# makes the fixture RED, which is the only truthful outcome.
#
# This directory is not itself a fixture: it has no fixture.json, so neither the
# runner nor the coverage count sees it.
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

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# stdbuf is not decoration: 4C writes result-test verdicts to raw std::cout and
# MPI_Abort discards a block-buffered stdout, so without line buffering the
# decisive lines are lost exactly on the runs that fail.  That is itself pitfall
# fourc::input_format#18.
run4c() { stdbuf -oL -eL "$BIN" "$1" "$2" 2>&1; }

# Run a deck and echo `EXIT_<label>=<status>`, keeping the log at $TMP/<label>.log
probe() {  # $1 = label, $2 = deck path
  run4c "$2" "$TMP/o_$1" > "$TMP/$1.log" 2>&1
  echo "EXIT_$1=$?"
}

# Locate 4C's own regression decks.  Several pitfalls concern physics whose
# minimal working input is hundreds of lines (XFEM, poroelasticity, SSI); for
# those the honest baseline is an upstream deck that is known to run, mutated in
# exactly one place.  Missing decks abort loudly, same rule as a missing binary.
DECKS=""
for _d in "${FOURC_INPUT_FILES:-}" "$HOME/4C/tests/input_files" \
          "$HOME/Schreibtisch/4C-src/4C/tests/input_files"; do
  [ -n "${_d:-}" ] && [ -d "$_d" ] && DECKS="$_d" && break
done

# upstream <name.4C.yaml> -> prints its path, or aborts loudly.
upstream() {
  if [ -z "$DECKS" ]; then
    echo "FIXTURE_ABORT=no_upstream_decks (set FOURC_INPUT_FILES)" >&2
    echo "FIXTURE_ABORT=no_upstream_decks (set FOURC_INPUT_FILES)"
    exit 3
  fi
  if [ ! -f "$DECKS/$1" ]; then
    echo "FIXTURE_ABORT=no_upstream_decks (missing $1)"
    exit 3
  fi
  printf '%s\n' "$DECKS/$1"
}
