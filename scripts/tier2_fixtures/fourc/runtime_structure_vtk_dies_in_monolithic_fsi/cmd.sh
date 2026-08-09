#!/bin/bash
# Tier-2 for fourc::input_format#13 — the rule holds, the quoted Signal does
# not, and the message 4C really prints sends you down a dead end.
#
# Claimed: a structural VTK section in an FSI input aborts with 'inconsistent
#          integration strategy' or similar, from the FSI setup phase.
# Observed: no such string exists.  What MONOLITHIC FSI prints is
#
#     Runtime output is not available in the old structure time integration!
#     You need to take the new one, i.e. set `INT_STRATEGY: Standard`!
#
#          from src/structure/4C_structure_timint.cpp -- and that advice does
#          not work.  INT_STRATEGY already DEFAULTS to Standard, and writing it
#          out explicitly changes nothing, because
#          FSI::MonolithicBase::create_structure_time_integrator constructs the
#          deprecated Adapter::StructureBaseAlgorithm unconditionally.
#
#          PARTITIONED FSI is not affected at all: the identical two sections on
#          a staggered deck exit 0 and write .vtu files.
#
# So the real rule is narrower than the entry: it is monolithic FSI that has to
# fall back to post_vtu, not FSI in general.
. "$(dirname "$0")/../_lib/preamble.sh"

MONO=$(upstream fsi_fp_mono_fs_ga_ga.4C.yaml) || exit 3
PART=$(upstream volmortar2D_fsi.4C.yaml) || exit 3
grep -q 'COUPALGO: "iter_monolithicfluidsplit"' "$MONO" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q 'COUPALGO: "iter_stagg' "$PART" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '^STRUCTURAL DYNAMIC:$' "$MONO" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

VTK='IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: ascii
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
'
mkdir -p "$TMP/mono" "$TMP/monovtk" "$TMP/monostd" "$TMP/part" "$TMP/partvtk"
cp "$MONO" "$TMP/mono/in.4C.yaml"
{ printf '%s' "$VTK"; cat "$MONO"; } > "$TMP/monovtk/in.4C.yaml"
# ...and the same thing again, this time taking the message's own advice.
{ printf '%s' "$VTK"; sed '0,/^STRUCTURAL DYNAMIC:$/s//STRUCTURAL DYNAMIC:\n  INT_STRATEGY: "Standard"/' "$MONO"; } > "$TMP/monostd/in.4C.yaml"
{ printf '%s' "$VTK"; cat "$PART"; } > "$TMP/partvtk/in.4C.yaml"
grep -q '  INT_STRATEGY: "Standard"' "$TMP/monostd/in.4C.yaml" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

go() {  # $1 = dir label
  run4c "$TMP/$1/in.4C.yaml" "$TMP/$1/out" > "$TMP/$1/log" 2>&1
  echo "EXIT_$1=$?"
}
go mono
go monovtk
go monostd
go partvtk

# Baseline monolithic FSI, no VTK sections: fine.
grep -m1 -F "processor 0 finished normally" "$TMP/mono/log"
echo "MONO_CORRECT=$(grep -c 'is CORRECT' "$TMP/mono/log")"
# Add the two runtime-VTK sections and it aborts before solving anything.
grep -m1 -F "Runtime output is not available in the old structure time integration! You need to take the new one, i.e. set \`INT_STRATEGY: Standard\`!" "$TMP/monovtk/log"
grep -m1 -F "4C_structure_timint.cpp" "$TMP/monovtk/log"
echo "MONOVTK_VTU=$(find "$TMP/monovtk" -name '*.vtu' | wc -l)"
# Taking the message's advice changes nothing at all.
echo "MONOSTD_SAME_ABORT=$(grep -c 'Runtime output is not available in the old structure time integration' "$TMP/monostd/log")"
echo "MONOSTD_VTU=$(find "$TMP/monostd" -name '*.vtu' | wc -l)"
# The claimed diagnostic does not exist.
echo "CLAIMED_INCONSISTENT_INT_STRATEGY_TEXT=$(cat "$TMP"/mono*/log | grep -ci 'inconsistent integration strategy')"
# Partitioned FSI takes the same two sections without complaint.
grep -m1 -F "processor 0 finished normally" "$TMP/partvtk/log"
echo "PARTVTK_CORRECT=$(grep -c 'is CORRECT' "$TMP/partvtk/log")"
echo "PARTVTK_WROTE_VTU=$([ "$(find "$TMP/partvtk" -name '*.vtu' | wc -l)" -gt 0 ] && echo yes || echo no)"
exit 0
