#!/bin/bash
# Tier-2 for fourc::sti#3 — and a FALSIFICATION of its Signal.
#
# Claimed: mixing the field prefixes (a thermal flux under TRANSPORT NEUMANN)
#          "aborts with 'condition does not apply to this field' from
#          4C_io_input_spec_builders.cpp — the vocabulary distinguishes by FIELD
#          before condition TYPE."
#
# Observed: no such string, and no parse-time distinction at all.  All the
# prefixed spellings are valid section names, so nothing is rejected when the
# file is read; the prefix chooses WHICH FIELD gets the load, silently.  The
# upstream STI deck applies its potential flux through the UNPREFIXED
# `DESIGN SURF NEUMANN CONDITIONS`.  Move that same block:
#
#   THERMO     -> DESIGN SURF THERMO NEUMANN CONDITIONS: parses, runs, then dies
#                 mid-solve with "The NUMDOF you have entered in your TRANSPORT
#                 NEUMANN CONDITION does not equal the number of scalars."
#                 from scatra_ele/4C_scatra_ele_boundary_calc.cpp — note the
#                 message says TRANSPORT although the section said THERMO,
#                 because the thermo field is itself a scatra discretisation.
#   TRANSPORT  -> DESIGN SURF TRANSPORT NEUMANN CONDITIONS: parses, runs to
#                 completion, exits with most of its result tests WRONG.  This
#                 is the dangerous one: a rename with no diagnostic and a
#                 different answer.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream sti_mono_3D_hex8_elch_s2i_butlervolmerpeltier_adiabatic_mortar_standard.4C.yaml) || exit 3
grep -q "^DESIGN SURF NEUMANN CONDITIONS:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/plain.yaml"
sed 's/^DESIGN SURF NEUMANN CONDITIONS:/DESIGN SURF THERMO NEUMANN CONDITIONS:/'    "$BASE" > "$TMP/thermo.yaml"
sed 's/^DESIGN SURF NEUMANN CONDITIONS:/DESIGN SURF TRANSPORT NEUMANN CONDITIONS:/' "$BASE" > "$TMP/transport.yaml"

probe PLAIN     "$TMP/plain.yaml"
probe THERMO    "$TMP/thermo.yaml"
probe TRANSPORT "$TMP/transport.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/PLAIN.log"
# Both prefixed spellings are valid section names: neither is rejected at parse.
echo "THERMO_SECTION_NAME_REJECTED=$(grep -c 'is not a valid section name' "$TMP/THERMO.log")"
echo "TRANSPORT_SECTION_NAME_REJECTED=$(grep -c 'is not a valid section name' "$TMP/TRANSPORT.log")"
# THERMO: fails mid-solve, and the diagnostic names the wrong field.
grep -m1 -F "The NUMDOF you have entered in your TRANSPORT NEUMANN CONDITION does not equal the number of scalars." "$TMP/THERMO.log"
grep -m1 -oF "4C_scatra_ele_boundary_calc.cpp" "$TMP/THERMO.log"
# TRANSPORT: no diagnostic at all, just a different answer.
grep -m1 -F "processor 0 finished normally" "$TMP/TRANSPORT.log"
echo "TRANSPORT_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/TRANSPORT.log")"
grep -m1 -F "is WRONG --> actresult=" "$TMP/TRANSPORT.log"
# The claimed diagnostic exists nowhere.
echo "CLAIMED_DOES_NOT_APPLY_TEXT=$(grep -ci 'condition does not apply to this field' "$TMP/THERMO.log" "$TMP/TRANSPORT.log" | awk -F: '{s+=$2} END {print s+0}')"
exit 0
