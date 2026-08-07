#!/bin/bash
# Tier-2 for fourc::sti#2 — a FALSIFICATION.
#
# Claimed: "Scalar transport and thermal fields must use COMPATIBLE time steps.
#          Mismatched TIMESTEP values lead to temporal interpolation errors at
#          the coupling ... Set TIMESTEP equal in both DYNAMIC sections."
#
# Observed: there are not two DYNAMIC sections to set.  STI has ONE time step,
# SCALAR TRANSPORT DYNAMIC's, and the thermo field inherits it.
#
#   STIDT     add TIMESTEP to STI DYNAMIC  -> parse abort, STI DYNAMIC has no
#             such parameter: "Could not match this input", from
#             core/io/src/4C_io_input_spec_builders.cpp
#   THERMDT   add a whole THERMAL DYNAMIC section with a different TIMESTEP and
#             NUMSTEP -> accepted and completely ignored: exit 0, all 40 result
#             tests pass, identical to the reference
#   SCATRADT  halve SCALAR TRANSPORT DYNAMIC's TIMESTEP -> the results move
#
# The last arm is the control: it shows the ONE time step that does govern, so
# THERMDT's zero change is an override and not an inert deck.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream sti_mono_3D_hex8_elch_s2i_butlervolmerpeltier_adiabatic_mortar_standard.4C.yaml) || exit 3
grep -q "^  TIMESTEP: 5$" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "^STI DYNAMIC:$"  "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/ref.yaml"
sed 's/^STI DYNAMIC:$/STI DYNAMIC:\n  TIMESTEP: 2.5/' "$BASE" > "$TMP/stidt.yaml"
sed 's/^STI DYNAMIC:$/THERMAL DYNAMIC:\n  TIMESTEP: 2.5\n  NUMSTEP: 8\nSTI DYNAMIC:/' "$BASE" > "$TMP/thermdt.yaml"
sed 's/^  TIMESTEP: 5$/  TIMESTEP: 2.5/' "$BASE" > "$TMP/scatradt.yaml"

probe REF      "$TMP/ref.yaml"
probe STIDT    "$TMP/stidt.yaml"
probe THERMDT  "$TMP/thermdt.yaml"
probe SCATRADT "$TMP/scatradt.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/REF.log"
# STI DYNAMIC has no TIMESTEP at all.
grep -m1 -F "Could not match this input" "$TMP/STIDT.log"
grep -m1 -oF "4C_io_input_spec_builders.cpp" "$TMP/STIDT.log"
# A THERMAL DYNAMIC section with its own time step is legal and inert.
grep -m1 -F "processor 0 finished normally" "$TMP/THERMDT.log"
echo "THERMAL_DYNAMIC_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/THERMDT.log")"
echo "THERMAL_DYNAMIC_WARNINGS=$(grep -ciE 'timestep.*(ignor|overrid|inconsist|mismatch)' "$TMP/THERMDT.log")"
# Control: the single governing time step does change the answer.
echo "SCATRA_TIMESTEP_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/SCATRADT.log")"
grep -m1 -F "is WRONG --> actresult=" "$TMP/SCATRADT.log"
if [ "$(grep -c 'is WRONG --> actresult=' "$TMP/THERMDT.log")" = "0" ]; then
  echo "VERDICT: STI_HAS_A_SEPARATE_THERMAL_TIMESTEP=no"
else
  echo "VERDICT: STI_HAS_A_SEPARATE_THERMAL_TIMESTEP=yes"
fi
exit 0
