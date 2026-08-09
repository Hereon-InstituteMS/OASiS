#!/bin/bash
# Tier-2 for fourc::particles#7 — swapping the phase labels is confirmed to be
# fatal, and the diagnostic points at a missing STATE, not at a wrong TYPE.
#
# Upstream particle_sph_2d_pdbody_gravity.4C.yaml has 444 boundaryphase wall
# particles and 162 pdphase body particles, mapped by
# PHASE_TO_MATERIAL_ID: "boundaryphase 1 pdphase 2".  The mutant exchanges the
# two TYPE labels and touches nothing else, exactly the mix-up the entry warns
# about.
#
# What happens:
#   * The wall becomes the peridynamic body.  The bond list grows from 1512 to
#     3588 pairs — the structure that was supposed to be rigid is now the thing
#     that deforms.
#   * The former body is now boundaryphase, and boundary particles do not move:
#     4C reports its probe particle at posx 6.50000000000000000e+00 with velx
#     and vely exactly 0.00000000000000000e+00 after 3000 steps.  That is the
#     entry's "sticks to it, no rebound", in numbers.
#   * The only complaint 4C makes is
#       state 'pd_damage_phi' not found in container!
#     from particle/src/algorithm/4C_particle_algorithm_result_test.cpp, because
#     the probed particle no longer carries peridynamic states.  It names a
#     state, never a phase, a TYPE or PHASE_TO_MATERIAL_ID, so an agent reading
#     it looks for a broken output request rather than a swapped label.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_gravity.4C.yaml) || exit 3
grep -q 'PHASE_TO_MATERIAL_ID: "boundaryphase 1 pdphase 2"' "$BASE" || \
  { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/asis.yaml"
python3 - "$BASE" "$TMP/swap.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
assert t.count("TYPE pdphase POS") == 162 and t.count("TYPE boundaryphase POS") == 444
s = (t.replace("TYPE pdphase POS", "TYPE @@@ POS")
      .replace("TYPE boundaryphase POS", "TYPE pdphase POS")
      .replace("TYPE @@@ POS", "TYPE boundaryphase POS"))
open(sys.argv[2], "w").write(s)
PY
echo "PD_PARTICLES_ASIS=$(grep -c 'TYPE pdphase POS' "$TMP/asis.yaml")"
echo "PD_PARTICLES_SWAP=$(grep -c 'TYPE pdphase POS' "$TMP/swap.yaml")"

probe ASIS "$TMP/asis.yaml"
probe SWAP "$TMP/swap.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/ASIS.log"
echo "ASIS_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/ASIS.log")"
grep -m1 -F "Number of initialized peridynamic bonds on this proc: 1512" "$TMP/ASIS.log"

# The wall is now the deformable body.
grep -m1 -F "Number of initialized peridynamic bonds on this proc: 3588" "$TMP/SWAP.log"
# The body is now a boundary phase and does not move at all.
grep -m1 -F "is WRONG --> actresult= 0.00000000000000000e+00, givenresult= 2.31537930774811720e+01" "$TMP/SWAP.log"
grep -m1 -F "is WRONG --> actresult= 6.50000000000000000e+00" "$TMP/SWAP.log"
echo "SWAP_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/SWAP.log")"
# ...and the diagnostic names a state, not a phase.
grep -m1 -F "state 'pd_damage_phi' not found in container!" "$TMP/SWAP.log"
grep -m1 -oE "4C_particle_algorithm_result_test\.cpp, line [0-9]+" "$TMP/SWAP.log"
echo "SWAP_DIAGNOSTIC_NAMES_A_PHASE=$(grep -ciE "not found in container.*(phase|TYPE)" "$TMP/SWAP.log")"
exit 0
