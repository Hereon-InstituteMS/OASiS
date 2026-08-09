#!/bin/bash
# Tier-2 for fourc::particle_pd#10 — boundaryphase can push a peridynamic body
# and cannot pull it, so it is useless for any opening or tensile load case.
#
# Self-contained 2-D deck: a 6 x 6 pdphase plate whose left column sits at
# x = 1.5, and a 6-particle boundaryphase wall at x = 0.5, i.e. exactly one
# PERIDYNAMIC_GRID_SPACING away.  The wall is driven by a Dirichlet function
# whose only difference between the two arms is the SIGN of the velocity:
#
#   PULL : wall displacement -25*t  -> wall retreats from the plate
#   PUSH : wall displacement +25*t  -> wall advances into the plate
#
# PULL: the plate does not move at all.  Its edge particle ends at posx = 1.5
# and velx = 0 with abs(diff) exactly 0.00000000000000000e+00, and the
# "Number of pd_neighbor_pairs in peridynamic evaluation on this proc" line
# reads 0 on every single step — no contact pair is ever created, because a pair
# only exists where the separation is BELOW the grid spacing.  There is no
# tensile branch to reach.
#
# PUSH: pairs appear and the plate is accelerated.  Same deck, same wall, same
# speed, opposite sign.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = signed wall velocity
python3 - "$1" <<'PY'
import sys
v = sys.argv[1]
pd = ['  - "TYPE pdphase POS %.1f %.1f 0.0 PDBODYID 0"' % (1.5 + ix, 0.5 + iy)
      for iy in range(6) for ix in range(6)]
bd = ['  - "TYPE boundaryphase POS 0.5 %.1f 0.0"' % (0.5 + iy) for iy in range(6)]
print(f"""PROBLEM TYPE:
  PROBLEMTYPE: "Particle"
IO:
  STDOUTEVERY: 500
  VERBOSITY: "Standard"
BINNING STRATEGY:
  BIN_SIZE_LOWER_BOUND: 3.0
  DOMAINBOUNDINGBOX: "-12.0 -12.0 -0.01 12.0 12.0 0.01"
PARTICLE DYNAMIC:
  DYNAMICTYPE: "VelocityVerlet"
  INTERACTION: "SPH"
  RESULTSEVERY: 500
  RESTARTEVERY: 500
  TIMESTEP: 8.0e-6
  NUMSTEP: 500
  MAXTIME: 10
  PHASE_TO_DYNLOADBALFAC: "boundaryphase 1.0 pdphase 1.0"
  PHASE_TO_MATERIAL_ID: "boundaryphase 1 pdphase 2"
  RIGID_BODY_MOTION: false
  PD_BODY_INTERACTION: true
PARTICLE DYNAMIC/INITIAL AND BOUNDARY CONDITIONS:
  DIRICHLET_BOUNDARY_CONDITION: "boundaryphase 1"
  CONSTRAINT: "Projection2D"
PARTICLE DYNAMIC/SPH:
  KERNEL: QuinticSpline
  KERNEL_SPACE_DIM: Kernel2D
  INITIALPARTICLESPACING: 1.0
  BOUNDARYPARTICLEFORMULATION: AdamiBoundaryFormulation
  TRANSPORTVELOCITYFORMULATION: StandardTransportVelocity
PARTICLE DYNAMIC/PD:
  INTERACTION_HORIZON: 3.0
  PERIDYNAMIC_GRID_SPACING: 1.0
  PD_DIMENSION: Peridynamic_2DPlaneStrain
  NORMALCONTACTLAW: NormalLinearSpring
  NORMAL_STIFF: 1.0e-3
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{v}*t"
  - COMPONENT: 1
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
  - COMPONENT: 2
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
MATERIALS:
  - MAT: 1
    MAT_ParticleSPHBoundary:
      INITRADIUS: 0.5
      INITDENSITY: 8000.0e-9
  - MAT: 2
    MAT_ParticlePD:
      INITRADIUS: 0.5
      INITDENSITY: 8000.0e-9
      YOUNG: 1.0e3
      CRITICAL_STRETCH: 10.0
RESULT DESCRIPTION:
  - PARTICLE:
      ID: 0
      QUANTITY: "posx"
      VALUE: 1.5
      TOLERANCE: 1e-14
  - PARTICLE:
      ID: 0
      QUANTITY: "velx"
      VALUE: 0.0
      TOLERANCE: 1e-14
PARTICLES:
""" + "\n".join(pd + bd))
PY
}

deck "-25.0" > "$TMP/pull.yaml"
deck "25.0"  > "$TMP/push.yaml"

probe PULL "$TMP/pull.yaml"
probe PUSH "$TMP/push.yaml"

# Retreating wall: the plate is untouched, to the last bit.
grep -m1 -F "processor 0 finished normally" "$TMP/PULL.log"
echo "PULL_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/PULL.log")"
echo "PULL_EXACT_ZERO_DIFFS=$(grep -c 'is CORRECT, abs(diff)= 0.00000000000000000e+00' "$TMP/PULL.log")"
grep -m1 -F "is CORRECT, abs(diff)= 0.00000000000000000e+00" "$TMP/PULL.log"
# ...because no contact pair is ever formed on any step.
echo "PULL_MAX_CONTACT_PAIRS=$(grep -oE 'pd_neighbor_pairs in peridynamic evaluation on this proc: [0-9]+' "$TMP/PULL.log" | grep -oE '[0-9]+$' | sort -n | tail -1)"

# Advancing wall: pairs form and the plate moves.
echo "PUSH_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/PUSH.log")"
echo "PUSH_MAX_CONTACT_PAIRS=$(grep -oE 'pd_neighbor_pairs in peridynamic evaluation on this proc: [0-9]+' "$TMP/PUSH.log" | grep -oE '[0-9]+$' | sort -n | tail -1)"
grep -m1 -E "velx .*is WRONG --> actresult= 4\.31" "$TMP/PUSH.log"

if [ "$(grep -c 'is WRONG --> actresult=' "$TMP/PULL.log")" = "0" ] \
   && [ "$(grep -c 'is WRONG --> actresult=' "$TMP/PUSH.log")" -gt 0 ]; then
  echo "VERDICT: BOUNDARYPHASE_CAN_LOAD_PDPHASE_IN_TENSION=no"
else
  echo "VERDICT: BOUNDARYPHASE_CAN_LOAD_PDPHASE_IN_TENSION=yes"
fi
exit 0
