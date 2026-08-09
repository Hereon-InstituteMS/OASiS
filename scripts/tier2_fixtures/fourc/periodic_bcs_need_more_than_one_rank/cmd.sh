#!/bin/bash
# Tier-2 for fourc::fluid_turbulence#3 — periodic boundary conditions are not
# only a modelling choice on this build, they are a RANK-COUNT constraint.
#
# The same deck, byte for byte, is run on 1, 2 and 4 MPI ranks. On one rank it
# dies inside Core::Conditions::PeriodicBoundaryConditions::balance_load with a
# bare `terminate called after throwing an instance of 'int'` and SIGABRT — no
# PROC 0 ERROR banner, no source line, nothing naming the periodic block. On two
# and four ranks it completes and writes its results.
#
# Why this matters more than an ordinary pitfall: a diagnostic that names
# nothing sends the reader looking at their periodic condition, which is
# correct, instead of at their `mpirun -np`. The LES channel template this
# project ships records np=2 for exactly this reason, and the pair of arms below
# is what that number rests on.
#
# The CONTRAST arm is the load-bearing half. Without it the fixture would only
# show that a deck failed on one rank, which any broken deck does. Running the
# identical file on two ranks and getting exit 0 is what makes it a statement
# about the rank count rather than about the deck.
. "$(dirname "$0")/../_lib/preamble.sh"

cat > "$TMP/pbc.yaml" <<'YAML'
PROBLEM TYPE:
  PROBLEMTYPE: "Fluid"
FLUID DYNAMIC:
  LINEAR_SOLVER: 1
  TIMEINTEGR: "One_Step_Theta"
  THETA: 1
  TIMESTEP: 0.01
  NUMSTEP: 1
  MAXTIME: 1
  ITEMAX: 2
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Fluid_Solver"
MATERIALS:
  - MAT: 1
    MAT_fluid:
      DYNVISCOSITY: 0.01
      DENSITY: 1
DESIGN SURF DIRICH CONDITIONS:
  - E: 3
    NUMDOF: 4
    ONOFF: [1, 1, 1, 0]
    VAL: [0, 0, 0, 0]
    FUNCT: [0, 0, 0, 0]
  - E: 4
    NUMDOF: 4
    ONOFF: [1, 1, 1, 0]
    VAL: [0, 0, 0, 0]
    FUNCT: [0, 0, 0, 0]
DESIGN SURF PERIODIC BOUNDARY CONDITIONS:
  - E: 1
    ID: 1
    MASTER_OR_SLAVE: "Master"
    PLANE: "yz"
    LAYER: 1
    ANGLE: 0
    ABSTREETOL: 1e-09
  - E: 2
    ID: 1
    MASTER_OR_SLAVE: "Slave"
    PLANE: "yz"
    LAYER: 1
    ANGLE: 0
    ABSTREETOL: 1e-09
  - E: 5
    ID: 2
    MASTER_OR_SLAVE: "Master"
    PLANE: "xy"
    LAYER: 1
    ANGLE: 0
    ABSTREETOL: 1e-09
  - E: 6
    ID: 2
    MASTER_OR_SLAVE: "Slave"
    PLANE: "xy"
    LAYER: 1
    ANGLE: 0
    ABSTREETOL: 1e-09
DESIGN VOL NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 4
    ONOFF: [1, 0, 0, 0]
    VAL: [1, 0, 0, 0]
    FUNCT: [0, 0, 0, 0]
    TYPE: "Dead"
DESIGN VOL MODE FOR KRYLOV SPACE PROJECTION:
  - E: 1
    DIS: "fluid"
    NUMMODES: 4
    ONOFF: [0, 0, 0, 1]
    WEIGHTVECDEF: "integration"
DSURF-NODE TOPOLOGY:
  - "SIDE fluid x- DSURFACE 1"
  - "SIDE fluid x+ DSURFACE 2"
  - "SIDE fluid y- DSURFACE 3"
  - "SIDE fluid y+ DSURFACE 4"
  - "SIDE fluid z- DSURFACE 5"
  - "SIDE fluid z+ DSURFACE 6"
DVOL-NODE TOPOLOGY:
  - "VOLUME fluid DVOL 1"
FLUID DOMAIN:
  bottom_corner_point: [0, -1, 0]
  top_corner_point: [2, 1, 1]
  subdivisions: [4, 4, 4]
  elements:
    FLUID:
      HEX8:
        MAT: 1
        NA: Euler
YAML

# --- the failing arm: one rank -------------------------------------------
NP1_DECK="$TMP/pbc.yaml"
stdbuf -oL -eL "$BIN" "$NP1_DECK" "$TMP/o1" > "$TMP/np1.log" 2>&1
echo "EXIT_NP1=$?"

# --- the contrast arms: two and four ranks, identical file ----------------
NP2=2
stdbuf -oL -eL mpirun -np "$NP2" --oversubscribe "$BIN" "$TMP/pbc.yaml" "$TMP/o2" \
    > "$TMP/np2.log" 2>&1
echo "EXIT_NP2=$?"
stdbuf -oL -eL mpirun -np 4 --oversubscribe "$BIN" "$TMP/pbc.yaml" "$TMP/o4" \
    > "$TMP/np4.log" 2>&1
echo "EXIT_NP4=$?"

# The deck really is one file: prove the arms are not two different decks.
echo "SAME_DECK_SHA=$(sha256sum "$TMP/pbc.yaml" | cut -c1-16)"

# What the single-rank failure looks like, and what it does NOT look like.
echo "NP1_HAS_BALANCE_LOAD=$(grep -c 'PeriodicBoundaryConditions12balance_load\|balance_load' "$TMP/np1.log")"
echo "NP1_HAS_TERMINATE_INT=$(grep -c "terminate called after throwing an instance of 'int'" "$TMP/np1.log")"
echo "NP1_NAMES_PERIODIC_SECTION=$(grep -c 'DESIGN SURF PERIODIC' "$TMP/np1.log")"
echo "NP1_HAS_PROC0_ERROR=$(grep -c 'PROC 0 ERROR' "$TMP/np1.log")"

# The contrast arms got all the way through the time loop.
echo "NP2_FINISHED=$(grep -c 'finished normally' "$TMP/np2.log")"
echo "NP4_FINISHED=$(grep -c 'finished normally' "$TMP/np4.log")"
