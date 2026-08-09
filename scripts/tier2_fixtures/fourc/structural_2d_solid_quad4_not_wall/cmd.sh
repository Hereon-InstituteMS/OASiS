#!/bin/bash
# Tier-2 for fourc::linear_elasticity#4 and fourc::structural_mechanics#7
# — WHICH element type owns 2D structural cells is version-dependent in
# 4C, and the two spellings share no keywords:
#
#   4C <= 2026.2:  WALL QUAD4 <n..> MAT m KINEM k EAS e THICK t
#                       STRESS_STRAIN s GP a b
#                  (SOLID registers 3D cell types only)
#   4C >= 2026.3:  SOLID QUAD4 <n..> MAT m KINEM k
#                       THICKNESS t PLANE_ASSUMPTION p
#                  ('WALL' is then an unknown eletype)
#
# This fixture does NOT hard-code either era. It probes both spellings
# against whatever binary is installed and asserts the INVARIANT that
# holds on both: exactly ONE of the two factories accepts quad4. It
# then prints a machine-readable verdict so a reader can see which era
# the host is on.
#
# 2026-06-01: written against 4C 2026.3.0-dev, where SOLID won.
# 2026-08-03: re-run against the deployed 4C 2026.2.0-dev
#             (git 89519cf) at /home/alexander/4C/build/4C, where WALL
#             wins — the original one-sided expectation was falsified
#             on this host, which is why the fixture is now
#             era-agnostic instead of being deleted or inverted.
set -u
# Resolve the 4C binary: explicit override first, then the paths this
# repo has been verified against (2026-08-03 verification host runs
# 4C 2026.2.0-dev at /home/alexander/4C/build/4C).
for _c in "${FOURC_BINARY:-}" "$HOME/4C/build/4C" "$HOME/Schreibtisch/4C-src/4C/build/4C"; do
  [ -x "$_c" ] && BIN="$_c" && break
done
: "${BIN:?4C binary not found — set FOURC_BINARY}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/4C-dependencies/lib}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
# stdbuf is mandatory here: 4C writes the result-test verdict to raw
# std::cout and MPI_Abort discards a block-buffered stdout, which is
# itself pitfall input_format#18.
run4c() { stdbuf -oL -eL "$BIN" "$1" "$2" 2>&1; }

head_of() {
cat <<YAML
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1
  NUMSTEP: 1
  MAXTIME: 1
  TOLDISP: 1e-10
  TOLRES: 1e-09
  MAXITER: 30
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "S"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0, 0]
    FUNCT: [0, 0]
DLINE-NODE TOPOLOGY:
  - "NODE 1 DLINE 1"
  - "NODE 4 DLINE 1"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
YAML
}

tail_of() {
cat <<YAML
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000
      NUE: 0.3
      DENS: 1
YAML
}

probe() {  # $1 = label, $2 = element line
  { head_of; echo "STRUCTURE ELEMENTS:"; echo "  - \"$2\""; tail_of; }       > "$TMP/$1.4C.yaml"
  run4c "$TMP/$1.4C.yaml" "$TMP/o_$1" > "$TMP/$1.log" 2>&1
  if grep -q "fill_complete() on discretization structure" "$TMP/$1.log"; then
    echo "yes"
  else
    echo "no"
  fi
}

# MUTATION CONTROL — a POSITIVE one, because the claim here is an INVARIANT over
# whatever 4C is installed ("exactly one of the two factories accepts quad4"),
# not an injected pathology that could be taken back out.  T2_MUTATE=1 gives the
# second probe the SAME element line as the first, so both arms necessarily
# agree, EXACTLY_ONE_2D_FACTORY flips to `no` — a forbidden token — and the
# expectation disappears.  That rules out the case that would make the fixture
# worthless: a verdict printed rather than derived from two independent 4C runs.
MUTATE="${T2_MUTATE:-0}"
WALL_LINE="1 WALL QUAD4 1 2 3 4 MAT 1 KINEM linear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 2 2"
SOLID_LINE="1 SOLID QUAD4 1 2 3 4 MAT 1 KINEM linear THICKNESS 1.0 PLANE_ASSUMPTION plane_strain"
[ "$MUTATE" = "1" ] && SOLID_LINE="$WALL_LINE"

WALL_OK=$(probe wall "$WALL_LINE")
SOLID_OK=$(probe solid "$SOLID_LINE")

echo "WALL_ACCEPTS_QUAD4=$WALL_OK"
echo "SOLID_ACCEPTS_QUAD4=$SOLID_OK"
if [ "$WALL_OK" = "yes" ] && [ "$SOLID_OK" = "no" ]; then
  echo "EXACTLY_ONE_2D_FACTORY=yes"
  echo "VERDICT: 2D_ELEMENT=WALL"
elif [ "$WALL_OK" = "no" ] && [ "$SOLID_OK" = "yes" ]; then
  echo "EXACTLY_ONE_2D_FACTORY=yes"
  echo "VERDICT: 2D_ELEMENT=SOLID"
else
  echo "EXACTLY_ONE_2D_FACTORY=no"
  echo "VERDICT: 2D_ELEMENT=UNRESOLVED"
fi

# Show the discriminating diagnostic so a human reader can see it.
grep -m1 -h -E "does not seem to know cell type|Unknown type 'WALL'" \
    "$TMP/wall.log" "$TMP/solid.log" || true
exit 0
