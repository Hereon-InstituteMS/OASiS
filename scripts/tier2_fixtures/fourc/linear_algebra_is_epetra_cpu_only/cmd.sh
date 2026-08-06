#!/bin/bash
# Tier-2 for fourc::input_format#14 — 4C's linear algebra is Epetra-based and
# CPU-only, so no GPU-targeted environment variable does anything at all.
#
# The observable for a claim about hardware is what the BINARY reports.  Three
# things are checked here, none of them wall-clock:
#
#   * what the shared library actually links: Epetra libraries are present,
#     and no CUDA / HIP / SYCL / ROCm runtime is linked at all;
#   * that Core::LinAlg is built on Epetra rather than Tpetra -- the Epetra
#     solver stack is what appears in the abort trace of a failing solve
#     (Amesos2::Umfpack<Epetra_CrsMatrix, Epetra_MultiVector>);
#   * that setting CUDA_VISIBLE_DEVICES / KOKKOS_NUM_DEVICES / HIP_VISIBLE_DEVICES
#     leaves the computed answer bit-identical and the run structurally
#     identical, which is what "zero effect" means without a stopwatch.
. "$(dirname "$0")/../_lib/preamble.sh"

DECK="$TMP/in.4C.yaml"
cat > "$DECK" <<'YAML'
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  NUMSTEP: 1
  MAXTIME: 0.1
  TOLDISP: 1e-10
  TOLRES: 1e-09
  MAXITER: 30
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "S"
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 3
    ONOFF: [0, 1, 0]
    VAL: [0, 1, 0]
    FUNCT: [0, 0, 0]
    TYPE: "Live"
DSURF-NODE TOPOLOGY:
  - "NODE 1 DSURFACE 1"
  - "NODE 4 DSURFACE 1"
  - "NODE 5 DSURFACE 1"
  - "NODE 8 DSURFACE 1"
  - "NODE 2 DSURFACE 2"
  - "NODE 3 DSURFACE 2"
  - "NODE 6 DSURFACE 2"
  - "NODE 7 DSURFACE 2"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
  - "NODE 5 COORD 0.0 0.0 1.0"
  - "NODE 6 COORD 1.0 0.0 1.0"
  - "NODE 7 COORD 1.0 1.0 1.0"
  - "NODE 8 COORD 0.0 1.0 1.0"
STRUCTURE ELEMENTS:
  - "1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM nonlinear"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000
      NUE: 0.3
      DENS: 1
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 3
      QUANTITY: "dispy"
      VALUE: 4.47909266337460053e-03
      TOLERANCE: 1e-12
YAML

# --- what the binary links -------------------------------------------------
LIB=$(dirname "$BIN")/lib4C.so
[ -f "$LIB" ] || LIB="$BIN"
ldd "$LIB" > "$TMP/ldd.txt" 2>/dev/null
echo "LINKED_EPETRA_LIBS=$(grep -c 'libepetra\|epetra\.so' "$TMP/ldd.txt")"
echo "LINKED_GPU_RUNTIMES=$(grep -ciE 'libcuda|libcudart|libamdhip|libhip|libsycl|librocm' "$TMP/ldd.txt")"
echo "TRILINOS_SONAME_MAJOR=$(grep -o 'libepetra\.so\.[0-9]*' "$TMP/ldd.txt" | head -1)"

# --- GPU environment variables change nothing ------------------------------
probe PLAIN "$DECK"
CUDA_VISIBLE_DEVICES=0 KOKKOS_NUM_DEVICES=1 HIP_VISIBLE_DEVICES=0 \
  KOKKOS_DEVICES=Cuda probe GPUENV "$DECK"

grep -m1 -F "processor 0 finished normally" "$TMP/PLAIN.log"
grep -m1 -F "processor 0 finished normally" "$TMP/GPUENV.log"
echo "PLAIN_CORRECT=$(grep -c 'is CORRECT' "$TMP/PLAIN.log")"
echo "GPUENV_CORRECT=$(grep -c 'is CORRECT' "$TMP/GPUENV.log")"
echo "GPU_ENV_CHANGED_THE_ANSWER=$([ "$(grep -o 'abs(diff)= [0-9.e+-]*' "$TMP/PLAIN.log")" = "$(grep -o 'abs(diff)= [0-9.e+-]*' "$TMP/GPUENV.log")" ] && echo no || echo yes)"
echo "GPU_ENV_PRODUCED_ANY_DEVICE_MESSAGE=$(grep -ciE 'cuda|gpu|device' "$TMP/GPUENV.log")"

# --- the stack that actually runs, in 4C's own words ------------------------
# Every run ends with 4C's Teuchos timer table, whose labels name the concrete
# classes the run went through.  Only the LABELS and the CALL COUNT are read
# here; the elapsed columns are ignored on purpose.
echo "TIMER_LABELS_EPETRA=$(grep -c '^Epetra_' "$TMP/PLAIN.log")"
echo "TIMER_LABELS_TPETRA=$(grep -c '^Tpetra' "$TMP/PLAIN.log")"
echo "EPETRA_MULTIPLY_CALLS=$(grep -o 'Epetra_CrsMatrix::Multiply(TransA,X,Y) *[0-9.e+-]* *([0-9]*)' "$TMP/PLAIN.log" | grep -o '([0-9]*)$')"
exit 0
