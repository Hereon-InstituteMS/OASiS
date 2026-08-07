#!/bin/bash
# Compile one shared translation unit against either deal.II build on this host,
# and cache the result so many fixture directories can share a single build.
#
#   usage: build.sh <name> <release|debug>   -> prints the executable path
#
# WHY THE DEBUG PATH IS HAND-ROLLED. deal.II's own CMake macros
# (deal_ii_setup_target) are not available for the Debug tree: it is a partial
# build — lib/libdeal_II.g.so and include/ were built, share/deal.II/macros was
# not — so find_package finds no macros there. Pointing find_package's HINTS at
# it does something WORSE than fail: CMake silently prefers the system
# /usr deal.II 9.1.1 and the program then links a seven-year-older library. That
# was measured here: the same source reported dealii_version=9.1.1 and asserted
# out of /usr/include/deal.II/lac/vector.h.
#
# So the Debug flags are taken from the WORKING Release link line (captured with
# `cmake --build --verbose`) with exactly three substitutions:
#   generated-header dir  build/include      -> dbgbuild/include
#   library               libdeal_II.so      -> libdeal_II.g.so
#   NDEBUG                                    -> DEBUG, and -O2 -> -O0 -g
# The source headers are shared: both builds are the same revision of
# /home/alexander/dealii (9.8.0-pre, shortrev 87abfb5e).
set -u

NAME="$1"
VARIANT="${2:-release}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_TREE=/home/alexander/dealii
REL=/home/alexander/dealii/build
DBG=${DEAL_II_DEBUG_DIR:-/media/alexander/PortableSSD/dealii-verify-r2/dbgbuild}
OUTDIR="$HERE/_build"
mkdir -p "$OUTDIR"
EXE="$OUTDIR/${NAME}_${VARIANT}"
SRC="$HERE/${NAME}.cc"

if [ ! -f "$SRC" ]; then echo "MISSING_SOURCE=$SRC" >&2; exit 90; fi
if [ "$VARIANT" = debug ] && [ ! -f "$DBG/lib/libdeal_II.g.so" ]; then
  echo "MISSING_DEBUG_LIBRARY=$DBG/lib/libdeal_II.g.so" >&2; exit 91
fi
if [ "$VARIANT" = release ] && [ ! -f "$REL/lib/libdeal_II.so" ]; then
  echo "MISSING_RELEASE_LIBRARY=$REL/lib/libdeal_II.so" >&2; exit 92
fi

# Cached build: reuse unless the source is newer.
if [ -x "$EXE" ] && [ "$EXE" -nt "$SRC" ]; then echo "$EXE"; exit 0; fi

INC="-isystem $SRC_TREE/include \
 -isystem $SRC_TREE/bundled/taskflow-3.10.0 \
 -isystem $SRC_TREE/bundled/boost-1.84.0/include \
 -isystem $SRC_TREE/bundled/kokkos-4.5.01/algorithms/src \
 -isystem $SRC_TREE/bundled/kokkos-4.5.01/containers/src \
 -isystem $SRC_TREE/bundled/kokkos-4.5.01/core/src \
 -isystem $SRC_TREE/bundled/kokkos-4.5.01/simd/src \
 -isystem $SRC_TREE/bundled/kokkos-4.5.01/tpls/mdspan/include \
 -isystem $SRC_TREE/bundled/kokkos-4.5.01/tpls/desul/include \
 -isystem $SRC_TREE/bundled/magic_enum-v0.9.7/include/magic_enum \
 -isystem /usr/include/suitesparse -isystem /usr/include/opencascade"

if [ "$VARIANT" = debug ]; then
  GEN="-isystem $DBG/include"
  DEFS="-DDEBUG -UNDEBUG"
  OPT="-O0 -g"
  LIBDIR="$DBG/lib"
  LIB="$DBG/lib/libdeal_II.g.so"
else
  GEN="-isystem $REL/include"
  DEFS="-DNDEBUG"
  OPT="-O2"
  LIBDIR="$REL/lib"
  LIB="$(ls "$REL"/lib/libdeal_II.so.* | head -1)"
fi

SYSLIBS="/usr/lib/x86_64-linux-gnu/libtbb.so /usr/lib/x86_64-linux-gnu/libz.so \
 /usr/lib/x86_64-linux-gnu/libumfpack.so /usr/lib/x86_64-linux-gnu/libcholmod.so \
 /usr/lib/x86_64-linux-gnu/libccolamd.so /usr/lib/x86_64-linux-gnu/libcolamd.so \
 /usr/lib/x86_64-linux-gnu/libcamd.so /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so \
 /usr/lib/x86_64-linux-gnu/libamd.so -lrt /usr/lib/x86_64-linux-gnu/libmetis.so \
 /usr/lib/x86_64-linux-gnu/libarpack.so /usr/lib/x86_64-linux-gnu/liblapack.so \
 /usr/lib/x86_64-linux-gnu/libblas.so /usr/lib/x86_64-linux-gnu/libgsl.so \
 /usr/lib/x86_64-linux-gnu/libgslcblas.so /usr/lib/x86_64-linux-gnu/libmuparser.so"

# shellcheck disable=SC2086
/usr/bin/c++ $DEFS $GEN $INC -fopenmp-simd -pthread $OPT \
  -Wno-unused-parameter -Wno-unused-variable \
  -o "$EXE.o" -c "$SRC" > "$OUTDIR/${NAME}_${VARIANT}.compile.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "COMPILE_FAILED rc=$rc" >&2
  grep -E "error" "$OUTDIR/${NAME}_${VARIANT}.compile.log" | head -20 >&2
  exit 93
fi
# shellcheck disable=SC2086
/usr/bin/c++ -rdynamic -pthread "$EXE.o" -o "$EXE" \
  -Wl,-rpath,"$LIBDIR" "$LIB" $SYSLIBS \
  > "$OUTDIR/${NAME}_${VARIANT}.link.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "LINK_FAILED rc=$rc" >&2
  tail -20 "$OUTDIR/${NAME}_${VARIANT}.link.log" >&2
  exit 94
fi
echo "$EXE"
