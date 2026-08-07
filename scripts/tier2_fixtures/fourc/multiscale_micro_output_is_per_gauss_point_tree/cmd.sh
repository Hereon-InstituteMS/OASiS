#!/bin/bash
# Tier-2 for fourc::multiscale#7 — the macro VTU really does hide the micro
# scale, and the switch that reveals it is RUNTIMEOUTPUT_GP on the material.
#
# The entry told an agent to "use the MULTISCALE micro-output writer (separate
# VTU per Gauss point)" without naming it. It is RUNTIMEOUTPUT_GP, a key on
# MAT_Struct_Multiscale with values all / none / first_gp_only, and what it
# produces is one WHOLE OUTPUT TREE PER GAUSS POINT, named
# res_microdis<N>_el<E>_gp<G>-vtk-files/. That naming is worth pinning because it
# is the only way to tell which RVE a file belongs to.
#
# Upstream sohex8_multiscale_macro assigns MAT 1 RUNTIMEOUTPUT_GP: all, MAT 2
# none and MAT 3 first_gp_only. Flipping MAT 1 from all to none drops the
# per-Gauss-point trees from 26 to 2 while the macro run is otherwise unchanged
# and both arms exit 0 — so an agent who never sets it gets a clean, complete,
# silent run with the micro scale simply absent.
#
# Each arm runs in its own directory: 4C writes its VTU trees relative to the
# working directory.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream sohex8_multiscale_macro.4C.yaml) || exit 3
MICRO=$(upstream sohex8_multiscale_micro.mat.4C.yaml) || exit 3
cd "$TMP" || exit 3
cp "$BASE" macro.yaml
grep -q "      RUNTIMEOUTPUT_GP: all" macro.yaml || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

mkdir -p d_all d_none
cp macro.yaml d_all/
sed 's/      RUNTIMEOUTPUT_GP: all/      RUNTIMEOUTPUT_GP: none/' macro.yaml > d_none/macro.yaml
cp "$MICRO" d_all/
cp "$MICRO" d_none/

( cd d_all  && stdbuf -oL -eL "$BIN" macro.yaml res > run.log 2>&1; echo "EXIT_ALL=$?" )
( cd d_none && stdbuf -oL -eL "$BIN" macro.yaml res > run.log 2>&1; echo "EXIT_NONE=$?" )

grep -m1 -F "processor 0 finished normally" d_all/run.log
echo "ALL_TESTS_CORRECT=$(grep -c 'is CORRECT' d_all/run.log)"
echo "NONE_TESTS_CORRECT=$(grep -c 'is CORRECT' d_none/run.log)"
echo "ALL_GP_TREES=$(find d_all  -maxdepth 1 -type d -name 'res_microdis*_gp*' | wc -l)"
echo "NONE_GP_TREES=$(find d_none -maxdepth 1 -type d -name 'res_microdis*_gp*' | wc -l)"
# The tree naming carries the discretisation, element and Gauss-point index.
find d_all -maxdepth 1 -type d -name 'res_microdis1_el*_gp*-vtk-files' | head -1 | sed 's|.*/||'
# Turning the micro output off is not reported anywhere.
echo "NONE_OUTPUT_WARNINGS=$(grep -ciE 'no micro output|micro output disabled' d_none/run.log)"
exit 0
