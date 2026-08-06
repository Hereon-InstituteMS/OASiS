#!/bin/bash
# Tier-2 for fourc::ehl#6 — a viscosity unit slip on a coupled EHL problem does
# not give you a quietly rescaled answer.  It destroys the linear algebra, and
# WHERE it dies depends on the very scaling the entry recommends.
#
# Upstream ehl3d_mixed.4C.yaml runs a monolithic EHL with
# MAT_lubrication_law_constant/VISCOSITY 4e-08 and passes seven result tests.
# Multiply the viscosity by 1000 — the size of a mPa.s-vs-Pa.s slip — and:
#
#   INFNORMSCALING on (the 4C default, and the "numerical scaling helps
#   conditioning" the entry recommends):
#       residual reaches 2.97939e+11, then
#       Signal: Floating point exception (8) / Invalid floating point operation
#       inside Epetra_CrsMatrix::InvRowSums, called from
#       EHL::Monolithic::scale_system — the row-sum scaling itself overflows.
#
#   INFNORMSCALING off:
#       the same deck dies instead inside umfdi_kernel_init, i.e. in UMFPACK's
#       factorisation.
#
# Either way the shell sees 136 and 4C prints no error line at all.  The lesson
# the entry gets right is that units matter; what it misses is that the failure
# is a process kill in a third-party library, not a "silent scaling error".
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ehl3d_mixed.4C.yaml) || exit 3
grep -q '      VISCOSITY: 4e-08' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_viscosity_changed"; exit 3; }
grep -q '^ELASTO HYDRO DYNAMIC/MONOLITHIC:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_monolithic_section_changed"; exit 3; }

# The pathology: the viscosity the bad arms are given.
BAD_VISCOSITY=4e-05

cp "$BASE" "$TMP/si.yaml"
sed "s/      VISCOSITY: 4e-08/      VISCOSITY: $BAD_VISCOSITY/" "$BASE" > "$TMP/slip.yaml"
sed 's|^ELASTO HYDRO DYNAMIC/MONOLITHIC:|ELASTO HYDRO DYNAMIC/MONOLITHIC:\n  INFNORMSCALING: false|' \
    "$TMP/slip.yaml" > "$TMP/slip_noscale.yaml"
grep -m1 '      VISCOSITY:' "$TMP/slip.yaml" | tr -d ' ' | sed 's/^/SLIP_ARM_/'

probe SI           "$TMP/si.yaml"
probe SLIP         "$TMP/slip.yaml"
probe SLIP_NOSCALE "$TMP/slip_noscale.yaml"

grep -m1 -F "OK (7)" "$TMP/SI.log"
grep -m1 -F "processor 0 finished normally" "$TMP/SI.log"
grep -m1 -F "Signal: Floating point exception (8)" "$TMP/SLIP.log"
grep -m1 -F "Signal code: Invalid floating point operation (7)" "$TMP/SLIP.log"
grep -m1 -oF "InvRowSums" "$TMP/SLIP.log"
grep -m1 -oF "scale_system" "$TMP/SLIP.log"
grep -m1 -oF "umfdi_kernel_init" "$TMP/SLIP_NOSCALE.log"

# No 4C-side message and no result-test verdict: the process is killed by the
# signal, so MPI_Abort and 4C's own error path never run.
echo "SLIP_4C_ERROR_LINES=$(grep -c 'PROC 0 ERROR in' "$TMP/SLIP.log")"
echo "SLIP_MPI_ABORT_LINES=$(grep -c 'MPI_ABORT was invoked' "$TMP/SLIP.log")"
echo "SLIP_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/SLIP.log")"
# The two arms die in different libraries, which is the conditioning story.
echo "SLIP_DIES_IN_ROW_SCALING=$(grep -c 'InvRowSums' "$TMP/SLIP.log")"
echo "SLIP_NOSCALE_DIES_IN_UMFPACK=$(grep -c 'umfdi_kernel_init' "$TMP/SLIP_NOSCALE.log")"
exit 0
