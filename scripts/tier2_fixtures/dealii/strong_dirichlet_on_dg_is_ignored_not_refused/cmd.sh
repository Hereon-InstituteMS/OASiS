#!/bin/bash
# Tier-2 for dealii dg_transport#7 -- probes "strong_bc_on_dg", "hier_bc_crash"
# and "bern_bc_crash" of the shared DG-transport translation unit
# _shared/dgtransport_family.cc, compiled once and cached so eleven fixture
# directories share ONE C++ build.
#
# VectorTools::interpolate_boundary_values on an FE_DGQ(1) DoFHandler, into both
# a boundary-value map and an AffineConstraints, and then a solve of the same
# transport problem with the weak inflow flux LEFT OUT -- the mistake a user
# coming from CG makes. The call returns normally having written nothing, and the
# solution is identically zero: the prescribed inflow datum never enters.
#
# The contrast the entry draws is checked here too, in separate processes because
# it is fatal: on the CONTINUOUS non-interpolatory elements FE_Q_Hierarchical(2)
# and FE_Bernstein(2) the very same call SEGFAULTS (rc=139) instead of returning
# an empty map. Both elements report has_support_points=false, exactly as FE_DGQ
# does, so that flag is not what separates them.
#
# Mutation control: T2_MUTATE=1 puts the weak inflow flux back, the inflow datum
# reaches the domain, and the fixture fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SHARED="$HERE/../_shared"

for probe in hier_bc_crash bern_bc_crash; do
  echo "=== probe=$probe"
  out="$(bash "$SHARED/run.sh" dgtransport_family release "$probe" 2>&1 \
         | grep -vE '^(/media/|/lib/|/usr/lib|\[0x|#[0-9])')"
  echo "$out"
  rc="$(printf '%s\n' "$out" | sed -n 's/^exit_code=//p' | tail -1)"
  echo "summary_${probe}_rc=${rc}"
done

echo "=== probe=strong_bc_on_dg"
exec bash "$SHARED/run.sh" dgtransport_family release strong_bc_on_dg
