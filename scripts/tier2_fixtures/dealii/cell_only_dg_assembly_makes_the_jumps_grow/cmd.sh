#!/bin/bash
# Tier-2 for dealii dg_transport#1 -- probe "feinterface_face_terms" of the
# shared DG-transport translation unit _shared/dgtransport_family.cc, compiled
# once and cached so eleven fixture directories share ONE C++ build.
#
# Upwind DG for beta.grad(u) + u = f on a 16x16 FE_DGQ(1) mesh with a smooth
# manufactured solution, assembled ONCE with the FEInterfaceValues face loop and
# once without it, in the same run. Without it the matrix has ZERO entries
# joining dofs of different cells.
#
# The Signal in the entry does NOT reproduce: it says the face jumps come out at
# ~1e-8, "a smooth (non-DG) solution". Measured, the cell-only operator leaves
# the jumps four orders of magnitude LARGER than the coupled reference, and the
# L2 error with them -- the cells are not smoothly joined, they are unjoined.
#
# The grep below checks the entry's other statement, that the function is called
# integrate_difference and that interpolate_difference does not exist. Both hold.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"

echo -n "phantom_interpolate_difference_files="
grep -rl "interpolate_difference" /home/alexander/dealii/include \
  /home/alexander/dealii/source 2>/dev/null | wc -l
if grep -rq "integrate_difference" \
     /home/alexander/dealii/include/deal.II/numerics/vector_tools_integrate_difference.h; then
  echo "integrate_difference_is_the_real_name=true"
else
  echo "integrate_difference_is_the_real_name=false"
fi

exec bash "$HERE/../_shared/run.sh" dgtransport_family release feinterface_face_terms
