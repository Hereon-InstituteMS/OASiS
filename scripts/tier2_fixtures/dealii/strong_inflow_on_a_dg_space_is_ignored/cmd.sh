#!/bin/bash
# Tier-2 for dealii dg_transport#7 -- probe "dg_strong_dirichlet" of the shared DG
# translation unit _shared/dg_family.cc, compiled once per build type and cached
# so every fixture that names a probe of that unit shares ONE C++ build.
#
# VectorTools::interpolate_boundary_values on an FE_DGQ(1) DoFHandler, by both
# routes -- into a std::map and straight into an AffineConstraints. Both return
# normally and both write NOTHING: boundary_values_size=0 and n_constraints=0 on
# a mesh whose whole boundary carries id 0. The same calls on FE_Q(1) write 16
# entries each, which is the mutation.
#
# The entry's downstream half is then shown on the DG space: the inflow datum is
# "set" through that empty map instead of through the numerical flux, and the
# transport solve returns the zero vector -- the prescribed value 1 appears
# nowhere in the answer.
#
# Both build types run, because the entry says this is silent in each: the Debug
# library reproduces the Release output line for line and exits 0. There is no
# Assert to switch on. The grep checks the entry's other statement, that the
# ExcMessage it used to promise, "strong boundary conditions not supported for
# DG", is in no header or source of this tree.
#
# Mutation control: T2_MUTATE=1 makes the probe use FE_Q(1) instead, where the
# strong route is the right one, and the fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SHARED="$HERE/../_shared"

echo -n "phantom_exception_string_files="
grep -rl "strong boundary conditions not supported for DG" \
  /home/alexander/dealii/include /home/alexander/dealii/source 2>/dev/null | wc -l

for variant in release debug; do
  echo "=== variant=$variant"
  out="$(bash "$SHARED/run.sh" dg_family "$variant" dg_strong_dirichlet 2>&1)"
  echo "$out"
  rc="$(printf '%s\n' "$out" | sed -n 's/^exit_code=//p' | tail -1)"
  echo "summary_${variant}_rc=${rc}"
  if printf '%s\n' "$out" | grep -q "^strong_route_wrote_nothing=true$"; then
    echo "summary_${variant}_strong_route_wrote_nothing=true"
  else
    echo "summary_${variant}_strong_route_wrote_nothing=false"
  fi
done
