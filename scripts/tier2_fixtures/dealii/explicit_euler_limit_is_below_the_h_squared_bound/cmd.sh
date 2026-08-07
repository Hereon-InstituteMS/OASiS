#!/bin/bash
# Tier-2 for dealii time_dependent_heat#1 -- probe "forward_euler_stability" of
# the shared transient-heat translation unit _shared/heat_family.cc, which is
# compiled once and cached, so this fixture adds no build of its own.
#
# theta = 0 (forward Euler) on the unit square with alpha = 1 and h = 1/16, a
# FIXED number of steps at each dt so a larger step cannot hide the instability
# by finishing sooner. The stability limit is found by scanning dt, and it lands
# BELOW the textbook finite-difference bound h^2/(2 alpha): the bound is about
# three times the measured limit, so a dt that satisfies "dt < h^2/(2 alpha)"
# still blows up with a consistent (non-lumped) FE mass matrix.
#
# THE SUNDIALS HALF OF THE ENTRY IS NOT TESTABLE ON THIS INSTALL. It names
# SUNDIALS::ARKode step rejection and SUNDIALS::IDA; this deal.II was configured
# without SUNDIALS, which the grep below reads straight out of config.h. The
# contrast the entry draws is instead made with theta = 1 inside the probe.
#
# T2_MUTATE=1 switches to theta = 1 (backward Euler), nothing blows up at any dt
# tried, and the fixture fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"

CONFIG=/home/alexander/dealii/build/include/deal.II/base/config.h
if grep -q "^#define DEAL_II_WITH_SUNDIALS" "$CONFIG"; then
  echo "deal_ii_with_sundials=true"
else
  echo "deal_ii_with_sundials=false"
fi
grep -m1 "DEAL_II_WITH_SUNDIALS" "$CONFIG"

exec bash "$HERE/../_shared/run.sh" heat_family release forward_euler_stability
