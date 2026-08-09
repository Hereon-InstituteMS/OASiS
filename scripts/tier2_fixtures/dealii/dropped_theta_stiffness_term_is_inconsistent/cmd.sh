#!/bin/bash
# Tier-2 for dealii heat#1 — probe "dropped_theta_term" of the shared transient-heat translation
# unit _shared/heat_family.cc, compiled once and cached so several fixtures
# share one build.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" heat_family release dropped_theta_term
