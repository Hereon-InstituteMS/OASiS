#!/bin/bash
# Tier-2 for dealii wave#1 — probe "reassemble_each_step" of the shared wave
# translation unit _shared/wave_family.cc, compiled once and cached so six
# fixtures share one build.
#
# Two loops on the same problem in one process, the reference (assemble once)
# timed FIRST and after an untimed warm-up so the ordering works against the
# result: re-assembling M, K and the effective matrix inside the time loop costs
# several times the wall time of assembling them once, for an identical answer
# (the CG iteration counts of the two loops are equal).
#
# The entry's two quantitative clauses are measured rather than repeated:
# "assemble_system dominating at 60-80%" and "scaling as O(ndof^2)". Both come
# out false here, and the fixture pins them false.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant
# (assemble once), and the fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" wave_family release reassemble_each_step
