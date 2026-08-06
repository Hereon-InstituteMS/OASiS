#!/bin/bash
# Tier-2 for dealii wave#5 — probe "vtu_every_step_cost" of the shared wave
# translation unit _shared/wave_family.cc, compiled once and cached so six
# fixtures share one build.
#
# The entry is explicit that the section LABEL is the author's, not deal.II's,
# so the probe creates the sections ("assemble_system", "solve",
# "output_results"), writes real .vtu files to a temporary directory, and reads
# the numbers back out of TimerOutput::get_summary_data after printing
# print_summary(). It also writes the .pvd the entry recommends via
# DataOutBase::write_pvd_record.
#
# ">50% of total" is a RATIO, so what the output is compared against decides it.
# Both comparisons are run: with CG+SSOR per step the output section is well
# under half; with the one-shot factorisation the catalog recommends for a
# constant dt and mesh, the same output section is the top entry at over half.
#
# Mutation control: T2_MUTATE=1 makes the probe write every 20th step instead,
# and the fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" wave_family release vtu_every_step_cost
