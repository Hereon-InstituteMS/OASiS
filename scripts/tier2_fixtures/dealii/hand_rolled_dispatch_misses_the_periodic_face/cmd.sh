#!/bin/bash
# Tier-2 for dealii dg_transport#2 -- probe "mesh_loop_dispatch" of the shared DG-transport
# translation unit _shared/dgtransport_family.cc, compiled once and cached so
# eleven fixture directories share ONE C++ build.
#
# A y-PERIODIC mesh (GridTools::collect_periodic_faces + add_periodicity), upwind
# DG transport with beta = (1,1) and the exact solution sin(2 pi (y-x)). The
# periodic face answers TRUE to cell->at_boundary(), so a hand-written dispatch
# puts inflow/outflow flux terms on it and never couples the two sides;
# MeshWorker::mesh_loop asks has_periodic_neighbor() and hands the same face to
# the face worker.
#
# The face-visit counts are MEASURED, not assumed: with
# assemble_own_interior_faces_once mesh_loop visits each interior face exactly
# once (periodic pairs included), and with assemble_own_interior_faces_both
# exactly twice.
#
# The entry also offers "the global system is non-symmetric" as a signal. It is
# not one: the CORRECT mesh_loop assembly of an upwind advection operator is
# just as non-symmetric, and the run prints both defects.
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" dgtransport_family release mesh_loop_dispatch
