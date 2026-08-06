#!/bin/bash
# Tier-2 for dealii dg_transport#2 -- probe "handrolled_forgets_periodic" of the
# shared DG translation unit _shared/dg_family.cc, compiled once and cached so
# every fixture that names a probe of that unit shares ONE C++ build.
#
# Upwind DG transport with b = (1, 0.5) on a unit square whose y = 0 and y = 1
# faces are matched by GridTools::collect_periodic_faces +
# Triangulation::add_periodicity, so the profile entering at x = 0 leaves through
# the top and must re-enter at the bottom. Two assemblies of that same problem
# run in one invocation:
#   handrolled  a cell/face loop that dispatches on cell->face(f)->at_boundary()
#               and never asks cell->has_periodic_neighbor(f). A periodic face
#               still reports at_boundary() == true, so the pair is assembled as
#               a physical boundary: 0 matrix couplings across it, the answer is
#               O(1) wrong (max difference 0.99971 on a field of amplitude ~2),
#               and the mean |u(x,1) - u(x,0)| mismatch is 0.37 against 0.016 for
#               the correct assembly -- the entry's "kinks at periodic-face
#               nodes", measured. Nothing is raised: DoFTools::
#               make_flux_sparsity_pattern DOES provide the periodic entries
#               (64 of them), the hand-rolled loop simply never writes them.
#   mesh_loop   MeshWorker::mesh_loop with the same three workers gets the
#               periodic faces for free -- its dispatch tests
#               has_periodic_neighbor() and routes the pair to the face worker.
#
# The entry's other clause, that the hand-rolled system is distinguishable by
# being "non-symmetric", is measured and is false: upwind transport is
# non-symmetric either way and the relative asymmetry of the two matrices agrees
# to three digits (1.31015 against 1.31075).
#
# Mutation control: T2_MUTATE=1 makes the probe assemble with
# MeshWorker::mesh_loop, and the fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" dg_family release handrolled_forgets_periodic
