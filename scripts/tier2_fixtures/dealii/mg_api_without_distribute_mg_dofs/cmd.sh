#!/bin/bash
# Tier-2 for dealii multigrid#0 -- probe "mg_no_level_dofs" of the shared
# multigrid translation unit _shared/multigrid_family.cc, compiled once and
# cached so five fixture directories share ONE C++ build.
#
# distribute_dofs() is called and distribute_mg_dofs() is NOT, and then exactly
# ONE mg_ API call is made -- nothing else touches an mg_ API first, so the crash
# is attributable to that call and no other. All three of the entry's calls are
# run, in both builds, in six separate processes:
#   dof_handler.n_dofs(level)              Debug 134 / Release 139
#   MGTransferPrebuilt::build(dof_handler) Debug 134 / Release 139
#   MGConstrainedDoFs::initialize(dof)     Debug 134 / Release 139
# The Release half of the entry reproduces exactly: SIGSEGV with no message at
# all.
#
# THE DEBUG HALF DOES NOT. The entry promises three DIFFERENT diagnoses. On this
# build all three produce the SAME one, and it does not come from any of the
# three named entry points -- it comes from DoFHandler::locally_owned_mg_dofs in
# dofs/dof_handler.h: "The level dofs are not set up properly! Did you call
# distribute_mg_dofs()?". The grep below shows why the transfer one cannot
# appear: the string "prerequisite for multigrid transfers" is in no file of this
# tree. ("n_dofs(level) can only be called after distribute_mg_dofs" does exist,
# in one file -- it is simply not the assert that fires first.)
#
# The entry's actual ADVICE holds: has_level_dofs() is false here, cheaply, in
# both builds, before anything crashes.
#
# Mutation control: T2_MUTATE=1 calls distribute_mg_dofs() first and every one of
# the six runs returns rc=0, so the fixture fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SHARED="$HERE/../_shared"

for s in "n_dofs(level) can only be called after distribute_mg_dofs" \
         "prerequisite for multigrid transfers" \
         "The level dofs are not set up properly"; do
  echo -n "message_files[$s]="
  grep -rl "$s" /home/alexander/dealii/include /home/alexander/dealii/source \
    2>/dev/null | wc -l
done

for variant in release debug; do
  for sub in n_dofs transfer constrained; do
    echo "=== variant=$variant call=$sub"
    out="$(bash "$SHARED/run.sh" multigrid_family "$variant" mg_no_level_dofs "$sub" 2>&1 \
           | grep -vE '^(/media/|/lib/|/usr/lib|\[0x|#[0-9])')"
    echo "$out"
    rc="$(printf '%s\n' "$out" | sed -n 's/^exit_code=//p' | tail -1)"
    echo "summary_${variant}_${sub}_rc=${rc}"
  done
done
